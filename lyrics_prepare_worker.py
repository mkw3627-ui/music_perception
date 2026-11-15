"""Prepare a user-selected multilingual Whisper model independently of music gating.

Run under the managed CPython -I -S and isolated vendor.  No N.E.K.O SDK,
no audio processing and no authentication secrets in the status file.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import threading
import time
import types
import uuid
from pathlib import Path

_report_lock = threading.Lock()


def report(path: Path, model: str, stage: str, state: str, note: str = '', *,
           source: str = '', source_name: str = '', filename: str = '', phase: str = '',
           received_bytes: int | None = None, estimated_total: int | None = None,
           progress_pct: int | None = None, verification_pct: int | None = None,
           weight_bytes: int | None = None, weight_mtime_ns: int | None = None,
           sidecar_signature: list[dict] | None = None,
           speed_bytes_s: float | None = None, eta_seconds: float | None = None,
           total_bytes: int | None = None, stalled: bool | None = None,
           last_progress_at: float | None = None):
    record = {'schema': 'music_model_prepare_v1', 'model': model,
              'stage': stage, 'state': state, 'updated_at': int(time.time())}
    if note:
        record['message'] = note[:350]
    if source_name: record['source_name'] = source_name[:60]
    if source: record['source'] = source[:70]
    if filename: record['filename'] = filename[:80]
    if phase: record['phase'] = phase[:40]
    if weight_bytes is not None and weight_mtime_ns is not None:
        record['weight_bytes'] = int(weight_bytes)
        record['weight_mtime_ns'] = int(weight_mtime_ns)
    if sidecar_signature is not None:
        record['sidecar_signature'] = sidecar_signature
    if received_bytes is not None:
        record['received_bytes'] = max(0, int(received_bytes))
    if estimated_total is not None:
        record['expected_bytes_estimate'] = max(0, int(estimated_total))
        record['progress_estimated'] = (state != 'ready')
    if speed_bytes_s is not None: record['speed_bytes_s'] = max(0,float(speed_bytes_s))
    if eta_seconds is not None: record['eta_seconds'] = max(0,float(eta_seconds))
    if total_bytes is not None: record['total_bytes'] = max(0,int(total_bytes))
    if stalled is not None: record['stalled'] = bool(stalled)
    if last_progress_at is not None: record['last_progress_at'] = float(last_progress_at)
    if progress_pct is not None:
        record['progress_pct'] = max(0, min(100, int(progress_pct)))
    if verification_pct is not None:
        record['verification_pct'] = max(0, min(100, int(verification_pct)))
    with _report_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Windows antivirus/readers can transiently hold a status file open.
        # Never abort a valid multi-GB download just because its progress JSON
        # could not be replaced in this one sampling interval.
        for attempt in range(5):
            tmp=path.with_name(path.name+'.'+uuid.uuid4().hex[:10]+'.tmp')
            try:
                tmp.write_text(json.dumps(record,ensure_ascii=False),encoding='utf-8')
                os.replace(tmp,path)
                break
            except OSError:
                time.sleep(0.05*(attempt+1))
            finally:
                try:tmp.unlink(missing_ok=True)
                except OSError:pass
    return record


def main() -> int:
    if len(sys.argv) != 6:
        return 2
    vendor, model_root, model, target, mirror = sys.argv[1:]
    if model not in {'tiny', 'base', 'small', 'large-v3-turbo'} or mirror not in {'0', '1'}:
        return 2
    target = Path(target)
    sys.path.insert(0, str(Path(vendor).resolve()))
    prior_package = sys.modules.get('music_perception')
    package = types.ModuleType('music_perception')
    package.__path__ = [str(Path(__file__).resolve().parent)]
    package.__package__ = 'music_perception'
    sys.modules['music_perception'] = package
    from music_perception.download_telemetry import TransferMeter
    transfer_meter = TransferMeter()
    finished = threading.Event()
    observer = None
    transfer = {'phase':'checking_existing','source':'','source_name':'','filename':''}
    transfer_lock = threading.Lock()
    try:
        from music_perception.lyrics_model_store import (
            EXPECTED_DOWNLOAD_BYTES,
            ensure_local_model,
            observed_download_bytes,
            sidecar_stat_signature,
        )
        total = EXPECTED_DOWNLOAD_BYTES[model]
        report(target, model, 'checking_existing', 'running',
               '正在核对已有权重；未确认完整之前不能启用模型',
               received_bytes=observed_download_bytes(model, Path(model_root)),
               estimated_total=total)

        def on_transfer(event):
            # The direct HTTPS streamer knows which file is currently being
            # written. Ignore any old 99%-complete .partial while a clean
            # .restart transfer is in progress.
            explicit_bytes=event.get('received_bytes')
            with transfer_lock:
                transfer.update({key: event.get(key,'') for key in
                                 ('phase','source','source_name','filename')})
                transfer['streaming'] = isinstance(explicit_bytes,int)
            done=event.get('verification_bytes')
            total_hash=event.get('verification_total')
            verification=(int(100 * done / total_hash)
                          if isinstance(done,int) and isinstance(total_hash,int) and total_hash else None)
            received=(explicit_bytes if isinstance(explicit_bytes,int) else
                      observed_download_bytes(model,Path(model_root)))
            actual_total=event.get('expected_bytes')
            estimate=int(actual_total) if isinstance(actual_total,int) and actual_total>0 else total
            # 100% means FULL SHA verified, never merely EOF or last network byte.
            stats=transfer_meter.update(received,estimate,restarted=bool(event.get('restarting')),
                                        estimated=not (isinstance(actual_total,int) and actual_total>0))
            pct=stats['progress_pct'] if stats['progress_pct'] is not None else 0
            restarting=bool(event.get('restarting'))
            note=('正在校验 SHA-256 权重' if verification is not None else
                  '重新完整下载，旧断点已保留' if restarting else
                  '正在尝试 '+event.get('source_name','公开模型源'))
            report(target,model,event.get('phase','connecting'),'running',note,
                   source=event.get('source',''),source_name=event.get('source_name',''),
                   filename=event.get('filename',''),phase=event.get('phase','connecting'),
                   received_bytes=received,estimated_total=estimate,
                   speed_bytes_s=stats['speed_bytes_s'],eta_seconds=stats['eta_seconds'],
                   total_bytes=stats['total_bytes'],stalled=stats['stalled'],
                   last_progress_at=stats['last_progress_at'],
                   verification_pct=verification,
                   progress_pct=99 if verification is not None else pct)

        def observe_bytes():
            last = (-1, -1)
            while not finished.wait(.35):
                received = observed_download_bytes(model, Path(model_root))
                # Percent is an approximation until SHA-256 succeeds, not a
                # claim about bytes successfully downloaded from a server.
                pct = min(99, int(received * 100 / total)) if received else None
                current = (received, pct)
                if current != last:
                    with transfer_lock:
                        phase=transfer['phase']; source=transfer['source']
                        source_name=transfer['source_name']; filename=transfer['filename']
                        streaming=transfer.get('streaming',False)
                    if phase == 'verifying_sha' or streaming:
                        # SHA owns its progress; direct HTTPS owns real bytes
                        # even when a corrupt 99% .partial still exists.
                        last = current
                        continue
                    report(target, model, phase, 'running',
                           '公开模型文件：'+(source_name or '读取本地缓存'),
                           source=source,source_name=source_name,filename=filename,
                           phase=phase,received_bytes=received,
                           estimated_total=total,progress_pct=pct)
                    last = current

        observer = threading.Thread(target=observe_bytes, name='lyrics-download-progress', daemon=True)
        observer.start()
        # Healthy local models are verified and returned without any download.
        # Some tests may supply a simplified model-store stub. Production
        # functions accept a progress callback to report actual source status.
        kwargs={'allow_public_mirror': mirror == '1'}
        if 'progress_callback' in inspect.signature(ensure_local_model).parameters:
            kwargs['progress_callback']=on_transfer
        verified_dir = ensure_local_model(model, Path(model_root),**kwargs)
        verified_stat = (Path(verified_dir)/'model.bin').stat() if (Path(verified_dir)/'model.bin').is_file() else None
        finished.set()
        observer.join(timeout=1.5)
        report(target, model, 'sha256_verified', 'ready',
               '权重完整且 SHA-256 通过；请进入下一步加载与试推理',
               received_bytes=observed_download_bytes(model, Path(model_root)),
               estimated_total=total, progress_pct=100,
               weight_bytes=verified_stat.st_size if verified_stat else None,
               weight_mtime_ns=verified_stat.st_mtime_ns if verified_stat else None,
               sidecar_signature=sidecar_stat_signature(verified_dir,model))
        return 0
    except Exception as exc:
        # Model-store errors are deliberately composed of public model IDs,
        # endpoint classes and verified file status; expose a short reason.
        # Never dump arbitrary third-party exception text (which can contain
        # user names, filesystem paths, proxy URLs or access credentials).
        from music_perception.lyrics_model_store import LyricsModelDownloadError
        if isinstance(exc, LyricsModelDownloadError):
            # This message is composed by the allowlisted downloader using
            # PUBLIC source titles, typed exceptions, and local file status;
            # never forward raw urllib/HF errors that may contain proxy URLs.
            raw = str(exc)
            if '（' in raw and '）' in raw:
                safe_source_results = raw.split('（',1)[1].rsplit('）',1)[0]
                note = '模型下载失败（各源诊断）：' + safe_source_results[:285]
            else:
                note = '模型下载/校验失败：' + raw.split('。',1)[0]
        elif isinstance(exc, PermissionError):
            note = 'PermissionError：模型目录被 Windows 拒绝访问，请检查文件占用或权限'
        else:
            note = type(exc).__name__ + '：模型文件尚未准备完成；请检查网络与模型缓存'
        try:
            received = observed_download_bytes(model, Path(model_root))
            total = EXPECTED_DOWNLOAD_BYTES[model]
        except (NameError, OSError, ValueError):
            received, total = None, None
        report(target, model, 'failed', 'failed', note,
               received_bytes=received, estimated_total=total)
        return 1
    finally:
        finished.set()
        if observer is not None and observer.is_alive():
            observer.join(timeout=1.5)
        # In the isolated CPython process this is not required, but direct
        # unit tests and reusable callers should not leak the fake package.
        if prior_package is None:
            sys.modules.pop('music_perception', None)
        else:
            sys.modules['music_perception'] = prior_package


if __name__ == '__main__':
    raise SystemExit(main())
