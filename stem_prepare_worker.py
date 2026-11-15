"""Isolated, user-confirmed downloader for one public UVR model.

Works with managed CPython -I -S using ONLY standard library modules; the
untrusted network cannot decide the model hash or destination.  Progress and
errors are written to plugin-owned JSON, never host logs with private data.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time
import uuid


def _write_status(path:Path, **fields):
    status={'schema':'music_stem_download_v1','updated_at':int(time.time()),**fields}
    path.parent.mkdir(parents=True,exist_ok=True)
    # Do not terminate a download because Hosted UI/antivirus briefly locked
    # a progress JSON.  Model SHA validation remains completely independent.
    for attempt in range(5):
        temp=path.with_name(path.name+'.'+uuid.uuid4().hex[:10]+'.tmp')
        try:
            temp.write_text(json.dumps(status,ensure_ascii=False),encoding='utf-8')
            os.replace(temp,path)
            break
        except OSError:
            time.sleep(.05*(attempt+1))
        finally:
            try:temp.unlink(missing_ok=True)
            except OSError:pass


def source_urls(priority, quality='ultra'):
    # Keep the previously released Ultra pinned revision and its three URLs.
    # This preserves existing resumable downloads after an in-place upgrade.
    if quality == 'ultra':
        file='/Politrees/UVR_resources/resolve/495f5b4625eb092fe1fbe336f0b67c847566e52e/MDXNet_models/UVR-MDX-NET-Inst_HQ_3.onnx?download=true'
        sources=[('Hugging Face 官方','https://huggingface.co'+file),
                 ('HF-Mirror 公益镜像','https://hf-mirror.com'+file),
                 ('HF-Mirror 备用节点','https://hf-mirror.net'+file)]
        if priority=='mirror':return [sources[1],sources[2],sources[0]]
        if priority=='official':return sources
        raise ValueError('无效下载来源')
    try:
        from .stem_catalog import urls_for
    except ImportError:
        from stem_catalog import urls_for
    return urls_for(quality,priority)


def main(argv=None):
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) not in (3,4):return 2
    model_dir,status_path,priority=args[:3]
    quality=args[3] if len(args)==4 else 'ultra'
    # All allowed destinations, hashes and URLs are defined by our catalogue.
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from stem_catalog import spec, verify_local_model
    profile=spec(quality)
    filename=profile['file']
    directory=Path(model_dir)
    target=directory/filename
    status_file=Path(status_path)
    from lyrics_model_store import _stream_to_file
    from download_telemetry import TransferMeter
    directory.mkdir(parents=True,exist_ok=True)
    if verify_local_model(target,quality):
        _write_status(status_file,state='ready',stage='sha256_verified',pct=100,
                      model=quality,bytes_received=target.stat().st_size,source='本地缓存',sha256_verified=True,model_bytes=target.stat().st_size,model_mtime_ns=target.stat().st_mtime_ns)
        return 0
    errors=[]
    expected=profile['bytes_estimate']
    for source_label,url in source_urls(priority,quality):
        current={'bytes_received':0, 'expected_bytes_estimate':expected, 'source':source_label,
                 'state':'running', 'stage':'connecting', 'sha256_verified':False}
        def update(**fields):
            current.update(fields)
            _write_status(status_file,model=quality,**current)
        update()
        meter=TransferMeter()
        try:
            def progress(received,total,fresh):
                actual=int(received)
                denom=int(total) if total else expected
                stats=meter.update(actual,denom,restarted=bool(fresh),estimated=not bool(total))
                update(state='running',stage='downloading',bytes_received=actual,
                       expected_bytes_estimate=denom,pct=stats['progress_pct'],
                       speed_bytes_s=stats['speed_bytes_s'],eta_seconds=stats['eta_seconds'],
                       total_bytes=stats['total_bytes'],stalled=stats['stalled'],
                       last_progress_at=stats['last_progress_at'],
                       transfer_restarted=bool(fresh))
            def verifying(done,total):
                update(state='running',stage='verifying',verification_pct=round(100*done/max(1,total)))
            _stream_to_file(url,target,
                           min_bytes=int(expected*0.9),max_bytes=int(expected*1.1),
                           expected_sha256=profile['sha256'],
                           transfer_progress=progress,verification_progress=verifying)
            if not verify_local_model(target,quality):
                raise ValueError('完整模型的SHA-256最终检查未通过')
            update(state='ready',stage='sha256_verified',bytes_received=target.stat().st_size,
                   expected_bytes_estimate=target.stat().st_size,pct=100,
                   verification_pct=100,sha256_verified=True,model_bytes=target.stat().st_size,model_mtime_ns=target.stat().st_mtime_ns)
            return 0
        except Exception as exc:
            # Only local generic error codes appear in UI: never leak proxy
            # URLs, credentials, or local user names from urllib exceptions.
            errors.append(source_label+'：'+type(exc).__name__)
            update(state='running',stage='trying_next',error=errors[-1])
    _write_status(status_file,state='failed',stage='failed',pct=0,sha256_verified=False,
                  model=quality,bytes_received=max((p.stat().st_size for p in (
                     target.with_name(filename+'.partial'),
                     target.with_name(filename+'.restart')) if p.is_file()),default=0),
                  expected_bytes_estimate=expected,
                  message='所有来源均未取得校验通过的模型；可切换下载源继续下载',
                  sources_failed=errors[:3])
    return 1


if __name__=='__main__':
    sys.exit(main())
