"""Independent sung-lyrics models, never shared with the video speech worker.

Four multilingual profiles. Tiers indicate resource / intended use, NOT a
promise that a larger model always has lower lyric CER/WER on every recording.
Public checkpoints only; no copyrighted lyrics database is bundled.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path

PROFILES = {
    'low': {'id':'sing-whisper-small','repo':'Systran/faster-whisper-small',
            'engine':'faster_whisper','name':'Whisper Small · 轻量歌声基线',
            'download_bytes_estimate':484_000_000,'suggested_free_vram_mib':2300,
            'minimum_weight_bytes':400_000_000},
    'medium': {'id':'sing-qwen-0.6b','repo':'Qwen/Qwen3-ASR-0.6B',
            'engine':'qwen_asr','name':'Qwen3-ASR 0.6B · 歌声识别',
            'download_bytes_estimate':1_900_000_000,'suggested_free_vram_mib':4200,
            'minimum_weight_bytes':1_200_000_000},
    'high': {'id':'sing-whisper-large-v3','repo':'Systran/faster-whisper-large-v3',
            'engine':'faster_whisper','name':'Whisper Large-v3 · 多语言精细转写',
            'download_bytes_estimate':3_100_000_000,'suggested_free_vram_mib':6500,
            'minimum_weight_bytes':2_500_000_000},
    'ultra': {'id':'sing-qwen-1.7b','repo':'Qwen/Qwen3-ASR-1.7B',
            'engine':'qwen_asr','name':'Qwen3-ASR 1.7B · 歌声专业档',
            'download_bytes_estimate':4_800_000_000,'suggested_free_vram_mib':9000,
            'minimum_weight_bytes':4_000_000_000},
}
MODEL_BY_ID = {entry['id']: dict(entry, quality=key) for key,entry in PROFILES.items()}
# Immutable hashes checked against model authors' public Hugging Face LFS
# metadata on 2026-10-21. They also allow Chinese mirror-first downloads
# when the official metadata API is inaccessible.
PINNED_WEIGHT_HASHES = {
 'low': {'model.bin':'3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671'},
 'medium': {'model.safetensors':'79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea'},
 'high': {'model.bin':'69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1'},
 'ultra': {'model-00001-of-00002.safetensors':'a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6',
           'model-00002-of-00002.safetensors':'6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc'},
}
PINNED_REVISION = {
 'low':'03feb38459a0c9abc5336827440d32db28b7c373',
 'medium':'9ba1d4a347453a45ef3fa6baabe6a1a5a132397e',
 'high':'edaa852ec7e145841d8ffdb056a99866b5f0a478',
 'ultra':'e91b24b78a0456a6b0e8d41c964277829fab1c43',
}
# 魔搭（ModelScope）上同名仓库的权重 SHA-256 与 PINNED_WEIGHT_HASHES 完全一致
# （2026-11 用魔搭文件 API 逐个核对）。国内实测 10 MB/s 以上，镜像优先时排第一。
MODELSCOPE = 'https://www.modelscope.cn'
MODELSCOPE_FILES = {
 'low': ('config.json','tokenizer.json','vocabulary.txt','model.bin'),
 'medium': ('chat_template.json','config.json','generation_config.json','merges.txt',
            'preprocessor_config.json','tokenizer_config.json','vocab.json','model.safetensors'),
 'high': ('config.json','preprocessor_config.json','tokenizer.json','vocabulary.json','model.bin'),
 'ultra': ('chat_template.json','config.json','generation_config.json','merges.txt',
           'preprocessor_config.json','tokenizer_config.json','vocab.json',
           'model.safetensors.index.json','model-00001-of-00002.safetensors',
           'model-00002-of-00002.safetensors'),
}
MANIFEST_NAME = '.neko_singing_sha256.json'
ACCEPTED_EXT = {'.json','.txt','.model','.tiktoken','.safetensors','.bin'}

class SingingModelError(RuntimeError):
    pass


def model_folder(root: Path, quality: str) -> Path:
    if quality not in PROFILES:
        raise ValueError('Unknown singing tier')
    return Path(root) / PROFILES[quality]['id']


def _digest(path: Path) -> str:
    hasher=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024*4),b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_rel(filename: str):
    part=Path(filename)
    if (part.is_absolute() or len(part.parts)!=1 or filename.startswith('.') or
            part.suffix not in ACCEPTED_EXT or '\\' in filename):
        raise SingingModelError('Invalid model filename')


def verified_folder(root: Path, quality: str, *, deep: bool = False) -> Path | None:
    """Always gate inference on an installed, bounded, immutable model manifest.

    deep=True is used on activation, sha256-checking every sidecar and weight.
    Normal UI refreshes compare file identity (size, modification time) only;
    that is not treated as a fresh independent SHA verdict.
    """
    folder=model_folder(root,quality)
    try:
        manifest=json.loads((folder/MANIFEST_NAME).read_text('utf-8'))
        spec=PROFILES[quality]
        if (manifest.get('schema')!='singing-model-v1' or
                manifest.get('repo')!=spec['repo'] or
                not isinstance(manifest.get('revision'),str) or
                manifest['revision']!=PINNED_REVISION[quality] or
                not isinstance(manifest.get('files'),dict)):
            return None
        files=manifest['files']
        pinned=PINNED_WEIGHT_HASHES[quality]
        if (not all(name in files and
                isinstance(files[name],dict) and
                files[name].get('sha256')==sha
                for name,sha in pinned.items())):
            return None
        if len(files)>24 or len(files)<3:
            return None
        for pinned_name,pinned_digest in PINNED_WEIGHT_HASHES[quality].items():
            info=files.get(pinned_name)
            if not isinstance(info,dict) or info.get('sha256')!=pinned_digest:
                return None
        total_weight=0
        for name,info in files.items():
            _validate_rel(name)
            file=folder/name
            if file.is_symlink() or not file.is_file() or not isinstance(info,dict):
                return None
            stat=file.stat()
            if (stat.st_size!=info.get('bytes') or
                    (not deep and stat.st_mtime_ns!=info.get('mtime_ns')) or
                    (deep and _digest(file)!=info.get('sha256'))):
                return None
            if name.endswith(('.bin','.safetensors')):
                total_weight+=stat.st_size
        if total_weight < spec['minimum_weight_bytes'] or not (folder/'config.json').is_file():
            return None
        if spec['engine']=='faster_whisper' and not (folder/'model.bin').is_file():
            return None
        if spec['engine']=='qwen_asr' and not any(f.endswith('.safetensors') for f in files):
            return None
        return folder
    except (OSError,ValueError,KeyError,TypeError,SingingModelError):
        return None


def inventory(root: Path) -> dict:
    result={}
    for quality, spec in PROFILES.items():
        folder=model_folder(root,quality)
        installed=verified_folder(root,quality,deep=False) is not None
        result[quality]={**spec,'quality':quality,
                         'state':'verified' if installed else ('incomplete' if folder.exists() else 'missing'),
                         'sha256_verified':installed,
                         'disk_required_bytes_estimate':int(spec['download_bytes_estimate']*2.15),
                         'gpu_vram_note':'显存仅为规划建议；以实际推理/可用显存为准'}
    return result


def _create_verified_manifest(folder: Path, spec: dict, revision: str, lfs_shas: dict | None = None):
    files={}
    weight=0
    for file in folder.iterdir():
        if file.is_dir() or file.name==MANIFEST_NAME or file.name.startswith('.'):
            continue
        _validate_rel(file.name)
        if file.is_symlink():
            raise SingingModelError('Refusing model symlink')
        stat=file.stat()
        digest=_digest(file)
        # Where available compare with the upstream git-lfs SHA, not merely
        # a self-generated digest of bytes served by a mirror.
        expected=(lfs_shas or {}).get(file.name)
        if file.suffix in {'.bin','.safetensors'}:
            if not expected or expected.lower()!=digest.lower():
                raise SingingModelError('模型权重 SHA-256 与官方 LFS 元数据不一致：'+file.name)
            weight+=stat.st_size
        files[file.name]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':digest}
    if len(files)<3 or weight<spec['minimum_weight_bytes']:
        raise SingingModelError('Weight file missing or too small')
    manifest={'schema':'singing-model-v1','repo':spec['repo'],
              'revision':revision,'files':files}
    staged=folder/(MANIFEST_NAME+'.new')
    try:
        staged.write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8')
        os.replace(staged,folder/MANIFEST_NAME)
    finally:
        staged.unlink(missing_ok=True)


def _download_modelscope(stage: Path, quality: str, *, lfs: dict, on_bytes=None):
    """逐个文件从魔搭直链下载到 stage；权重用固定 SHA 校验，支持断点续传。"""
    try:
        from .lyrics_model_store import _stream_to_file
    except ImportError:
        from lyrics_model_store import _stream_to_file
    from urllib.parse import quote
    spec=PROFILES[quality]
    stage.mkdir(parents=True,exist_ok=True)
    done=0
    for name in MODELSCOPE_FILES[quality]:
        _validate_rel(name)
        target=stage/name
        url=MODELSCOPE+'/models/'+spec['repo']+'/resolve/master/'+quote(name,safe='')
        pinned=lfs.get(name)
        if pinned and target.is_file() and _digest(target)==pinned:
            done+=target.stat().st_size
            continue
        base=done
        def progress(received,_total,_fresh,base=base):
            if on_bytes:
                on_bytes(base+int(received))
        if pinned:
            _stream_to_file(url,target,min_bytes=1_000_000,max_bytes=6_000_000_000,
                            expected_sha256=pinned,transfer_progress=progress)
        else:
            _stream_to_file(url,target,min_bytes=2,max_bytes=20_000_000)
            head=target.read_bytes()[:256].lstrip().lower()
            if head.startswith((b'<html',b'<!doctype',b'{"code"',b'{"error"')):
                target.unlink(missing_ok=True)
                raise SingingModelError('魔搭返回了错误页而不是模型文件：'+name)
        done+=target.stat().st_size
    return done


def acquire_model(root: Path, quality: str, *, source='official', on_stage=None, on_progress=None):
    """Install only after user consent; HTTPS Hub snapshot pinned to a commit.

    Keep old verified folder unchanged on download error. We do not download
    any .py scripts and never enable trust_remote_code.
    """
    if quality not in PROFILES or source not in {'mirror','official'}:
        raise ValueError('Invalid model choice')
    old=verified_folder(root,quality,deep=True)
    if old is not None:
        return {'state':'verified','reused':True,'model':PROFILES[quality]['id']}
    from huggingface_hub import HfApi, snapshot_download
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    spec=PROFILES[quality]
    endpoints= ((MODELSCOPE,'https://hf-mirror.com','https://huggingface.co') if source=='mirror'
                else ('https://huggingface.co','https://hf-mirror.com',MODELSCOPE))
    # The upstream model card / revision / LFS SHA comes from the official
    # endpoint. Mirrors may only supply bytes validated against these hashes.
    revision=PINNED_REVISION[quality]
    lfs=dict(PINNED_WEIGHT_HASHES[quality])
    if on_stage:
        on_stage('verifying_upstream_manifest')
    try:
        if source=='mirror':
            # 国内直连 huggingface.co 的 API 常卡到超时；镜像优先时只用固定 SHA。
            raise TimeoutError('skip official metadata in mirror mode')
        official=HfApi(endpoint='https://huggingface.co').model_info(
            spec['repo'],revision=revision,files_metadata=True)
        if getattr(official,'sha',None)!=revision:
            raise SingingModelError('Official commit differs from pinned revision')
        for sibling in official.siblings or []:
            data=getattr(sibling,'lfs',None) or {}
            if isinstance(data,dict) and sibling.rfilename in lfs and data.get('sha256'):
                if lfs[sibling.rfilename]!=data['sha256']:
                    raise SingingModelError('Upstream LFS SHA does not match pinned model')
    except SingingModelError:
        raise
    except Exception:
        # The immutable public weight SHA pins remain mandatory; missing
        # official API metadata never means blindly trusting a mirror.
        if on_stage:
            on_stage('using_pinned_sha256_offline')
    final=model_folder(root,quality)
    last_error=None
    for endpoint in endpoints:
        stage=root / (spec['id']+'.partial')
        try:
            if on_stage:
                on_stage('downloading:'+endpoint)
            # snapshot_download uses a content-addressed cache and supports
            # repeated / interrupted runs. Keep it entirely under data/models.
            # Read only local file sizes while the Hub downloader works.  The
            # Hub's parallel Xet downloads do not expose a stable callback;
            # this is explicitly an estimated transfer view, not a fake ETA.
            from .download_telemetry import TransferMeter, snapshot_bytes
            monitor_stop=threading.Event()
            meter=TransferMeter()
            expected=int(spec['download_bytes_estimate'])
            def poll_progress():
                while not monitor_stop.is_set():
                    count=snapshot_bytes(stage)
                    stats=meter.update(count,expected,estimated=True)
                    if on_progress:
                        on_progress('downloading',dict(stats,source_name=(
                            '魔搭 ModelScope' if endpoint==MODELSCOPE else
                            '国内镜像' if 'mirror' in endpoint else 'Hugging Face 官方')))
                    monitor_stop.wait(1.5)
            monitor=threading.Thread(target=poll_progress,name='neko-sing-transfer-meter',daemon=True)
            monitor.start()
            try:
                if endpoint==MODELSCOPE:
                    _download_modelscope(stage,quality,lfs=lfs)
                else:
                    snapshot_download(repo_id=spec['repo'],revision=revision,
                        endpoint=endpoint,local_dir=str(stage),
                        allow_patterns=['*.json','*.txt','*.model','*.tiktoken','*.safetensors','model.bin'],
                        local_files_only=False)
            finally:
                monitor_stop.set()
                monitor.join(timeout=3)
            if on_progress:
                on_progress('verifying',{'bytes_received':snapshot_bytes(stage),
                      'total_bytes':expected,'progress_pct':99,'progress_estimated':True,
                      'speed_bytes_s':None,'eta_seconds':None})
            if on_stage:
                on_stage('sha256_verification')
            # 前一个来源（如魔搭）中断留下的断点文件，换源成功后不再需要。
            for leftover in list(stage.glob('*.partial'))+list(stage.glob('*.restart')):
                leftover.unlink(missing_ok=True)
            # snapshot_download also leaves .cache/ which is excluded from the
            # manifest and never copied to an executable location.
            _create_verified_manifest(stage,spec,revision,lfs_shas=lfs)
            if verified_folder(root,quality,deep=True) is not None:
                # Do not replace a verified installation during race/retry.
                shutil.rmtree(stage,ignore_errors=True)
                return {'state':'verified','reused':True,'model':spec['id']}
            if final.exists():
                # Preserve user's pre-existing/incomplete directory rather
                # than deleting it: rename for manual inspection.
                stale=root/(spec['id']+'.previous')
                if stale.exists():
                    raise SingingModelError('An old backup needs manual cleanup')
                final.rename(stale)
            stage.rename(final)
            return {'state':'verified','reused':False,'model':spec['id'],
                    'revision':revision}
        except Exception as exc:
            last_error=exc
            if on_stage:
                on_stage('retry_or_failed:'+type(exc).__name__)
    raise SingingModelError('模型下载或验证失败；未修改原来的可用模型：'+type(last_error).__name__)


def verify_manual_import(root: Path, quality: str) -> dict:
    """Offline, full SHA-256 validation of user-supplied official files.

    Does not download, infer, upload audio, or overwrite an existing model.
    A later activation performs a true CPU/CUDA preflight before selection.
    """
    if quality not in PROFILES:
        raise ValueError('Unknown singer quality tier')
    folder=model_folder(root,quality)
    if verified_folder(root,quality,deep=True) is not None:
        return {'ok':True,'reused':True,'quality':quality}
    if not folder.is_dir() or folder.is_symlink():
        raise SingingModelError('指定目录中没有完整的手动导入模型')
    pins=PINNED_WEIGHT_HASHES[quality]
    for name,pin in pins.items():
        weight=folder/name
        if not weight.is_file() or weight.is_symlink() or _digest(weight)!=pin:
            raise SingingModelError('手动模型权重缺失或 SHA-256 与官方固定版本不一致：'+name)
    if not (folder/'config.json').is_file():
        raise SingingModelError('手动模型缺少 config.json')
    _create_verified_manifest(folder,PROFILES[quality],PINNED_REVISION[quality],pins)
    if verified_folder(root,quality,deep=True) is None:
        raise SingingModelError('手动文件未通过完整校验；尚未启用模型')
    return {'ok':True,'quality':quality,'sha256_verified':True}


def uninstall_model(root: Path, quality: str) -> dict:
    """Never follow links or remove any directory except an exact tier folder."""
    folder=model_folder(root,quality)
    if not folder.exists():
        return {'ok':True,'already_missing':True}
    if folder.is_symlink() or folder.resolve()!=folder.absolute():
        return {'ok':False,'error':'拒绝删除符号链接或非预期路径'}
    try:
        shutil.rmtree(folder)
    except OSError as exc:
        return {'ok':False,'error':'Windows 文件占用或删除失败：'+type(exc).__name__}
    return {'ok':True,'model':PROFILES[quality]['id']}


def verify_manual_model(root: Path, quality: str) -> dict:
    """Validate user-copied files offline against the exact public SHA pins.

    Only looks in the tier-specific local model folder; no networking, no
    model decoding or inference. A successful result creates only a manifest.
    Both installed-video Whisper and pre-existing song checkpoints remain intact.
    """
    folder=model_folder(root,quality)
    if not folder.is_dir() or folder.is_symlink():
        return {'ok':False,'error':'请先把官方模型文件放入该档位文件夹'}
    try:
        spec=PROFILES[quality]
        _create_verified_manifest(folder,spec,PINNED_REVISION[quality],
                                  PINNED_WEIGHT_HASHES[quality])
        if verified_folder(root,quality,deep=True) is None:
            raise SingingModelError('模型本地检验尚未通过')
        return {'ok':True,'state':'verified','quality':quality,
                'note':'本地文件通过官方固定 SHA-256 校验；尚未执行真实推理'}
    except (OSError,ValueError,SingingModelError) as exc:
        return {'ok':False,'quality':quality,
                'error':type(exc).__name__+': '+str(exc)[:180]}
