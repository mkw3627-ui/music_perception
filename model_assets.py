"""Model install plans and strictly scoped model-file removal.

Only public, pinned model artifacts are managed.  Hardware/VRAM values are
planning estimates, not benchmarked minimums or evidence of GPU compatibility.
No N.E.K.O SDK, internet, audio recording or model inference occurs here.
"""
from __future__ import annotations

import secrets
import shutil
from pathlib import Path

from .lyrics_model_store import EXPECTED_DOWNLOAD_BYTES, MODELS
from .stem_catalog import DEFAULT_STEM_QUALITY, spec

# Conservative *planning* figures, MiB, not measured on this user machine. Actual peak varies with runtime,
# batching, beam size, available CUDA libraries, and other GPU applications.
ASR_GPU_PLANS = {
    'tiny': (1024, '0.5–1.5 GiB'),
    'base': (1536, '0.8–2 GiB'),
    'small': (3072, '1.5–3 GiB'),
    'large-v3-turbo': (6144, '约 4–8+ GiB（视量化及其他GPU任务而变）'),
}
STEM_EXPECTED_BYTES = 66_800_000
STEM_GPU_PLAN = (2048, '约 1–3+ GiB（未在目标机器实测）')


def model_plan(model: str, *, source: str = 'mirror') -> dict:
    if source not in {'mirror','official'}:
        raise ValueError('不支持的下载来源')
    if model == 'stems' or model.startswith('stems:'):
        quality=model.split(':',1)[1] if ':' in model else DEFAULT_STEM_QUALITY
        profile=spec(quality)
        bytes_estimate = profile['bytes_estimate']
        vram=profile['vram_estimate_mib']
        range_text='约 '+str(vram)+' MiB 空闲起估，需实际验证，试听决定质量'
        return {
            'id': 'stems:'+quality, 'kind': 'stems', 'name': profile['label'],
            'quality': quality, 'filename': profile['file'], 'sha256': profile['sha256'],
            'download_bytes_estimate': bytes_estimate, 'disk_required_bytes_estimate': bytes_estimate * 2 + 32_000_000,
            'suggested_free_vram_mib': vram, 'estimated_gpu_peak': range_text,
            'cpu_supported': True, 'nvidia_supported': '需要独立验证 ONNX Runtime CUDA Provider；默认 CPU',
            'download_source': source, 'scope': '仅公开模型文件，不上传系统音频',
            'disclaimer': '显存数字为保守估计，不是严格最低硬件门槛；GPU需真实加载测试。',
        }
    if model not in MODELS:
        raise ValueError('未知歌词模型')
    bytes_estimate = EXPECTED_DOWNLOAD_BYTES[model]
    vram, range_text = ASR_GPU_PLANS[model]
    return {
        'id': model, 'kind': 'lyrics', 'name': model,
        'download_bytes_estimate': bytes_estimate,
        'disk_required_bytes_estimate': bytes_estimate * 2 + 32_000_000,
        'suggested_free_vram_mib': vram, 'estimated_gpu_peak': range_text,
        'cpu_supported': True, 'nvidia_supported': '需要 CTranslate2、CUDA/cuBLAS/cuDNN 与驱动兼容',
        'download_source': source, 'scope': '仅下载公开的 CTranslate2 模型文件',
        'disclaimer': '显存数据为估计/建议空闲量，不是保证可运行的官方最低要求。',
    }


def free_disk_check(target: Path, plan: dict) -> tuple[bool, int | None]:
    """Disk capacity check before a download, never as part of UI refresh."""
    directory = Path(target)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        available = shutil.disk_usage(directory).free
        return available >= plan['disk_required_bytes_estimate'], available
    except OSError:
        return False, None


def remove_plugin_model_dir(root: Path, model: str) -> dict:
    """Rename-then-remove only one exact plugin-owned tier directory.

    Never access external HF caches, sibling model folders, environment/
    managed Python, or the rest of the installed plugin.  Do not follow
    directory/symlink redirections.  Active workers are checked by caller.
    """
    if model not in MODELS:
        raise ValueError('禁止删除未登记模型')
    root = Path(root)
    directory = root / model
    if root.is_symlink() or root.parent.is_symlink() or directory.is_symlink():
        raise OSError('模型路径包含符号链接，拒绝删除')
    if not directory.exists():
        return {'ok':True, 'already_missing':True, 'model':model, 'bytes_removed':0}
    if not directory.is_dir():
        raise OSError('模型路径并非目录')
    # An exact name and lexical parent boundary; no user-supplied paths.
    if directory.parent.absolute() != root.absolute():
        raise OSError('卸载路径越界')
    # Directory removal must fail closed when it contains junctions/links.
    # Avoid recursive traversal of Windows reparse points.
    for item in directory.rglob('*'):
        if item.is_symlink():
            raise OSError('模型目录包含符号链接，已停止卸载')
    total = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    trash = root / ('.pending_uninstall_' + model + '_' + secrets.token_hex(4))
    directory.rename(trash)  # If Windows locks this folder, leave old folder intact.
    try:
        shutil.rmtree(trash)
    except OSError as exc:
        # Rename may succeed even when a DLL/file handle prevents final removal.
        # If any data remains, attempt to restore the expected model path so
        # status and the on-disk directory do not silently diverge.
        restored=False
        if trash.exists() and not directory.exists():
            try:trash.rename(directory);restored=True
            except OSError:pass
        return {'ok':False,
                'error':('卸载未完成，已恢复模型目录，请停止插件后重试' if restored else
                         '卸载未完成，请关闭插件后检查临时目录 '+trash.name),
                'model':model, 'bytes_removed':0, 'cause':type(exc).__name__,
                'restored':restored}
    return {'ok':True, 'model':model, 'bytes_removed':total}


def remove_stem_model(stems_directory: Path, quality: str = DEFAULT_STEM_QUALITY) -> dict:
    """Delete ONE known model and resumable partials, never the whole folder.

    A Windows lock must be reported as a failure, never silently accepted.
    """
    directory=Path(stems_directory)
    filename=spec(quality)['file']
    if directory.is_symlink():
        raise OSError('模型目录为符号链接，拒绝删除')
    paths=[directory/filename, directory/(filename+'.partial'),
           directory/(filename+'.restart'), directory/(filename+'.rejected')]
    total=0
    for path in paths:
        if path.is_symlink():
            raise OSError('分离模型文件是符号链接，拒绝删除')
    for path in paths:
        if path.is_file():
            total+=path.stat().st_size
            path.unlink()
    # Successful file removal must also clear its status, not a neighbouring tier.
    return {'ok':True,'bytes_removed':total,'model':'stems', 'quality':quality}
