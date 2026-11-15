"""Audited, hash-pinned ONNX MDX model choices for local source separation.

The tier names describe a *resource budget/preset*, not an independently
measured accuracy ranking. We intentionally show the original model identity
and provide an A/B preview; a more expensive preset can sound worse on some
music. Four different published ONNX weights, no silent downloads.
"""
from __future__ import annotations

from pathlib import Path

# SHA-256 and file size sourced from the pinned published Hugging Face file
# listings; these checks MUST happen before a weight is activated.
# Names are hardcoded to prevent a caller choosing arbitrary remote objects.
REPO = 'notabilia/uvr5-models'
STEM_PROFILES = {
    # MDX architectural parameters are model-specific. Each combination below
    # is backed by published UVR model_data.json, not inferred from file size.
    'low': {
        'label': '低 · Karaoke 2', 'file': 'UVR_MDXNET_KARA_2.onnx',
        'repo': 'notabilia/uvr5-models',
        'sha256': 'bf32e15105a09c0f7dddd2b67346146334d6f3ecb399ed7638eba2ab07cbf5f4',
        'bytes_estimate': 52_800_000, 'output': 'instrumental',
        'n_fft': 5120, 'fft': 5120, 'dim_f': 2048, 'compensation': 1.035,
        'vram_estimate_mib': 1024, 'note': '低频谱维度卡拉 OK 分离；对部分歌曲残留较多',
    },
    'medium': {
        'label': '中 · Inst HQ2', 'file': 'UVR-MDX-NET-Inst_HQ_2.onnx',
        'repo': 'seanghay/uvr_models',
        'sha256': '197f8ab296df850f961e68c595f6649acb7d9e621b5600b460f3458967299112',
        'bytes_estimate': 66_800_000, 'output': 'instrumental',
        'n_fft': 6144, 'fft': 6144, 'dim_f': 3072, 'compensation': 1.033,
        'vram_estimate_mib': 1536, 'note': '另一个侧重伴奏的分离版本，需试听对比',
    },
    'high': {
        'label': '高 · Vocal FT', 'file': 'UVR-MDX-NET-Voc_FT.onnx',
        'repo': 'Blane187/all_public_uvr_models',
        'sha256': '534b2070fcc7df514b13ef660dc8cbb328679c2374d04354a5c42bb14ecce111',
        'bytes_estimate': 66_800_000, 'output': 'vocals',
        'n_fft': 7680, 'fft': 7680, 'dim_f': 3072, 'compensation': 1.021,
        'vram_estimate_mib': 2048, 'note': '直接提取人声的专用模型；不保证所有歌曲最佳',
    },
    'ultra': {
        'label': '极高 · Inst HQ3', 'file': 'UVR-MDX-NET-Inst_HQ_3.onnx',
        'repo': 'notabilia/uvr5-models',
        'sha256': '317554b07fe1ea5279a77f2b1520a41ea4b93432560c4ffd08792c30fddf9adc',
        'bytes_estimate': 66_800_000, 'output': 'instrumental',
        'n_fft': 6144, 'fft': 6144, 'dim_f': 3072, 'compensation': 1.022,
        'vram_estimate_mib': 3072, 'note': '兼容旧版 Inst HQ3；并非保证最优，可对比试听',
    },
}
DEFAULT_STEM_QUALITY = 'ultra'  # Retains existing model/filename after upgrade.


def spec(quality: str) -> dict:
    if quality not in STEM_PROFILES:
        raise ValueError('未知人声分离模型档位')
    return STEM_PROFILES[quality]


def path_for(root: str | Path, quality: str) -> Path:
    return Path(root) / spec(quality)['file']


def urls_for(quality: str, priority: str = 'mirror') -> list[tuple[str,str]]:
    profile=spec(quality)
    filename = profile['file']
    path = f"/{profile['repo']}/resolve/main/{filename}?download=true"
    entries = [
        ('HF Mirror', 'https://hf-mirror.com'+path),
        ('Hugging Face 官方', 'https://huggingface.co'+path),
    ]
    if priority == 'mirror':
        return entries
    if priority == 'official':
        return entries[::-1]
    raise ValueError('无效下载来源')


def overview(root: str | Path) -> dict:
    """Cheap file inventory; SHA verified only when user requests it."""
    result = {}
    for quality, item in STEM_PROFILES.items():
        path = Path(root)/item['file']
        try:
            size = path.stat().st_size if path.is_file() and not path.is_symlink() else 0
        except OSError:
            size = 0
        result[quality] = { 'quality': quality, 'filename': item['file'],
                            'label': item['label'], 'present': size > 0,
                            'bytes': size, 'expected_bytes_estimate': item['bytes_estimate'],
                            'vram_estimate_mib': item['vram_estimate_mib'],
                            'note': item['note'] }
    return result


def verify_local_model(path: str | Path, quality: str) -> bool:
    """Dependency-free SHA check for the isolated downloader and plugin worker."""
    import hashlib
    info=spec(quality)
    path=Path(path)
    if path.name != info['file'] or path.is_symlink() or not path.is_file():
        return False
    try:
        size=path.stat().st_size
        if not .9*info['bytes_estimate']<=size<=1.1*info['bytes_estimate']:
            return False
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024), b''):digest.update(block)
        return digest.hexdigest()==info['sha256']
    except OSError:
        return False
