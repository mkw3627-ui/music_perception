"""One-shot CTranslate2 CUDA capability probe. Does not load models or audio.

Started by the plugin's isolated managed Python. The result is preliminary:
the first real model inference still needs compatible GPU DLLs (CUDA 12/cuDNN 9).
"""
from __future__ import annotations
import json
import pathlib
import sys


def inspect_cuda(ctranslate2):
    """No side effects: never install drivers, select GPU or download weights."""
    try:
        count = int(ctranslate2.get_cuda_device_count())
        if count < 1:
            return {'cuda_candidate': False, 'gpu_count': 0,
                    'message': '未检测到可用的 NVIDIA CUDA 显卡。可继续使用 CPU INT8。'}
        types = sorted(str(x) for x in ctranslate2.get_supported_compute_types('cuda'))
        return {'cuda_candidate': ('float16' in types), 'gpu_count': count,
                'compute_types': types,
                'message': ('检测到 NVIDIA CUDA 候选，仍需首次实际歌词推理确认 cuBLAS/cuDNN 可用。'
                            if 'float16' in types else
                            '找到 CUDA 设备，但未检测到当前歌词模式所需的 FP16。')}
    except (Exception,) as exc:
        return {'cuda_candidate': False, 'gpu_count': 0,
                'message': ('GPU 后端检查失败：' + type(exc).__name__ +
                            '。请核对驱动、CUDA 12 和 cuDNN 9；CPU 仍可使用。')}


def main():
    if len(sys.argv) != 2:
        return 2
    vendor = pathlib.Path(sys.argv[1]).resolve()
    if not vendor.is_dir():
        return 2
    sys.path.insert(0, str(vendor))
    try:
        import ctranslate2
        info = inspect_cuda(ctranslate2)
    except (ImportError, OSError) as exc:
        info = {'cuda_candidate': False, 'gpu_count': 0,
                'message': '歌词推理依赖不可用：' + type(exc).__name__}
    info['version'] = 1
    sys.stdout.buffer.write((json.dumps(info, ensure_ascii=False) + '\n').encode('utf-8'))
    sys.stdout.buffer.flush()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
