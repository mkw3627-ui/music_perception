"""Isolated ONNX GPU diagnostic worker; reports CUDA execution, not merely availability.

Windows note: ONNX Runtime can retain a profiling JSON file handle until the
InferenceSession is destroyed, and antivirus/indexers may hold the file a bit
longer. Therefore *diagnostic cleanup must not turn a successful GPU inference
into an erroneous WinError 32 failure*. No audio content is logged.
"""
from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

PROFILE_PREFIX = 'neko-stem-gpu-check-'
PROFILE_MAX_BYTES = 32_000_000


def cuda_profile_evidence(profile_path):
    """Verify actual CUDA-executed ONNX nodes from a bounded ORT trace."""
    path = Path(profile_path)
    for attempt in range(4):
        try:
            if not path.is_file() or path.stat().st_size > PROFILE_MAX_BYTES:
                return {'cuda_nodes': 0, 'cpu_nodes': 0,
                        'error': 'GPU profile missing or oversized'}
            trace = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(trace, list):
                return {'cuda_nodes': 0, 'cpu_nodes': 0,
                        'error': 'GPU profile format unexpected'}
            nodes = [row for row in trace if isinstance(row, dict)
                     and row.get('cat') == 'Node'
                     and isinstance(row.get('args'), dict)
                     and isinstance(row.get('name'), str)
                     and row['name'].endswith('_kernel_time')]
            providers = [row['args'].get('provider') for row in nodes]
            return {'cuda_nodes': providers.count('CUDAExecutionProvider'),
                    'cpu_nodes': providers.count('CPUExecutionProvider')}
        except PermissionError:
            if attempt == 3:
                return {'cuda_nodes': 0, 'cpu_nodes': 0,
                        'error': 'GPU profile file is temporarily locked by Windows'}
            time.sleep(0.12 * (attempt + 1))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return {'cuda_nodes': 0, 'cpu_nodes': 0,
                    'error': f'Cannot read GPU profile: {type(exc).__name__}'}
    return {'cuda_nodes': 0, 'cpu_nodes': 0, 'error': 'GPU profile read failed'}


def _remove_profile_dir(path, retries=5):
    """Best-effort cleanup; a locked profiling file must never fail the probe.

    Returns True when removed. A Windows lock may outlive this worker's Python
    object references; stale diagnostic-only traces are retried on later runs.
    """
    path = Path(path)
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if attempt + 1 >= retries:
                return False
            gc.collect()
            time.sleep(min(0.4, 0.08 * (attempt + 1)))
    return False


def _clean_old_profile_dirs(base=None, max_dirs=12, min_age_s=3600):
    """Remove old, owned temporary diagnostics without touching active probes."""
    root = Path(base) if base is not None else Path(tempfile.gettempdir())
    now = time.time()
    try:
        candidates = list(root.glob(PROFILE_PREFIX + '*'))[:max_dirs]
    except OSError:
        return
    for path in candidates:
        try:
            # Never follow links, remove an unrecognized name or delete a
            # directory that another probe may still be using.
            if (not path.is_dir() or path.is_symlink()
                    or not path.name.startswith(PROFILE_PREFIX)
                    or now - path.stat().st_mtime < min_age_s):
                continue
            _remove_profile_dir(path, retries=1)
        except OSError:
            continue


def run_real_cuda_probe(model_path, result):
    """Return profiling evidence; release the ONNX session before cleanup.

    Updates result in-place. A cleanup warning is independent of the reported
    CUDA inference outcome; no trace includes actual captured PCM.
    """
    import numpy as np
    from music_perception.stem_mdx import TwoStemMDX

    _clean_old_profile_dirs()
    temp = Path(tempfile.mkdtemp(prefix=PROFILE_PREFIX))
    separator = None
    try:
        result['stage'] = 'create_cuda_session'
        separator = TwoStemMDX(model_path, request_gpu=True, profile_dir=temp)
        result['stage'] = 'run_cuda_inference'
        separated = separator.separate(np.zeros((44100, 2), dtype=np.float32),
                                       44100, strength='off')
        # ORT end_profiling finalizes its JSON. On Windows the session can still
        # own the file handle; it must be released *before directory deletion*.
        result['stage'] = 'read_cuda_profile'
        trace_path = separator.session.end_profiling()
        evidence = cuda_profile_evidence(trace_path)
        result['profile_evidence'] = evidence
        result['actual_device'] = separated.get('device')
        result['real_inference'] = bool(separated.get('device') == 'cuda'
                                        and evidence['cuda_nodes'] > 0)
        result['ok'] = result['real_inference']
        result['stage'] = 'complete'
        if not result['ok']:
            if evidence.get('error'):
                result['error_code'] = 'cuda_profile_unavailable'
                result['error'] = ('模型推理后未能可靠读取 CUDA 运行记录：'
                                   + evidence['error'])
            else:
                result['error_code'] = 'cuda_nodes_not_executed'
                result['error'] = ('CUDA 会话存在，但没有检测到 CUDA 运算节点；'
                                   '不能认定 GPU 已启用')
    finally:
        # Crucial: releasing *both* wrappers lets ORT close Windows handles.
        separator = None
        gc.collect()
        if not _remove_profile_dir(temp):
            result['profile_cleanup_warning'] = (
                'CUDA 检测临时日志仍被 Windows 占用；不影响已经取得的推理结论。'
                '后台临时文件将在下次检测时尝试清理。')


def main():
    if len(sys.argv) != 4:
        return 2
    vendor, overlay, model = map(Path, sys.argv[1:])
    if not vendor.is_dir():
        raise RuntimeError('audio environment not found')
    sys.path.insert(0, str(vendor))
    if str(overlay) != '-' and overlay.is_dir():
        sys.path.insert(0, str(overlay))
    # Discover only the plugin-private user-mode NVIDIA DLLs. Never silently
    # rely on a different CUDA Toolkit on the Windows system PATH.
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from stem_cuda_bundle import configure_private_cuda, cuda_runtime_path
    from stem_gpu_runtime import overlay_path
    private_env = (overlay.parent.parent if str(overlay) != '-' else vendor.parent)
    onnx_gpu = overlay_path(private_env)
    private_cuda = cuda_runtime_path(private_env)
    package = types.ModuleType('music_perception')
    package.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules['music_perception'] = package
    result = {'ok': False, 'cuda_provider': False, 'real_inference': False,
              'actual_device': None, 'stage': 'import', 'ort_version': None,
              'error': None}
    try:
        result['cuda_bundle_installed'] = private_cuda is not None
        result['onnx_gpu_overlay_installed'] = onnx_gpu is not None
        if onnx_gpu is None or private_cuda is None:
            result['stage'] = 'missing_private_cuda_runtime'
            result['error_code'] = 'private_cuda_not_installed'
            result['error'] = ('分离 GPU 需要同时安装插件私有 ONNX Runtime GPU + CUDA 12.x + cuDNN 9 DLL。'
                               '无需向 Windows 系统目录安装 CUDA Toolkit。')
            print(json.dumps(result,ensure_ascii=False))
            return 1
        try:
            result['stage'] = 'configure_private_dll_search'
            configure_private_cuda(private_cuda)
            result['cuda_bundle_path'] = str(private_cuda)
        except (OSError, RuntimeError) as exc:
            result['error_code'] = 'private_cuda_dll_loader_failed'
            result['error'] = f'{type(exc).__name__}: {str(exc)[:240]}'
            print(json.dumps(result,ensure_ascii=False))
            return 1
        import onnxruntime as ort
        if not str(Path(ort.__file__).resolve()).casefold().startswith(str(onnx_gpu.resolve()).casefold()):
            raise RuntimeError('ONNX imported from audio vendor instead of verified GPU overlay')
        result['ort_version'] = ort.__version__
        result['ort_source'] = ('gpu_overlay'
                                if str(overlay) != '-' and
                                str(Path(ort.__file__).resolve()).startswith(str(overlay.resolve()))
                                else 'audio_vendor')
        if hasattr(ort, 'preload_dlls'):
            result['stage'] = 'preload_cuda_dlls'
            try:
                import contextlib
                with contextlib.redirect_stdout(sys.stderr):
                    ort.preload_dlls(directory='')
            except (OSError, RuntimeError) as exc:
                result['preload_warning'] = f'{type(exc).__name__}: {str(exc)[:150]}'
        result['stage'] = 'probe_providers'
        result['providers'] = list(ort.get_available_providers())
        result['cuda_provider'] = 'CUDAExecutionProvider' in result['providers']
        if not result['cuda_provider']:
            result['error_code'] = 'cuda_provider_missing'
            result['error'] = ('分离进程没有 CUDAExecutionProvider；请先安装分离专用 ONNX GPU 运行包，'
                               '如果已安装，请检查 CUDA 12.8+ / cuDNN 9 与 Visual C++ 运行库')
        elif str(model) != '-' and model.is_file():
            run_real_cuda_probe(model, result)
        else:
            result['ok'] = False
            result['stage'] = 'provider_only'
            result['error_code'] = 'real_inference_not_run'
            result['note'] = ('只检测到 CUDA Provider，模型缺失或分离进程正在使用中，'
                              '本次未验证真实 CUDA 计算')
    except Exception as exc:
        result['error_code'] = 'gpu_runtime_or_model_error'
        result['error'] = f'{type(exc).__name__}: {str(exc)[:280]}'
        if any(word in result['error'].lower()
               for word in ('dll', 'cudnn', 'cublas', 'loadlibrary', 'cuda')):
            result['suggestion'] = ('优先检查 CUDA 12.8+、cuDNN 9 和 Visual C++ DLL；'
                                    '仅有显卡驱动不代表这些运行库齐全')
    # Every diagnostic path must print exactly one JSON response to the host.
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
