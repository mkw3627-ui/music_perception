"""Plugin-private CUDA 12.x / cuDNN 9 Windows runtime for ONNX stem separation.

The installed Windows NVIDIA *display driver* is necessarily shared. CUDA
user-mode DLLs, cuDNN, cuBLAS, cuFFT, NVRTC and cuRAND are unpacked only into
this plugin's `data/environment/stem-onnxruntime-gpu/` tree. We NEVER run a
system-wide CUDA installer or pip against the host's Python.

Only exact PyPI Windows x64 wheels with fixed public SHA-256 values are
accepted, including from explicitly selected Chinese PyPI mirrors. Files are
staged and atomically activated; interrupted downloads retain .partial bytes.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
from html.parser import HTMLParser
from pathlib import Path

CUDA_BUNDLE_VERSION = 'cuda12.9-cudnn9.22-win64-r1'
CUDA_MARKER = '.neko-private-cuda-verified.json'
CUDA_DIR_NAME = 'cuda-' + CUDA_BUNDLE_VERSION
# Win x64 wheel SHA-256 from PyPI's published release metadata, not the mirror.
CUDA_WHEELS = (
    dict(package='nvidia-cuda-runtime-cu12', version='12.9.79', family='cuda_runtime',
         sha256='8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891',
         approximate_bytes=3_600_000, dlls=('cudart64_12.dll',)),
    dict(package='nvidia-cublas-cu12', version='12.9.2.10', family='cublas',
         sha256='623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661',
         approximate_bytes=553_200_000, dlls=('cublasLt64_12.dll', 'cublas64_12.dll')),
    dict(package='nvidia-cufft-cu12', version='11.4.1.4', family='cufft',
         sha256='8e5bfaac795e93f80611f807d42844e8e27e340e0cde270dcb6c65386d795b80',
         approximate_bytes=200_100_000, dlls=('cufft64_11.dll',)),
    dict(package='nvidia-cudnn-cu12', version='9.22.0.52', family='cudnn',
         sha256='5d10117314c861245992dbcf8a6f8ae1f54852137a7c9f80cc9de9fa596f7d62',
         approximate_bytes=687_200_000,
         dlls=('cudnn64_9.dll', 'cudnn_ops64_9.dll', 'cudnn_adv64_9.dll',
               'cudnn_graph64_9.dll', 'cudnn_heuristic64_9.dll')),
    dict(package='nvidia-curand-cu12', version='10.3.10.19', family='curand',
         sha256='e8129e6ac40dc123bd948e33d3e11b4aa617d87a583fa2f21b3210e90c743cde',
         approximate_bytes=68_800_000, dlls=('curand64_10.dll',)),
    dict(package='nvidia-cuda-nvrtc-cu12', version='12.9.86', family='cuda_nvrtc',
         sha256='72972ebdcf504d69462d3bcd67e7b81edd25d0fb85a2c46d3ea3517666636349',
         approximate_bytes=76_400_000, dlls=('nvrtc64_120_0.dll',)),
)
CUDA_ESTIMATED_DOWNLOAD_BYTES = sum(x['approximate_bytes'] for x in CUDA_WHEELS)
CUDA_MIN_FREE_BYTES = 6_000_000_000  # downloads + temporary extraction; user must consent
_DLL_HANDLES = []


class PrivateCudaError(RuntimeError):
    pass


def _gpu_root(environment_dir):
    return Path(environment_dir) / 'stem-onnxruntime-gpu'


def _folder(environment_dir):
    return _gpu_root(environment_dir) / CUDA_DIR_NAME


def wheel_name(spec):
    return f"{spec['package'].replace('-', '_')}-{spec['version']}-py3-none-win_amd64.whl"


def library_directories(runtime_dir):
    root = Path(runtime_dir)
    return tuple(root / 'nvidia' / spec['family'] / 'bin' for spec in CUDA_WHEELS)


def missing_private_cuda_files(runtime_dir):
    root = Path(runtime_dir)
    missing = []
    for spec in CUDA_WHEELS:
        for dll in spec['dlls']:
            file = root / 'nvidia' / spec['family'] / 'bin' / dll
            try:
                if (not file.is_file() or file.is_symlink() or file.stat().st_size < 1024):
                    missing.append(str(file.relative_to(root)))
            except OSError:
                missing.append(str(file.relative_to(root)))
    return missing


def cuda_runtime_path(environment_dir):
    root = _folder(environment_dir)
    try:
        marker = json.loads((root / CUDA_MARKER).read_text(encoding='utf-8'))
        if (marker.get('version') != CUDA_BUNDLE_VERSION or
                marker.get('wheels') != {wheel_name(w): w['sha256'] for w in CUDA_WHEELS}):
            return None
        if missing_private_cuda_files(root):
            return None
        return root
    except (OSError, ValueError, TypeError):
        return None


def runtime_inventory(environment_dir):
    path = cuda_runtime_path(environment_dir)
    return {
        'cuda_dll_installed': bool(path),
        'cuda_runtime_path': str(path) if path is not None else None,
        'cuda_bundle_version': CUDA_BUNDLE_VERSION,
        'cuda_download_estimate_bytes': CUDA_ESTIMATED_DOWNLOAD_BYTES,
        'gpu_disk_space_estimate_bytes': CUDA_MIN_FREE_BYTES,
        'cuda_isolation': 'plugin_private_process_only',
        'gpu_driver_note': 'NVIDIA Windows display driver and MSVC system runtime are shared; CUDA/cuDNN are plugin-private',
    }


class _Links(HTMLParser):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href', '')
            if Path(urllib.parse.unquote(urllib.parse.urlsplit(href).path)).name == self.filename:
                self.urls.append(href)


def _sources(package, version, priority):
    if priority not in {'mirror', 'official'}:
        raise PrivateCudaError('Unknown download priority')
    sources = (
        ('清华镜像', f'https://pypi.tuna.tsinghua.edu.cn/simple/{package}/'),
        ('阿里云镜像', f'https://mirrors.aliyun.com/pypi/simple/{package}/'),
        ('PyPI 官方', f'https://pypi.org/pypi/{package}/{version}/json'),
    )
    return sources if priority == 'mirror' else sources[2:] + sources[:2]


def _fetch_url(index, spec):
    filename = wheel_name(spec)
    request = urllib.request.Request(index, headers={'User-Agent': 'NEKO-private-CUDA/0.9.5'})
    with urllib.request.urlopen(request, timeout=22) as reply:
        if urllib.parse.urlsplit(reply.geturl()).scheme != 'https':
            raise PrivateCudaError('Non-HTTPS wheel index blocked')
        content = reply.read(3_000_001)
    if len(content) > 3_000_000:
        raise PrivateCudaError('Wheel index too large')
    if index.endswith('/json'):
        obj = json.loads(content)
        matching = [r for r in obj.get('urls', ())
                    if r.get('filename') == filename and
                    r.get('digests', {}).get('sha256') == spec['sha256']]
        if not matching:
            raise PrivateCudaError('Official wheel SHA does not match pinned value')
        url = matching[0]['url']
        if urllib.parse.urlsplit(url).hostname != 'files.pythonhosted.org':
            raise PrivateCudaError('Unexpected official CDN')
        return url
    html = _Links(filename)
    html.feed(content.decode('utf-8', errors='replace'))
    host = urllib.parse.urlsplit(index).hostname
    for href in html.urls:
        url = urllib.parse.urljoin(index, href)
        parts = urllib.parse.urlsplit(url)
        if parts.scheme == 'https' and parts.hostname == host:
            return url
    raise PrivateCudaError('Wheel not found on chosen mirror')


def _safe_extract_wheel(archive_path, stage, spec):
    """Never follow links or allow ZIP slip, even after verifying SHA."""
    # Distribution files from pinned PyPI win64 wheel only; 3 GiB/wheel cap.
    count = 0
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            name = entry.filename
            parts = Path(name).parts
            if (not parts or name.startswith(('/', '\\')) or '\\' in name or
                    ':' in name or '..' in parts or entry.file_size > 3_000_000_000 or
                    (entry.external_attr >> 16) & 0o170000 == 0o120000):
                raise PrivateCudaError('Unsafe CUDA wheel ZIP entry')
            # Do not permit binary wheels to populate unrelated Python modules.
            distinfo = spec['package'].replace('-', '_') + '-' + spec['version'] + '.dist-info'
            if parts[0] != 'nvidia' and not parts[0].startswith(distinfo):
                raise PrivateCudaError('CUDA wheel contains unexpected filesystem paths')
            count += entry.file_size
            if count > 3_500_000_000:
                raise PrivateCudaError('CUDA wheel expansion too large')
            output = Path(stage).joinpath(*parts)
            if entry.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, output.open('wb') as target:
                shutil.copyfileobj(source, target, 1024 * 1024)


def _check_target():
    if (sys.platform != 'win32' or sys.version_info[:2] != (3, 11) or
            platform.machine().upper() not in {'AMD64', 'X86_64'}):
        raise PrivateCudaError('Plugin-private CUDA wheels require Windows x64 Python 3.11')


def install_cuda_runtime(environment_dir, status_path, priority='mirror'):
    """Install six NVIDIA user-mode runtimes with full digest verification.

    Called in the plugin's managed Python installer process only; no global
    environment mutation. A partial download never becomes a runtime.
    """
    _check_target()
    environment_dir = Path(environment_dir)
    root = _gpu_root(environment_dir)
    root.mkdir(parents=True, exist_ok=True)
    if cuda_runtime_path(environment_dir) is not None:
        return True
    try:
        from .lyrics_model_store import _file_sha256, _stream_to_file
        from .stem_gpu_runtime import _status
    except ImportError:
        from lyrics_model_store import _file_sha256, _stream_to_file
        from stem_gpu_runtime import _status
    destination = _folder(environment_dir)
    if destination.exists() and cuda_runtime_path(environment_dir) is None:
        # No overwriting a live (or locked) DLL directory. Park stale runtime.
        backup = root / (destination.name + '.invalid-' + uuid.uuid4().hex[:8])
        try:
            os.replace(destination, backup)
        except OSError as exc:
            raise PrivateCudaError('上次 CUDA 环境未完整安装且被锁定，请关闭插件重试：' + str(exc)[:110]) from exc
    downloads = root / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    remaining = sum(spec['approximate_bytes'] for spec in CUDA_WHEELS
                    if not (downloads / wheel_name(spec)).is_file())
    # With very large wheels, reserve space for staging and unpacking.
    try:
        free = shutil.disk_usage(root).free
    except OSError:
        free = None
    if free is not None and free < max(3_500_000_000, remaining + 2_000_000_000):
        raise PrivateCudaError('插件数据盘剩余空间不足。首次 CUDA/cuDNN 安装建议预留至少 6 GB 空间')
    try:
        from .download_telemetry import TransferMeter
    except ImportError:
        from download_telemetry import TransferMeter
    progress_base = 0
    wheels = []
    for spec in CUDA_WHEELS:
        filename = wheel_name(spec)
        wheel = downloads / filename
        verified = False
        try:
            verified = wheel.is_file() and _file_sha256(wheel) == spec['sha256']
        except OSError:
            pass
        if not verified:
            errors = []
            for source, index in _sources(spec['package'], spec['version'], priority):
                try:
                    url = _fetch_url(index, spec)
                    _status(status_path, state='running', stage='cuda_downloading',
                            package=spec['package'], source=source, progress_pct=min(98,10+int(88*progress_base/CUDA_ESTIMATED_DOWNLOAD_BYTES)))
                    meter=TransferMeter()
                    def on_progress(done, total, restarted, spec=spec, source=source):
                        stats=meter.update(done,total or spec['approximate_bytes'],
                                           restarted=bool(restarted),estimated=not bool(total))
                        overall = progress_base + min(done, spec['approximate_bytes'])
                        _status(status_path, state='running', stage='cuda_downloading',
                                package=spec['package'], source=source,
                                bytes_received=done, total_bytes=total or spec['approximate_bytes'],
                                speed_bytes_s=stats['speed_bytes_s'],eta_seconds=stats['eta_seconds'],
                                stalled=stats['stalled'],transfer_pct=stats['progress_pct'],
                                last_progress_at=stats['last_progress_at'],
                                cuda_bytes_completed=overall,
                                cuda_download_bytes_total=CUDA_ESTIMATED_DOWNLOAD_BYTES,
                                progress_pct=min(98,10+int(88*overall/CUDA_ESTIMATED_DOWNLOAD_BYTES)))
                    _stream_to_file(url, wheel,
                                    min_bytes=int(spec['approximate_bytes']*0.7),
                                    max_bytes=int(spec['approximate_bytes']*1.2),
                                    expected_sha256=spec['sha256'],
                                    transfer_progress=on_progress)
                    if _file_sha256(wheel) != spec['sha256']:
                        raise PrivateCudaError('SHA verification failed')
                    verified = True
                    break
                except Exception as exc:
                    errors.append(f'{source}: {type(exc).__name__} {str(exc)[:70]}')
                    _status(status_path, state='running',stage='cuda_next_mirror',
                            package=spec['package'],message=errors[-1])
            if not verified:
                raise PrivateCudaError('CUDA/cuDNN 组件下载失败：' + spec['package'] + '；' + '; '.join(errors[-2:]))
        wheels.append((spec,wheel))
        progress_base += spec['approximate_bytes']
    stage = root / ('.cuda-stage-' + uuid.uuid4().hex)
    stage.mkdir(parents=True, exist_ok=False)
    try:
        for i, (spec, wheel) in enumerate(wheels, start=1):
            _status(status_path,state='running',stage='cuda_extracting',
                    package=spec['package'],progress_pct=99,
                    message=f'正在解包 CUDA/cuDNN 组件 {i}/{len(wheels)}')
            _safe_extract_wheel(wheel, stage, spec)
        if missing_private_cuda_files(stage):
            raise PrivateCudaError('Pinned NVIDIA wheel is missing necessary runtime DLLs: ' +
                                   ', '.join(missing_private_cuda_files(stage)[:3]))
        (stage/CUDA_MARKER).write_text(json.dumps({
            'version': CUDA_BUNDLE_VERSION,
            'wheels': {wheel_name(w):w['sha256'] for w in CUDA_WHEELS},
            'created_at':int(time.time()),
        }), encoding='utf-8')
        os.replace(stage,destination)
        if cuda_runtime_path(environment_dir) is None:
            raise PrivateCudaError('CUDA environment activated but final integrity check failed')
        return True
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def configure_private_cuda(runtime_dir):
    """Configure DLL search *only inside a dedicated worker subprocess*.

    No persistent environment variables, global PATH changes or system CUDA.
    Returns DLL directories. Keep add_dll_directory handles alive.
    """
    root = Path(runtime_dir).resolve()
    if not root.is_dir() or missing_private_cuda_files(root):
        raise PrivateCudaError('插件独立 CUDA/cuDNN DLL 不完整，不能启用 GPU')
    if sys.platform != 'win32':
        raise PrivateCudaError('Windows CUDA loader was requested on another OS')
    dll_dirs = library_directories(root)
    for folder in dll_dirs:
        _DLL_HANDLES.append(os.add_dll_directory(str(folder)))
    # Only modifies the worker's own inherited environment. No registry edits.
    os.environ['PATH'] = os.pathsep.join([*(str(p) for p in dll_dirs),os.environ.get('PATH','')])
    if str(root) not in sys.path:
        sys.path.insert(0,str(root))
    return dll_dirs
