"""Isolated, explicitly approved ONNX Runtime GPU wheel for music separation.

No host-wide pip calls, no modifications to the existing audio vendor, models or
Whisper environment.  Only the fixed Windows CPython 3.11 wheel whose published
SHA-256 is pinned is accepted. A failed download never activates an overlay.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
from html.parser import HTMLParser
from pathlib import Path

GPU_PACKAGE_VERSION = '1.26.0'  # pinned CUDA 12.8/cuDNN 9 GPU binary
GPU_WHEEL = 'onnxruntime_gpu-1.26.0-cp311-cp311-win_amd64.whl'
# Published PyPI sha256 for the cp311 Win x64 GPU wheel. Never bypass it on mirrors.
GPU_WHEEL_SHA256 = 'cc5329aad02d9745cc3ae9cdb185bfa1aad242a7bf89b8c471280002ec40f98a'
GPU_ESTIMATED_WHEEL_BYTES = 226_500_000
GPU_ROOT_NAME = 'stem-onnxruntime-gpu'
GPU_ACTIVE_NAME = 'ort-1.26.0-cp311-win-amd64'
MARKER = '.stem-gpu-wheel-verified.json'


class StemGpuInstallError(RuntimeError):
    pass


def gpu_root(environment_dir):
    return Path(environment_dir) / GPU_ROOT_NAME


def overlay_path(environment_dir):
    """Cheap read-only active overlay check; no network or GPU initialization."""
    dest = gpu_root(environment_dir) / GPU_ACTIVE_NAME
    marker = dest / MARKER
    try:
        data = json.loads(marker.read_text(encoding='utf-8'))
        required = (dest / 'onnxruntime' / '__init__.py',
                    dest / 'onnxruntime' / 'capi' / 'onnxruntime_providers_cuda.dll')
        if (data.get('wheel') == GPU_WHEEL and
                data.get('sha256') == GPU_WHEEL_SHA256 and
                all(item.is_file() and not item.is_symlink() for item in required)):
            return dest
    except (ValueError, OSError, TypeError):
        pass
    return None


def inventory(environment_dir):
    try:
        from .stem_cuda_bundle import CUDA_ESTIMATED_DOWNLOAD_BYTES, cuda_runtime_path, runtime_inventory
    except ImportError:
        from stem_cuda_bundle import CUDA_ESTIMATED_DOWNLOAD_BYTES, cuda_runtime_path, runtime_inventory
    installed = overlay_path(environment_dir)
    cuda = cuda_runtime_path(environment_dir)
    return {'installed': installed is not None and cuda is not None,
            'ort_wheel_installed': installed is not None,
            **runtime_inventory(environment_dir),
            'version': GPU_PACKAGE_VERSION,
            'wheel_bytes_estimate': GPU_ESTIMATED_WHEEL_BYTES,
            'disk_required_bytes_estimate': 6_000_000_000,
            'download_bytes_estimate': GPU_ESTIMATED_WHEEL_BYTES + CUDA_ESTIMATED_DOWNLOAD_BYTES,
            'python_required': 'Windows 64-bit CPython 3.11',
            'cuda_required': 'CUDA 12.8+ runtime + cuDNN 9.x (installed privately); NVIDIA driver remains system-wide',
            'overlay_path': str(installed) if installed else None,
            'note': ('Private ONNX CUDA and cuDNN wheels verified; actual GPU inference still needs a model probe.'
                     if installed and cuda else 'GPU runtime incomplete; requires both GPU ORT and private CUDA/cuDNN libraries.')}


def _status(path, **data):
    """Best-effort atomic progress file, safe against Windows reader locks.

    GUI status readers and antivirus software can briefly prevent os.replace.
    Keep the installer progressing and leave the last valid status untouched.
    A unique temporary filename also avoids collisions with an interrupted
    previous preparation attempt. Neither a failure nor retry changes the
    already-verified runtime installation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({'schema': 'stem_gpu_install_v1',
                          'updated_at': int(time.time()), **data},
                         ensure_ascii=False)
    for attempt in range(5):
        tmp = path.with_name(path.name + '.' + uuid.uuid4().hex[:12] + '.tmp')
        try:
            tmp.write_text(payload, encoding='utf-8')
            os.replace(tmp, path)
            return True
        except (OSError, PermissionError):
            if attempt < 4:
                time.sleep(0.07 * (attempt + 1))
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
    return False


class _WheelLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls=[]
    def handle_starttag(self, tag, attrs):
        if tag != 'a':
            return
        href = dict(attrs).get('href', '')
        if Path(urllib.parse.unquote(urllib.parse.urlsplit(href).path)).name == GPU_WHEEL:
            self.urls.append(href)


def _index_urls(priority):
    """Only pinned wheel URLs from trusted PyPI and HTTPS mirror indexes."""
    if priority not in ('mirror','official'):
        raise StemGpuInstallError('invalid source preference')
    providers = (('清华镜像', 'https://pypi.tuna.tsinghua.edu.cn/simple/onnxruntime-gpu/'),
                 ('阿里云镜像', 'https://mirrors.aliyun.com/pypi/simple/onnxruntime-gpu/'),
                 ('PyPI 官方', 'https://pypi.org/pypi/onnxruntime-gpu/1.26.0/json'))
    return providers if priority == 'mirror' else providers[2:] + providers[:2]


def _fetch_source(index_url):
    request = urllib.request.Request(index_url, headers={'User-Agent':'NEKO-stem-GPU-installer/0.9.5'})
    with urllib.request.urlopen(request,timeout=15) as response:
        if urllib.parse.urlsplit(response.geturl()).scheme != 'https':
            raise StemGpuInstallError('non-HTTPS index redirect blocked')
        index = response.read(2_000_000)
        if len(index) >= 2_000_000:
            raise StemGpuInstallError('wheel index too large')
    if index_url.endswith('/json'):
        info=json.loads(index)
        candidates=[x for x in info.get('urls',[]) if x.get('filename') == GPU_WHEEL
                    and x.get('digests',{}).get('sha256') == GPU_WHEEL_SHA256]
        if not candidates:
            raise StemGpuInstallError('official wheel hash not found')
        url=candidates[0]['url']
        if urllib.parse.urlsplit(url).hostname != 'files.pythonhosted.org':
            raise StemGpuInstallError('untrusted official wheel CDN')
        return url
    parser=_WheelLinks()
    parser.feed(index.decode('utf-8','replace'))
    host=urllib.parse.urlsplit(index_url).hostname
    for href in parser.urls:
        url=urllib.parse.urljoin(index_url,href)
        split=urllib.parse.urlsplit(url)
        if split.scheme == 'https' and split.hostname == host:
            return url
    raise StemGpuInstallError('pinned wheel not found at this mirror')


def _check_platform():
    if sys.platform != 'win32' or sys.version_info[:2] != (3,11) or __import__('platform').machine().upper() not in ('AMD64','X86_64'):
        raise StemGpuInstallError('GPU auto-install currently supports Windows x64 CPython 3.11 only; CPU remains available')


def _install_wheel(wheel: Path, directory: Path):
    """Safely extract a verified wheel to a new directory; don't touch live DLLs."""
    if not wheel.is_file():
        raise StemGpuInstallError('GPU wheel missing')
    directory.mkdir(parents=True,exist_ok=False)
    unpacked=0
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            name=member.filename
            parts=Path(name).parts
            if (not parts or name.startswith(('/', '\\')) or '\\' in name or
                    '..' in parts or ':' in name or member.file_size > 350_000_000):
                raise StemGpuInstallError('unsafe GPU wheel member rejected')
            unpacked+=member.file_size
            if unpacked > 1_500_000_000:
                raise StemGpuInstallError('GPU wheel expanded size limit exceeded')
            dest=directory.joinpath(*parts)
            if member.is_dir():
                dest.mkdir(parents=True,exist_ok=True)
                continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(member) as src, dest.open('wb') as out:
                shutil.copyfileobj(src,out,1024*1024)
    if not (directory/'onnxruntime'/'capi'/'onnxruntime_providers_cuda.dll').is_file():
        raise StemGpuInstallError('this wheel is missing the CUDA execution provider')
    if not (directory/'onnxruntime'/'__init__.py').is_file():
        raise StemGpuInstallError('this wheel does not contain the expected package')


def install_gpu(environment_dir, status_path, priority='mirror'):
    """Explicitly approved install, run only in an isolated CPython subprocess."""
    _check_platform()
    root=gpu_root(environment_dir)
    root.mkdir(parents=True,exist_ok=True)
    try:
        from .stem_cuda_bundle import cuda_runtime_path, install_cuda_runtime
    except ImportError:
        from stem_cuda_bundle import cuda_runtime_path, install_cuda_runtime
    if overlay_path(environment_dir) and cuda_runtime_path(environment_dir):
        _status(status_path,state='ready',stage='reused',progress_pct=100,
                message='插件专用 ORT + CUDA + cuDNN 已装好；还需真实推理检测')
        return True
    wheel=root/'downloads'/GPU_WHEEL
    wheel.parent.mkdir(parents=True,exist_ok=True)
    errors=[]
    try:
        from .lyrics_model_store import _file_sha256, _stream_to_file
    except ImportError:  # independent -I -S installer imports by local filename
        from lyrics_model_store import _file_sha256, _stream_to_file
    try:
        from .download_telemetry import TransferMeter
    except ImportError:
        from download_telemetry import TransferMeter
    if overlay_path(environment_dir):
        _status(status_path,state='running',stage='gpu_wheel_reused',progress_pct=10)
    elif wheel.is_file() and _file_sha256(wheel)==GPU_WHEEL_SHA256:
        _status(status_path,state='running',stage='verifying_ort',progress_pct=9,source='缓存')
    elif not overlay_path(environment_dir):
        sources=_index_urls(priority)
        for label,index in sources:
            try:
                _status(status_path,state='running',stage='resolving',progress_pct=0,source=label)
                url=_fetch_source(index)
                last=[0.0]
                meter=TransferMeter()
                def progress(done,total,restarted):
                    now=time.monotonic()
                    if now-last[0]>.4 or done==(total or -1):
                        last[0]=now
                        stats=meter.update(done,total or GPU_ESTIMATED_WHEEL_BYTES,
                                           restarted=bool(restarted),estimated=not bool(total))
                        _status(status_path,state='running',stage='downloading',source=label,
                                bytes_received=done,total_bytes=total or GPU_ESTIMATED_WHEEL_BYTES,
                                speed_bytes_s=stats['speed_bytes_s'],eta_seconds=stats['eta_seconds'],
                                stalled=stats['stalled'],transfer_pct=stats['progress_pct'],
                                last_progress_at=stats['last_progress_at'],
                                progress_pct=min(9,int(10*done/max(1,total or GPU_ESTIMATED_WHEEL_BYTES))),
                                restarted=bool(restarted))
                _stream_to_file(url,wheel,min_bytes=200_000_000,max_bytes=255_000_000,
                                expected_sha256=GPU_WHEEL_SHA256,transfer_progress=progress)
                break
            except Exception as exc:
                errors.append(f'{label}: {type(exc).__name__}')
                _status(status_path,state='running',stage='trying_next',source=label,error=errors[-1])
        if not overlay_path(environment_dir) and (not wheel.is_file() or _file_sha256(wheel)!=GPU_WHEEL_SHA256):
            _status(status_path,state='failed',stage='download',progress_pct=0,
                    message='GPU wheel could not be downloaded and SHA-verified; existing audio environment unchanged',errors=errors)
            return False
    final=root/GPU_ACTIVE_NAME
    staging=root/('.gpu-stage-'+uuid.uuid4().hex)
    try:
        if overlay_path(environment_dir) is None:
            _status(status_path,state='running',stage='extracting',progress_pct=9)
            _install_wheel(wheel,staging)
            (staging/MARKER).write_text(json.dumps({'wheel':GPU_WHEEL,'sha256':GPU_WHEEL_SHA256,'version':GPU_PACKAGE_VERSION}),encoding='utf-8')
            if not final.exists():
                os.replace(staging,final)
            elif not overlay_path(environment_dir):
                raise StemGpuInstallError('old GPU provider directory invalid; close the plugin before repair')
        if overlay_path(environment_dir) is None:
            raise StemGpuInstallError('ONNX GPU overlay activation failed')
        _status(status_path,state='running',stage='cuda_libraries',progress_pct=10,
                message='ONNX GPU 包就绪，开始准备本插件专属 CUDA/cuDNN DLL')
        if not install_cuda_runtime(environment_dir,status_path,priority):
            raise StemGpuInstallError('插件专用 CUDA/cuDNN 安装未完成')
        if cuda_runtime_path(environment_dir) is None:
            raise StemGpuInstallError('插件专用 CUDA/cuDNN DLL 未完成验证')
        _status(status_path,state='ready',stage='private_cuda_installed',progress_pct=100,
                message='全部 CUDA/cuDNN 文件已校验安装；仍须进行真实 GPU 模型推理检测')
        return True
    except (OSError,ValueError,RuntimeError,zipfile.BadZipFile) as exc:
        _status(status_path,state='failed',stage='install',progress_pct=0,
                message=f'GPU 安装未启用：{type(exc).__name__}: {str(exc)[:340]}')
        return False
    finally:
        if staging.exists():
            shutil.rmtree(staging,ignore_errors=True)
