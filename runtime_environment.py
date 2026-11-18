"""Optional music engine bootstrap for 为她所听.

The N.E.K.O SDK plugin shell needs no third-party imports and can load even
when the optional music stack is absent.  It prepares *only* this plugin's
isolated data directory, never the host environment or other plugins.

Automatic downloads are opt-in at the product level via installing/enabling
this plugin and are clearly advertised on the plugin panel. Audio capture
still requires a separate, explicit consent.

N.E.K.O's official distributable workflow is still build-time vendoring.
This fallback runtime feature store is not claimed to be a host-managed vendor.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import urllib.request
import uuid
from importlib import metadata
from pathlib import Path
from urllib.parse import urlparse

ALLOWLIST = {
    "librosa", "numpy", "scipy", "soundfile", "soxr", "soundcard", "ai-edge-litert", "imageio-ffmpeg", "faster-whisper", "winrt-windows-media-control",
}
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"
MODEL_MIRROR = "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"
# Exact small LiteRT YAMNet binary; copied from its pinned publicly inspectable
# model repository. All third-party sources require the same SHA-256 digest.
YAMNET_PINNED_REVISION = 'e90ce378296ec8c5130f2bbceed762a2b0d383c5'
YAMNET_FILE = 'lite-model_yamnet_classification_tflite_1.tflite'
YAMNET_SHA256 = '10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de'
YAMNET_SOURCES = (
    (MODEL_URL, None),
    (MODEL_MIRROR, None),
    *tuple((f'https://{host}/thelou1s/yamnet/resolve/{YAMNET_PINNED_REVISION}/{YAMNET_FILE}',YAMNET_SHA256)
      for host in ('hf-mirror.com','hf-mirror.net','huggingface.co')),
)


def ordered_yamnet_sources(priority: str = 'mirror') -> tuple:
    """按下载偏好排序 YAMNet 源：mirror 时 hf-mirror 系列在前，official 时 Google 在前。"""
    if priority == 'official':
        return YAMNET_SOURCES
    mirror_hosts = tuple((f'https://{host}/thelou1s/yamnet/resolve/{YAMNET_PINNED_REVISION}/{YAMNET_FILE}',YAMNET_SHA256)
                         for host in ('hf-mirror.com', 'hf-mirror.net', 'huggingface.co'))
    return mirror_hosts + ((MODEL_URL, None), (MODEL_MIRROR, None))


def ordered_pip_sources(priority: str = 'mirror') -> tuple:
    """按下载偏好排序 pip 源：mirror 国内组在前，official 国外组在前。"""
    if priority == 'official':
        return PIP_SOURCES_OFFICIAL + PIP_SOURCES_DOMESTIC
    return PIP_SOURCES_DOMESTIC + PIP_SOURCES_OFFICIAL
MODEL_MAX_BYTES = 80 * 1024 * 1024
MODEL_MIN_BYTES = 1_000_000
READY_RUNTIME_MARKER = ".neko-audio-runtime-ready.json"

# 零环境兜底：python-build-standalone 便携版（与 uv 管理的 Python 同源）。
# 版本号与 SHA-256 取自 20261110 发行版的 SHA256SUMS；南大镜像 2026-11 实测
# 国内 30MB/s+，GitHub 官方作为第二来源。
PORTABLE_PYTHON_RELEASE = '20261110'
PORTABLE_PYTHON = {
    '3.11': ('cpython-3.11.18+20261110-x86_64-pc-windows-msvc-install_only.tar.gz',
             '5ca7e23bbcc8933a75f4ae9c3a5e5d1ffae5fd143aec9a9d77cf5124cc6e0eef'),
    '3.12': ('cpython-3.12.16+20261110-x86_64-pc-windows-msvc-install_only.tar.gz',
             'e23fe5b16926de1d81661b1a97352e89fb167549c55a6825796ccd0f32ec7f1b'),
    '3.13': ('cpython-3.13.18+20261110-x86_64-pc-windows-msvc-install_only.tar.gz',
             '7e93813bd362c6413f30788b248590e2f169df2d17dcac81aafbcc7a6eeb542c'),
}
PORTABLE_PYTHON_MIRROR = 'https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone'
PORTABLE_PYTHON_SOURCES = (
    ('南京大学镜像', PORTABLE_PYTHON_MIRROR + '/' + PORTABLE_PYTHON_RELEASE),
    ('GitHub 官方', 'https://github.com/astral-sh/python-build-standalone/releases/download/'
                    + PORTABLE_PYTHON_RELEASE),
)

# 内置的可信包源：国内优先、官方兜底。刻意不读用户机器的 PIP_INDEX/UV_INDEX
# （见 _attempt_install 里的环境变量剥离），来源必须固定，防"源注入"。
# 2026-11-13：拆成两组，按用户的下载偏好决定哪组在前；国内组加阿里云镜像，
# 国外组保持官方 PyPI。
PIP_SOURCES_DOMESTIC = (
    ("国内清华镜像", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    ("国内阿里云镜像", "https://mirrors.aliyun.com/pypi/simple"),
)
PIP_SOURCES_OFFICIAL = (
    ("官方 PyPI", "https://pypi.org/simple"),
)
# 兼容旧代码：PIP_SOURCES 保留为「国内优先」的完整顺序（mirror 默认）。
PIP_SOURCES = PIP_SOURCES_DOMESTIC + PIP_SOURCES_OFFICIAL

# Probe in a separate interpreter, so a corrupted native wheel cannot crash
# the main N.E.K.O plugin worker during preparation. No SDK/host imports.
PROBE_SCRIPT = r'''
import importlib.metadata as md
import pathlib
import sys
p = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(p))
from packaging.requirements import Requirement
requirements = sys.argv[2:]
dists = {d.metadata.get('Name', '').lower().replace('_', '-').replace('.', '-'): d.version
         for d in md.distributions(path=[str(p)])}
for spec in requirements:
    req = Requirement(spec)
    name = req.name.lower().replace('_', '-').replace('.', '-')
    if req.marker is not None and not req.marker.evaluate():
        continue
    if name not in dists or not req.specifier.contains(dists[name], prereleases=False):
        raise RuntimeError('Dependency not satisfied: ' + req.name)
import numpy, scipy, scipy.signal, scipy.ndimage, scipy.fft
import librosa, soundfile, soxr, soundcard, imageio_ffmpeg
import faster_whisper, ctranslate2
if sys.platform == 'win32':
    # The media-session wheel can be installed but unusable when its WinRT
    # native binding is unavailable; validate the real namespace import.
    from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager
    assert GlobalSystemMediaTransportControlsSessionManager is not None
# Checking only `import scipy` missed environments where scipy.signal cannot
# be resolved. A full music analysis requires these compiled submodules.
assert hasattr(scipy.signal, 'resample_poly')
assert hasattr(scipy.ndimage, 'median_filter')
import os
assert os.path.isfile(imageio_ffmpeg.get_ffmpeg_exe()), "FFmpeg bundled executable missing"
from ai_edge_litert.interpreter import Interpreter
print('ISOLATED_AUDIO_IMPORT_OK')
'''

MODEL_PROBE_SCRIPT = r'''
import pathlib
import sys
import numpy as np
from ai_edge_litert.interpreter import Interpreter
model=pathlib.Path(sys.argv[1])
i=Interpreter(model_path=str(model),num_threads=2)
inp=i.get_input_details()[0]
shape=tuple(int(x) for x in inp['shape'])
if shape not in {(15600,), (1,15600)} or inp['dtype']!=np.float32:
    raise RuntimeError('Model input is not YAMNet float32 15600')
if not any(int(t['shape'][-1]) == 521 for t in i.get_output_details()):
    raise RuntimeError('Model does not output YAMNet classes')
i.allocate_tensors()
i.set_tensor(inp['index'],np.zeros(shape,dtype=np.float32))
i.invoke()
print('YAMNET_INFERENCE_OK')
'''


class EnvironmentError(RuntimeError):
    pass


def model_valid(path: Path) -> bool:
    try:
        if not (MODEL_MIN_BYTES <= path.stat().st_size <= MODEL_MAX_BYTES):
            return False
        with path.open('rb') as stream:
            return stream.read(8)[4:8] == b'TFL3'
    except OSError:
        return False


def requirements_from_manifest(root: Path) -> list[str]:
    """Whitelist and validate all automatic downloads. Ignore arbitrary indexes."""
    try:
        doc = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))
        if doc['project'].get('dependencies') != []:
            raise EnvironmentError('引导层不应声明需要先安装的依赖；请使用 v0.6.2 配置')
        reqs = doc['project']['optional-dependencies']['audio']
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, EnvironmentError):
            raise
        raise EnvironmentError('插件可选音频依赖清单缺失或损坏') from exc
    if not isinstance(reqs, list) or len(reqs) != len(ALLOWLIST):
        raise EnvironmentError('拒绝未经审查或不完整的音频依赖清单')
    names = []
    import re
    for value in reqs:
        if value == "winrt-Windows.Media.Control>=3.2.1,<4; sys_platform == 'win32'":
            names.append('winrt-windows-media-control')
            continue
        if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9-]*(?:[<>=!~.,0-9 ]+)?', value):
            raise EnvironmentError('依赖来源或版本格式不安全；只允许固定的 PyPI 库')
        names.append(re.split(r'[<>=!~ ]', value, maxsplit=1)[0].lower())
    if set(names) != ALLOWLIST or len(names) != len(set(names)):
        raise EnvironmentError('不匹配的依赖清单；未执行任何下载')
    return reqs


def current_distributions(location: Path) -> dict[str, str]:
    if not location.is_dir():
        return {}
    found: dict[str, str] = {}
    for dist in metadata.distributions(path=[str(location)]):
        name = dist.metadata.get('Name', '').lower().replace('_', '-').replace('.', '-')
        if name:
            found[name] = dist.version
    return found


class PythonResolutionError(RuntimeError):
    """No runnable, ABI-compatible CPython was available for the audio worker."""


class PythonResolver:
    """Locate a *real* CPython interpreter; never execute a Nuitka host as Python.

    uv can manage an independent CPython in the plugin's private data folder.
    All discovered executables must pass an actual `-I -S -c` probe and match
    the host's CPython major/minor and pointer width before selecting wheels.
    """

    _MARKER = 'NEKO_REAL_CPYTHON:'
    _SCRIPT = (
        "import sys,struct;"
        "print('NEKO_REAL_CPYTHON:%d.%d:%d:%s' % "
        "(sys.version_info[0],sys.version_info[1],struct.calcsize('P')*8,sys.implementation.name))"
    )

    def __init__(self, directory: Path, *, cancel=None, notify=None):
        self.directory = Path(directory)
        self.cancel = cancel
        self.notify = notify or (lambda label, rc, out: None)

    @property
    def version(self):
        return '%d.%d' % (sys.version_info.major, sys.version_info.minor)

    def _run(self, args, *, timeout=15, env=None):
        """Bound output, honor shutdown and never invoke a shell."""
        variables = dict(os.environ)
        # Avoid inheriting a virtualenv or the frozen server's fake Python.
        for key in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONEXECUTABLE', 'PYTHONSTARTUP',
                    'VIRTUAL_ENV', 'CONDA_PREFIX', 'UV_PYTHON', 'UV_PYTHON_INSTALL_DIR'):
            variables.pop(key, None)
        variables.update({'UV_NO_CONFIG': '1', 'UV_NO_PROGRESS': '1', 'UV_NO_COLOR': '1'})
        if env:
            variables.update(env)
        try:
            with tempfile.TemporaryFile(mode='w+b') as log:
                child = subprocess.Popen(args, cwd=str(self.directory), env=variables,
                                         stdout=log, stderr=subprocess.STDOUT)
                end = time.monotonic() + timeout
                while child.poll() is None:
                    if self.cancel is not None and self.cancel.is_set():
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=5)
                        raise PythonResolutionError('N.E.K.O 正在关闭，已取消 Python 环境准备')
                    if time.monotonic() > end:
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=5)
                        return -2, '命令超时（自动停止）'
                    time.sleep(0.15)
                log.seek(0, os.SEEK_END)
                log.seek(max(log.tell() - 8192, 0))
                tail = log.read().decode('utf8', errors='replace')
                return int(child.returncode), tail
        except OSError as exc:
            return -1, f'{type(exc).__name__}: {exc}'

    def _validate(self, candidate):
        if not candidate:
            return False
        executable = Path(candidate)
        # Reject the N.E.K.O Nuitka executable without ever invoking it.
        if not re.fullmatch(r'python(?:\d+(?:\.\d+)?)?(?:\.exe)?',
                            executable.name, re.IGNORECASE):
            return False
        if not executable.is_file():
            return False
        code, output = self._run([str(executable), '-I', '-S', '-c', self._SCRIPT], timeout=12)
        expected = f'{self._MARKER}{self.version}:64:cpython'
        return code == 0 and expected in output.splitlines()

    def _uv_find(self, uv, *, own_dir=None):
        opts = ['--no-project', '--no-python-downloads']
        env = None
        if own_dir is not None:
            opts.append('--managed-python')
            env = {'UV_PYTHON_INSTALL_DIR': str(own_dir)}
        code, output = self._run([uv, 'python', 'find', *opts, self.version], timeout=40, env=env)
        if code == 0:
            for line in output.strip().splitlines():
                candidate = line.strip()
                if self._validate(candidate):
                    return candidate
        return None

    def resolve(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        # In development sys.executable is normally python.exe; in packaged
        # Nuitka it is projectneko_server.exe and must NEVER be used with -c/-m.
        candidates = [sys.executable]
        for binary in ('python', 'python3', f'python{self.version}', 'python.exe'):
            found = shutil.which(binary)
            if found and found not in candidates:
                candidates.append(found)
        if os.name == 'nt' and shutil.which('py'):
            # Resolve the Windows launcher to the true interpreter first.
            code, out = self._run(['py', '-' + self.version, '-I', '-S', '-c',
                                    'import sys;print(sys.executable)'], timeout=12)
            if code == 0:
                candidates.extend(out.strip().splitlines()[-1:])
        for candidate in candidates:
            if self._validate(candidate):
                return str(Path(candidate).resolve())

        uv = shutil.which('uv')
        if not uv:
            # 零环境（Steam 版用户常见）：没有系统 Python 也没有 uv。
            # 直接下载固定版本、固定 SHA-256 的便携 CPython 到插件私有目录。
            return self._portable_python()
        existing = self._uv_find(uv)
        if existing:
            return existing

        own_dir = self.directory / 'managed-python'
        existing = self._uv_find(uv, own_dir=own_dir)
        if existing:
            return existing
        self.notify('python-bootstrap', 0, f'正在由 uv 准备独立 CPython {self.version}（不修改系统 Python）')
        # uv 默认从 GitHub 拉 Python，国内常超时；指向南大镜像（同一套发行包）。
        code, output = self._run([uv, 'python', 'install', '--no-bin', '--no-registry',
                                   '--install-dir', str(own_dir), self.version], timeout=900,
                                 env={'UV_PYTHON_INSTALL_MIRROR': PORTABLE_PYTHON_MIRROR})
        self.notify('python-bootstrap', code, output)
        if code != 0:
            # uv 失败时再用自带的便携版下载器试一次（带固定 SHA-256）。
            try:
                return self._portable_python()
            except PythonResolutionError:
                pass
            brief = MusicEnvironment._error_excerpt(MusicEnvironment._redact_install_output(output))
            raise PythonResolutionError('无法自动准备独立 CPython ' + self.version + '：' + brief[-600:])
        installed = self._uv_find(uv, own_dir=own_dir)
        if not installed:
            raise PythonResolutionError('uv 已报告 Python 安装完成，但未找到可执行的 64 位 CPython '
                                        + self.version + '；请查看诊断日志')
        return installed

    def _portable_python(self) -> str:
        """下载 python-build-standalone 便携版到 <data>/portable-python。

        只接受 PORTABLE_PYTHON 里固定的版本和 SHA-256；国内镜像优先、GitHub 兜底。
        解压前先校验摘要，解压后必须通过 _validate 的真实解释器探测。
        """
        import tarfile
        target = self.directory / 'portable-python'
        exe = target / 'python.exe'
        if self._validate(str(exe)):
            return str(exe.resolve())
        pinned = PORTABLE_PYTHON.get(self.version)
        if os.name != 'nt' or pinned is None:
            raise PythonResolutionError(
                f'未找到可用的独立 CPython {self.version}，也没有检测到 uv；'
                '插件不会把 projectneko_server.exe 当 Python 使用。请安装 uv 或 Python 后重试。')
        filename, digest = pinned
        archive = self.directory / (filename + '.download')
        errors: list[str] = []
        for label, base in PORTABLE_PYTHON_SOURCES:
            if self.cancel is not None and self.cancel.is_set():
                raise PythonResolutionError('N.E.K.O 正在关闭，已取消 Python 环境准备')
            url = base + '/' + filename.replace('+', '%2B')
            self.notify('python-bootstrap', 0, f'正在下载便携 CPython {self.version}（{label}）')
            try:
                archive.unlink(missing_ok=True)
                hasher = hashlib.sha256()
                request = urllib.request.Request(url, headers={'User-Agent': 'WeiTaSuoTing-portable-python'})
                with urllib.request.urlopen(request, timeout=30) as response, archive.open('wb') as out:
                    if urlparse(response.geturl()).scheme != 'https':
                        raise OSError('非 HTTPS 重定向')
                    while True:
                        if self.cancel is not None and self.cancel.is_set():
                            raise PythonResolutionError('N.E.K.O 正在关闭，已取消 Python 环境准备')
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        hasher.update(chunk)
                if hasher.hexdigest() != digest:
                    raise OSError('SHA-256 不匹配')
                staging = Path(tempfile.mkdtemp(prefix='.portable-python-', dir=self.directory))
                try:
                    with tarfile.open(archive, 'r:gz') as tar:
                        tar.extractall(staging, filter='data')
                    if target.exists():
                        shutil.rmtree(target)
                    os.replace(staging / 'python', target)
                finally:
                    shutil.rmtree(staging, ignore_errors=True)
                if not self._validate(str(exe)):
                    raise OSError('解压后的 python.exe 未通过解释器探测')
                self.notify('python-bootstrap', 0, f'便携 CPython {self.version} 已就绪（{label}）')
                return str(exe.resolve())
            except PythonResolutionError:
                raise
            except (OSError, tarfile.TarError, ValueError) as exc:
                errors.append(f'{label}: {type(exc).__name__} {str(exc)[:80]}')
            finally:
                archive.unlink(missing_ok=True)
        raise PythonResolutionError('便携 CPython 下载失败：' + ' / '.join(errors)[-500:])


class MusicEnvironment:
    """Thread-safe preparation state, single-shot on startup, manual retry."""

    def __init__(self, *, source: Path, data: Path, models: Path, download_source: str = 'mirror'):
        self.source = Path(source)
        self.data = Path(data)
        self.models = Path(models)
        self.download_source = download_source if download_source in {'mirror', 'official'} else 'mirror'
        self.data.mkdir(parents=True, exist_ok=True)
        self._state_lock = threading.Lock()
        self._install_lock = threading.Lock()
        self._cancel = threading.Event()
        self._state = {
            'phase': 'waiting', 'ready': False, 'model_ready': False,
            'stage': '等待自动检查', 'missing': [], 'error': None,
            'repair_attempts': 0, 'downloaded': False, 'installer': None,
            'error_details': None, 'diagnostics_path': None, 'python_runtime': None,
            'vendor_path': None,
            'model_download': {'state':'idle'},
        }
        self._active_vendor: Path | None = None
        self._python_path: str | None = None

    def snapshot(self) -> dict:
        with self._state_lock:
            return dict(self._state)

    def _report(self, **fields):
        with self._state_lock:
            self._state.update(fields)

    def stop(self):
        self._cancel.set()

    def _interpreter(self) -> None:
        if sys.version_info[:2] not in {(3, 11), (3, 12), (3, 13)} or platform.architecture()[0] != '64bit':
            raise EnvironmentError('需要 64 位 Python 3.11～3.13 的 N.E.K.O 宿主（与可用二进制 wheel 匹配）')
        self._report(stage='定位真实 Python 解释器；排除 N.E.K.O 编译版主程序')
        resolver = PythonResolver(self.data, cancel=self._cancel, notify=self._diagnostic)
        try:
            self._python_path = resolver.resolve()
        except PythonResolutionError as exc:
            raise EnvironmentError(str(exc)) from exc
        self._report(python_runtime=self._python_path, stage='已验证独立 CPython，正在检查音频依赖')

    def worker_runtime(self) -> tuple[str, Path]:
        """Return the verified dedicated CPython and plugin-private audio wheels.

        Windows loopback must run outside the frozen N.E.K.O host process.
        Never fall back to sys.executable (projectneko_server.exe).
        """
        status = self.snapshot()
        if not status.get('ready') or not self._python_path or self._active_vendor is None:
            raise EnvironmentError('隔离音频采集环境尚未就绪，请先完成环境管理检查')
        return self._python(), self._active_vendor.resolve()

    def _python(self) -> str:
        # Important: never silently fall back to a frozen N.E.K.O executable.
        # All subprocess call sites must first resolve and validate CPython.
        if not self._python_path:
            raise EnvironmentError('尚未找到经过验证的独立 Python；禁止使用 N.E.K.O 主程序执行 pip 或 -c')
        return self._python_path

    def _probe(self, location: Path, reqs: list[str]):
        cmd = [self._python(), '-I', '-S', '-c', PROBE_SCRIPT, str(location), *reqs]
        try:
            finished = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True, errors='replace', timeout=180, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise EnvironmentError('音频库独立导入检查失败或超时；检查 Python/系统权限') from exc
        if finished.returncode or 'ISOLATED_AUDIO_IMPORT_OK' not in finished.stdout:
            details = self._redact_install_output((finished.stderr or '') + '\n' + (finished.stdout or ''))
            self._report(error_details=self._error_excerpt(details))
            self._diagnostic('native-import-check', finished.returncode, details)
            raise EnvironmentError('依赖版本不满足或原生扩展无法加载：' + self._error_excerpt(details)[-500:])

    def _model_path(self) -> Path:
        packaged = self.source / 'models' / 'yamnet.tflite'
        if model_valid(packaged):
            return packaged
        return self.models / 'yamnet.tflite'

    def _model_probe(self, model: Path, vendor: Path):
        script = 'import sys;sys.path.insert(0,sys.argv[1]);sys.argv.pop(1);' + MODEL_PROBE_SCRIPT
        proc = subprocess.run([self._python(), '-I', '-S', '-c', script, str(vendor), str(model)],
                              capture_output=True, text=True, errors='replace', timeout=90)
        if proc.returncode or 'YAMNET_INFERENCE_OK' not in proc.stdout:
            raise EnvironmentError('YAMNet 模型虽已下载，但 LiteRT 推理验证失败')

    def _download_model(self) -> Path:
        current = self._model_path()
        if model_valid(current):
            return current
        self.models.mkdir(parents=True, exist_ok=True)
        target = self.models / 'yamnet.tflite'
        temporary = target.with_suffix('.download')
        from .download_telemetry import TransferMeter
        attempts: list[dict] = []
        for url, pinned_digest in ordered_yamnet_sources(self.download_source):
            if self._cancel.is_set():
                raise EnvironmentError('插件正在关闭，已中止环境修复')
            temporary.unlink(missing_ok=True)
            meter=TransferMeter()
            label=('Google 官方' if 'googleapis' in url or 'tfhub.dev' in url else '公开镜像')
            self._report(model_download={'state':'running','stage':'connecting',
                 'source_name':label,'updated_at':time.time(),
                 'bytes_received':0,'total_bytes':None,'progress_pct':None})
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'WeiTaSuoTing/0.6.5'})
                with urllib.request.urlopen(req, timeout=45) as response, temporary.open('wb') as output:
                    final_host=urlparse(response.geturl())
                    if final_host.scheme != 'https':
                        raise EnvironmentError('模型下载必须使用 HTTPS')
                    if (not pinned_digest and final_host.hostname not in
                            {'storage.googleapis.com','tfhub.dev'}):
                        raise EnvironmentError('Google 模型来源重定向到未授权域名')
                    digest=hashlib.sha256()
                    length = 0
                    declared=response.headers.get('Content-Length')
                    expected=int(declared) if declared and declared.isdigit() else None
                    if expected is not None and expected > MODEL_MAX_BYTES:
                        raise EnvironmentError('YAMNet 下载大小超出模型限制')
                    while True:
                        if self._cancel.is_set():
                            raise EnvironmentError('插件已停止，取消下载')
                        chunk = response.read(256 * 1024)
                        if not chunk:
                            break
                        length += len(chunk)
                        if length > MODEL_MAX_BYTES:
                            raise EnvironmentError('模型下载文件超出大小限制')
                        output.write(chunk)
                        digest.update(chunk)
                        stat=meter.update(length,expected,estimated=not bool(expected))
                        self._report(model_download=dict(stat,state='running',stage='downloading',
                            source_name=label,updated_at=time.time()))
                self._report(model_download={'state':'running','stage':'verifying',
                    'bytes_received':length,'total_bytes':expected,
                    'updated_at':time.time(),'source_name':label})
                if pinned_digest and digest.hexdigest()!=pinned_digest:
                    raise EnvironmentError("YAMNet 镜像 SHA-256 不匹配")
                if not model_valid(temporary):
                    raise EnvironmentError('模型格式不正确，未替换已有模型')
                os.replace(temporary, target)
                attempts.append({"source": label, "ok": True, "error": ""})
                self._report(model_download={'state':'ready','stage':'verified',
                    'bytes_received':length,'total_bytes':length,'progress_pct':100,
                    'source_name':label,'updated_at':time.time()},
                    download_attempts=attempts)
                return target
            except (OSError, ValueError, EnvironmentError):
                attempts.append({"source": label, "ok": False,
                                 "error": "下载或校验失败"})
                if self._cancel.is_set():
                    raise EnvironmentError('插件正在关闭，已停止下载')
            finally:
                temporary.unlink(missing_ok=True)
        self._report(model_download={'state':'failed','stage':'failed',
            'message':'全部公开下载源不可用或模型未通过校验', 'updated_at':time.time()})
        raise EnvironmentError('YAMNet 模型下载失败，已尝试 Google 官方与固定 SHA 的多条备用源；请检查网络或稍后重试')

    @staticmethod
    def _redact_install_output(raw: str) -> str:
        """Show useful installer errors without exposing common local secrets.

        Installer subprocesses may print authenticated indexes, tokens or
        profile paths. Keep a bounded, redacted diagnostic, never raw CLI logs.
        """
        raw = re.sub(r'(?i)(https?://)[^/@\s]+@', r'\1[隐藏凭据]@', raw)
        raw = re.sub(r'(?i)(bearer\s+)\S+', r'\1[隐藏凭据]', raw)
        raw = re.sub(r'(?i)(\b(?:token|password|secret|api[_-]?key)\s*[=:]\s*)\S+',
                     r'\1[隐藏凭据]', raw)
        raw = re.sub(r'(?i)([a-z]:\\users\\)[^\\/\s]+', r'\1[用户]', raw)
        raw = re.sub(r'(?i)(/(?:home|Users)/)[^/\s]+', r'\1[用户]', raw)
        raw = re.sub(r'(https?://[^\s?#]+)\?[^\s#]+', r'\1?[隐藏查询参数]', raw)
        return raw[-24_000:]

    @staticmethod
    def _error_excerpt(output: str) -> str:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        # Errors are often printed last; keep enough context to identify
        # unknown uv flags, unsupported wheels and name-resolution failures.
        return ' | '.join(lines[-7:])[-900:] or '安装器没有返回可读错误输出'

    def _diagnostic(self, label: str, returncode: int | None, output: str):
        """Write short, sanitized local logs. Never fail an install for logging."""
        try:
            path = self.data / 'install_diagnostics.log'
            sanitized = self._redact_install_output(output)
            # 2026-11-14: never call platform.system()/machine() here. On
            # Windows those can spawn a ``ver`` child process, which shows up
            # as an extra installer launch in tests and can fail installs.
            # sys.platform + sys.maxsize give the same information cheaply.
            os_name = sys.platform
            arch = '64bit' if sys.maxsize > 2**32 else '32bit'
            with path.open('a', encoding='utf-8') as stream:
                stream.write(f'\n[{label}] exit={returncode} Python={sys.version_info.major}.{sys.version_info.minor} '
                             f'OS={os_name} architecture={arch}\n')
                stream.write(sanitized + '\n')
            self._report(diagnostics_path=str(path))
        except Exception:  # noqa: BLE001 - 写日志绝不能让安装失败
            return

    def _attempt_install(self, command: list[str], *, deadline: float) -> tuple[int, str]:
        """Run one installer with capped logs, deadline and shutdown support.

        逐行进度回调通过 ``self._pip_progress_cb``（可选）传递，不在签名里，
        避免与既有测试的 ``_attempt_install(cmd, *, deadline)`` mock 冲突。
        """
        progress_cb = getattr(self, '_pip_progress_cb', None)
        environment = {**os.environ, 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
                       'UV_NO_PROGRESS': '1', 'UV_NO_COLOR': '1', 'UV_NO_CONFIG': '1',
                       'UV_LINK_MODE': 'copy',
                       'PIP_NO_INPUT': '1'}
        # Nuitka hosts may set Python paths pointing at the compiled server.
        # They must not leak into the independent CPython used for dependencies.
        for key in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONEXECUTABLE', 'PYTHONSTARTUP',
                    'VIRTUAL_ENV', 'CONDA_PREFIX', 'UV_PYTHON', 'UV_PYTHON_INSTALL_DIR'):
            environment.pop(key, None)
        # We explicitly choose the public PyPI index; ignore third-party index
        # settings that may inject credentials or change package provenance.
        for key in list(environment):
            if key.upper().startswith(('PIP_INDEX', 'PIP_EXTRA_INDEX', 'PIP_FIND_LINKS',
                                       'UV_INDEX', 'UV_DEFAULT_INDEX', 'UV_EXTRA_INDEX')):
                environment.pop(key, None)
        if progress_cb is None:
            return self._attempt_install_buffered(command, deadline=deadline, environment=environment)
        # 逐行读 stdout，实时回传进度，同时保留受限诊断日志。
        lines: list[str] = []
        try:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     cwd=str(self.data), env=environment, text=True,
                                     encoding='utf-8', errors='replace', bufsize=1)
        except OSError as exc:
            return -1, f'无法启动安装器：{type(exc).__name__}: {exc}'
        try:
            assert child.stdout is not None
            for line in child.stdout:
                if self._cancel.is_set():
                    raise EnvironmentError('插件正在关闭，已取消自动依赖安装')
                if time.monotonic() >= deadline:
                    raise EnvironmentError('自动安装超过总计 25 分钟限制，已停止以保护宿主')
                lines.append(line)
                try:
                    progress_cb(line)
                except Exception:  # noqa: BLE001 - 进度解析失败不影响安装
                    pass
            child.wait()
        except EnvironmentError:
            child.terminate()
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            raise
        finally:
            if child.poll() is None:
                child.terminate()
        tail = self._redact_install_output(''.join(lines[-4096:]))
        return int(child.returncode), tail

    def _attempt_install_buffered(self, command: list[str], *, deadline: float,
                                  environment: dict) -> tuple[int, str]:
        """原整段日志方式（无逐行进度回调），保持既有测试与行为不变。"""
        with tempfile.TemporaryFile(mode='w+b') as logs:
            try:
                child = subprocess.Popen(command, stdout=logs, stderr=subprocess.STDOUT,
                                         cwd=str(self.data), env=environment)
            except OSError as exc:
                return -1, f'无法启动安装器：{type(exc).__name__}: {exc}'
            try:
                while child.poll() is None:
                    if self._cancel.is_set():
                        raise EnvironmentError('插件正在关闭，已取消自动依赖安装')
                    if time.monotonic() >= deadline:
                        raise EnvironmentError('自动安装超过总计 25 分钟限制，已停止以保护宿主')
                    time.sleep(0.3)
            except EnvironmentError:
                child.terminate()
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
                raise
            finally:
                if child.poll() is None:
                    child.terminate()
            logs.seek(0, os.SEEK_END)
            logs.seek(max(0, logs.tell() - 65_536))
            tail = logs.read().decode('utf-8', errors='replace')
            return int(child.returncode), self._redact_install_output(tail)

    def _pip_supports_raw_progress(self) -> bool:
        """私有解释器里的 pip 是否 ≥ 24.1（支持 ``--progress-bar raw`` 字节进度）。"""
        cached = getattr(self, '_pip_raw_cached', None)
        if cached is not None:
            return cached
        try:
            proc = subprocess.run(
                [self._python(), '-m', 'pip', '--version'],
                capture_output=True, text=True, errors='replace', timeout=15, check=False)
            m = re.search(r'pip\s+(\d+)\.(\d+)', proc.stdout or '')
            ok = bool(m and (int(m.group(1)), int(m.group(2))) >= (24, 1))
        except Exception:  # noqa: BLE001 - 版本检测失败一律按不支持处理，不影响安装
            ok = False
        self._pip_raw_cached = ok
        return ok

    def _install(self, stage: Path, reqs: list[str]):
        uv = shutil.which('uv')
        sources = ordered_pip_sources(self.download_source)
        # 安装器(uv→pip) 有先后；源(国内→官方) 只在 pip 层换源重试。
        # uv 的参数错误 / 装好后原生导入失败都与源无关，换源重试 uv 是浪费，
        # 所以 uv 只试首选源，失败就交给 pip 按全部源依次重试。
        installers: list[tuple[str, list[str], list[tuple[str, str]]]] = []
        if uv:
            installers.append(('uv', [uv, 'pip', 'install', '--python', self._python(),
                                      '--target', str(stage), '--only-binary', ':all:'],
                               [sources[0]]))
        # Using -m pip does not bootstrap or modify pip in the N.E.K.O host.
        pip_base = [self._python(), '-m', 'pip', '--isolated', 'install',
                    '--target', str(stage), '--only-binary=:all:',
                    '--disable-pip-version-check', '--no-input']
        pip_raw = self._pip_supports_raw_progress()
        if pip_raw:
            pip_base.append('--progress-bar')
            pip_base.append('raw')
        installers.append(('pip', pip_base, sources))
        options: list[tuple[str, str, list[str]]] = [
            (name, source_name, base + ['--index-url', url, *reqs])
            for name, base, srcs in installers
            for source_name, url in srcs
        ]
        deadline = time.monotonic() + 1500
        self._report(stage='从国内镜像/官方 PyPI 准备插件专用音频依赖', installer=None,
                     error_details=None, diagnostics_path=None)
        logfile = self.data / 'install_diagnostics.log'
        try:
            logfile.unlink(missing_ok=True)
        except OSError:
            pass
        failures: list[str] = []
        attempts: list[dict] = []
        used_installers: list[str] = []
        for index, (name, source_name, cmd) in enumerate(options):
            if self._cancel.is_set():
                raise EnvironmentError('插件关闭，已中止自动安装')
            if name not in used_installers:
                used_installers.append(name)
            if self._cancel.is_set():
                raise EnvironmentError('插件关闭，已中止自动安装')
            remaining = deadline - time.monotonic()
            if remaining <= 5:
                failures.append(f'{name}@{source_name} 未尝试：总计 25 分钟时限已到')
                break
            if index:
                # A failed run may leave a half-populated stage. Never mix
                # installed files from two different resolvers or sources.
                try:
                    shutil.rmtree(stage)
                    stage.mkdir(parents=True)
                except OSError as exc:
                    raise EnvironmentError('无法清理失败安装的临时目录，未覆盖原有 vendor') from exc
            # Split the shared 25-minute budget so the first combination cannot
            # consume all of it and starve the fallbacks.
            combos_left = len(options) - index
            attempt_deadline = min(deadline,
                                   time.monotonic() + max(120.0, remaining / combos_left))
            shown = f'{name}@{source_name}'
            self._report(installer=shown,
                         stage=f'正在使用 {name} 安装独立音乐环境（源：{source_name}）')
            progress_cb = None
            if name == 'pip' and pip_raw:
                pkg_total = max(1, len(reqs))
                pkg_index = [0]

                def _pip_line(line: str) -> None:
                    stripped = line.strip()
                    m = re.search(r'Progress\s+(\d+)\s+of\s+(\d+)', stripped)
                    if m:
                        self._report(pip_progress={"phase": "downloading",
                                                   "bytes": int(m.group(1)),
                                                   "total_bytes": int(m.group(2)),
                                                   "indeterminate": False})
                        return
                    if stripped.startswith('Collecting'):
                        pkg_index[0] += 1
                        self._report(pip_progress={"phase": "collecting",
                                                   "package_index": pkg_index[0],
                                                   "package_total": pkg_total,
                                                   "indeterminate": True})
                    elif stripped.startswith('Installing collected packages'):
                        self._report(pip_progress={"phase": "installing",
                                                   "package_index": pkg_index[0],
                                                   "package_total": pkg_total,
                                                   "indeterminate": True})

                progress_cb = _pip_line
            try:
                self._pip_progress_cb = progress_cb
                rc, output = self._attempt_install(cmd, deadline=attempt_deadline)
            except EnvironmentError as exc:
                if self._cancel.is_set() or time.monotonic() >= deadline:
                    raise
                failures.append(f'{shown} 超时：{exc}')
                self._report(error_details='\n'.join(failures)[-1500:])
                continue
            finally:
                self._pip_progress_cb = None
            self._diagnostic(shown, rc, output)
            if rc == 0:
                attempts.append({"source": shown, "ok": True, "error": ""})
                self._report(stage=f'{shown} 已下载依赖，正在验证二进制模块与 FFmpeg',
                             download_attempts=attempts)
                try:
                    self._probe(stage, reqs)
                except EnvironmentError as exc:
                    failures.append(f'{shown} 安装完成但导入验证失败：{exc}')
                    self._report(error_details='\n'.join(failures)[-1500:])
                    continue  # next resolver must start with an empty stage
                self._report(error_details=None, installer=' → '.join(used_installers),
                             stage='原生扩展导入成功，准备提交已验证的插件环境')
                return
            brief = self._error_excerpt(output)
            failures.append(f'{shown} 退出码 {rc}：{brief}')
            attempts.append({"source": shown, "ok": False, "error": brief[:120]})
            self._report(error_details='\n'.join(failures)[-1500:], download_attempts=attempts)
        details = '；'.join(failures)
        raise EnvironmentError('全部可用安装方式均失败：' + details[-1500:])

    def _recover_interrupted_vendor_update(self, reqs: list[str]):
        """Recover a crash between replacing the old and verified new vendor.

        The recovery only touches this plugin's private data folder. Never
        treat a leftover .vendor.previous as an automatic reinstall blocker.
        """
        vendor = self.data / 'vendor'
        backup = self.data / '.vendor.previous'
        if not backup.exists():
            return
        if not vendor.exists():
            backup.rename(vendor)
            self._report(stage='检测到上次修复中断，已恢复旧依赖以便重新检查')
            return
        try:
            self._probe(vendor, reqs)
        except EnvironmentError:
            try:
                self._probe(backup, reqs)
            except EnvironmentError:
                # Both copies are currently unusable. Keep the old data for
                # diagnosis without preventing a new, verified install.
                name = self.data / ('.vendor.saved-' + uuid.uuid4().hex[:12])
                backup.rename(name)
                self._report(stage='旧环境与备份均未通过验证，已保留备份，准备自动修复')
                return
            broken = self.data / ('.vendor.broken-' + uuid.uuid4().hex[:12])
            vendor.rename(broken)
            try:
                backup.rename(vendor)
            except OSError:
                broken.rename(vendor)
                raise
            # The previously valid copy is active; a deletion failure is not
            # a reason to stop music analysis.
            shutil.rmtree(broken, ignore_errors=True)
            self._report(stage='已从上次中断的更新恢复可用依赖')
        else:
            # The new install had already committed successfully. Remove only
            # the old backup, not the functioning live vendor.
            shutil.rmtree(backup, ignore_errors=True)

    @staticmethod
    def _runtime_fingerprint(requirements: list[str]) -> str:
        """Bind the side-by-side directory to the exact audited dependency list."""
        return hashlib.sha256("\n".join(requirements).encode("utf-8")).hexdigest()

    def _verified_runtime_directories(self, requirements: list[str]) -> list[Path]:
        """Discover *completed* independent installs from earlier sessions.

        A marker merely makes a directory eligible: every candidate still
        passes our isolated imports, distribution-version and DLL probes.
        Never execute scripts or read arbitrary paths named by marker files.
        """
        wanted = self._runtime_fingerprint(requirements)
        matches = []
        for location in (*self.data.glob('.audio-runtime-*'),
                         *self.data.glob('.audio-staging-*')):
            if not location.is_dir() or location.is_symlink():
                continue
            marker = location / READY_RUNTIME_MARKER
            try:
                record = json.loads(marker.read_text(encoding='utf-8'))
                if (record.get('schema') != 1 or record.get('requirements_sha256') != wanted
                        or record.get('python_abi') != self._python_abi_tag()):
                    continue
                matches.append(location)
            except (OSError, ValueError, TypeError, AttributeError):
                continue
        def safe_mtime(location: Path) -> float:
            try:
                return location.stat().st_mtime
            except OSError:
                return 0.0
        return sorted(matches, key=safe_mtime, reverse=True)

    def _orphaned_staging_directories(self) -> list[Path]:
        """Recover a fully installed v0.6.4 stage after Windows rejected rename.

        Check at most two recent candidates, and do real DLL and version probes.
        Never accept a partially populated wheel set.
        """
        locations = []
        for path in self.data.glob('.audio-staging-*'):
            if path.is_dir() and not path.is_symlink() and not (path / READY_RUNTIME_MARKER).exists():
                try:
                    packages = current_distributions(path)
                    if not {'faster-whisper', 'ai-edge-litert'}.issubset(packages):
                        continue
                    locations.append(path)
                except (OSError, ValueError):
                    continue
        def safe_mtime(location: Path) -> float:
            try:
                return location.stat().st_mtime
            except OSError:
                return 0.0
        return sorted(locations, key=safe_mtime, reverse=True)[:2]

    @staticmethod
    def _python_abi_tag() -> str:
        return f'cpython-{sys.version_info.major}.{sys.version_info.minor}-64'

    def _accept_verified_runtime(self, stage: Path, requirements: list[str]) -> Path:
        """Commit by marking a *new* directory, not by renaming vendor.

        Windows DLLs can keep the directory open even after a worker returns.
        We never rename/delete an active vendor or the freshly probed staging
        directory. The stable marker is tiny; existing paths stay untouched.
        """
        marker = stage / READY_RUNTIME_MARKER
        temporary = stage / '.neko-audio-runtime-ready.tmp'
        data = {'schema': 1,
                'requirements_sha256': self._runtime_fingerprint(requirements),
                'python_abi': self._python_abi_tag()}
        try:
            with temporary.open('x', encoding='utf-8') as stream:
                json.dump(data, stream, ensure_ascii=True, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, marker)
        except OSError as exc:
            raise EnvironmentError(
                '新依赖已通过导入检测，但 Windows 拒绝保存就绪标记；'
                '请检查插件数据目录的写权限：' + type(exc).__name__) from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        self._diagnostic('runtime-activation', 0,
                         '已验证并启用独立依赖目录；未移动或覆盖可能被进程占用的 vendor')
        return stage

    def _replace_vendor(self, stage: Path) -> Path:
        vendor = self.data / 'vendor'
        backup = self.data / '.vendor.previous'
        if backup.exists():
            raise EnvironmentError('发现未清理的依赖备份；已停止更新以避免丢失可用环境')
        had_vendor = vendor.exists()
        if had_vendor:
            vendor.rename(backup)
        try:
            stage.rename(vendor)
        except OSError:
            if had_vendor and backup.exists() and not vendor.exists():
                backup.rename(vendor)
            raise
        if backup.exists():
            try:
                shutil.rmtree(backup)
            except OSError:
                self._report(stage='依赖已更新；旧环境备份仍占用，等待下次清理')
        return vendor

    def _enable(self, location: Path):
        # The SDK plugin runs in its own worker. No imports of third-party
        # audio modules happen in the bootstrap module before this point.
        value = str(location.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)
        importlib.invalidate_caches()
        self._active_vendor = location
        self._report(vendor_path=str(location.resolve()))

    @staticmethod
    def _immutable_runtime() -> bool:
        """On Windows never rename an installed directory with loaded DLLs."""
        return os.name == 'nt'

    def ensure(self) -> dict:
        if not self._install_lock.acquire(blocking=False):
            return self.snapshot()
        try:
            self._cancel.clear()
            state = self.snapshot()
            self._report(phase='checking', stage='扫描 Python 环境与音频依赖', error=None,
                         repair_attempts=state['repair_attempts'] + 1, error_details=None)
            self._interpreter()
            reqs = requirements_from_manifest(self.source)
            # Windows cannot reliably rename an in-use vendor directory when
            # its .pyd/.dll files are loaded by another plugin worker.
            # Keep any legacy .vendor.previous intact; we can probe and reuse
            # it without a rename. POSIX retains the older crash-recovery path.
            if not self._immutable_runtime():
                self._recover_interrupted_vendor_update(reqs)
            installed = self.source / 'vendor'
            local = self.data / 'vendor'
            backup = self.data / '.vendor.previous'
            candidate = None
            candidates = [*self._verified_runtime_directories(reqs), local]
            if backup.is_dir():
                candidates.append(backup)
            if installed.is_dir():
                candidates.append(installed)
            for path in candidates:
                available = current_distributions(path)
                required_now = (ALLOWLIST if sys.platform == 'win32' else
                                ALLOWLIST - {'winrt-windows-media-control'})
                self._report(missing=sorted(required_now.difference(available)))
                try:
                    self._probe(path, reqs)
                except EnvironmentError:
                    continue
                candidate = path
                break
            if candidate is None and self._immutable_runtime():
                # v0.6.4 may have completed installation into staging before
                # its final directory rename failed with [WinError 5].
                for previous_stage in self._orphaned_staging_directories():
                    self._report(phase='checking', stage='发现上次残留依赖，正在验证能否直接恢复')
                    try:
                        self._probe(previous_stage, reqs)
                        candidate = self._accept_verified_runtime(previous_stage, reqs)
                        self._report(stage='已恢复上次下载的完整依赖，无需重复安装', missing=[])
                        break
                    except (EnvironmentError, OSError):
                        continue
            if candidate is None:
                self._report(phase='installing', stage='检测到缺失/损坏或版本不匹配，自动安装完整依赖')
                immutable = self._immutable_runtime()
                prefix = '.audio-runtime-' if immutable else '.audio-staging-'
                stage = Path(tempfile.mkdtemp(prefix=prefix, dir=self.data))
                try:
                    self._install(stage, reqs)
                    self._report(stage='验证全部依赖版本、原生 DLL、FFmpeg 与媒体会话')
                    self._probe(stage, reqs)
                    if immutable:
                        candidate = self._accept_verified_runtime(stage, reqs)
                    else:
                        candidate = self._replace_vendor(stage)
                    self._report(downloaded=True, missing=[], error_details=None)
                finally:
                    # The verified Windows runtime IS the staging directory.
                    # Do not delete/rename it after activation. A failed install
                    # is best-effort cleaned without touching old working dirs.
                    if stage.exists() and candidate != stage:
                        try:
                            shutil.rmtree(stage)
                        except OSError as exc:
                            self._diagnostic('cleanup-staging', -1, type(exc).__name__)
            self._enable(candidate)
            self._report(phase='checking', ready=True, missing=[], error=None, error_details=None,
                         stage='音频依赖就绪，检查 YAMNet 分类模型')
            try:
                model = self._download_model()
                self._report(stage='验证 YAMNet 实际推理')
                self._model_probe(model, candidate)
                self._report(model_ready=True, phase='ready', stage='全部完成，音乐识别可用', error=None)
            except (EnvironmentError, OSError, subprocess.TimeoutExpired) as exc:
                self._report(model_ready=False, phase='partial', stage='歌曲分析可用，音乐分类模型待修复',
                             error=str(exc))
            return self.snapshot()
        except (EnvironmentError, OSError, ValueError) as exc:
            self._report(phase='failed', ready=False, model_ready=False,
                         stage='环境修复未完成，可在界面点击重试', error=str(exc))
            return self.snapshot()
        finally:
            self._install_lock.release()
