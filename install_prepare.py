"""Host-local preparation for N.E.K.O's *visual* package installer.

This module only prepares this plugin's own development SOURCE directory.
It does NOT install into the runtime, mutate the host's configs, or listen to audio.
The browser then asks the official N.E.K.O server to build, plan and install.

This is intentionally a separate module because building a plugin is a host-side,
explicitly user-authorized operation, not part of plugin startup or LLM tools.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from .runtime_environment import PIP_SOURCES, MusicEnvironment, PythonResolutionError, PythonResolver

PLUGIN_ID = "music_perception"
MAX_MODEL_BYTES = 80 * 1024 * 1024
MIN_MODEL_BYTES = 1_000_000
# An official Google-hosted model plus a pinned third-party fallback; no arbitrary URLs.
MODEL_SOURCES = (
    ("Google MediaPipe", "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite", None),
    ("Google TFHub", "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite", None),
    ("国内备用镜像 A · 固定哈希", "https://hf-mirror.com/thelou1s/yamnet/resolve/e90ce378296ec8c5130f2bbceed762a2b0d383c5/lite-model_yamnet_classification_tflite_1.tflite", "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de"),
    ("国内备用镜像 B · 固定哈希", "https://hf-mirror.net/thelou1s/yamnet/resolve/e90ce378296ec8c5130f2bbceed762a2b0d383c5/lite-model_yamnet_classification_tflite_1.tflite", "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de"),
    ("固定哈希官方仓库备份", "https://huggingface.co/thelou1s/yamnet/resolve/e90ce378296ec8c5130f2bbceed762a2b0d383c5/lite-model_yamnet_classification_tflite_1.tflite", "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de"),
)


class PreparationError(RuntimeError):
    """Actionable, redacted diagnostics safe for the local installer UI."""


def plugin_checkout(source: Path) -> Path:
    """Find the canonical full-source checkout if source happens to live there.

    N.E.K.O can ALSO register an external source folder in its visual
    Development pane, so a checkout is optional for this version.
    """
    source = Path(source).resolve(strict=True)
    if source.name != PLUGIN_ID or not (source / "plugin.toml").is_file() or not (source / "pyproject.toml").is_file():
        raise PreparationError("开发源码路径必须指向完整的 music_perception 插件文件夹")
    root = source.parent.parent.parent
    if source.parent.name == "plugins" and source.parent.parent.name == "plugin" \
       and (root / "plugin/neko_plugin_cli").is_dir() and (root / "pyproject.toml").is_file():
        return root
    return source


def _dependencies(source: Path) -> list[str]:
    """Only install expected pinned/semver third-party runtime requirements.

    Reject arbitrary pip/VCS/file URLs embedded in modified manifests.
    """
    allowed = {"librosa", "numpy", "scipy", "soundfile", "soxr", "soundcard", "ai-edge-litert", "imageio-ffmpeg", "faster-whisper", "winrt-windows-media-control"}
    try:
        contents = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
        values = contents["project"]["optional-dependencies"]["audio"]
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise PreparationError("pyproject.toml 中没有合法的运行依赖声明") from exc
    if not isinstance(values, list) or len(values) != len(allowed):
        raise PreparationError("插件依赖清单不完整或包含未经审核的包")
    names = []
    for dep in values:
        if dep == "winrt-Windows.Media.Control>=3.2.1,<4; sys_platform == 'win32'":
            names.append('winrt-windows-media-control')
            continue
        if not isinstance(dep, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*(?:[<>=!~.,0-9 ]+)?", dep):
            raise PreparationError("插件依赖存在非标准来源，请重新检查 pyproject.toml")
        names.append(re.split(r"[<>=!~ ]", dep, maxsplit=1)[0].lower().replace("_", "-"))
    if set(names) != allowed or len(names) != len(allowed):
        raise PreparationError("插件依赖与已审核的音乐分析库不一致")
    return values


def valid_model(path: Path) -> bool:
    try:
        size = Path(path).stat().st_size
        if not (MIN_MODEL_BYTES <= size <= MAX_MODEL_BYTES):
            return False
        with Path(path).open("rb") as stream:
            return stream.read(8)[4:8] == b"TFL3"
    except OSError:
        return False


def download_yamnet(source: Path, progress: Callable[[str], None]) -> str:
    target = Path(source) / "models" / "yamnet.tflite"
    if valid_model(target):
        progress("模型文件已存在，跳过下载")
        return "cached"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".download")
    errors = []
    try:
        for description, url, digest in MODEL_SOURCES:
            temporary.unlink(missing_ok=True)
            progress(f"正在下载音乐识别模型：{description}")
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "N.E.K.O-WeiTaSuoTing/0.6.2"})
                sha = hashlib.sha256()
                length = 0
                with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
                    redirect = urllib.parse.urlparse(response.geturl())
                    if redirect.scheme.lower() != "https":
                        raise PreparationError("模型下载链接重定向到不安全的连接")
                    while True:
                        chunk = response.read(512 * 1024)
                        if not chunk:
                            break
                        length += len(chunk)
                        if length > MAX_MODEL_BYTES:
                            raise PreparationError("模型文件超过大小限制")
                        output.write(chunk)
                        sha.update(chunk)
                if not valid_model(temporary):
                    raise PreparationError("下载内容不是有效的 TFLite 模型")
                if digest and sha.hexdigest().lower() != digest:
                    raise PreparationError("备用镜像文件的 SHA256 不一致")
                os.replace(temporary, target)
                progress(f"模型准备成功：{length // 1024 // 1024} MiB")
                return description
            except (OSError, ValueError, PreparationError) as exc:
                errors.append(f"{description}: {type(exc).__name__}")
        raise PreparationError("模型下载失败；检查本机网络、HTTPS 和 Google 模型地址。来源：" + "、".join(errors))
    finally:
        temporary.unlink(missing_ok=True)


def _run_command(repo: Path, command: list[str], *, timeout: int) -> None:
    env = {**os.environ, "UV_NO_PROGRESS": "1", "UV_NO_COLOR": "1", "UV_NO_CONFIG": "1",
           "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1", "UV_LINK_MODE": "copy"}
    for key in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONEXECUTABLE', 'PYTHONSTARTUP',
                'VIRTUAL_ENV', 'CONDA_PREFIX', 'UV_PYTHON', 'UV_PYTHON_INSTALL_DIR'):
        env.pop(key, None)
    try:
        completed = subprocess.run(command, cwd=str(repo), env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise PreparationError("依赖准备超时；可以检查网络或 N.E.K.O 构建日志") from exc
    except OSError as exc:
        raise PreparationError("无法执行 N.E.K.O 的 Python 构建工具") from exc
    if completed.returncode != 0:
        clean = MusicEnvironment._redact_install_output(completed.stdout or '')
        excerpt = MusicEnvironment._error_excerpt(clean)
        raise PreparationError(f"依赖准备或校验失败（退出码 {completed.returncode}）：{excerpt}")


def _run_uv(uv: str, repo: Path, params: list[str], *, timeout: int) -> None:
    _run_command(repo, [uv, *params], timeout=timeout)


_NATIVE_PROBE = r'''
import pathlib,sys,os
vendor=pathlib.Path(sys.argv[1]).resolve();model=pathlib.Path(sys.argv[2]).resolve()
if not vendor.is_dir(): raise SystemExit("missing vendor")
sys.path.insert(0,str(vendor))
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import librosa,soundcard,scipy,scipy.signal,scipy.ndimage,scipy.fft,soundfile,soxr,imageio_ffmpeg
import faster_whisper, ctranslate2
if sys.platform == 'win32':
    from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager
    assert GlobalSystemMediaTransportControlsSessionManager is not None
assert hasattr(scipy.signal, 'resample_poly') and hasattr(scipy.ndimage, 'median_filter')
assert os.path.isfile(imageio_ffmpeg.get_ffmpeg_exe())
model_obj=Interpreter(model_path=str(model),num_threads=2)
inputs=model_obj.get_input_details();outputs=model_obj.get_output_details()
assert len(inputs)==1 and tuple(inputs[0]["shape"]) in ((15600,),(1,15600))
assert inputs[0]["dtype"]==np.float32
assert any(len(x["shape"]) and int(x["shape"][-1])==521 for x in outputs)
model_obj.allocate_tensors()
model_obj.set_tensor(inputs[0]["index"],np.zeros(tuple(inputs[0]["shape"]),dtype=np.float32))
model_obj.invoke()
print("YAMNet LiteRT and audio stack OK")
'''


def prepare(source: Path, progress: Callable[[str], None] | None = None) -> dict:
    """Prepare just this plugin's model/vendor; official host builds & installs."""
    progress = progress or (lambda _: None)
    source = Path(source).resolve(strict=True)
    root_or_source = plugin_checkout(source)
    deps = _dependencies(source)
    uv = shutil.which("uv")
    if not uv:
        progress("没有检测到 uv，将查找独立 CPython 的 pip（不调用 N.E.K.O 主程序）")
    progress("插件源码合法，正在准备本地 YAMNet 分类模型")
    source_name = download_yamnet(source, progress)
    # Developer/build action may be invoked by a Nuitka-packaged N.E.K.O.
    # Never use sys.executable for pip or native probes unless it was verified.
    toolchain_root = Path(os.environ.get('LOCALAPPDATA') or tempfile.gettempdir()) / 'N.E.K.O' / PLUGIN_ID
    progress('正在检查独立 CPython；已打包的 N.E.K.O server.exe 不能执行 pip')
    try:
        build_python = PythonResolver(toolchain_root, notify=lambda label, rc, out: progress(
            '正在准备 Python 环境' if rc == 0 else 'Python 环境准备结果：' +
            MusicEnvironment._error_excerpt(MusicEnvironment._redact_install_output(out))[-300:])).resolve()
    except PythonResolutionError as exc:
        raise PreparationError(str(exc)) from exc
    # This plugin intentionally has no *mandatory* project.dependencies.
    # Therefore official `sync` would install nothing; instead vendor the
    # optional music stack explicitly before the official build API runs.
    already_valid = False
    existing_vendor = source / "vendor"
    if existing_vendor.is_dir():
        try:
            _run_command(source, [build_python, "-c", _NATIVE_PROBE,
                                  str(existing_vendor), str(source / "models/yamnet.tflite")], timeout=180)
            already_valid = True
            progress("检测到完整可用的 vendor 与模型，跳过重新安装")
        except PreparationError:
            progress("已有 vendor 不完整或版本不匹配，开始重新构建")
    if not already_valid:
        progress("正在准备此插件专有的可选音频依赖 vendor（不修改宿主 Python）")
        staging = Path(tempfile.mkdtemp(prefix=".vendor-build-", dir=source))
        try:
            uv_failure = None
            installed_and_tested = False
            if uv:
                for source_name, index_url in PIP_SOURCES:
                    try:
                        _run_uv(uv, source, ["pip", "install", "--python", build_python,
                                             "--target", str(staging), "--only-binary", ":all:",
                                             "--index-url", index_url, *deps], timeout=900)
                        progress(f"uv 安装完成（源：{source_name}），正在检查原生音频库、FFmpeg 和 YAMNet 模型")
                        _run_command(source, [build_python, "-c", _NATIVE_PROBE,
                                              str(staging), str(source / "models/yamnet.tflite")], timeout=180)
                        installed_and_tested = True
                        break
                    except PreparationError as exc:
                        uv_failure = ((uv_failure + "；") if uv_failure else "") + f"{source_name}：{exc}"
                        progress(f"uv 使用{source_name}失败，清理后换源重试")
                        shutil.rmtree(staging, ignore_errors=True)
                        staging.mkdir(parents=True)
            if not installed_and_tested:
                pip_failure = None
                for source_name, index_url in PIP_SOURCES:
                    try:
                        _run_command(source, [build_python, "-m", "pip", "--isolated", "install",
                                              "--target", str(staging), "--only-binary=:all:",
                                              "--index-url", index_url,
                                              "--disable-pip-version-check", "--no-input", *deps], timeout=600)
                        progress(f"pip 安装完成（源：{source_name}），正在检查原生音频库、FFmpeg 和 YAMNet 模型")
                        _run_command(source, [build_python, "-c", _NATIVE_PROBE,
                                              str(staging), str(source / "models/yamnet.tflite")], timeout=180)
                        break
                    except PreparationError as exc:
                        pip_failure = ((pip_failure + "；") if pip_failure else "") + f"{source_name}：{exc}"
                        progress(f"pip 使用{source_name}失败，清理后换源重试")
                        shutil.rmtree(staging, ignore_errors=True)
                        staging.mkdir(parents=True)
                else:
                    if uv_failure is not None or pip_failure is not None:
                        raise PreparationError('uv 与 pip 均未通过安装和原生验证。uv：' +
                                               str(uv_failure or '未启用') +
                                               '；pip：' + str(pip_failure or '未启用'))
                    raise
            vendor = source / "vendor"
            backup = source / ".vendor-previous"
            if backup.exists():
                raise PreparationError("检测到旧版依赖备份，为防止丢失可用环境，本次构建已停止")
            if vendor.exists():
                vendor.rename(backup)
            try:
                staging.rename(vendor)
            except OSError:
                if backup.exists() and not vendor.exists():
                    backup.rename(vendor)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    # Core's official build/install actions remain exclusively in the host UI.
    if root_or_source != source and uv:
        progress("正在运行 N.E.K.O 官方开发检查")
        _run_uv(uv, root_or_source, ["run", "neko-plugin", "check", PLUGIN_ID], timeout=300)
    progress("依赖与模型检查通过；等待 N.E.K.O 官方构建和安装")
    return {"ready": True, "plugin_id": PLUGIN_ID, "model_source": source_name,
            "installed": False, "mode": "checkout_source" if root_or_source != source else "registered_source",
            "note": "音乐分析依赖和模型已准备到插件源码，正式打包/安装仍由 N.E.K.O 官方接口执行。"}
