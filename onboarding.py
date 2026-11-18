# -*- coding: utf-8 -*-
"""为她所听 · 新手引导后端（2026-11-13 新增）。

只做三件事：环境检测、依赖下载编排、首次引导状态。全部复用本插件已有的
探测/下载函数，不新写任何网络或硬件探测逻辑。

- 目录选择器 ``picker_start / picker_wait / picker_cancel`` 从 music_synth 的
  ``setup.py`` 复制而来（原函数 ``browse_start / browse_wait / browse_cancel``），
  仅改名，逻辑不变；文件头保留来源说明。
- ``OnboardingStore`` 读写 ``data_path("onboarding.json")``，固定放在 SDK 数据
  目录（因为它本身要记录用户选择的位置），写入用临时文件 + ``os.replace``。
- ``scan_environment`` 与 ``recommend`` 是纯函数/薄封装，只读不改。
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 目录选择器（复制自 music_synth/setup.py，原 browse_start/wait/cancel）
# ---------------------------------------------------------------------------
_PICKER_LOCK = threading.Lock()
_PICKER: dict = {"state": "idle", "path": "", "error": "", "hwnd": 0, "gen": 0}
_PICKER_THREAD: list = [None]


def _browse_dialog_win32(start_path: str, title: str, gen: int = 0) -> tuple[str, str]:
    """Windows 原生文件夹选择框（shell32!SHBrowseForFolderW，纯 ctypes）。

    宿主是 PyInstaller 打的 exe，tkinter 只带了 __init__，没有 filedialog 子模块，
    所以 Windows 上直接调系统 API，不依赖任何额外组件。
    """
    from ctypes import wintypes

    class BROWSEINFOW(ctypes.Structure):
        _fields_ = [
            ("hwndOwner", wintypes.HWND),
            ("pidlRoot", ctypes.c_void_p),
            ("pszDisplayName", wintypes.LPWSTR),
            ("lpszTitle", wintypes.LPCWSTR),
            ("ulFlags", ctypes.c_uint),
            ("lpfn", ctypes.c_void_p),
            ("lParam", ctypes.c_void_p),
            ("iImage", ctypes.c_int),
        ]

    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    ole32 = ctypes.WinDLL("ole32")
    user32 = ctypes.WinDLL("user32", use_last_error=True)

    shell32.SHBrowseForFolderW.argtypes = [ctypes.POINTER(BROWSEINFOW)]
    shell32.SHBrowseForFolderW.restype = ctypes.c_void_p
    shell32.SHGetPathFromIDListW.argtypes = [ctypes.c_void_p, wintypes.LPWSTR]
    shell32.SHGetPathFromIDListW.restype = wintypes.BOOL
    user32.SendMessageW.argtypes = [wintypes.HWND, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p]
    user32.SendMessageW.restype = ctypes.c_void_p
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.PostMessageW.argtypes = [wintypes.HWND, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p]
    user32.PostMessageW.restype = wintypes.BOOL

    BFFM_INITIALIZED = 1
    BFFM_SETSELECTIONW = 0x400 + 103
    WM_CLOSE = 0x0010
    initial = str(start_path or "")
    if not (initial and Path(initial).is_dir()):
        initial = ""
    dlg = [0]

    owner = None
    try:
        fg = user32.GetForegroundWindow()
        if fg and user32.IsWindow(fg):
            owner = fg
    except Exception:  # noqa: BLE001
        owner = None

    @ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HWND, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p)
    def on_msg(hwnd, msg, lparam, data):  # noqa: ANN001
        if msg == BFFM_INITIALIZED:
            dlg[0] = int(hwnd or 0)
            try:
                with _PICKER_LOCK:
                    _PICKER["hwnd"] = dlg[0]
                    stale = bool(gen) and gen != _PICKER["gen"]
                if stale:
                    user32.PostMessageW(hwnd, WM_CLOSE, 0, None)
                    return 0
                user32.SetForegroundWindow(hwnd)
                if initial:
                    ptr = ctypes.cast(ctypes.c_wchar_p(initial), ctypes.c_void_p)
                    user32.SendMessageW(hwnd, BFFM_SETSELECTIONW, 1, ptr)
            except Exception:  # noqa: BLE001
                pass
        return 0

    display = ctypes.create_unicode_buffer(260)
    bi = BROWSEINFOW()
    bi.hwndOwner = owner
    bi.pszDisplayName = ctypes.cast(display, wintypes.LPWSTR)
    bi.lpszTitle = title or "选择安装文件夹"
    bi.ulFlags = 0x1 | 0x10 | 0x40
    bi.lpfn = ctypes.cast(on_msg, ctypes.c_void_p)
    bi.lParam = None
    bi.iImage = 0

    try:
        ole32.CoInitialize(None)
    except Exception:  # noqa: BLE001
        pass
    try:
        pidl = shell32.SHBrowseForFolderW(ctypes.byref(bi))
    finally:
        with _PICKER_LOCK:
            if _PICKER.get("hwnd") == dlg[0]:
                _PICKER["hwnd"] = 0
    if not pidl:
        return "", ""
    try:
        buf = ctypes.create_unicode_buffer(32768)
        if not shell32.SHGetPathFromIDListW(pidl, buf):
            return "", "无法读取所选目录"
        return os.path.normpath(buf.value), ""
    finally:
        try:
            ole32.CoTaskMemFree(ctypes.c_void_p(pidl))
        except Exception:  # noqa: BLE001
            pass


def _browse_dialog_tk(start_path: str, title: str) -> tuple[str, str]:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:  # noqa: BLE001
        pass
    root.update_idletasks()
    initial = str(start_path or "")
    if not (initial and Path(initial).is_dir()):
        initial = ""
    try:
        picked = filedialog.askdirectory(
            parent=root, title=title or "选择安装文件夹",
            initialdir=initial or None, mustexist=False)
        return (os.path.normpath(picked) if picked else ""), ""
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001
            pass


def _browse_dialog(start_path: str, title: str, gen: int = 0) -> tuple[str, str]:
    """弹系统目录选择框。返回 (路径, 错误)；路径为空表示用户取消。"""
    errors: list[str] = []
    if sys.platform == "win32":
        try:
            return _browse_dialog_win32(start_path, title, gen)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"win32: {type(exc).__name__}: {exc}")
    try:
        return _browse_dialog_tk(start_path, title)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"tkinter: {type(exc).__name__}: {exc}")
    return "", "；".join(errors) or "当前环境没有可用的目录选择器"


def picker_start(start_path: str = "", title: str = "", restart: bool = False) -> dict:
    """确保系统目录框已打开（已开着就复用），立即返回当前状态。"""
    with _PICKER_LOCK:
        th = _PICKER_THREAD[0]
        alive = th is not None and th.is_alive()
        if alive and not restart:
            return dict(_PICKER)
        if alive and _PICKER["state"] not in {"cancelled", "error"}:
            return dict(_PICKER)
        if not restart and _PICKER["state"] in {"picked", "cancelled", "error"}:
            return dict(_PICKER)

        gen = int(_PICKER.get("gen") or 0) + 1

        def worker() -> None:
            path, err = _browse_dialog(start_path, title, gen)
            with _PICKER_LOCK:
                if int(_PICKER.get("gen") or 0) != gen:
                    return
                if err:
                    _PICKER.update(state="error", path="", error=err)
                elif path:
                    _PICKER.update(state="picked", path=path, error="")
                else:
                    _PICKER.update(state="cancelled", path="", error="")

        _PICKER.update(state="open", path="", error="", hwnd=0, gen=gen)
        th = threading.Thread(target=worker, name="mp-onboard-browse", daemon=True)
        _PICKER_THREAD[0] = th
        th.start()
        return dict(_PICKER)


def picker_cancel() -> dict:
    """关掉还没选完的系统目录框，并把状态复位成 cancelled。"""
    with _PICKER_LOCK:
        _PICKER["gen"] = int(_PICKER.get("gen") or 0) + 1
        hwnd = int(_PICKER.get("hwnd") or 0)
        _PICKER.update(state="cancelled", path="", error="", hwnd=0)
        result = dict(_PICKER)
    if hwnd:
        try:
            from ctypes import wintypes
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.PostMessageW.argtypes = [wintypes.HWND, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p]
            user32.PostMessageW.restype = wintypes.BOOL
            user32.PostMessageW(hwnd, 0x0111, 2, None)  # WM_COMMAND / IDCANCEL
            user32.PostMessageW(hwnd, 0x0010, 0, None)  # WM_CLOSE
        except Exception:  # noqa: BLE001
            pass
    return result


def picker_wait(seconds: float = 20.0) -> dict:
    """等用户选完（最多 seconds 秒）。没等到就返回 state=open，面板可以再调。"""
    deadline = time.time() + max(0.0, float(seconds))
    while time.time() < deadline:
        with _PICKER_LOCK:
            if _PICKER["state"] in {"picked", "cancelled", "error"}:
                return dict(_PICKER)
        time.sleep(0.1)
    with _PICKER_LOCK:
        return dict(_PICKER)


# ---------------------------------------------------------------------------
# 位置校验
# ---------------------------------------------------------------------------
def validate_storage_root(path: str, need_bytes: int = 0) -> dict:
    """校验用户选择的下载位置：可写、空间够、不在插件或 Steam 目录里。

    目录不存在就尝试创建；写一个探测文件再删掉；用 shutil.disk_usage 取剩余
    空间。路径含非 ASCII 字符只加 warning，不拦截。
    """
    try:
        root = Path(str(path or "")).expanduser().resolve()
    except (OSError, ValueError):
        return {"ok": False, "writable": False, "free_bytes": 0,
                "warnings": [], "error": "路径无效或无法解析"}
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {"ok": False, "writable": False, "free_bytes": 0,
                "warnings": [], "error": f"无法创建目录：{type(exc).__name__}"}
    if root.is_symlink():
        return {"ok": False, "writable": False, "free_bytes": 0,
                "warnings": [], "error": "拒绝符号链接路径"}

    warnings: list[str] = []
    # 拒绝插件安装目录及其子目录
    plugin_dir = Path(__file__).resolve().parent
    try:
        if root == plugin_dir or root.is_relative_to(plugin_dir):
            return {"ok": False, "writable": False, "free_bytes": 0, "warnings": [],
                    "error": "不能选插件自己的安装目录，请另选位置"}
    except (OSError, ValueError):
        pass
    # 拒绝任何 steamapps 路径（Steam 更新会清空）
    lowered = str(root).lower().replace("\\", "/")
    if "steamapps" in lowered:
        return {"ok": False, "writable": False, "free_bytes": 0, "warnings": [],
                "error": "不能选 Steam 目录（Steam 更新可能清空该处）"}
    # 写探测文件
    probe = root / ".neko-onboarding-write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        writable = True
    except OSError as exc:
        return {"ok": False, "writable": False, "free_bytes": 0, "warnings": [],
                "error": f"目录不可写：{type(exc).__name__}"}

    free_bytes = 0
    try:
        free_bytes = shutil.disk_usage(root).free
    except OSError:
        pass
    if int(need_bytes or 0) > 0 and free_bytes > 0 and free_bytes < int(need_bytes):
        return {"ok": False, "writable": True, "free_bytes": free_bytes, "warnings": [],
                "error": "磁盘剩余空间不足，需要更大的空间"}

    if any(ord(ch) > 127 for ch in str(root)):
        warnings.append("路径含非 ASCII 字符，部分 CUDA 组件可能不兼容（待实测）")
    return {"ok": True, "writable": writable, "free_bytes": free_bytes,
            "warnings": warnings, "error": ""}


# ---------------------------------------------------------------------------
# 引导状态存储
# ---------------------------------------------------------------------------
class OnboardingStore:
    """读写 ``data_path("onboarding.json")``。固定放 SDK 数据目录，不迁移。"""

    _DEFAULTS = {
        "version": 1, "completed": False, "step": 1,
        "storage_root": "", "source_pref": "mirror", "updated_at": 0.0,
    }

    def __init__(self, data_dir: str | Path, source_pref: str = "mirror"):
        self._path = Path(data_dir) / "onboarding.json"
        self._default_source = source_pref if source_pref in {"mirror", "official"} else "mirror"

    def load(self) -> dict:
        try:
            doc = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(doc, dict):
                return dict(self._DEFAULTS, source_pref=self._default_source)
        except (OSError, ValueError, TypeError):
            return dict(self._DEFAULTS, source_pref=self._default_source)
        out = dict(self._DEFAULTS)
        for key in ("version", "completed", "step", "storage_root", "source_pref", "updated_at"):
            if key in doc:
                out[key] = doc[key]
        out["source_pref"] = (out["source_pref"] if out["source_pref"] in {"mirror", "official"}
                              else self._default_source)
        return out

    def save(self, **fields) -> dict:
        doc = self.load()
        doc.update(fields)
        doc["updated_at"] = time.time()
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._path)
        return doc

    def completed(self) -> bool:
        return bool(self.load().get("completed"))

    def storage_root(self) -> str:
        return str(self.load().get("storage_root") or "")

    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# 环境检测
# ---------------------------------------------------------------------------
def _item(pid: str, name: str, purpose: str, required: bool, state: str,
          size_bytes: int = 0, detail: str = "") -> dict:
    return {"id": pid, "name": name, "purpose": purpose, "required": required,
            "state": state, "size_bytes": int(size_bytes or 0), "detail": str(detail or "")}


def scan_environment(plugin: Any) -> dict:
    """生成检测项，全部调用现有函数；不写新的探测逻辑，不下载。"""
    from .runtime_environment import model_valid
    from .adaptive_resources import gpu_memory, free_system_ram_mib
    from .stem_catalog import overview as stem_overview
    from .singing_catalog import inventory as singing_inventory
    from .stem_cuda_bundle import missing_private_cuda_files

    items: list[dict] = []
    data_dir = Path(plugin.data_path(".")).resolve()

    # python：运行时
    env_snap = {}
    env = getattr(plugin, "_environment", None)
    if env is not None:
        try:
            env_snap = env.snapshot() or {}
        except Exception:  # noqa: BLE001
            env_snap = {}
    py = env_snap.get("python_runtime")
    items.append(_item(
        "python", "运行环境", "独立的 Python 解释器，跑音频分析",
        True, "ok" if py else ("broken" if env_snap.get("phase") == "failed" else "missing"),
        0, str(py or "")))

    # audio_deps：音频依赖
    ready = bool(env_snap.get("ready"))
    items.append(_item(
        "audio_deps", "音频依赖", "让她能听见电脑里的声音",
        True, "ok" if ready else ("broken" if env_snap.get("error") else "missing"),
        0, str(env_snap.get("error") or "")))

    # yamnet：声音分类模型
    yamnet = data_dir / "models" / "yamnet.tflite"
    yamnet_ok = False
    try:
        yamnet_ok = model_valid(yamnet)
    except OSError:
        yamnet_ok = False
    items.append(_item(
        "yamnet", "声音分类模型", "认出是音乐还是说话",
        True, "ok" if yamnet_ok else "missing", 0, ""))

    # gpu：显卡（只检测，不算必需项）
    gpu = gpu_memory(timeout=1.8)
    ram = free_system_ram_mib()
    items.append(_item(
        "gpu", "显卡", "决定能不能用 GPU 加速（只检测，不下任何东西）",
        False, "ok" if gpu.get("available") else "optional",
        0, f"显存空闲 {gpu.get('free_mib')} MiB；内存空闲 {ram} MiB" if gpu.get("available") else "未检测到 NVIDIA 显卡"))

    # app_capture：应用声道组件
    cap = getattr(plugin, "_process_capture_install", None)
    cap_ok = bool(cap and cap.installed())
    items.append(_item(
        "app_capture", "应用声道组件", "只听你指定的那个软件",
        False, "ok" if cap_ok else "optional", 0, ""))

    # lyrics_model：视频语音模型（可选，任一档位安装即 ok）
    lyrics_ok = False
    try:
        from .lyrics_model_store import inspect_model_files, MODELS
        lroot = data_dir / "models" / "whisper_multilingual"
        for _q, m in MODELS.items():
            rec = inspect_model_files(m, lroot, deep=False)
            if rec.get("state") in {"present_verified", "present_unverified"}:
                lyrics_ok = True
                break
    except Exception:  # noqa: BLE001
        lyrics_ok = False
    items.append(_item(
        "lyrics_model", "视频语音模型", "听懂视频里在说什么",
        False, "ok" if lyrics_ok else "optional", 0, ""))

    # stem_model：人声分离模型
    stem_ok = False
    try:
        for _q, d in stem_overview(data_dir / "models" / "stems").items():
            if d.get("present"):
                stem_ok = True
                break
    except Exception:  # noqa: BLE001
        stem_ok = False
    items.append(_item(
        "stem_model", "人声分离模型", "把人声和伴奏拆开",
        False, "ok" if stem_ok else "optional", 0, ""))

    # singing：唱歌识别环境 + 模型
    sing_env_ok = False
    try:
        from .singing_install import private_environment_status
        sing_status = private_environment_status(data_dir / "environment")
        sing_env_ok = bool(sing_status.get("present"))
        for _q, d in singing_inventory(data_dir / "models" / "singing_specialist").items():
            if d.get("state") == "verified":
                sing_env_ok = True
                break
    except Exception:  # noqa: BLE001
        sing_env_ok = False
    items.append(_item(
        "singing", "唱歌识别环境", "听清歌里唱的词",
        False, "ok" if sing_env_ok else "optional", 0, ""))

    # cuda_bundle：GPU 加速包（有 N 卡时才列出）
    if gpu.get("available"):
        missing = []
        try:
            missing = missing_private_cuda_files(data_dir / "environment")
        except Exception:  # noqa: BLE001
            missing = []
        items.append(_item(
            "cuda_bundle", "GPU 加速包", "给分离/识别上 CUDA 加速",
            False, "ok" if not missing else "optional", 0,
            f"缺 {len(missing)} 个 CUDA 文件" if missing else ""))

    return {"state": "done", "at": time.time(), "items": items}


# ---------------------------------------------------------------------------
# 推荐配置
# ---------------------------------------------------------------------------
_TIER_ORDER = ("low", "medium", "high", "ultra")


def recommend(hardware: dict, catalogs: dict, free_bytes: int, reserve_mib: int) -> dict:
    """从高档往低档，选第一个「显存够 + 磁盘够」的档位；否则落到 CPU 最低档。"""
    gpu_free = hardware.get("gpu_free_mib") if hardware.get("gpu_available") else None
    device = "cuda" if isinstance(gpu_free, int) and gpu_free is not None else "cpu"
    reserve = max(0, int(reserve_mib or 0))
    free = max(0, int(free_bytes or 0))
    choices: dict[str, str] = {}
    total = 0
    reasons: list[str] = []
    for domain in ("video", "singing", "stem"):
        tiers = catalogs.get(domain) if isinstance(catalogs, dict) else {}
        if not isinstance(tiers, dict):
            choices[domain] = "low"
            continue
        picked: str | None = None
        if device == "cuda":
            for q in reversed(_TIER_ORDER):
                item = tiers.get(q)
                if not isinstance(item, dict):
                    continue
                vram = int(item.get("vram_mib") or 0)
                size = int(item.get("bytes") or 0)
                if vram + reserve <= gpu_free and size <= free:
                    picked = q
                    total += size
                    break
        if picked is None:
            for q in _TIER_ORDER:
                if q in tiers:
                    picked = q
                    total += int(tiers[q].get("bytes") or 0) if isinstance(tiers[q], dict) else 0
                    break
        choices[domain] = picked or "low"
    reasons.append(f"空闲显存 {gpu_free} MiB" if device == "cuda" else "未检测到可用 GPU，选用 CPU 最低档")
    return {"video_quality": choices.get("video", "low"),
            "singing_quality": choices.get("singing", "low"),
            "stem_quality": choices.get("stem", "low"),
            "device": device, "total_bytes": total, "reason": "；".join(reasons)}


# ---------------------------------------------------------------------------
# 下载编排器
# ---------------------------------------------------------------------------
def _normalize_job(job: dict) -> dict:
    base = {"id": str(job.get("id") or ""), "label": str(job.get("label") or ""),
            "state": str(job.get("state") or "queued"),
            "progress_pct": job.get("progress_pct") or 0,
            "indeterminate": bool(job.get("indeterminate", False)),
            "stage": str(job.get("stage") or ""), "bytes": int(job.get("bytes") or 0),
            "total_bytes": int(job.get("total_bytes") or 0),
            "speed_bytes_s": job.get("speed_bytes_s"), "eta_seconds": job.get("eta_seconds"),
            "source": str(job.get("source") or ""), "attempts": list(job.get("attempts") or [])}
    if base["state"] not in {"queued", "running", "done", "failed", "cancelled"}:
        base["state"] = "queued"
    return base


class OnboardingInstaller:
    """一个后台线程，按顺序执行下载任务；支持取消；进度统一成 jobs[] 格式。

    ``add_job(id, label, run, progress)``：``run`` 是同步可调用对象，返回
    ``{ok: bool, error?: str, attempts?: [...], source?: str}``；``progress``
    返回当前进度 dict（bytes / total_bytes / stage / source 等，可为 None）。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._jobs: list[dict] = []

    def add_job(self, job_id: str, label: str, run, progress=None) -> None:
        with self._lock:
            if any(j["id"] == job_id for j in self._jobs):
                return
            self._jobs.append({
                "id": job_id, "label": label, "state": "queued",
                "progress_pct": 0, "indeterminate": False, "stage": "",
                "bytes": 0, "total_bytes": 0, "speed_bytes_s": None,
                "eta_seconds": None, "source": "", "attempts": [],
                "_run": run, "_progress": progress})

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [_normalize_job({k: v for k, v in j.items()
                                    if not k.startswith("_")}) for j in self._jobs]

    def cancel(self) -> None:
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _poll(self, job: dict) -> None:
        progress = job.get("_progress")
        if not callable(progress):
            return
        try:
            info = progress() or {}
        except Exception:  # noqa: BLE001
            return
        if not isinstance(info, dict):
            return
        with self._lock:
            if info.get("state"):
                # 下载中阶段不外露 failed；由 run 返回决定最终态
                if info.get("state") == "running":
                    job["state"] = "running"
            pct = info.get("progress_pct")
            if pct is None:
                # 通用兜底：只给 bytes/total 时也能算出进度卡百分比。
                received = info.get("bytes") or info.get("bytes_received") or 0
                total = info.get("total_bytes") or info.get("expected_bytes_estimate") or 0
                try:
                    pct = (round(float(received) * 100.0 / float(total), 1)
                           if float(total) > 0 else None)
                except (TypeError, ValueError):
                    pct = None
            if pct is not None:
                job["progress_pct"] = pct
            job["indeterminate"] = bool(info.get("indeterminate", False))
            job["stage"] = str(info.get("stage") or "")
            job["bytes"] = int(info.get("bytes") or info.get("bytes_received") or 0)
            job["total_bytes"] = int(info.get("total_bytes") or info.get("expected_bytes_estimate") or 0)
            job["speed_bytes_s"] = info.get("speed_bytes_s")
            job["eta_seconds"] = info.get("eta_seconds")
            job["source"] = str(info.get("source") or info.get("source_name") or "")

    def run_all(self, wait: bool = False) -> None:
        """启动后台线程按顺序跑；wait=True 时同步等完（仅测试用）。

        重复调用时：已有线程在跑就直接返回；新一轮开始会重置上一次的取消
        标志，并且只处理还没结束的任务。每个任务运行期间由独立线程每 0.5
        秒调用一次 ``_poll``，让进度卡能拿到中间进度，而不是只在结束时跳 100。
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._cancel.is_set():
                # 上一轮被取消过；用户重新发起下载时允许新一轮继续。
                self._cancel = threading.Event()
            jobs = list(self._jobs)

        def work() -> None:
            for job in jobs:
                with self._lock:
                    if job.get("state") in {"done", "failed", "cancelled"}:
                        continue
                if self._cancel.is_set():
                    with self._lock:
                        job["state"] = "cancelled"
                    continue
                with self._lock:
                    job["state"] = "running"
                run = job.get("_run")
                if not callable(run):
                    with self._lock:
                        job["state"] = "failed"
                    continue
                monitor_stop = threading.Event()
                monitor = None
                if callable(job.get("_progress")):
                    def poll_loop(target=job, stop=monitor_stop):
                        while not stop.is_set():
                            self._poll(target)
                            stop.wait(0.5)
                    monitor = threading.Thread(target=poll_loop,
                                               name="mp-onboard-progress", daemon=True)
                    monitor.start()
                try:
                    result = run() or {}
                except Exception as exc:  # noqa: BLE001
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                finally:
                    monitor_stop.set()
                    if monitor is not None:
                        monitor.join(timeout=2)
                # 通用约定：run() 返回 dict 时读 ok/error/attempts/source；
                # 返回其它真值（例如下载函数直接返回文件路径）一律视为成功。
                if not isinstance(result, dict):
                    result = {"ok": True}
                with self._lock:
                    job["attempts"] = list(result.get("attempts") or [])
                    if result.get("source"):
                        job["source"] = str(result["source"])
                    if result.get("ok"):
                        job["state"] = "done"
                        job["progress_pct"] = 100
                    else:
                        job["state"] = "failed"
                        job["stage"] = str(result.get("error") or "")[:200]

        self._thread = threading.Thread(target=work, name="mp-onboard-install", daemon=True)
        self._thread.start()
        if wait:
            self._thread.join()

    def done(self) -> bool:
        with self._lock:
            return all(j["state"] in {"done", "failed", "cancelled"} for j in self._jobs)
