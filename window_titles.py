"""Read-only top-level window titles, used only as a *hint* for app tags.

Windows never reports which window produced which PCM, and a window title is
not proof of content. A title can only suggest "this audible process is
probably playing a video", so callers must keep the resulting tag unconfirmed.

Only the title-bar text Windows already publishes is read here: no screen
pixels, no page content and no keystrokes.
"""
from __future__ import annotations

import sys

MAX_TITLE = 200
MAX_DEPTH = 6


def _user32():
    if sys.platform != "win32":
        raise OSError("window titles are only available on Windows")
    import ctypes
    from ctypes import wintypes
    return ctypes, wintypes, ctypes.WinDLL("user32", use_last_error=True)


def titles_by_pid(limit=400):
    """Map pid -> every visible titled window title, topmost first.

    One process can own several windows (a browser has one per window), and the
    audible tab is not necessarily the front one, so all titles are kept.
    Failures return what was collected.
    """
    collected = {}
    try:
        ct, w, u = _user32()
    except (OSError, AttributeError):
        return collected
    try:
        u.EnumWindows.argtypes = [ct.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM), w.LPARAM]
        u.EnumWindows.restype = w.BOOL
        u.IsWindowVisible.argtypes = [w.HWND]
        u.IsWindowVisible.restype = w.BOOL
        u.GetWindowTextLengthW.argtypes = [w.HWND]
        u.GetWindowTextLengthW.restype = ct.c_int
        u.GetWindowTextW.argtypes = [w.HWND, w.LPWSTR, ct.c_int]
        u.GetWindowTextW.restype = ct.c_int
        u.GetWindowThreadProcessId.argtypes = [w.HWND, ct.POINTER(w.DWORD)]
        u.GetWindowThreadProcessId.restype = w.DWORD
    except (AttributeError, OSError):
        return collected

    def _collect(hwnd, _param):
        if len(collected) >= limit:
            return False
        try:
            if not u.IsWindowVisible(hwnd):
                return True
            length = u.GetWindowTextLengthW(hwnd)
            if length <= 0 or length > 1024:
                return True
            buffer = ct.create_unicode_buffer(MAX_TITLE + 1)
            if not u.GetWindowTextW(hwnd, buffer, MAX_TITLE + 1):
                return True
            pid = w.DWORD(0)
            u.GetWindowThreadProcessId(hwnd, ct.byref(pid))
            title = buffer.value.strip()
            if pid.value > 4 and title:
                collected.setdefault(int(pid.value), [])
                if title not in collected[int(pid.value)] and len(collected[int(pid.value)]) < 8:
                    collected[int(pid.value)].append(title)
        except (OSError, ValueError, AttributeError):
            return True
        return True

    try:
        callback = ct.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)(_collect)
        u.EnumWindows(callback, 0)
    except (OSError, ValueError, AttributeError):
        pass
    return collected


def foreground_window():
    """(pid, title) of the window in the foreground, or (0, "")."""
    try:
        ct, w, u = _user32()
        u.GetForegroundWindow.restype = w.HWND
        u.GetWindowTextLengthW.argtypes = [w.HWND]
        u.GetWindowTextLengthW.restype = ct.c_int
        u.GetWindowTextW.argtypes = [w.HWND, w.LPWSTR, ct.c_int]
        u.GetWindowTextW.restype = ct.c_int
        u.GetWindowThreadProcessId.argtypes = [w.HWND, ct.POINTER(w.DWORD)]
        u.GetWindowThreadProcessId.restype = w.DWORD
        hwnd = u.GetForegroundWindow()
        if not hwnd:
            return 0, ""
        length = u.GetWindowTextLengthW(hwnd)
        if length <= 0 or length > 1024:
            return 0, ""
        buffer = ct.create_unicode_buffer(MAX_TITLE + 1)
        if not u.GetWindowTextW(hwnd, buffer, MAX_TITLE + 1):
            return 0, ""
        pid = w.DWORD(0)
        u.GetWindowThreadProcessId(hwnd, ct.byref(pid))
        return int(pid.value), buffer.value.strip()
    except (OSError, AttributeError, ValueError):
        return 0, ""


def parent_pid(pid):
    """Kernel parent pid; 0 when unavailable (exited or protected process)."""
    try:
        if sys.platform != "win32":
            return 0
        import ctypes
        from ctypes import wintypes
        ct, w = ctypes, wintypes
        if ct.sizeof(ct.c_void_p) != 8:
            return 0
        k = ct.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        k.OpenProcess.restype = w.HANDLE
        k.CloseHandle.argtypes = [w.HANDLE]
        k.CloseHandle.restype = w.BOOL
        handle = k.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
        if not handle:
            return 0
        try:
            ntdll = ct.WinDLL("ntdll")
            info = ct.create_string_buffer(48)
            needed = w.ULONG(0)
            ntdll.NtQueryInformationProcess.argtypes = [w.HANDLE, ct.c_int, ct.c_void_p,
                                                        w.ULONG, ct.POINTER(w.ULONG)]
            if ntdll.NtQueryInformationProcess(handle, 0, info, len(info),
                                               ct.byref(needed)) != 0:
                return 0
            return int.from_bytes(info.raw[0x28:0x30], "little")
        finally:
            k.CloseHandle(handle)
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


def _as_titles(value):
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item or "").strip())
    return ()


def image_name(pid):
    """Executable file name of a process, or "" when it cannot be read."""
    try:
        if sys.platform != "win32":
            return ""
        import ctypes
        from ctypes import wintypes
        ct, w = ctypes, wintypes
        k = ct.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        k.OpenProcess.restype = w.HANDLE
        k.QueryFullProcessImageNameW.argtypes = [w.HANDLE, w.DWORD, w.LPWSTR,
                                                 ct.POINTER(w.DWORD)]
        k.QueryFullProcessImageNameW.restype = w.BOOL
        k.CloseHandle.argtypes = [w.HANDLE]
        k.CloseHandle.restype = w.BOOL
        handle = k.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ""
        try:
            buffer = ct.create_unicode_buffer(32768)
            size = w.DWORD(len(buffer))
            if not k.QueryFullProcessImageNameW(handle, 0, buffer, ct.byref(size)):
                return ""
            return buffer.value.replace("/", "\\").split("\\")[-1].casefold()
        finally:
            k.CloseHandle(handle)
    except (OSError, ValueError, TypeError, AttributeError):
        return ""


def resolve_titles(pid, titles, *, foreground=None, max_depth=MAX_DEPTH):
    """Every window title that belongs to a process, falling back to ancestors.

    Chromium renders audio in a child process, so the audible pid often has no
    window of its own. Returns (titles, owner_pid, hops); ((), 0, 0) if unknown.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return (), 0, 0
    titles = titles or {}
    own = _as_titles(titles.get(pid))
    if own:
        return own, pid, 0
    # A child process may only borrow a title from the SAME executable: that is
    # the Chromium case (renderer -> browser). Any other parent (explorer, a
    # launcher, an installer) says nothing about what the child is playing, so
    # walking it would fabricate video evidence.
    family = image_name(pid)
    if not family:
        return (), 0, 0
    chain = []
    current = pid
    for _ in range(max(1, int(max_depth))):
        parent = parent_pid(current)
        if parent <= 4 or parent in chain:
            break
        if image_name(parent) != family:
            break
        chain.append(parent)
        current = parent
    if not chain:
        return (), 0, 0
    if isinstance(foreground, (tuple, list)) and len(foreground) == 2:
        fg_pid = int(foreground[0] or 0)
        fg_title = str(foreground[1] or "")
        if fg_title and (fg_pid == pid or fg_pid in chain):
            return (fg_title,), fg_pid, 0
    for hop, ancestor in enumerate(chain, start=1):
        found = _as_titles(titles.get(ancestor))
        if found:
            return found, ancestor, hop
    return (), 0, 0

def windows_by_pid(limit=400):
    """pid -> [{"title": str, "minimized": bool}]。

    最小化的窗口同样在列表里，标题也读得到——用它判断「在放什么」，
    不需要还原窗口、更不需要截图。
    """
    rows = {}
    try:
        ct, w, u = _user32()
    except (OSError, AttributeError):
        return rows
    try:
        u.IsIconic.argtypes = [w.HWND]
        u.IsIconic.restype = w.BOOL
    except AttributeError:
        return rows

    def _collect(hwnd, _param):
        if sum(len(v) for v in rows.values()) >= limit:
            return False
        try:
            if not u.IsWindowVisible(hwnd):
                return True
            length = u.GetWindowTextLengthW(hwnd)
            if length <= 0 or length > 1024:
                return True
            buffer = ct.create_unicode_buffer(MAX_TITLE + 1)
            if not u.GetWindowTextW(hwnd, buffer, MAX_TITLE + 1):
                return True
            pid = w.DWORD(0)
            u.GetWindowThreadProcessId(hwnd, ct.byref(pid))
            title = buffer.value.strip()
            if pid.value > 4 and title:
                rows.setdefault(int(pid.value), [])
                if len(rows[int(pid.value)]) < 8:
                    rows[int(pid.value)].append(
                        {"title": title, "minimized": bool(u.IsIconic(hwnd))})
        except (OSError, ValueError, AttributeError):
            return True
        return True

    try:
        callback = ct.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)(_collect)
        u.EnumWindows(callback, 0)
    except (OSError, ValueError, AttributeError):
        pass
    return rows
