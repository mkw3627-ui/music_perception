# -*- coding: utf-8 -*-
"""统一读取版本号，避免测试里写死（2026-11-13 新手引导改造）。"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def plugin_version() -> str:
    """pyproject.toml 里的 [project].version。"""
    try:
        import tomllib
    except ModuleNotFoundError:  # py3.10
        import tomli as tomllib  # type: ignore
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def ui_build() -> str:
    """__init__.py 里的 UI_PROTOCOL_BUILD 常量值。"""
    text = (_ROOT / "__init__.py").read_text(encoding="utf-8", errors="replace")
    m = re.search(r'UI_PROTOCOL_BUILD\s*=\s*["\']([^"\']+)["\']', text)
    if not m:
        raise AssertionError("__init__.py 里找不到 UI_PROTOCOL_BUILD")
    return m.group(1)


def panel_ui_build() -> str:
    """ui/panel.tsx 里的 UI_PROTOCOL_BUILD 常量值（前端）。"""
    text = (_ROOT / "ui" / "panel.tsx").read_text(encoding="utf-8", errors="replace")
    m = re.search(r'UI_PROTOCOL_BUILD\s*=\s*["\']([^"\']+)["\']', text)
    if not m:
        raise AssertionError("panel.tsx 里找不到 UI_PROTOCOL_BUILD")
    return m.group(1)
