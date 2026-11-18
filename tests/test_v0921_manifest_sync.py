# -*- coding: utf-8 -*-
"""清单同步（2026-11-13）：__init__.py 的 @plugin_entry 与 plugin.toml 双向一致。"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

cm = importlib.import_module("check_manifest")


def test_entry_ids_extracted_from_init():
    ids = cm.extract_entry_ids(ROOT / "__init__.py")
    assert ids, "应能从 __init__.py 提取到 @plugin_entry"
    assert "music_dashboard" in ids or len(ids) > 10


def test_manifest_entries_extracted():
    ids = cm.extract_manifest_entries(ROOT / "plugin.toml")
    assert isinstance(ids, set)


def test_compare_entries_returns_shape():
    errors, warnings = cm.compare_entries(ROOT)
    assert isinstance(errors, list) and isinstance(warnings, list)


def test_no_entries_only_in_manifest():
    """plugin.toml 不应出现 __init__.py 里没有的入口。"""
    code_ids = cm.extract_entry_ids(ROOT / "__init__.py")
    toml_ids = cm.extract_manifest_entries(ROOT / "plugin.toml")
    extra = toml_ids - code_ids
    assert not extra, "plugin.toml 多出入口：%s" % sorted(extra)[:8]
