"""Checks plugin.toml against the official N.E.K.O source-layout basics."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / 'tools' / 'check_manifest.py'
spec = importlib.util.spec_from_file_location('check_manifest', FILE)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def scratch_source(tmp_path: Path) -> Path:
    source = tmp_path / 'music_perception'
    source.mkdir()
    for relative in ('plugin.toml', 'pyproject.toml', '__init__.py', 'ui/panel.tsx'):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return source


def test_current_manifest_passes_static_official_shape():
    errors, warnings = checker.check(ROOT)
    assert not errors, errors
    assert not any('必需依赖 vendor' in item for item in warnings)


def test_rejects_manifest_not_supporting_sdk_010(tmp_path):
    source = scratch_source(tmp_path)
    p = source / 'plugin.toml'
    s = p.read_text(encoding='utf8')
    s = s.replace('supported = ">=0.1.0,<0.3.0"', 'supported = ">=0.1.1,<0.3.0"')
    p.write_text(s,encoding='utf8')
    errors, _ = checker.check(source)
    assert any('0.1.0' in line for line in errors)


def test_rejects_missing_ui_entry(tmp_path):
    source = scratch_source(tmp_path)
    (source / 'ui' / 'panel.tsx').unlink()
    errors, _ = checker.check(source)
    assert any('UI' in line for line in errors)


def test_misplaced_data_folder_shows_actionable_warning(tmp_path):
    source = tmp_path / 'N.E.K.O' / 'plugins' / 'music_perception'
    source.mkdir(parents=True)
    for relative in ('plugin.toml', 'pyproject.toml', '__init__.py', 'ui/panel.tsx'):
        dst=source/relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative,dst)
    errors, warnings = checker.check(source)
    assert not errors
    assert any('运行数据目录' in line for line in warnings)
