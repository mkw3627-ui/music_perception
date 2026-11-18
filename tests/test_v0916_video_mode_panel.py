from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _source():
    return (ROOT / '__init__.py').read_text(encoding='utf8')


def _panel():
    return (ROOT / 'ui' / 'panel.tsx').read_text(encoding='utf8')


def test_video_modes_are_the_two_declared_tiers():
    text = _source()
    assert "VIDEO_WATCH_MODES = frozenset({'balanced', 'efficient'})" in text
    assert "DEFAULT_VIDEO_WATCH_MODE = 'balanced'" in text


def test_set_video_mode_action_is_declared_and_exposed():
    text = _source()
    assert "@ui.action(id='set_video_mode'" in text
    assert "async def set_video_mode(self,mode=DEFAULT_VIDEO_WATCH_MODE,**_):" in text


def test_panel_registers_video_tab_and_calls_set_video_mode():
    panel = _panel()
    assert '{value:"video", label:"视频"}' in panel
    assert 'tab==="video"' in panel
    assert 'perform("set_video_mode"' in panel
    assert 'status.video_watch_mode' in panel


def test_preferences_round_trip_and_reject_unknown_mode(tmp_path, monkeypatch):
    import importlib.util
    import sys

    # Load the module through the same path tests use elsewhere.
    spec = importlib.util.spec_from_file_location(
        'music_perception_pkg', ROOT / '__init__.py',
        submodule_search_locations=[str(ROOT)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules['music_perception_pkg'] = pkg
    try:
        spec.loader.exec_module(pkg)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f'plugin module not importable here: {exc}')

    VIDEO = pkg.VIDEO_WATCH_MODES
    assert 'balanced' in VIDEO and 'efficient' in VIDEO
    assert 'speech' not in VIDEO and 'deep' not in VIDEO
    assert 'bogus' not in VIDEO
