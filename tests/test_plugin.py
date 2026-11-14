from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import soundfile as sf
import numpy as np

from music_perception import MusicPerceptionPlugin
from plugin.sdk.plugin import Ok


class FakeConfig:
    def __init__(self, roots):
        self.roots = roots

    async def dump(self, timeout=5.0):
        return {"music_perception": {"allowed_roots": [str(x) for x in self.roots], "segment_seconds": 4}}


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


@pytest.fixture
def plugin_fixture(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    plugin = object.__new__(MusicPerceptionPlugin)
    plugin.config = FakeConfig([music])
    plugin.logger = FakeLogger()
    plugin._default_roots = lambda: []
    base = tmp_path / "data"
    def fake_data_path(*components):
        return base.joinpath(*components)
    plugin.data_path = fake_data_path
    called = []
    def push_message(**kwargs):
        called.append(kwargs)
    plugin.push_message = push_message
    plugin.ctx = SimpleNamespace(plugins=SimpleNamespace())
    try:
        yield plugin, music, other, called
    finally:
        # Some legacy integration tests explicitly start daemon workers but
        # do not call the N.E.K.O shutdown lifecycle. Avoid leaving workers
        # running when pytest (and its Python interpreter) is shutting down.
        if (getattr(plugin, '_background_worker', None) is not None or
                getattr(plugin, '_lyrics_worker_thread', None) is not None):
            plugin._shutdown_requested = True
            try:
                plugin._stop_background_processor()
            except (RuntimeError, OSError):
                pass


def test_authorized_and_unauthorized_file(plugin_fixture):
    plugin, music, other, _ = plugin_fixture
    good = music / "good.wav"
    bad = other / "outside.wav"
    for path in [good, bad]:
        sf.write(path, np.zeros(2 * 16000, dtype=np.float32), 16000)
    assert asyncio.run(plugin._resolve_audio(str(good)))[0] == good.resolve()
    with pytest.raises(Exception, match="不在允许"):
        asyncio.run(plugin._resolve_audio(str(bad)))
    with pytest.raises(Exception, match="绝对路径"):
        asyncio.run(plugin._resolve_audio("good.wav"))
    with pytest.raises(Exception, match="不允许 URL"):
        asyncio.run(plugin._resolve_audio("https://example.org/song.wav"))


def test_symlink_outside_blocked(plugin_fixture):
    plugin, music, other, _ = plugin_fixture
    target = other / "nope.wav"
    sf.write(target, np.zeros(16000, dtype=np.float32), 16000)
    symlink = music / "symlink.wav"
    try:
        symlink.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink unavailable")
    with pytest.raises(Exception, match="不在允许"):
        asyncio.run(plugin._resolve_audio(str(symlink)))


def test_scan_files_only_authorized(plugin_fixture):
    plugin, music, other, _ = plugin_fixture
    sf.write(music / "one.wav", np.zeros(16000, dtype=np.float32), 16000)
    sf.write(other / "two.wav", np.zeros(16000, dtype=np.float32), 16000)
    r = asyncio.run(plugin.scan_music()).value
    assert r["ok"] is True
    assert [f["file_name"] for f in r["files"]] == ["one.wav"]


def test_analyze_and_query_segment(plugin_fixture):
    plugin, music, _other, called = plugin_fixture
    p = music / "positive-ish.wav"
    sr = 16000
    times = np.arange(16000 * 7) / sr
    samples = .25 * np.sin(2 * np.pi * 440 * times)
    sf.write(p, samples.astype(np.float32), sr)
    asyncio.run(plugin.startup())
    value = asyncio.run(plugin.analyze_song(str(p), "brief")).value
    assert value["ok"] is True
    assert value["report"]["processed_entire_song"] is True
    full = asyncio.run(plugin.get_last_report()).value
    assert full["ok"] is True
    assert full["report"]["duration_s"] == 7
    result = asyncio.run(plugin.inspect_segment(0, 4)).value
    assert result["ok"] is True
    assert len(result["spectral_band_levels_low_to_high_0_100"]) == 32
    assert asyncio.run(plugin.inspect_segment(0, float("nan"))).value["ok"] is False
    legacy = asyncio.run(plugin.tell_ai()).value
    assert legacy['ok'] and legacy['status'] == 'silent_data_only'
    assert 'analysis' in legacy and not called  # old tool never forces a chat response


def test_short_system_capture_is_opt_in(plugin_fixture):
    plugin, _music, _other, _called = plugin_fixture
    calls = []
    async def call_entry(entry_id, arguments, *, timeout):
        calls.append((entry_id, arguments, timeout))
        return Ok({"available": True, "interpretation": "有声音"})
    plugin.ctx.plugins.call_entry = call_entry
    result = asyncio.run(plugin.listen_now(4)).value
    assert result["ok"] is True
    assert calls[0][0] == "keyboard_controller:analyze_audio"
    assert calls[0][1] == {"duration": 4}
