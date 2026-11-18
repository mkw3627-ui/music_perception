"""Light-weight N.E.K.O SDK contract simulation; not a real host integration test."""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def build_plugin(monkeypatch, tmp_path, *, music_roots=None):
    def wrap_dec(**kwargs):
        return lambda func: func

    class Ok:
        def __init__(self, value):
            self.value = value

    class FakeBase:
        def data_path(self, name):
            return tmp_path / "data" / name
        def push_message(self, **kw):
            self.sent.append(kw)
            return {"submitted": True}

    fake_pkg = types.ModuleType("plugin")
    fake_pkg.__path__ = []
    fake_sdk = types.ModuleType("plugin.sdk")
    fake_sdk.__path__ = []
    fake_subpkg = types.ModuleType("plugin.plugins")
    fake_subpkg.__path__ = []
    sdk_plugin = types.ModuleType("plugin.sdk.plugin")
    sdk_plugin.NekoPluginBase = FakeBase
    sdk_plugin.Ok = Ok
    sdk_plugin.lifecycle = wrap_dec
    sdk_plugin.llm_tool = wrap_dec
    sdk_plugin.plugin_entry = wrap_dec
    sdk_plugin.ui = SimpleNamespace(action=wrap_dec, context=wrap_dec)
    sdk_plugin.neko_plugin = lambda cls: cls
    for key, value in {"plugin": fake_pkg, "plugin.sdk": fake_sdk,
                       "plugin.plugins": fake_subpkg, "plugin.sdk.plugin": sdk_plugin}.items():
        monkeypatch.setitem(sys.modules, key, value)
    spec = importlib.util.spec_from_file_location(
        "plugin.plugins.music_perception", ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    from conftest import ReadyTestEnvironment
    monkeypatch.setattr(module, 'MusicEnvironment', ReadyTestEnvironment)
    # This file loads the plugin under its own module name, so the conftest
    # autouse stub does not reach this class. Disable the Windows scanner here.
    monkeypatch.setattr(module.MusicPerceptionPlugin, '_start_audible_scanner', lambda self: None)
    plugin = module.MusicPerceptionPlugin()
    plugin.sent = []
    plugin.logger = logging.getLogger("test-music-perception")
    cfg = {"music_perception": {"allowed_roots": [str(x) for x in (music_roots or [])],
                                "segment_seconds": 4, "max_seconds": 300}}
    async def dump(timeout):
        return cfg
    plugin.config = SimpleNamespace(dump=dump)
    return plugin, Ok


def synthetic_wav(file: Path):
    file.parent.mkdir(parents=True, exist_ok=True)
    sr = 16000
    t = np.arange(sr * 7, dtype=np.float32) / sr
    y = .3 * np.sin(2 * np.pi * 440 * t)
    for click in range(1, 7):
        y[click*sr:click*sr+160] += .4
    sf.write(file, np.clip(y, -.99, .99), sr)
    return file


def test_authorized_local_analysis_and_ai_push(monkeypatch, tmp_path):
    allowed = tmp_path / "authorized"
    allowed.mkdir()
    track = synthetic_wav(allowed / "sample.wav")
    plugin, Ok = build_plugin(monkeypatch, tmp_path, music_roots=[allowed])

    async def run():
        res = await plugin.startup()
        assert isinstance(res, Ok)
        assert plugin.sent == []  # Never auto-push/audio-record on startup.
        full = await plugin.analyze_song(path=str(track), detail="full")
        assert full.value["ok"] is True
        report = full.value["report"]
        assert report["segments"][-1]["end_s"] == 7
        assert report["file_name"] == "sample.wav"
        saved = await plugin.get_last_report()
        assert saved.value["report"] == report
        brief = await plugin.analyze_song(path=str(track), detail="brief")
        assert "spectrum" not in brief.value["report"]
        assert brief.value["cache_hit"] is True
        query = await plugin.inspect_segment(start_s=1, end_s=3)
        assert query.value["ok"] is True
        assert query.value["start_s"] == 1
        assert plugin.sent == []
        pushed = await plugin.tell_ai()
        assert pushed.value["ok"] is True
        assert pushed.value['status'] == 'silent_data_only'
        assert pushed.value['analysis']['processed_entire_song'] is True
        assert plugin.sent == []  # no unsolicited or forced chat messages
    asyncio.run(run())


def test_cog_poll_denies_without_monitor_or_ai_authorization(monkeypatch, tmp_path):
    # Regression: cog_poll_v1 is the only cognition-bridge entry point. It must
    # refuse to return observations unless both monitoring and AI delivery are
    # explicitly authorized, and must never fall through to a success shape.
    plugin, Ok = build_plugin(monkeypatch, tmp_path)

    async def run():
        plugin._auto_consented = False
        plugin._ai_delivery = False
        denied = await plugin.cog_poll_v1()
        assert denied.value['status'] == 'denied'
        assert denied.value['reason_code'] == 'monitor_not_authorized'
        assert denied.value['observations'] == []

        plugin._auto_consented = True
        plugin._ai_delivery = False
        denied = await plugin.cog_poll_v1()
        assert denied.value['status'] == 'denied'
        assert denied.value['reason_code'] == 'ai_delivery_not_authorized'
        assert denied.value['observations'] == []
        assert plugin.sent == []

    asyncio.run(run())


def test_root_validation_rejects_escapes(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = synthetic_wav(tmp_path / "secret" / "track.wav")
    (allowed / "link.wav").symlink_to(outside)
    plugin, _ = build_plugin(monkeypatch, tmp_path, music_roots=[allowed])

    async def run():
        await plugin.startup()
        escaped = await plugin.analyze_song(path=str(allowed / "link.wav"))
        assert escaped.value["ok"] is False
        assert "允许" in escaped.value["error"]
        remote = await plugin.analyze_song(path="https://example.com/test.mp3")
        assert remote.value["ok"] is False
        assert "本地" in remote.value["error"]
        denied = await plugin.analyze_song(path=str(outside))
        assert denied.value["ok"] is False
    asyncio.run(run())


def test_loopback_only_when_explicitly_invoked(monkeypatch, tmp_path):
    plugin, Ok = build_plugin(monkeypatch, tmp_path)
    calls = []
    async def call_entry(name, args, timeout):
        calls.append((name, args, timeout))
        return Ok({"available": True, "interpretation": "音乐的频谱特征"})
    plugin.ctx = SimpleNamespace(plugins=SimpleNamespace(call_entry=call_entry))

    async def run():
        await plugin.startup()
        assert not calls
        result = await plugin.listen_now(duration=5)
        assert result.value["ok"] is True
        assert calls[0][0] == "keyboard_controller:analyze_audio"
        assert calls[0][1]["duration"] == 5
        assert result.value["audio"]["available"] is True
    asyncio.run(run())


def test_repeated_analysis_uses_fingerprint_cache_and_invalidates(monkeypatch, tmp_path):
    """Second pass skips CPU work; different file mtimes or settings trigger fresh analysis."""
    allowed = tmp_path / "authorized"
    allowed.mkdir()
    track = synthetic_wav(allowed / "cached.wav")
    plugin, _ = build_plugin(monkeypatch, tmp_path, music_roots=[allowed])
    module = sys.modules[plugin.__class__.__module__]
    actual_analysis = module.analyze_audio
    counter = {"calls": 0}

    def spy(*args, **kwargs):
        counter["calls"] += 1
        return actual_analysis(*args, **kwargs)

    monkeypatch.setattr(module, "analyze_audio", spy)

    async def run():
        await plugin.startup()
        one = (await plugin.analyze_song(path=str(track), detail="brief")).value
        two = (await plugin.analyze_song(path=str(track), detail="full")).value
        assert one["ok"] and two["ok"]
        assert one["cache_hit"] is False and two["cache_hit"] is True
        assert counter["calls"] == 1

        # Change only the segmentation config -- the cache must not reuse stale segments.
        old_dump = plugin.config.dump
        async def changed_cfg(timeout):
            cfg = await old_dump(timeout)
            cfg["music_perception"]["segment_seconds"] = 5
            return cfg
        plugin.config.dump = changed_cfg
        three = (await plugin.analyze_song(path=str(track), detail="brief")).value
        assert three["ok"] and three["cache_hit"] is False
        assert counter["calls"] == 2

        # A modified audio file must be reanalyzed even if its path is unchanged.
        original_mtime = track.stat().st_mtime_ns
        import os
        os.utime(track, ns=(original_mtime + 3_000_000_000, original_mtime + 3_000_000_000))
        four = (await plugin.analyze_song(path=str(track), detail="brief")).value
        assert four["ok"] and four["cache_hit"] is False
        assert counter["calls"] == 3
    asyncio.run(run())
