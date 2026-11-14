"""Confirm N.E.K.O can get a startup acknowledgement before any slow installer.

These tests mock the SDK and do not perform network I/O or audio capture.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import music_perception as plugin_module
from _version import ui_build, panel_ui_build, plugin_version  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _instance(tmp_path):
    plugin = object.__new__(plugin_module.MusicPerceptionPlugin)
    plugin.data_path = lambda name='': tmp_path / name
    plugin.logger = logging.getLogger('test-startup-readiness')
    # 安装门新规则：只有引导已完成/位置已确认才允许自动下载。本文件的用例
    # 验证的是「授权之后」的启动、修复与守护行为，所以准备步骤里先写一条
    # 「引导已完成」，断言本身不动。
    store = plugin_module.OnboardingStore(tmp_path)
    store.save(completed=True, step=5, storage_root=str(tmp_path))
    plugin._onboarding_store = store
    return plugin


def test_manifest_allows_longer_worker_handshake():
    manifest = tomllib.loads((ROOT / 'plugin.toml').read_text(encoding='utf8'))
    runtime = manifest['plugin_runtime']
    assert manifest['plugin']['version'] == plugin_version()
    assert 10 < runtime['timeout'] <= 300
    assert runtime['auto_start'] is True


def test_environment_does_not_run_in_startup_hook(monkeypatch, tmp_path):
    calls = []

    class SlowInstaller:
        def __init__(self, **kwargs):
            self.called = False

        def snapshot(self):
            return {'ready': False, 'model_ready': False, 'phase': 'waiting', 'stage': 'waiting'}

        def ensure(self):
            self.called = True
            calls.append('start')
            time.sleep(0.12)  # pretend to be a slow network operation
            calls.append('end')
            return {'ready': False, 'error': 'no network'}

        def stop(self):
            calls.append('stop')

    monkeypatch.setattr(plugin_module, 'MusicEnvironment', SlowInstaller)
    monkeypatch.setattr(plugin_module, 'ENVIRONMENT_START_DELAY_SECONDS', 0.10)
    plugin = _instance(tmp_path)

    async def check():
        began = time.monotonic()
        result = await plugin.startup()
        assert time.monotonic() - began < 0.5
        assert result.value['status'] == 'ready'
        assert calls == []
        # Still within the post-start handshake grace period.
        await asyncio.sleep(0.02)
        assert calls == []
        await asyncio.wait_for(plugin._environment_task, 2.0)
        assert calls == ['start', 'end']
        await plugin.shutdown()

    asyncio.run(check())


def test_shutdown_before_prepare_never_runs_installer(monkeypatch, tmp_path):
    called = []

    class NoInstaller:
        def __init__(self, **kwargs):
            pass

        def snapshot(self):
            return {'phase': 'waiting', 'ready': False, 'model_ready': False}

        def ensure(self):
            called.append('ran')
            raise AssertionError('installer should never run after shutdown')

        def stop(self):
            called.append('stopped')

    monkeypatch.setattr(plugin_module, 'MusicEnvironment', NoInstaller)
    monkeypatch.setattr(plugin_module, 'ENVIRONMENT_START_DELAY_SECONDS', 0.3)
    plugin = _instance(tmp_path)

    async def check():
        await plugin.startup()
        await plugin.shutdown()
        await asyncio.sleep(0.31)
        assert called == ['stopped']
        assert plugin._environment_task.cancelled()

    asyncio.run(check())


def test_repair_failure_does_not_crash_plugin(monkeypatch, tmp_path):
    class FailingEnv:
        def __init__(self, **kwargs):
            pass

        def snapshot(self):
            return {'phase': 'waiting', 'ready': False, 'model_ready': False}

        def ensure(self):
            raise RuntimeError('synthetic native extension failure')

        def stop(self):
            pass

    monkeypatch.setattr(plugin_module, 'MusicEnvironment', FailingEnv)
    monkeypatch.setattr(plugin_module, 'ENVIRONMENT_START_DELAY_SECONDS', 0.01)
    plugin = _instance(tmp_path)

    async def check():
        assert (await plugin.startup()).value['status'] == 'ready'
        await asyncio.wait_for(plugin._environment_task, 2.0)
        assert 'RuntimeError' in plugin._monitor_start_error
        await plugin.shutdown()

    asyncio.run(check())


def test_autostart_watchdog_recovers_when_lifecycle_task_does_not(monkeypatch, tmp_path):
    plugin = _instance(tmp_path)
    ready = threading.Event()

    class Environment:
        def snapshot(self):
            return {"ready": False, "model_ready": False}

        def ensure(self):
            return {"ready": True, "model_ready": True}

    async def start_live():
        ready.set()
        return {"ok": True}

    plugin._environment = Environment()
    plugin._auto_consented = True
    plugin._plugin_paused = False
    plugin._shutdown_requested = False
    plugin._listener = None
    plugin._start_live = start_live
    plugin._start_audible_scanner = lambda: None
    monkeypatch.setattr(plugin_module, "ENVIRONMENT_START_DELAY_SECONDS", -4.0)

    plugin._start_autostart_watchdog()
    try:
        assert ready.wait(2.0)
    finally:
        plugin._autostart_stop.set()
        plugin._autostart_thread.join(timeout=2.0)


def test_watchdog_waiting_source_branch_defines_result_and_starts_scanner(monkeypatch, tmp_path):
    # Regression: the watchdog reads `result` in the waiting_source branch. It
    # must be defined on every path, and that branch must start the scanner.
    plugin = _instance(tmp_path)
    scanner_started = threading.Event()
    done = threading.Event()

    class Environment:
        def snapshot(self):
            return {'ready': True, 'model_ready': True}

    def start_live():
        done.set()
        return {'ok': False, 'waiting_source': True, 'reason': 'waiting_for_saved_app_audio'}

    async def fake_start_live():
        return start_live()

    plugin._environment = Environment()
    plugin._auto_consented = True
    plugin._plugin_paused = False
    plugin._shutdown_requested = False
    plugin._listener = None
    plugin._start_live = fake_start_live
    plugin._start_audible_scanner = lambda: scanner_started.set()
    plugin._set_capture_recovery = lambda *a, **k: None
    monkeypatch.setattr(plugin_module, 'ENVIRONMENT_START_DELAY_SECONDS', -4.0)

    plugin._start_autostart_watchdog()
    try:
        assert done.wait(2.0)
        assert scanner_started.wait(2.0)
        assert plugin._autostart_thread.is_alive()
    finally:
        plugin._autostart_stop.set()
        plugin._autostart_thread.join(timeout=2.0)


def test_shutdown_stops_autostart_watchdog(monkeypatch, tmp_path):
    plugin = _instance(tmp_path)
    stopped = threading.Event()
    plugin._shutdown_requested = False
    plugin._autostart_stop = stopped
    plugin._start_resource_monitor = lambda: None

    asyncio.run(plugin.shutdown())

    assert stopped.is_set()
