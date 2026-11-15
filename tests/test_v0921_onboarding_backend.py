# -*- coding: utf-8 -*-
"""新手引导后端单测（2026-11-13）：OnboardingStore / validate / recommend / installer。"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

_SOURCE = str(Path(__file__).resolve().parents[1])
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

import onboarding  # noqa: E402


# ── OnboardingStore ────────────────────────────────────────────────────
def test_store_roundtrip_and_defaults(tmp_path):
    store = onboarding.OnboardingStore(tmp_path, source_pref="mirror")
    assert store.load()["completed"] is False
    assert store.load()["step"] == 1
    assert store.load()["storage_root"] == ""
    store.save(completed=True, storage_root=str(tmp_path / "models"), step=5)
    doc = store.load()
    assert doc["completed"] is True
    assert doc["step"] == 5
    assert doc["storage_root"] == str(tmp_path / "models")
    assert doc["source_pref"] == "mirror"
    assert (tmp_path / "onboarding.json").is_file()


def test_store_rejects_bad_source_pref(tmp_path):
    store = onboarding.OnboardingStore(tmp_path, source_pref="evil")
    assert store.load()["source_pref"] == "mirror"


def test_store_survives_corrupt_file(tmp_path):
    (tmp_path / "onboarding.json").write_text("{not json", encoding="utf-8")
    store = onboarding.OnboardingStore(tmp_path)
    assert store.load()["completed"] is False


# ── validate_storage_root ──────────────────────────────────────────────
def test_validate_accepts_writable_directory(tmp_path):
    res = onboarding.validate_storage_root(str(tmp_path / "data"))
    assert res["ok"] is True
    assert res["writable"] is True
    assert res["free_bytes"] > 0


def test_validate_rejects_steamapps(tmp_path):
    steam = tmp_path / "steamapps" / "common"
    steam.mkdir(parents=True)
    res = onboarding.validate_storage_root(str(steam))
    assert res["ok"] is False
    assert "Steam" in res["error"]


def test_validate_rejects_plugin_dir():
    plugin_dir = Path(onboarding.__file__).resolve().parent
    res = onboarding.validate_storage_root(str(plugin_dir))
    assert res["ok"] is False


def test_validate_insufficient_space(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda p: _FakeUsage(100))
    res = onboarding.validate_storage_root(str(tmp_path / "d"), need_bytes=1000)
    assert res["ok"] is False
    assert "空间" in res["error"]


class _FakeUsage:
    def __init__(self, free):
        self.free = free
        self.total = free
        self.used = 0


# ── recommend ──────────────────────────────────────────────────────────
def _catalogs():
    return {
        "video": {"low": {"vram_mib": 1024, "bytes": 1000},
                  "medium": {"vram_mib": 1536, "bytes": 2000},
                  "high": {"vram_mib": 3072, "bytes": 3000}},
        "singing": {"low": {"vram_mib": 2300, "bytes": 1000},
                    "medium": {"vram_mib": 4200, "bytes": 2000}},
        "stem": {"low": {"vram_mib": 1024, "bytes": 1000},
                 "ultra": {"vram_mib": 3072, "bytes": 3000}},
    }


def test_recommend_gpu_picks_highest_fitting_tier():
    hw = {"gpu_available": True, "gpu_free_mib": 3000}
    rec = onboarding.recommend(hw, _catalogs(), free_bytes=10_000_000, reserve_mib=512)
    assert rec["device"] == "cuda"
    # video: high 3072+512=3584 > 3000 不满足 → medium 1536+512=2048 ≤ 3000
    assert rec["video_quality"] == "medium"
    # stem: ultra 3072+512 > 3000 → low 1024+512 ok
    assert rec["stem_quality"] == "low"


def test_recommend_cpu_falls_back_to_lowest():
    hw = {"gpu_available": False, "gpu_free_mib": None}
    rec = onboarding.recommend(hw, _catalogs(), free_bytes=10_000_000, reserve_mib=512)
    assert rec["device"] == "cpu"
    assert rec["video_quality"] == "low"
    assert rec["singing_quality"] == "low"
    assert rec["stem_quality"] == "low"


# ── OnboardingInstaller ────────────────────────────────────────────────
def test_installer_runs_jobs_in_order():
    inst = onboarding.OnboardingInstaller()
    order = []
    inst.add_job("a", "A", lambda: (order.append("a"), {"ok": True})[1])
    inst.add_job("b", "B", lambda: (order.append("b"), {"ok": True})[1])
    inst.run_all(wait=True)
    assert order == ["a", "b"]
    states = {j["id"]: j["state"] for j in inst.snapshot()}
    assert states == {"a": "done", "b": "done"}


def test_installer_marks_failure_and_records_error():
    inst = onboarding.OnboardingInstaller()
    inst.add_job("x", "X", lambda: {"ok": False, "error": "boom"})
    inst.run_all(wait=True)
    job = inst.snapshot()[0]
    assert job["state"] == "failed"
    assert job["stage"] == "boom"


def test_installer_cancel_stops_remaining():
    inst = onboarding.OnboardingInstaller()
    started = threading.Event()
    release = threading.Event()

    def slow():
        started.set()
        release.wait(timeout=3)
        return {"ok": True}

    inst.add_job("s", "slow", slow)
    inst.add_job("n", "never", lambda: {"ok": True})
    inst.run_all(wait=False)
    assert started.wait(timeout=2)
    inst.cancel()
    release.set()
    # 等线程结束
    deadline = time.time() + 3
    while not inst.done() and time.time() < deadline:
        time.sleep(0.05)
    states = {j["id"]: j["state"] for j in inst.snapshot()}
    assert states["s"] in {"done", "cancelled"}
    assert states["n"] == "cancelled"


def test_installer_maps_progress():
    inst = onboarding.OnboardingInstaller()
    progress = {"state": "running", "bytes": 50, "total_bytes": 100,
                "stage": "正在下载", "source": "清华"}
    inst.add_job("p", "P", lambda: {"ok": True}, progress=lambda: progress)
    inst.run_all(wait=True)
    job = inst.snapshot()[0]
    assert job["state"] == "done"
    assert job["progress_pct"] == 100  # done 强制 100


# ── 安装门（B-6 返工）：门关着不调 ensure，但保留可读提示 ─────────────
def test_install_gate_blocks_ensure_but_keeps_prompt(monkeypatch, tmp_path):
    """门关着（onboarding.json 存在、completed=false、无 storage_root）且环境
    未就绪时：不调用会下载的 ensure，只做只读扫描，并留下可读的引导提示。"""
    import asyncio
    import logging

    import music_perception as module

    plugin = object.__new__(module.MusicPerceptionPlugin)
    plugin.data_path = lambda name='': tmp_path / name
    plugin.logger = logging.getLogger('test-gate')

    ensure_calls = []

    class Env:
        def snapshot(self):
            return {'ready': False, 'model_ready': False}

        def ensure(self):
            ensure_calls.append(1)
            return {'ready': True, 'model_ready': True}

        def stop(self):
            pass

    plugin._environment = Env()
    plugin._shutdown_requested = False
    plugin._auto_consented = True
    plugin._plugin_paused = False
    plugin._environment_prepare_started = False

    # 门关着：onboarding.json 存在、completed=false、storage_root 空。
    store = onboarding.OnboardingStore(tmp_path)
    store.save(completed=False, step=1, storage_root='')
    plugin._onboarding_store = store

    scanned = []
    monkeypatch.setattr(plugin, '_scan_onboarding_only', lambda: scanned.append(1))

    async def run():
        await plugin._prepare_runtime_environment(defer=False)

    asyncio.run(run())

    assert ensure_calls == []            # 门关着，不触发下载
    assert scanned == [1]                # 但做了只读扫描
    assert '新手引导' in (plugin._monitor_start_error or '')  # 有可读提示
    assert getattr(plugin, '_auto_health', {}).get('state') == 'pending_onboarding'


def test_new_user_gate_closed_and_startup_prompt(monkeypatch, tmp_path):
    """全新用户（没有 onboarding.json、没有就绪标记）：启动绝不下载，显示引导。

    这是安装门的核心承诺：位置还没确认，就不许自动下载几百 MB。
    """
    import asyncio
    import logging

    import music_perception as module

    plugin = object.__new__(module.MusicPerceptionPlugin)
    plugin.data_path = lambda name='': tmp_path / name
    plugin.logger = logging.getLogger('test-gate-new-user')

    ensure_calls = []

    class Env:
        def snapshot(self):
            return {'ready': False, 'model_ready': False}

        def ensure(self):
            ensure_calls.append(1)
            return {'ready': True, 'model_ready': True}

        def stop(self):
            pass

    plugin._environment = Env()
    plugin._shutdown_requested = False
    plugin._auto_consented = True
    plugin._plugin_paused = False
    plugin._environment_prepare_started = False
    plugin._onboarding_store = onboarding.OnboardingStore(tmp_path)  # 文件不存在

    scanned = []
    monkeypatch.setattr(plugin, '_scan_onboarding_only', lambda: scanned.append(1))

    async def run():
        await plugin._prepare_runtime_environment(defer=False)

    asyncio.run(run())

    assert ensure_calls == []            # 门关：绝不触发下载
    assert scanned == [1]                # 只做只读检测
    assert '新手引导' in (plugin._monitor_start_error or '')  # 有可读引导提示
    assert getattr(plugin, '_auto_health', {}).get('state') == 'pending_onboarding'
    assert not plugin._onboarding_store.path().is_file()      # 没替用户「确认」什么
    assert plugin._onboarding_public()['first_run'] is True   # dashboard 仍是首次运行


def test_old_user_migration_opens_gate(tmp_path):
    """老用户（有就绪标记）：启动迁移自动写成 completed=true，安装门随之打开。"""
    import music_perception as module
    from music_perception.runtime_environment import READY_RUNTIME_MARKER

    plugin = object.__new__(module.MusicPerceptionPlugin)
    plugin.data_path = lambda name='': tmp_path / name
    plugin._download_source_preference = 'mirror'
    plugin._onboarding_store = onboarding.OnboardingStore(tmp_path)

    runtime_dir = tmp_path / 'environment' / '.audio-runtime-olduser'
    runtime_dir.mkdir(parents=True)
    (runtime_dir / READY_RUNTIME_MARKER).write_text('{"schema": 1}', encoding='utf-8')

    plugin._maybe_migrate_onboarding()

    doc = plugin._onboarding_store.load()
    assert doc['completed'] is True
    assert doc['storage_root']                  # 迁移写入默认位置
    assert plugin._install_allowed() is True    # 老用户不受新门影响


def test_install_gate_closed_but_ready_still_restores_listener(monkeypatch, tmp_path):
    """门关着但环境已经就绪：门只挡下载，绝不挡恢复监听。

    这是复核单的核心回归点：`_prepare_runtime_environment` 不许在门关时
    整个提前 return，把已经能用的用户一起挡在监听之外。
    """
    import asyncio
    import logging

    import music_perception as module

    plugin = object.__new__(module.MusicPerceptionPlugin)
    plugin.data_path = lambda name='': tmp_path / name
    plugin.logger = logging.getLogger('test-gate-ready')

    started = []

    class Env:
        def snapshot(self):
            return {'ready': True, 'model_ready': True}

        def ensure(self):
            raise AssertionError('环境已就绪时不该触发下载（ensure）')

        def stop(self):
            pass

    plugin._environment = Env()
    plugin._shutdown_requested = False
    plugin._auto_consented = True
    plugin._plugin_paused = False
    plugin._environment_prepare_started = False

    # 门关着：onboarding.json 存在、completed=false、storage_root 空。
    store = onboarding.OnboardingStore(tmp_path)
    store.save(completed=False, step=1, storage_root='')
    plugin._onboarding_store = store

    async def fake_start_live():
        started.append(1)
        return {'ok': True}

    plugin._start_live = fake_start_live
    plugin._automatic_audio_check_task = lambda **kwargs: asyncio.sleep(0)
    scanned = []
    monkeypatch.setattr(plugin, '_scan_onboarding_only', lambda: scanned.append(1))

    async def run():
        await plugin._prepare_runtime_environment(defer=False)

    asyncio.run(run())

    assert started == [1]                # 就绪 → 照常恢复监听
    assert scanned == []                 # 门关着也不该走「只检测」的降级路径
