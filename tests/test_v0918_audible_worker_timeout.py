"""Audible session scan must never hang the host: it runs in a killable worker."""
import subprocess
import types
from pathlib import Path

import pytest

import importlib

plugin_mod = importlib.import_module("music_perception")


class _Env:
    def __init__(self, ready=True):
        self._ready = ready

    def snapshot(self):
        return {"ready": self._ready}

    def worker_runtime(self):
        return ("python.exe", Path("vendor"))


def _plugin(env):
    obj = object.__new__(plugin_mod.MusicPerceptionPlugin)
    obj._environment = env
    obj._audible_scan_error = ""
    return obj


def test_worker_timeout_returns_empty_and_records_reason(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=6)

    monkeypatch.setattr(subprocess, "run", boom)
    plugin = _plugin(_Env())
    assert plugin._run_audible_sessions_worker() == []
    assert plugin._audible_scan_error == "session_scan_timeout"


def test_worker_failure_returns_empty(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=1, stdout=b"", stderr=b""))
    plugin = _plugin(_Env())
    assert plugin._run_audible_sessions_worker() == []
    assert plugin._audible_scan_error == "session_scan_worker_failed"


def test_worker_success_returns_sessions(monkeypatch):
    payload = b'{"ok": true, "sessions": [{"pid": 42, "peak": 0.5, "sample_count": 2}]}\n'
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=0, stdout=payload, stderr=b""))
    plugin = _plugin(_Env())
    rows = plugin._run_audible_sessions_worker()
    assert rows == [{"pid": 42, "peak": 0.5, "sample_count": 2}]


def test_environment_not_ready_uses_inprocess_stdlib_scan(monkeypatch):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(1))
    monkeypatch.setattr(plugin_mod, "scan_audible_sessions",
                        lambda **k: [{"pid": 7, "peak": 0.3}])
    plugin = _plugin(_Env(ready=False))
    assert plugin._run_audible_sessions_worker() == [{"pid": 7, "peak": 0.3}]
    assert called == []
