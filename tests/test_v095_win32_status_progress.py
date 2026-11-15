"""Windows antivirus/WebView file-lock resilience for private GPU install."""
import json
import os
from pathlib import Path
from music_perception.stem_gpu_runtime import _status


def test_gpu_status_writes_atomically(tmp_path):
    status = tmp_path / 'gpu-install.json'
    assert _status(status, state='running', stage='downloading', progress_pct=23) is True
    assert json.loads(status.read_text('utf-8'))['progress_pct'] == 23
    assert not list(tmp_path.glob('*.tmp'))


def test_gpu_status_retries_transient_windows_lock(monkeypatch, tmp_path):
    status = tmp_path / 'gpu-install.json'
    original = os.replace
    count = {'n':0}
    def race(src,dst):
        count['n'] += 1
        if count['n'] < 3:
            raise PermissionError(32, 'Win32 temporary lock')
        return original(src,dst)
    monkeypatch.setattr(os, 'replace',race)
    assert _status(status,state='ready',progress_pct=100) is True
    assert count['n'] == 3
    assert json.loads(status.read_text('utf-8'))['state'] == 'ready'
    assert not list(tmp_path.glob('*.tmp'))


def test_gpu_status_persistent_lock_does_not_break_model_runtime(monkeypatch, tmp_path):
    status = tmp_path / 'gpu-install.json'
    status.write_text('{"state":"running"}',encoding='utf-8')
    def deny(src,dst):
        raise PermissionError(32,'other Windows process has an open file')
    monkeypatch.setattr(os,'replace',deny)
    assert _status(status,state='ready') is False
    assert json.loads(status.read_text(encoding='utf-8'))['state'] == 'running'
    assert not list(tmp_path.glob('*.tmp'))
