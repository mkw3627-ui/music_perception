"""Windows side-by-side vendor acceptance tests; no network or Windows host needed."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('runtime_side_by_side_v065', ROOT / 'runtime_environment.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _runtime(monkeypatch, tmp_path):
    env = module.MusicEnvironment(source=ROOT, data=tmp_path / 'data', models=tmp_path / 'models')
    env._python_path = module.sys.executable
    monkeypatch.setattr(env, '_immutable_runtime', lambda: True)
    monkeypatch.setattr(env, '_interpreter', lambda: None)
    monkeypatch.setattr(env, '_enable', lambda p: (setattr(env, '_active_vendor', p), env._report(vendor_path=str(p))))
    monkeypatch.setattr(env, '_model_probe', lambda *a: None)
    monkeypatch.setattr(env, '_download_model', lambda: ROOT / 'models' / 'dummy.tflite')
    return env


def _probe(path, reqs):
    if not (path / 'complete.lock-test').exists():
        raise module.EnvironmentError('missing lyrics and media dependencies')


def test_windows_full_install_never_renames_busy_vendor(monkeypatch, tmp_path):
    env = _runtime(monkeypatch, tmp_path)
    original = env.data / 'vendor'
    original.mkdir()
    (original / 'old-music-stack').write_text('keep old files')
    attempted = []

    def fake_install(stage, reqs):
        assert stage.name.startswith('.audio-runtime-')
        assert original.is_dir()
        (stage / 'complete.lock-test').write_text('all 54 packages')

    def forbidden_rename(*a, **k):
        attempted.append('rename called')
        raise PermissionError('[WinError 5] old active vendor locked')

    monkeypatch.setattr(env, '_probe', _probe)
    monkeypatch.setattr(env, '_install', fake_install)
    monkeypatch.setattr(env, '_replace_vendor', forbidden_rename)
    state = env.ensure()
    assert state['ready'] and state['model_ready'] and state['phase'] == 'ready'
    assert state['downloaded'] and state['missing'] == []
    assert not attempted
    assert (original / 'old-music-stack').read_text() == 'keep old files'
    active = env._active_vendor
    assert active != original and active.parent == env.data
    assert active.name.startswith('.audio-runtime-')
    assert (active / module.READY_RUNTIME_MARKER).is_file()
    assert (active / 'complete.lock-test').read_text() == 'all 54 packages'


def test_windows_restarts_use_verified_runtime_without_reinstall(monkeypatch, tmp_path):
    env = _runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(env, '_probe', _probe)
    monkeypatch.setattr(env, '_install', lambda stage, _: (stage / 'complete.lock-test').write_text('ready'))
    assert env.ensure()['ready']
    first = env._active_vendor
    restarted = _runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(restarted, '_probe', _probe)
    monkeypatch.setattr(restarted, '_install', lambda *a: pytest.fail('no repeated download expected'))
    state = restarted.ensure()
    assert state['phase'] == 'ready' and state['downloaded'] is False
    assert restarted._active_vendor == first
    assert len(list(env.data.glob('.audio-runtime-*'))) == 1


def test_windows_failed_import_preserves_old_install(monkeypatch, tmp_path):
    env = _runtime(monkeypatch, tmp_path)
    original = env.data / 'vendor'
    original.mkdir()
    (original / 'old').write_text('unchanged')
    monkeypatch.setattr(env, '_probe', _probe)
    monkeypatch.setattr(env, '_install', lambda stage, reqs: (stage / 'partial').write_text('bad'))
    state = env.ensure()
    assert state['phase'] == 'failed'
    assert state['ready'] is False
    assert (original / 'old').read_text() == 'unchanged'
    assert not list(env.data.glob('.audio-runtime-*'))


def test_windows_interrupted_old_vendor_backup_is_reused(monkeypatch, tmp_path):
    env = _runtime(monkeypatch, tmp_path)
    backup = env.data / '.vendor.previous'
    backup.mkdir()
    (backup / 'complete.lock-test').write_text('backup works')
    monkeypatch.setattr(env, '_probe', _probe)
    monkeypatch.setattr(env, '_install', lambda *a: pytest.fail('backup should be probed, never renamed'))
    assert env.ensure()['ready']
    assert env._active_vendor == backup
    assert backup.exists()


def test_windows_unverified_runtime_not_accepted(monkeypatch, tmp_path):
    env = _runtime(monkeypatch, tmp_path)
    abandoned = env.data / '.audio-runtime-cancelled'
    abandoned.mkdir()
    (abandoned / 'complete.lock-test').write_text('unmarked runtime')
    monkeypatch.setattr(env, '_probe', _probe)
    monkeypatch.setattr(env, '_install', lambda stage, _: (stage / 'complete.lock-test').write_text('verified'))
    assert env.ensure()['ready']
    assert env._active_vendor != abandoned


def test_windows_requirements_change_invalidates_old_ready_marker(monkeypatch, tmp_path):
    env = _runtime(monkeypatch, tmp_path)
    location = env.data / '.audio-runtime-old-version'
    location.mkdir()
    (location / 'complete.lock-test').write_text('only old spec')
    (location / module.READY_RUNTIME_MARKER).write_text(json.dumps({
        'schema':1,'python_abi':env._python_abi_tag(),
        'requirements_sha256':'invalid_sha'}))
    monkeypatch.setattr(env, '_probe', _probe)
    monkeypatch.setattr(env, '_install', lambda stage, _: (stage / 'complete.lock-test').write_text('new'))
    assert env.ensure()['ready']
    assert env._active_vendor != location


def test_probe_contains_actual_winrt_and_lyrics_native_imports():
    assert 'from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager' in module.PROBE_SCRIPT
    assert 'import faster_whisper, ctranslate2' in module.PROBE_SCRIPT
    assert 'scipy.signal' in module.PROBE_SCRIPT
    assert 'ISOLATED_AUDIO_IMPORT_OK' in module.PROBE_SCRIPT


def test_install_uses_copy_mode_to_avoid_windows_cross_volume_hardlinks(monkeypatch, tmp_path):
    env = _runtime(monkeypatch, tmp_path)
    seen = {}
    class Child:
        returncode = 0
        def __init__(self, argv, **kwargs):
            seen['env'] = kwargs['env']
        def poll(self):
            return 0
    monkeypatch.setattr(module.subprocess, 'Popen', Child)
    code, _ = env._attempt_install(['uv', 'pip', 'install'], deadline=module.time.monotonic() + 5)
    assert code == 0
    assert seen['env']['UV_LINK_MODE'] == 'copy'


def test_windows_salvages_complete_old_staging_without_download(monkeypatch, tmp_path):
    env = _runtime(monkeypatch, tmp_path)
    orphan = env.data / '.audio-staging-d5ds5h4i'
    orphan.mkdir()
    (orphan / 'complete.lock-test').write_text('previous uv exit=0')
    monkeypatch.setattr(env, '_probe', _probe)
    monkeypatch.setattr(module, 'current_distributions', lambda path: {
        'faster-whisper': '1.2.1', 'ai-edge-litert': '2.3.0',
        'winrt-windows-media-control': '3.2.1'})
    monkeypatch.setattr(env, '_install', lambda *a: pytest.fail('complete old wheel set should be reused'))
    state = env.ensure()
    assert state['ready'] and not state['downloaded']
    assert env._active_vendor == orphan
    assert (orphan / module.READY_RUNTIME_MARKER).is_file()
    restarted = _runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(restarted, '_probe', _probe)
    monkeypatch.setattr(restarted, '_install', lambda *a: pytest.fail('recovered stage lost after restart'))
    assert restarted.ensure()['ready']
    assert restarted._active_vendor == orphan


def test_windows_incomplete_old_staging_not_reused(monkeypatch, tmp_path):
    env = _runtime(monkeypatch, tmp_path)
    orphan = env.data / '.audio-staging-broken'
    orphan.mkdir()
    (orphan / 'complete.lock-test').write_text('incomplete on disk')
    monkeypatch.setattr(env, '_probe', _probe)
    monkeypatch.setattr(module, 'current_distributions', lambda path: {'ai-edge-litert':'2.3.0'})
    installed = []
    def fresh(stage, reqs):
        installed.append(stage)
        (stage / 'complete.lock-test').write_text('fully repaired')
    monkeypatch.setattr(env, '_install', fresh)
    assert env.ensure()['ready']
    assert len(installed) == 1
    assert env._active_vendor != orphan
