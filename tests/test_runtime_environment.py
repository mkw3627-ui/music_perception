"""Runtime bootstrap tests: intentionally offline; never call pip or PyPI."""
from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import threading
import tomllib

import pytest
from _version import ui_build, panel_ui_build, plugin_version  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('local_runtime_environment', ROOT / 'runtime_environment.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def boot(tmp_path) -> module.MusicEnvironment:
    env = module.MusicEnvironment(source=ROOT, data=tmp_path / 'data', models=tmp_path / 'models')
    # Unit tests that bypass _interpreter must explicitly inject a known-good
    # interpreter, mirroring the production resolver contract.
    env._python_path = module.sys.executable
    return env


def patch_platform(monkeypatch, env):
    # These tests exercise the POSIX-style rename/backup recovery path. On
    # Windows the production code deliberately uses side-by-side runtimes
    # (see test_windows_locked_vendor_v065), so pin the platform explicitly.
    monkeypatch.setattr(env, '_immutable_runtime', lambda: False)
    monkeypatch.setattr(env, '_interpreter', lambda: None)
    monkeypatch.setattr(env, '_enable', lambda p: setattr(env, '_active_vendor', p))
    monkeypatch.setattr(env, '_model_probe', lambda *a: None)
    monkeypatch.setattr(env, '_download_model', lambda: ROOT / 'models' / 'fake.tflite')


def test_bootstrap_manifest_declares_only_optional_audio_requirements():
    data = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf8'))
    assert data['project']['dependencies'] == []
    assert len(module.requirements_from_manifest(ROOT)) == 10
    assert len(data['project']['optional-dependencies']['audio']) == 10
    assert tomllib.loads((ROOT / 'plugin.toml').read_text(encoding='utf8'))['plugin']['version'] == plugin_version()


def test_all_requirements_must_be_known_from_pypi(tmp_path):
    plugin = tmp_path / 'music_perception'
    plugin.mkdir()
    manifest = (ROOT / 'pyproject.toml').read_text(encoding='utf8')
    manifest = manifest.replace('"numpy>=1.24"', '"numpy>=1.24,not-a-real-package"')
    (plugin / 'pyproject.toml').write_text(manifest, encoding='utf8')
    with pytest.raises(module.EnvironmentError):
        module.requirements_from_manifest(plugin)
    # No VCS / local file packages may be supplied by a changed manifest.
    (plugin / 'pyproject.toml').write_text(
        manifest.replace('numpy>=1.24,not-a-real-package', 'numpy @ https://unsafe.local/wheel'), encoding='utf8')
    with pytest.raises(module.EnvironmentError):
        module.requirements_from_manifest(plugin)


def test_uses_existing_valid_local_vendor_without_install(monkeypatch, tmp_path):
    env = boot(tmp_path)
    local = env.data / 'vendor'
    local.mkdir()
    patch_platform(monkeypatch, env)
    calls = []
    monkeypatch.setattr(env, '_probe', lambda location, req: calls.append(location))
    monkeypatch.setattr(env, '_install', lambda *args: pytest.fail('should not install'))
    s = env.ensure()
    assert s['phase'] == 'ready' and s['ready'] and s['model_ready']
    assert s['downloaded'] is False and calls == [local]


def test_repairs_absent_deps_into_isolated_data_not_host(monkeypatch, tmp_path):
    env = boot(tmp_path)
    patch_platform(monkeypatch, env)
    calls = []
    def fake_probe(loc, req):
        calls.append(loc)
        if not (loc / 'verified.marker').exists():
            raise module.EnvironmentError('not installed')
    def fake_install(stage, reqs):
        assert stage.parent == env.data
        assert len(reqs) == 10
        (stage / 'verified.marker').write_text('yes')
    monkeypatch.setattr(env, '_probe', fake_probe)
    monkeypatch.setattr(env, '_install', fake_install)
    s = env.ensure()
    assert s['phase'] == 'ready' and s['downloaded']
    assert (env.data / 'vendor' / 'verified.marker').exists()
    assert (ROOT / 'vendor').exists() is False  # no write to source
    assert len(calls) == 2


def test_broken_local_deps_replace_atomically(monkeypatch, tmp_path):
    env = boot(tmp_path)
    patch_platform(monkeypatch, env)
    original = env.data / 'vendor'
    original.mkdir()
    (original / 'broken').write_text('bad')
    monkeypatch.setattr(env, '_probe', lambda path, _: (
        None if (path / 'good').exists() else (_ for _ in ()).throw(module.EnvironmentError('bad deps'))))
    monkeypatch.setattr(env, '_install', lambda path, _: (path / 'good').write_text('ready'))
    s = env.ensure()
    assert s['phase'] == 'ready'
    assert not (original / 'broken').exists()
    assert (original / 'good').exists()
    assert not (env.data / '.vendor.previous').exists()


def test_failed_install_keeps_old_vendor(monkeypatch, tmp_path):
    env = boot(tmp_path)
    patch_platform(monkeypatch, env)
    original = env.data / 'vendor'
    original.mkdir()
    (original / 'prior').write_text('keep')
    monkeypatch.setattr(env, '_probe', lambda *a: (_ for _ in ()).throw(module.EnvironmentError('broken')))
    monkeypatch.setattr(env, '_install', lambda *a: (_ for _ in ()).throw(module.EnvironmentError('no internet')))
    s = env.ensure()
    assert s['phase'] == 'failed' and s['error'] == 'no internet'
    assert (original / 'prior').read_text() == 'keep'
    assert not list(env.data.glob('.audio-staging-*'))


def test_model_failure_keeps_file_analysis_ready(monkeypatch, tmp_path):
    env = boot(tmp_path)
    patch_platform(monkeypatch, env)
    (env.data / 'vendor').mkdir()
    monkeypatch.setattr(env, '_probe', lambda *a: None)
    monkeypatch.setattr(env, '_download_model', lambda: (_ for _ in ()).throw(module.EnvironmentError('offline')))
    s = env.ensure()
    assert s['phase'] == 'partial'
    assert s['ready'] and not s['model_ready']
    assert s['error'] == 'offline'


def test_retry_after_failed_attempt(monkeypatch, tmp_path):
    env = boot(tmp_path)
    patch_platform(monkeypatch, env)
    local = env.data / 'vendor'
    local.mkdir()
    monkeypatch.setattr(env, '_probe', lambda *a: None)
    offline = True
    def download():
        if offline:
            raise module.EnvironmentError('offline')
        return ROOT / 'models' / 'fake.tflite'
    monkeypatch.setattr(env, '_download_model', download)
    assert env.ensure()['phase'] == 'partial'
    offline = False
    result = env.ensure()
    assert result['phase'] == 'ready' and result['error'] is None
    assert result['repair_attempts'] == 2


def test_concurrent_start_no_duplicate_install(tmp_path):
    env = boot(tmp_path)
    assert env._install_lock.acquire(blocking=False)
    try:
        assert env.ensure()['repair_attempts'] == 0
    finally:
        env._install_lock.release()


def test_model_file_rejects_truncated_and_html(tmp_path):
    p = tmp_path / 'yamnet.tflite'
    p.write_bytes(b'not-a-model')
    assert not module.model_valid(p)
    p.write_bytes(b'bad!TFL3' + b'0' * 1_200_000)
    assert module.model_valid(p)
    p.write_bytes(b'12345678' + b'0' * 1_200_000)
    assert not module.model_valid(p)


def test_installer_uses_target_and_binary_only(monkeypatch, tmp_path):
    env = boot(tmp_path)
    stage = env.data / 'scratch'; stage.mkdir()
    seen = {}
    monkeypatch.setattr(module.shutil, 'which', lambda name: 'C:/mock/uv.exe' if name == 'uv' else None)
    class FakeChild:
        def __init__(self, argv, **kw):
            seen['argv'] = argv
            seen['cwd'] = kw['cwd']
            self.returncode = 0
        def poll(self): return 0
    monkeypatch.setattr(module.subprocess, 'Popen', FakeChild)
    monkeypatch.setattr(env, '_probe', lambda location, specs: None)
    env._install(stage, module.requirements_from_manifest(ROOT))
    assert '--target' in seen['argv']
    assert str(stage) in seen['argv']
    assert '--only-binary' in seen['argv']
    assert '--python' in seen['argv']
    assert '--index-url' in seen['argv']
    assert '--system' not in seen['argv']


def test_bootstrap_module_has_no_thirdparty_imports_at_module_scope():
    import ast
    for file in ('__init__.py', 'runtime_environment.py'):
        content = ast.parse((ROOT / file).read_text(encoding='utf8'))
        names = []
        for node in content.body:
            if isinstance(node, ast.Import): names += [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom): names.append(node.module or '')
        assert not any(n in {'numpy', 'scipy', 'librosa', 'soundfile', 'soundcard', 'soxr', 'ai_edge_litert'} for n in names)


def test_user_audio_monitor_not_started_by_environment_import():
    import ast
    root = ast.parse((ROOT / '__init__.py').read_text(encoding='utf8'))
    assert any(isinstance(n, ast.ClassDef) and n.name == 'MusicPerceptionPlugin' for n in root.body)


def test_valid_private_vendor_reused_when_bundled_vendor_is_broken(monkeypatch, tmp_path):
    """A stale installed vendor must not trigger endless redownloads."""
    source = tmp_path / 'sources' / 'music_perception'
    source.mkdir(parents=True)
    (source / 'pyproject.toml').write_bytes((ROOT / 'pyproject.toml').read_bytes())
    bundled = source / 'vendor'
    bundled.mkdir()
    env = module.MusicEnvironment(source=source, data=tmp_path / 'data', models=tmp_path / 'models')
    private = env.data / 'vendor'
    private.mkdir()
    patch_platform(monkeypatch, env)
    tried = []

    def probe(path, specs):
        tried.append(path)
        if path == bundled:
            raise module.EnvironmentError('bundled vendor is broken')

    monkeypatch.setattr(env, '_probe', probe)
    monkeypatch.setattr(env, '_install', lambda *a: pytest.fail('valid private vendor is already usable'))
    s = env.ensure()
    assert s['phase'] == 'ready' and not s['downloaded']
    # Prefer a verified private install before trying an older bundled copy.
    assert tried == [private]
    assert env._active_vendor == private


def test_restore_previous_vendor_after_interrupted_update(monkeypatch, tmp_path):
    env = boot(tmp_path)
    old = env.data / '.vendor.previous'
    old.mkdir()
    (old / 'good').write_text('restored')
    patch_platform(monkeypatch, env)
    monkeypatch.setattr(env, '_probe', lambda path, req: (
        None if (path / 'good').exists() else (_ for _ in ()).throw(module.EnvironmentError('invalid'))))
    monkeypatch.setattr(env, '_install', lambda *a: pytest.fail('should restore previous copy'))
    s = env.ensure()
    assert s['phase'] == 'ready'
    assert (env.data / 'vendor' / 'good').read_text() == 'restored'
    assert not old.exists()


def test_interrupted_update_prefers_last_known_good_backup(monkeypatch, tmp_path):
    env = boot(tmp_path)
    broken = env.data / 'vendor'
    backup = env.data / '.vendor.previous'
    broken.mkdir(); backup.mkdir()
    (broken / 'bad').write_text('corrupt')
    (backup / 'good').write_text('usable')
    patch_platform(monkeypatch, env)
    monkeypatch.setattr(env, '_probe', lambda path, _: (
        None if (path / 'good').exists() else (_ for _ in ()).throw(module.EnvironmentError('bad'))))
    monkeypatch.setattr(env, '_install', lambda *a: pytest.fail('should recover good backup'))
    state = env.ensure()
    assert state['phase'] == 'ready'
    assert (broken / 'good').read_text() == 'usable'
    assert not backup.exists()


def test_corrupt_vendor_and_backup_do_not_block_new_repair(monkeypatch, tmp_path):
    env = boot(tmp_path)
    broken = env.data / 'vendor'
    backup = env.data / '.vendor.previous'
    broken.mkdir(); backup.mkdir()
    (backup / 'old').write_text('forensics')
    patch_platform(monkeypatch, env)
    monkeypatch.setattr(env, '_probe', lambda path, _: (
        None if (path / 'good').exists() else (_ for _ in ()).throw(module.EnvironmentError('invalid'))))
    monkeypatch.setattr(env, '_install', lambda stage, _: (stage / 'good').write_text('repaired'))
    state = env.ensure()
    assert state['phase'] == 'ready' and state['downloaded']
    assert (broken / 'good').exists()
    assert not backup.exists()
    stale = list(env.data.glob('.vendor.saved-*'))
    assert len(stale) == 1
    assert (stale[0] / 'old').read_text() == 'forensics'


def test_uv_exit_2_automatically_falls_back_to_pip(monkeypatch, tmp_path):
    env = boot(tmp_path)
    stage = env.data / 'temp'; stage.mkdir()
    monkeypatch.setattr(module.shutil, 'which', lambda name: 'old-uv' if name == 'uv' else None)
    attempts = []

    def run(command, *, deadline):
        attempts.append(command)
        if command[0] == 'old-uv':
            (stage / 'from_uv_incomplete').write_text('remove me')
            return 2, 'error: unexpected argument --target\nUsage: uv pip install'
        assert command[:3] == [module.sys.executable, '-m', 'pip']
        assert '--isolated' in command
        assert not (stage / 'from_uv_incomplete').exists()
        (stage / 'from_pip').write_text('pip wheel')
        return 0, 'Successfully installed a package'

    monkeypatch.setattr(env, '_attempt_install', run)
    monkeypatch.setattr(env, '_probe', lambda location, specs: None)
    env._install(stage, module.requirements_from_manifest(ROOT))
    assert len(attempts) == 2
    assert env.snapshot()['installer'] == 'uv → pip'
    assert env.snapshot()['error_details'] is None
    assert (stage / 'from_pip').is_file()
    assert 'unexpected argument' in (env.data / 'install_diagnostics.log').read_text(encoding='utf8')


def test_uv_exit_zero_but_native_import_fails_then_pip_retries(monkeypatch, tmp_path):
    env = boot(tmp_path)
    stage = env.data / 'temp'; stage.mkdir()
    monkeypatch.setattr(module.shutil, 'which', lambda _: 'uv')
    installers = []

    def run(cmd, *, deadline):
        installers.append(cmd[0])
        (stage / ('success' if cmd[0] != 'uv' else 'broken')).write_text('x')
        return 0, 'Installed'

    def probe(location, specs):
        if (location / 'broken').exists():
            raise module.EnvironmentError('DLL load failed: cannot load native module')

    monkeypatch.setattr(env, '_attempt_install', run)
    monkeypatch.setattr(env, '_probe', probe)
    env._install(stage, module.requirements_from_manifest(ROOT))
    assert installers == ['uv', module.sys.executable]
    assert (stage / 'success').exists() and not (stage / 'broken').exists()


def test_both_installers_fail_with_actual_errors_and_preserve_vendor(monkeypatch, tmp_path):
    env = boot(tmp_path)
    vendor = env.data / 'vendor'; vendor.mkdir()
    (vendor / 'prior').write_text('keep')
    monkeypatch.setattr(module.shutil, 'which', lambda _: 'uv')
    patch_platform(monkeypatch, env)
    monkeypatch.setattr(env, '_probe', lambda *a: (_ for _ in ()).throw(module.EnvironmentError('not installed')))

    def run(cmd, *, deadline):
        return (2, 'error: unexpected argument --default-index') if cmd[0] == 'uv' else (1, 'No module named pip')

    monkeypatch.setattr(env, '_attempt_install', run)
    result = env.ensure()
    assert result['phase'] == 'failed'
    assert 'unexpected argument' in result['error_details']
    assert 'No module named pip' in result['error']
    assert result['diagnostics_path']
    assert (vendor / 'prior').read_text() == 'keep'
    assert not list(env.data.glob('.audio-staging-*'))


def test_private_credentials_and_user_names_redacted():
    secret = 'https://alice:12345@mirror.example/simple?token=very-secret C:\\Users\\JaneDoe\\Music '
    scrubbed = module.MusicEnvironment._redact_install_output(secret)
    for sensitive in ('alice', '12345', 'very-secret', 'JaneDoe'):
        assert sensitive not in scrubbed
    assert 'mirror.example' in scrubbed
    assert '[隐藏凭据]' in scrubbed


def test_no_uv_uses_only_pip(monkeypatch, tmp_path):
    env = boot(tmp_path)
    stage = env.data / 'temp'; stage.mkdir()
    monkeypatch.setattr(module.shutil, 'which', lambda _: None)
    called = []
    monkeypatch.setattr(env, '_attempt_install', lambda cmd, **kw: (called.append(cmd) or (0, 'ok')))
    monkeypatch.setattr(env, '_probe', lambda *a: None)
    env._install(stage, module.requirements_from_manifest(ROOT))
    assert len(called) == 1 and called[0][0] == module.sys.executable
    assert '--target' in called[0] and '--only-binary=:all:' in called[0]


def test_shutdown_interrupts_install_without_fallback(monkeypatch, tmp_path):
    env = boot(tmp_path)
    stage = env.data / 'temp'; stage.mkdir()
    monkeypatch.setattr(module.shutil, 'which', lambda _: 'uv')
    calls = []

    def canceled(cmd, **kw):
        calls.append(cmd)
        env._cancel.set()  # 模拟 shutdown：第一次尝试时就收到停止信号
        raise module.EnvironmentError('用户退出 N.E.K.O')

    monkeypatch.setattr(env, '_attempt_install', canceled)
    with pytest.raises(module.EnvironmentError, match='退出'):
        env._install(stage, module.requirements_from_manifest(ROOT))
    assert len(calls) == 1


def test_native_import_probe_keeps_specific_error_without_full_raw_log(monkeypatch, tmp_path):
    from types import SimpleNamespace
    env = boot(tmp_path)
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kw: SimpleNamespace(
        returncode=1, stdout='', stderr='ImportError: DLL load failed while importing scipy'))
    with pytest.raises(module.EnvironmentError, match='DLL load failed'):
        env._probe(tmp_path, module.requirements_from_manifest(ROOT))
    assert 'scipy' in env.snapshot()['error_details']
    assert 'native-import-check' in (env.data / 'install_diagnostics.log').read_text(encoding='utf8')


def test_real_subprocess_exit_2_preserves_diagnostic_tail(tmp_path):
    env = boot(tmp_path)
    rc, out = env._attempt_install(
        [module.sys.executable, '-c', "import sys;print('unexpected argument --mock');sys.exit(2)"],
        deadline=module.time.monotonic() + 10,
    )
    assert rc == 2
    assert 'unexpected argument --mock' in out


def test_no_fallback_to_frozen_server_if_interpreter_unresolved(tmp_path, monkeypatch):
    env = boot(tmp_path)
    env._python_path = None
    monkeypatch.setattr(module.sys, 'executable', 'E:/Steam/NEKO/projectneko_server.exe')
    with pytest.raises(module.EnvironmentError, match='禁止使用 N.E.K.O 主程序'):
        env._probe(tmp_path, module.requirements_from_manifest(ROOT))
    with pytest.raises(module.EnvironmentError, match='禁止使用 N.E.K.O 主程序'):
        env._install(tmp_path, module.requirements_from_manifest(ROOT))
