"""No-network simulations of the real packaged N.E.K.O / Nuitka failure."""
import sys
from pathlib import Path

import pytest
from music_perception import install_prepare as builder
from music_perception import runtime_environment as rt


def fake_python(tmp_path):
    p = tmp_path / 'python.exe'
    p.write_bytes(b'fake executable')
    return p


def test_frozen_exe_skipped_and_uv_prepares_private_matching_python(monkeypatch, tmp_path):
    server = tmp_path / 'projectneko_server.exe'
    server.write_text('fake packaged exe')
    python = fake_python(tmp_path)
    commands = []
    version = f'{sys.version_info.major}.{sys.version_info.minor}'

    def fake_run(self, command, *, timeout=15, env=None):
        commands.append((command, env))
        if command[0] == 'uv' and command[1:3] == ['python', 'install']:
            return 0, 'Installed CPython'
        if command[0] == 'uv' and command[1:3] == ['python', 'find']:
            if '--managed-python' in command and any('install' in item[0][2:3] for item in commands):
                return 0, str(python)
            return 2, 'No interpreter found'
        if command[0] == str(python):
            return 0, f'NEKO_REAL_CPYTHON:{version}:64:cpython\n'
        raise AssertionError(f'Unexpected attempted executable: {command!r}')

    monkeypatch.setattr(rt.sys, 'executable', str(server))
    monkeypatch.setattr(rt.shutil, 'which', lambda s: 'uv' if s == 'uv' else None)
    monkeypatch.setattr(rt.PythonResolver, '_run', fake_run)
    r = rt.PythonResolver(tmp_path / 'environment')
    got = r.resolve()
    assert got == str(python)
    assert not any(item[0][0] == str(server) for item in commands)
    install = [item for item in commands if item[0][1:3] == ['python', 'install']]
    assert len(install) == 1
    assert '--install-dir' in install[0][0]
    assert '--no-bin' in install[0][0] and '--no-registry' in install[0][0]
    managed_find = next(c for c, env in commands if '--managed-python' in c)
    assert managed_find[-1] == version
    assert any(env and 'UV_PYTHON_INSTALL_DIR' in env for c, env in commands)


def test_existing_external_python_avoids_install(monkeypatch, tmp_path):
    packaged = tmp_path / 'projectneko_server.exe'
    packaged.write_text('frozen')
    python = fake_python(tmp_path)
    seen = []
    def run(self, c, *, timeout=15, env=None):
        seen.append(c)
        if c[0] == 'uv' and c[1:3] == ['python', 'find']:
            return 0, str(python)
        if c[0] == str(python):
            return 0, f'NEKO_REAL_CPYTHON:{self.version}:64:cpython'
        raise AssertionError(c)
    monkeypatch.setattr(rt.sys, 'executable', str(packaged))
    monkeypatch.setattr(rt.shutil,'which', lambda x: 'uv' if x == 'uv' else None)
    monkeypatch.setattr(rt.PythonResolver,'_run',run)
    assert rt.PythonResolver(tmp_path / 'private').resolve() == str(python)
    assert not any(c[1:3] == ['python', 'install'] for c in seen)


def test_no_python_no_uv_falls_back_to_portable_never_calls_packaged_exe(monkeypatch, tmp_path):
    packaged = tmp_path / 'projectneko_server.exe'
    packaged.write_text('frozen')
    monkeypatch.setattr(rt.sys,'executable',str(packaged))
    monkeypatch.setattr(rt.shutil,'which',lambda x: None)
    monkeypatch.setattr(rt.PythonResolver,'_run', lambda *a, **k: pytest.fail('must not spawn packaged app'))
    portable = []
    monkeypatch.setattr(rt.PythonResolver,'_portable_python', lambda self: portable.append(1) or 'portable')
    assert rt.PythonResolver(tmp_path / 'private').resolve() == 'portable'
    assert portable == [1]


def test_portable_python_rejects_wrong_sha_and_never_extracts(monkeypatch, tmp_path):
    import io
    if rt.os.name != 'nt':
        pytest.skip('portable CPython is Windows only')

    class Reply(io.BytesIO):
        def geturl(self):
            return 'https://mirror.example/python.tar.gz'
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    monkeypatch.setattr(rt.urllib.request,'urlopen', lambda *a, **k: Reply(b'not the pinned archive'))
    monkeypatch.setattr(rt.PythonResolver,'_validate', lambda self, c: False)
    resolver = rt.PythonResolver(tmp_path / 'private')
    resolver.directory.mkdir(parents=True)
    with pytest.raises(rt.PythonResolutionError, match='SHA-256'):
        resolver._portable_python()
    assert not (tmp_path / 'private' / 'portable-python').exists()


def test_portable_python_pins_every_supported_version():
    for version in ('3.11', '3.12', '3.13'):
        filename, digest = rt.PORTABLE_PYTHON[version]
        assert filename.startswith('cpython-' + version + '.') and filename.endswith('windows-msvc-install_only.tar.gz')
        assert len(digest) == 64
    assert rt.PORTABLE_PYTHON_SOURCES[0][1].startswith('https://mirror.nju.edu.cn/')


def test_rejects_incompatible_native_abi(monkeypatch, tmp_path):
    python = fake_python(tmp_path)
    p = rt.PythonResolver(tmp_path / 'env')
    monkeypatch.setattr(rt.PythonResolver, '_run', lambda *a, **k: (0, 'NEKO_REAL_CPYTHON:3.10:32:cpython'))
    assert p._validate(python) is False


def test_real_interpreter_is_validated_without_network(tmp_path):
    p = rt.PythonResolver(tmp_path)
    # The test runner already uses a real interpreter and must be found before uv.
    assert Path(p.resolve()).is_file()


def test_build_preparation_uses_resolved_python_for_all_probes(monkeypatch, tmp_path):
    src=tmp_path/'music_perception'
    src.mkdir()
    root=Path(__file__).resolve().parents[1]
    for name in ('plugin.toml','pyproject.toml'):
        (src/name).write_bytes((root/name).read_bytes())
    real = tmp_path/'python.exe'
    real.touch()
    monkeypatch.setattr(rt.PythonResolver,'resolve', lambda self: str(real))
    monkeypatch.setattr(builder, 'download_yamnet',lambda *a:'fake-model')
    monkeypatch.setattr(builder.shutil,'which', lambda x: 'uv' if x == 'uv' else None)
    commands = []
    def fake_uv(executable, root, params, *, timeout):
        commands.append([executable,*params])
        if params[:2]==['pip','install']:
            Path(params[params.index('--target')+1],'installed').touch()
    monkeypatch.setattr(builder,'_run_uv',fake_uv)
    monkeypatch.setattr(builder,'_run_command',lambda repo, args, *, timeout: commands.append(args))
    builder.prepare(src)
    assert all(str(real) in cmd for cmd in commands)
    assert not any(cmd and cmd[0].lower().endswith('projectneko_server.exe') for cmd in commands)


def test_runtime_uses_validated_python_in_install_and_native_probe(monkeypatch,tmp_path):
    env=rt.MusicEnvironment(source=Path(__file__).resolve().parents[1], data=tmp_path/'env', models=tmp_path/'models')
    env._python_path='C:/verified/python.exe'
    commands=[]
    monkeypatch.setattr(rt.shutil, 'which', lambda x:'uv' if x=='uv' else None)
    monkeypatch.setattr(env, '_attempt_install', lambda cmd, *,deadline: (commands.append(cmd) or (0,'ok')))
    monkeypatch.setattr(env, '_probe',lambda *a: None)
    stage=tmp_path/'staging'
    stage.mkdir()
    env._install(stage,rt.requirements_from_manifest(env.source))
    assert commands[0][commands[0].index('--python')+1] == env._python_path


def test_uv_windows_user_profile_path_must_not_be_redacted_during_discovery(monkeypatch,tmp_path):
    """Logs are scrubbed, but the internal interpreter path must stay runnable."""
    p=rt.PythonResolver(tmp_path/'env')
    candidate=r'C:\Users\Administrator\AppData\Roaming\uv\python\cpython-3.13\python.exe'
    seen=[]
    monkeypatch.setattr(p,'_run',lambda cmd, **kw: (0,candidate))
    monkeypatch.setattr(p,'_validate',lambda path: (seen.append(path) or True))
    assert p._uv_find('uv')==candidate
    assert seen==[candidate]
    assert 'Administrator' not in rt.MusicEnvironment._redact_install_output(candidate)
