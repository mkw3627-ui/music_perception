"""Offline Windows isolated-WASAPI checks; no physical audio device needed."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
import struct
import subprocess
import sys
import types
import tomllib

import numpy as np
import pytest

from music_perception.auto_listener import ListenerError, SoundCardLoopback
import music_perception.isolated_loopback as iso
import music_perception.loopback_worker as worker
from _version import ui_build, panel_ui_build, plugin_version  # noqa: E402


class FakeProcess:
    def __init__(self, stdout):
        self.stdout = io.BytesIO(stdout)
        self.terminated = False
        self.killed = False
        self.waited = False
        self.argv = None
    def poll(self):
        return -15 if self.terminated or self.killed else None
    def terminate(self):
        self.terminated = True
    def kill(self):
        self.killed = True
    def wait(self, timeout=0):
        self.waited = True
        return 0


def pack(tag, obj):
    data = obj if isinstance(obj, bytes) else json.dumps(obj).encode('utf-8')
    return struct.pack('!cI', tag, len(data)) + data


def fake_files(tmp_path):
    v = tmp_path / 'vendor';v.mkdir()
    script = tmp_path / 'loopback_worker.py';script.write_text('# fake')
    python = tmp_path / 'python.exe';python.write_bytes(b'placeholder')
    return python, v, script, tmp_path / 'diagnostics.log'


def prepare(monkeypatch, tmp_path, output):
    python, vendor, script, log = fake_files(tmp_path)
    monkeypatch.setattr(iso.sys, 'platform', 'win32')
    processes = []
    def launch(argv, **kwargs):
        proc = FakeProcess(output)
        proc.argv = argv
        processes.append(proc)
        return proc
    monkeypatch.setattr(iso.subprocess, 'Popen', launch)
    obj = iso.IsolatedWindowsLoopback(python=str(python), vendor=vendor,
                                      worker_script=script, diagnostics=log, start_timeout=.25)
    return obj, processes


def test_frontend_refresh_is_defined_and_version_matches():
    root = Path(__file__).resolve().parents[1]
    html = (root / 'static/index.html').read_text('utf-8')
    assert 'let report=null, lastReportRevision=null, lastLiveCount=null;' in html
    panel = (root / 'ui/panel.tsx').read_text('utf8')
    assert 'props.api.refresh()' in panel
    assert "if(lastLiveCount===null)" in html
    doc = tomllib.loads((root / 'plugin.toml').read_text('utf-8'))
    assert doc['plugin']['version'] == plugin_version()
    assert (root / 'loopback_worker.py').exists()


def test_windows_worker_uses_real_python_and_reads_speaker_audio(monkeypatch, tmp_path):
    pcm = np.zeros(32000, dtype='<f4').tobytes()
    capture, procs = prepare(monkeypatch, tmp_path,
                             pack(b'R', {'device':'Real speaker (loopback)'}) + pack(b'D', pcm))
    with capture as source:
        assert source.device == 'Real speaker (loopback)'
        samples = source.read()
        assert samples.shape == (32000,) and samples.dtype == np.float32
        assert procs[0].argv[0].endswith('python.exe')
        assert procs[0].argv[1:3] == ['-I','-S']
        assert procs[0].argv[-1].endswith('vendor')
        assert procs[0].argv[0] != sys.executable
    assert procs[0].terminated and procs[0].waited


def test_worker_error_identifies_failure_stage_without_guessing(monkeypatch, tmp_path):
    capture, procs = prepare(monkeypatch, tmp_path,
                             pack(b'E', {'stage':'打开回环录音器', 'error':'RuntimeError: Error 0x10000001'}))
    with pytest.raises(ListenerError, match=r'打开回环录音器.*0x10000001'):
        with capture: pass
    assert procs[0].terminated


def test_worker_read_error_is_reported_with_diagnostics(monkeypatch, tmp_path):
    capture, _ = prepare(monkeypatch, tmp_path, pack(b'R', {'device':'Spkr'}) +
                         pack(b'E', {'stage':'正在读取扬声器回环帧', 'error':'driver invalidated'}))
    with capture as source:
        with pytest.raises(ListenerError, match='正在读取扬声器回环帧'):
            source.read()


def test_corrupted_pipe_and_wrong_audio_length_are_rejected(monkeypatch, tmp_path):
    cap, _ = prepare(monkeypatch, tmp_path, pack(b'R', {'device':'Spkr'}) + pack(b'D', b'noise'))
    with cap as source:
        with pytest.raises(ListenerError, match='帧长度不匹配'):
            source.read()


def test_no_nuitka_server_fallback_and_absent_vendor(monkeypatch, tmp_path):
    python, vendor, script, log = fake_files(tmp_path)
    monkeypatch.setattr(iso.sys, 'platform','win32')
    source = iso.IsolatedWindowsLoopback(python=tmp_path/'projectneko_server.exe',
                                        vendor=vendor,worker_script=script,diagnostics=log)
    with pytest.raises(ListenerError,match='独立 python.exe'):
        with source: pass
    vendor.rmdir()
    source = iso.IsolatedWindowsLoopback(python=python,vendor=vendor,worker_script=script,diagnostics=log)
    with pytest.raises(ListenerError,match='音频依赖目录'):
        with source: pass


def test_loopback_capture_rate_44100_preserves_16khz_size(monkeypatch):
    import music_perception.auto_listener as module
    monkeypatch.setattr(module.sys,'platform','linux')
    class Sc:
        def default_speaker(self):return types.SimpleNamespace(id='s',channels=2)
        def all_microphones(self,**kwargs):return [types.SimpleNamespace(id='s.monitor',name='s.monitor')]
    class Rec:
        def record(self,**kwargs):
            assert kwargs['numframes'] == 88200
            return np.zeros((88200,2),dtype=np.float32)
        def __enter__(self):return self
        def __exit__(self,*_):pass
    class Monitor:
        id='s.monitor'
        def recorder(self, **kwargs):
            assert kwargs['samplerate']==44100
            return Rec()
    fake = Sc();fake.all_microphones=lambda **kwargs:[Monitor()]
    monkeypatch.setitem(sys.modules,'soundcard',fake)
    with SoundCardLoopback(capture_rate=44100) as source:
        audio=source.read()
    assert audio.shape==(32000,)


def test_worker_protocol_error_on_other_os(monkeypatch,tmp_path):
    # A direct worker call on a non-Windows host cannot fall back to microphone.
    fake_out = io.BytesIO()
    monkeypatch.setattr(worker,'_OUT',fake_out)
    monkeypatch.setattr(worker.sys,'platform','linux')
    assert worker.main() == 1
    tag, length = struct.unpack('!cI',fake_out.getvalue()[:5])
    assert tag==b'E'
    info=json.loads(fake_out.getvalue()[5:5+length])
    assert 'Windows' in info['error']


def test_environment_worker_config_never_selects_host_executable(tmp_path):
    from music_perception.runtime_environment import MusicEnvironment, EnvironmentError
    data=tmp_path/'data';source=tmp_path/'source';source.mkdir()
    env=MusicEnvironment(source=source,data=data,models=data/'models')
    with pytest.raises(EnvironmentError,match='尚未就绪'):
        env.worker_runtime()
    env._python_path = '/fake/python.exe'
    env._active_vendor = data/'vendor';env._active_vendor.mkdir()
    env._report(ready=True)
    p, v = env.worker_runtime()
    assert p=='/fake/python.exe' and v == env._active_vendor.resolve()


def test_timeout_kills_stalled_worker_without_hanging_ui(monkeypatch, tmp_path):
    cap, procs = prepare(monkeypatch,tmp_path,pack(b'R', {'device':'Spkr'}))
    monkeypatch.setattr(iso,'FRAME_TIMEOUT_SECONDS',.05)
    with cap as source:
        # A worker that never produces an audio frame must not trap the listener.
        import io
        class FrozenPipe:
            def read(self, n):
                time.sleep(.2)
                return b''
            def close(self):pass
        procs[0].stdout=FrozenPipe()
        with pytest.raises(ListenerError,match='没有收到扬声器音频帧'):
            source.read()
    assert procs[0].terminated


def test_windows_plugin_chooses_isolated_worker_not_inprocess_com(monkeypatch, tmp_path):
    import asyncio
    import music_perception as plugin_module
    import music_perception.auto_listener as auto
    from test_plugin_contract import build_plugin
    plugin, _ = build_plugin(monkeypatch, tmp_path)
    actual_module = sys.modules[plugin.__class__.__module__]
    monkeypatch.setattr(actual_module.sys, 'platform', 'win32')
    model = tmp_path / 'models' / 'yamnet.tflite'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'fake data')
    plugin._yamnet_model = lambda: model
    monkeypatch.setattr(sys.modules[actual_module.__name__+'.auto_listener'],
                        'YamnetMusicGate', lambda *_a, **_kw: object())
    vendor = tmp_path / 'data' / 'environment' / 'vendor'
    vendor.mkdir(parents=True)

    class Env:
        def snapshot(self):return {'ready':True, 'model_ready':True}
        def worker_runtime(self):return str(tmp_path / 'python.exe'), vendor

    class FakeListener:
        def __init__(self, **kwargs):
            self.source_factory=kwargs.get('source_factory')
            self.running=False
        def start(self):
            self.running=True

    plugin._environment = Env()
    # Listening requires explicit user consent; the gate under test is only
    # reached after that consent is granted.
    plugin._auto_consented = True
    plugin._settings = lambda: asyncio.sleep(0, result={'music_gate_score':.24, 'max_seconds':90})
    monkeypatch.setattr(actual_module, 'LiveListener', FakeListener)
    result = asyncio.run(plugin._start_live())
    assert result['ok'] is True, result
    source = plugin._listener.source_factory()
    assert source.__class__.__name__ == 'IsolatedWindowsLoopback'
    assert source.python.name == 'python.exe'
    assert source.vendor == vendor.resolve()
    assert source.script.name == 'loopback_worker.py'
    # This test bypasses the plugin_fixture and host lifecycle. Stop the two
    # standalone queues so the Python interpreter can exit cleanly after
    # running the entire regression suite (not merely print '215 passed').
    plugin._stop_background_processor()
    assert not plugin._background_worker.is_alive()
    assert not plugin._lyrics_worker_thread.is_alive()


def test_worker_imports_package_relative_modules_under_real_isolated_python(tmp_path):
    """Regression: -I -S worker used to crash before SoundCard with relative import.

    The worker is executed in a *real* standalone interpreter, not just an
    in-process mock. A safe fake soundcard returns no speaker, proving module
    loading completes and the worker advances to device enumeration. No real
    audio device or network is accessed, and host plugin SDK is unavailable.
    """
    source = Path(__file__).resolve().parents[1]
    vendor = tmp_path / 'vendor'
    vendor.mkdir()
    (vendor / 'soundcard.py').write_text(
        'def default_speaker():\n    return None\n', encoding='utf-8')
    worker_script = source / 'loopback_worker.py'
    probe = (
        'import runpy,sys; '
        'm=runpy.run_path(sys.argv[1],run_name="loopback_probe"); '
        'sys.argv=[sys.argv[1],sys.argv[2]]; '
        'sys.platform="win32"; '
        'raise SystemExit(m["main"]())'
    )
    completed = subprocess.run(
        [sys.executable, '-I', '-S', '-c', probe,
         str(worker_script), str(vendor)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=12, check=False)
    assert completed.returncode == 1  # expected fake no-speaker error
    assert completed.stdout.startswith(b'E')
    tag, length = struct.unpack('!cI', completed.stdout[:5])
    assert tag == b'E'
    detail = json.loads(completed.stdout[5:5 + length].decode('utf-8'))
    assert detail['stage'].startswith('打开扬声器回环')
    assert '系统找不到默认扬声器' in detail['error']
    assert 'attempted relative import' not in completed.stderr.decode('utf-8', errors='replace')
    assert 'No module named' not in detail['error']


def test_stream_onset_packets_form_rolling_envelope(monkeypatch, tmp_path):
    """O 包（25ms 起音）要能被后台线程收进滚动环，供 Live2D 低延迟读取。"""
    pcm = np.zeros(32000, dtype='<f4').tobytes()
    output = (pack(b'R', {'device': 'Real speaker (loopback)', 'capture_rate': 48000})
              + pack(b'O', np.array([0.2, 0.9], dtype='<f4').tobytes())
              + pack(b'O', np.array([0.4, 0.1], dtype='<f4').tobytes())
              + pack(b'D', pcm))
    capture, _ = prepare(monkeypatch, tmp_path, output)
    capture.stream_onsets = True
    with capture as source:
        deadline = time.time() + 3
        snap = None
        while time.time() < deadline and snap is None:
            snap = source.take_onsets()
            time.sleep(0.01)
        assert snap is not None, '没有收到任何起音包'
        env, at = snap
        assert len(env) == 4
        assert env[-1] == pytest.approx(0.1, abs=1e-3)
        assert abs(at - time.time()) < 1.0
