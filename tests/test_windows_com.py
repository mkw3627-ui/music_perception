"""Regression for SoundCard's S_FALSE double-COM-init bug (offline)."""
from __future__ import annotations

import builtins
import threading
import time
import types

import music_perception.auto_listener as listener_module
import numpy as np
import pytest
from music_perception.auto_listener import Decision, ListenerError, LiveListener, SoundCardLoopback


class FakeOle32:
    """CoInitializeEx returns S_FALSE on repeated initialization, per MS docs."""
    def __init__(self):
        self.initialized = False
        self.init_calls = []
        self.uninit_calls = 0

    def CoInitializeEx(self, _reserved, flag):
        self.init_calls.append((threading.get_ident(), flag))
        if self.initialized:
            return 1  # S_FALSE is a successful call, but SoundCard 0.4.6 rejects it
        self.initialized = True
        return 0


class Recorder:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append('recorder-open')
        return self

    def __exit__(self, *_):
        self.events.append('recorder-close')


def mock_soundcard_import(monkeypatch, ole32: FakeOle32, *, loopback=True):
    """Simulate SoundCard's import-time COM init and actual loopback selection."""
    monkeypatch.setattr(listener_module.sys, 'platform', 'win32')
    fake = types.ModuleType('soundcard')
    events = []
    device = types.SimpleNamespace(id='speaker-id', name='Speakers', channels=2)
    fake.default_speaker = lambda: device

    class LoopbackMonitor:
        id = 'speaker-id' if loopback else 'real-hardware-microphone'
        isloopback = loopback

        def recorder(self, **kwargs):
            assert kwargs['samplerate'] in (48000, 44100)
            events.append('recorder-create')
            return Recorder(events)

    fake.all_microphones = lambda **kwargs: [LoopbackMonitor()]
    actual_import = builtins.__import__
    initialized_import = False

    def intercepted(name, *args, **kwargs):
        nonlocal initialized_import
        if name == 'soundcard':
            if not initialized_import:
                initialized_import = True
                hr = ole32.CoInitializeEx(None, 0)
                # Mirrors SoundCard 0.4.6 _COMLibrary.check_error:
                if hr != 0:
                    raise RuntimeError(f'Error {hex(hr + (1 << 32))}')
            return fake
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', intercepted)
    return events


def test_double_initialization_reproduces_uploaded_0x100000001(monkeypatch):
    ole32 = FakeOle32()
    # This emulates the v0.4.6 WindowsComApartment pre-init.
    assert ole32.CoInitializeEx(None, 0) == 0
    mock_soundcard_import(monkeypatch, ole32)
    with pytest.raises(ListenerError, match='0x100000001'):
        with SoundCardLoopback():
            pass
    assert len(ole32.init_calls) == 2


def test_soundcard_initializes_com_itself_only_once(monkeypatch):
    ole32 = FakeOle32()
    events = mock_soundcard_import(monkeypatch, ole32)
    with SoundCardLoopback() as source:
        assert source.device_name == 'Speakers'
        assert ole32.init_calls == [(threading.get_ident(), 0)]
    assert events == ['recorder-create', 'recorder-open', 'recorder-close']


def test_absent_true_loopback_does_not_grab_mic(monkeypatch):
    ole32 = FakeOle32()
    events = mock_soundcard_import(monkeypatch, ole32, loopback=False)
    with pytest.raises(ListenerError, match='不会改用麦克风'):
        with SoundCardLoopback():
            pass
    assert events == []
    assert len(ole32.init_calls) == 1


def test_capture_thread_stays_alive_without_manual_com_init(monkeypatch, tmp_path):
    ole32 = FakeOle32()
    events = mock_soundcard_import(monkeypatch, ole32)
    monkeypatch.setattr(listener_module.sys, 'platform', 'win32')

    def read(self):
        time.sleep(.005)
        return np.zeros(32000, dtype=np.float32)

    monkeypatch.setattr(SoundCardLoopback, 'read', read)

    class FakeGate:
        def __init__(self, *_args, **_kwargs):
            pass

        def classify(self, _pcm):
            return Decision(False, 0.0, 0.0)

    listener = LiveListener(model_file=tmp_path / 'unused', max_seconds=30,
                            threshold=.24, on_clip=lambda clip: None,
                            gate_factory=FakeGate)
    listener.start()
    try:
        for _ in range(200):
            if listener.status()['status'] == 'listening':
                break
            time.sleep(.005)
        assert listener.status()['status'] == 'listening'
    finally:
        assert listener.stop(timeout=3)
    assert len(ole32.init_calls) == 1
    assert ole32.init_calls[0][0] != threading.get_ident()
    assert events[-1] == 'recorder-close'


def test_44100_also_avoids_duplicate_com_initialization(monkeypatch):
    ole32 = FakeOle32()
    events = mock_soundcard_import(monkeypatch, ole32)
    with SoundCardLoopback(capture_rate=44100):
        pass
    assert len(ole32.init_calls) == 1
    assert events[-1] == 'recorder-close'


def test_no_com_preinitialization_in_soundcard_entrypoint():
    import inspect
    source = inspect.getsource(SoundCardLoopback.__enter__)
    assert 'WindowsComApartment(' not in source
    assert 'CoInitializeEx(' not in source.replace('CoInitializeEx returns S_FALSE', '')


def test_missing_soundcard_has_clear_error_without_crashing_host(monkeypatch):
    monkeypatch.setattr(listener_module.sys, 'platform', 'win32')
    actual_import = builtins.__import__
    def unavailable(name, *args, **kwargs):
        if name == 'soundcard':
            raise ImportError('soundcard package absent')
        return actual_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', unavailable)
    with pytest.raises(ListenerError, match='SoundCard 回环采集依赖'):
        with SoundCardLoopback():
            pass


def test_invalid_sample_rate_never_opens_recording_device():
    with pytest.raises(ValueError, match='44100 或 48000'):
        SoundCardLoopback(capture_rate=22050)
