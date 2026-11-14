from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from test_plugin import plugin_fixture

from music_perception.auto_listener import (
    Decision, FinishedClip, LiveListener, ListenerError, MusicClipCollector,
    YamnetMusicGate, SoundCardLoopback, SAMPLE_RATE,
)


def chunk(n=2, amp=.1):
    return np.full(SAMPLE_RATE * n, amp, np.float32)


def test_only_music_delivered_to_downstream():
    collector = MusicClipCollector(min_clip_seconds=8)
    # None of the first eight nonmusic windows is forwarded.
    for i in range(8):
        assert collector.feed(chunk(amp=.02 * (i + 1)), False) is None
    assert collector.feed(chunk(), True) is None
    assert collector.recording is False  # Need 2 consecutive music windows.
    assert collector.feed(chunk(), True) is None
    assert collector.recording is True
    for i in range(2):
        assert collector.feed(chunk(), True) is None
    assert collector.feed(chunk(), False) is None
    assert collector.feed(chunk(), False) is None
    result = collector.feed(chunk(), False)
    assert result is not None
    assert result.end_reason == 'non_music_boundary'
    assert len(result.samples) == 8 * SAMPLE_RATE
    # The rejected pre-music audio and trailing speech are absent.
    assert np.all(result.samples == .1)
    assert not collector.recording


def test_short_noise_never_becomes_a_song():
    collector = MusicClipCollector(min_clip_seconds=8)
    labels = [True, False, False, True, True, False, False, False]
    assert [collector.feed(chunk(), b) for b in labels] == [None] * len(labels)
    assert not collector.recording


def test_one_false_window_is_ignored_but_never_analyzed():
    # Focus this legacy test on the final boundary; separate tests cover
    # earlier 10-second incremental delivery while music keeps playing.
    c = MusicClipCollector(min_clip_seconds=4, first_progress_seconds=20)
    for yes in [True, True, False, True, True, False, False, False]:
        result = c.feed(chunk(), yes)
    assert result is not None
    assert len(result.samples) == 4 * 2 * SAMPLE_RATE  # 4 music, excludes interior non-music
    assert result.music_windows == 4


def test_session_cap_rotates_without_stopping_continuous_playback():
    c = MusicClipCollector(max_seconds=10, min_clip_seconds=4)
    found = []
    for _ in range(12):
        res = c.feed(chunk(), True)
        if res:
            found.append(res)
    assert len(found) == 2
    assert [clip.end_reason for clip in found] == ['duration_limit', 'duration_limit']
    assert all(len(clip.samples) == SAMPLE_RATE * 10 for clip in found)
    assert [clip.sequence for clip in found] == [1, 2]
    # After each rotated session, listening continues without a silence requirement.
    assert c.recording is True  # two further windows were collected as a new session
    assert c._sequence == 3
    assert c.feed(chunk(), True) is None  # capture continues after session rollover


def test_pending_audio_discarded_on_manual_stop():
    c = MusicClipCollector()
    c.feed(chunk(), True)
    c.feed(chunk(), True)
    assert c.recording
    c.discard()
    assert not c.recording
    assert c.feed(chunk(), False) is None


class FakeInterpreter:
    def __init__(self, *a, **kw):
        self.input_shape = np.asarray([15600], np.int32)
        self.data = None
        self.scores = np.zeros((1, 521), np.float32)

    def get_input_details(self):
        return [{"shape": self.input_shape, "index": 0}]

    def get_output_details(self):
        return [{"shape": np.asarray([1, 521]), "index": 1}]

    def allocate_tensors(self):
        pass

    def resize_tensor_input(self, idx, shape, strict=False):
        self.input_shape = np.asarray(shape)

    def set_tensor(self, index, data):
        self.data = data

    def invoke(self):
        pass

    def get_tensor(self, idx):
        return self.scores


def test_yamnet_music_and_speech_filter(monkeypatch, tmp_path):
    instance = FakeInterpreter()
    interpreter_module = types.ModuleType('ai_edge_litert.interpreter')
    interpreter_module.Interpreter = lambda **kwargs: instance
    package = types.ModuleType('ai_edge_litert')
    package.interpreter = interpreter_module
    monkeypatch.setitem(sys.modules, 'ai_edge_litert', package)
    monkeypatch.setitem(sys.modules, 'ai_edge_litert.interpreter', interpreter_module)
    model = tmp_path / 'yamnet.tflite'
    model.write_bytes(b'fake model')
    gate = YamnetMusicGate(model)
    sound = chunk()
    instance.scores[0, 0] = .85  # Speech
    assert not gate.classify(sound).music
    instance.scores[0, 132] = .9  # Music independently detected
    assert gate.classify(sound).music
    assert gate.classify(np.zeros_like(sound)).music is False  # Silence fails closed
    instance.scores[0, 132] = .06
    instance.scores[0, 211] = .90  # Pop music, not just generic Music
    instance.scores[0, 0] = .05
    assert gate.classify(sound).music


def test_missing_model_is_blocking(tmp_path):
    with pytest.raises(ListenerError, match='YAMNet'):
        YamnetMusicGate(tmp_path / 'model.tflite')


def test_loopback_no_microphone_fallback(monkeypatch):
    recorder_called = []
    class Speaker:
        id = 'speaker-001'
    class Microphone:
        id = 'mic-001'
        def recorder(self, **kwargs):
            recorder_called.append(kwargs)
    fake_soundcard = types.ModuleType('soundcard')
    fake_soundcard.default_speaker = lambda: Speaker()
    fake_soundcard.all_microphones = lambda **kwargs: [Microphone()]
    monkeypatch.setitem(sys.modules, 'soundcard', fake_soundcard)
    with pytest.raises(ListenerError, match='回环'):
        with SoundCardLoopback():
            pass
    assert not recorder_called


def test_thread_lifecycle_synth_source(tmp_path):
    # The fake source produces 8 seconds music and 6 seconds non-music.
    clips = []
    class Source:
        def __enter__(self):
            self.i = 0
            return self
        def __exit__(self, *_): pass
        def read(self):
            import time
            if self.i >= 8:
                time.sleep(.004)
            self.i += 1
            return chunk()
    class Classifier:
        def __init__(self, *_a, **_kw):
            self.i = 0
        def classify(self, _):
            self.i += 1
            return Decision(self.i <= 4, .9 if self.i <= 4 else 0, 0)
    listener = LiveListener(model_file=tmp_path / 'fake', max_seconds=900,
                            threshold=.24, on_clip=clips.append,
                            source_factory=Source, gate_factory=Classifier)
    listener.start()
    import time
    for _ in range(100):
        if clips: break
        time.sleep(.007)
    assert len(clips) == 1
    assert listener.status()['music_clips_completed'] == 1
    assert listener.stop()
    assert listener.status()['status'] == 'stopped'


def test_live_report_never_pretends_whole_song(plugin_fixture, monkeypatch):
    # Plugin callback generates a JSON report from RAM and no raw WAV file.
    from music_perception import MusicPerceptionPlugin
    import music_perception as module
    plugin, music, other, pushed = plugin_fixture
    real_analyze = module.analyze_waveform
    plugin._analyze_live_audio_isolated = lambda wave, cfg: real_analyze(
        wave, display_name='电脑系统正在播放的音乐（实时分段）',
        segment_seconds=cfg['segment_seconds'], max_seconds=cfg['max_seconds'],
        emotion_mode=cfg['emotion_mode'],emotion_models_dir=cfg['emotion_models_dir'])
    samples = np.sin(2*np.pi*440*np.arange(10*SAMPLE_RATE)/SAMPLE_RATE).astype(np.float32)*.25
    async def run():
        await plugin.startup()
        plugin._auto_consented = True
        await plugin._process_live_clip(FinishedClip(samples,'non_music_boundary',5))
        response = (await plugin.get_last_report()).value['report']
        assert response['duration_s'] == 10.0
        assert response['processed_entire_song'] is False
        assert response['processed_entire_capture'] is True
        assert response['source'] == 'speaker_loopback_after_yamnet_music_gate'
        assert pushed  # AI perception defaults on but does not force a user chat response.
        assert not list(plugin.data_path().glob('*.wav'))
    asyncio.run(run())


def test_explicit_consent_required_and_persisted(plugin_fixture, monkeypatch):
    import music_perception as module
    plugin, *_ = plugin_fixture
    class FakeGate:
        def __init__(self,*a,**kw): pass
    class FakeLive:
        def __init__(self, **kwargs): self.running=False
        def start(self): self.running=True
        def stop(self): self.running=False;return True
        def status(self): return {'status':'listening' if self.running else 'stopped', 'capturing_music':False}
    import music_perception.auto_listener as listener_module
    monkeypatch.setattr(listener_module, 'YamnetMusicGate', FakeGate)
    monkeypatch.setattr(module, 'LiveListener', FakeLive)
    fake_model=plugin.data_path('models') / 'yamnet.tflite'
    fake_model.parent.mkdir(parents=True)
    fake_model.write_bytes(b'dummy')
    async def run():
        await plugin.startup()
        denied = (await plugin.start_auto_listen(consent=False)).value
        assert denied['ok'] is False
        assert not plugin._preferences_path().exists()
        enabled = (await plugin.start_auto_listen(consent=True)).value
        assert enabled['ok'] is True
        assert json.loads(plugin._preferences_path().read_text())['speaker_loopback_enabled'] is True
        assert (await plugin.set_ai_delivery(enabled=True)).value['ok']
        assert (await plugin.get_status()).value['ai_delivery_enabled']
        await plugin.shutdown()
        # Explicit saved consent auto-resumes on new startup.
        await plugin.startup()
        await plugin._environment_task
        assert plugin._listener.running
        stopped = (await plugin.stop_auto_listen()).value
        assert stopped['ok'] is True
        assert json.loads(plugin._preferences_path().read_text())['speaker_loopback_enabled'] is False
        assert json.loads(plugin._preferences_path().read_text())['ai_delivery_enabled'] is True
        assert not plugin._auto_consented  # Explicit stop still disables sensing
        await plugin.shutdown()
    asyncio.run(run())


def test_yamnet_uses_multiple_fixed_15600_sample_windows(monkeypatch, tmp_path):
    """Regression: the Lite classifier must NEVER be fed raw 2s (32k) PCM."""
    from music_perception.auto_listener import YAMNET_FRAME_SAMPLES

    class RecordingInterpreter(FakeInterpreter):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.resize_called = False
        def set_tensor(self, index, data):
            self.calls.append(tuple(data.shape))
            assert tuple(data.shape) == (YAMNET_FRAME_SAMPLES,)
        def resize_tensor_input(self, *args, **kwargs):
            self.resize_called = True
            raise AssertionError("This fixed-length model must not be resized")
    fake = RecordingInterpreter()
    mock = types.ModuleType('ai_edge_litert.interpreter')
    mock.Interpreter = lambda **kw: fake
    package = types.ModuleType('ai_edge_litert')
    package.interpreter = mock
    monkeypatch.setitem(sys.modules, 'ai_edge_litert', package)
    monkeypatch.setitem(sys.modules, 'ai_edge_litert.interpreter', mock)
    model = tmp_path / 'yamnet.tflite'
    model.write_bytes(b'test')
    fake.scores[0, 132] = .92
    gate = YamnetMusicGate(model)
    result = gate.classify(chunk())
    assert result.music is True
    assert len(fake.calls) >= 3
    assert fake.resize_called is False


def test_yamnet_rejects_wrong_fixed_input(monkeypatch, tmp_path):
    class WrongInput(FakeInterpreter):
        def __init__(self):
            super().__init__()
            self.input_shape = np.asarray([32000], np.int32)
    fake = WrongInput()
    mock = types.ModuleType('ai_edge_litert.interpreter')
    mock.Interpreter = lambda **kw: fake
    package = types.ModuleType('ai_edge_litert')
    package.interpreter = mock
    monkeypatch.setitem(sys.modules, 'ai_edge_litert', package)
    monkeypatch.setitem(sys.modules, 'ai_edge_litert.interpreter', mock)
    model = tmp_path / 'wrong.tflite'
    model.write_bytes(b'test')
    with pytest.raises(ListenerError, match='15600|0.975'):
        YamnetMusicGate(model)


def test_exact_loopback_windows_and_linux_only():
    from music_perception.auto_listener import pick_speaker_monitor
    class Device:
        def __init__(self, device_id, loopback=False):
            self.id = device_id
            self.isloopback = loopback
    win_speaker = Device('output-A')
    physical_mic = Device('output-A', False)  # Must reject even if ID is identical
    speaker_monitor = Device('output-A', True)
    assert pick_speaker_monitor(win_speaker, [physical_mic], 'win32') is None
    assert pick_speaker_monitor(win_speaker, [physical_mic, speaker_monitor], 'win32') is speaker_monitor
    pulse_sink = Device('alsa_output.usb-123')
    pulse_monitor = Device('alsa_output.usb-123.monitor')
    assert pick_speaker_monitor(pulse_sink, [physical_mic], 'linux') is None
    assert pick_speaker_monitor(pulse_sink, [physical_mic, pulse_monitor], 'linux') is pulse_monitor
    assert pick_speaker_monitor(pulse_sink, [pulse_monitor], 'darwin') is None
