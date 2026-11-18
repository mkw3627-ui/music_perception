"""Regression of AI-first spectrum ingestion and silent context delivery."""
import asyncio
import json

import numpy as np
from music_perception.auto_listener import SAMPLE_RATE, FinishedClip
from music_perception.spectral_context import ai_auditory_context, auxiliary_cues, spectrum_for_ai


def test_spectrogram_time_frequency_axes_are_consistent():
    t = np.arange(10 * SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    src = .12 * np.sin(2 * np.pi * 440 * t)
    spec = spectrum_for_ai(src)
    assert spec['primary_evidence'] is True and spec['not_native_audio'] is True
    assert len(spec['levels_time_by_frequency_0_100']) == 24
    assert all(len(r) == 48 for r in spec['levels_time_by_frequency_0_100'])
    assert len(spec['frequency_centers_hz_low_to_high']) == 48
    assert len(spec['time_centers_s']) == 24
    assert all(0 <= n <= 100 for r in spec['levels_time_by_frequency_0_100'] for n in r)
    target = min(range(48), key=lambda j: abs(spec['frequency_centers_hz_low_to_high'][j] - 440))
    # 440 Hz sinusoidal tone should dominate this near-440 Hz band.
    mean_levels = np.mean(spec['levels_time_by_frequency_0_100'], axis=0)
    assert mean_levels[target] > mean_levels[0]
    assert spec['source'] == 'wasapi_default_render_endpoint_mix'
    assert len(json.dumps(spec, ensure_ascii=False)) < 16000


def test_spectrum_is_not_emotion_label_and_rejects_bad_inputs():
    import pytest
    with pytest.raises(ValueError):
        spectrum_for_ai(np.array([float('nan')]*16000, dtype=np.float32))
    with pytest.raises(ValueError):
        spectrum_for_ai(np.ones(100, dtype=np.float32))
    data = spectrum_for_ai(np.zeros(SAMPLE_RATE * 10, dtype=np.float32))
    assert max(max(r) for r in data['levels_time_by_frequency_0_100']) == 0
    assert 'emotion' not in data and 'happy' not in str(data).lower()


def test_ai_context_has_measurements_first_auxiliary_second():
    spec = spectrum_for_ai(np.ones(SAMPLE_RATE * 10, dtype=np.float32) * .02)
    hints = auxiliary_cues({'emotion':{'label':'偏欢快','method':'fast'},
                            'rhythm':{'bpm_estimate':121}})
    context = ai_auditory_context(spec, hints, {'offset_s':0,'end_s':10, 'part_index':1})
    assert list(context)[:2] == ['schema', 'auditory_observation']
    assert context['auditory_observation'] is spec
    assert 'auxiliary' in list(context)[2]
    assert hints['emotion_hint']['hypothesis'] == '偏欢快'
    assert 'warning' in hints['emotion_hint']


def test_ai_context_is_invisible_and_no_forced_reply(plugin_fixture):
    plugin, *_rest, pushes = plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._auto_consented = True
        plugin._ai_delivery = True
        c = ai_auditory_context({'levels_time_by_frequency_0_100': [[20]*48]*24},
                                {'emotion_hint': {'warning': 'uncertain'}},
                                {'music_sequence':1, 'part_index':1},music_events={'schema':'music_events_v1','timeline_basis':'music_only_elapsed','events':[{'type':'beat','at_s':1.0}],'beat_times_s':[1.0],'downbeat_times_s':[],'downbeat_status':'unavailable_without_trained_model','mood_sections':[],'sound_layers':{'status':'estimated'},'lyrics':{'status':'not_available'}})
        c['musical_event_timeline'] = {'schema':'music_events_v2', 'timeline_basis':'confirmed_music_elapsed',
           'beat_times_s':[1.0,1.5,2.0], 'downbeat_times_s':[], 'downbeat_status':'unavailable_without_trained_model',
           'mood_sections':[], 'sound_layers':{'method':'hpss_spectral_proxy'},
           'lyrics':{'status':'not_available'}}
        plugin._recent_auditory_context = c
        assert plugin._deliver_auditory_context(c, FinishedClip(np.ones(10), 'live_progress', 5), 10)
        assert len(pushes) == 1
        msg = pushes[0]
        assert msg['ai_behavior'] == 'read'
        assert msg['visibility'] == []
        assert msg['coalesce_key']
        assert msg['parts'][0]['type'] == 'text'
        assert 'musical_event_timeline' in msg['parts'][0]['text']
        assert 'levels_time_by_frequency_0_100' not in msg['parts'][0]['text']
        data = (await plugin.get_ai_auditory_context()).value
        assert data['available'] and data['context']['musical_event_timeline']==c['musical_event_timeline']
        assert 'auditory_observation' not in data['context']
        raw = (await plugin.get_ai_auditory_context(include_raw_spectrum=True)).value
        assert raw['context'] == c
        await plugin.stop_auto_listen()
        assert (await plugin.get_ai_auditory_context()).value['available'] is False
        assert plugin._recent_auditory_context is None
        await plugin.shutdown()
    asyncio.run(scenario())


def test_no_ai_delivery_when_not_allowed(plugin_fixture):
    plugin, *_rest, pushed = plugin_fixture
    async def scenario():
        await plugin.startup()
        context = {'auditory_observation':{'levels_time_by_frequency_0_100':[]}}
        assert plugin._deliver_auditory_context(context, FinishedClip([], 'live_progress', 0), 10) is False
        plugin._auto_consented = True
        assert plugin._deliver_auditory_context(context, FinishedClip([], 'live_progress', 0), 10) is False
        assert not pushed
        await plugin.shutdown()
    asyncio.run(scenario())


def test_isolated_listener_callback_handles_closed_start_loop(plugin_fixture, monkeypatch, tmp_path):
    """N.E.K.O may close the startup AND entry loops before music arrives.

    Native capture thread must remain able to queue the FIRST 10s chunk.
    """
    import time

    import music_perception as module
    from music_perception import auto_listener as listener_module
    plugin, *_ = plugin_fixture
    captured = {}
    received = []

    class FakeGate:
        def __init__(self, *args, **kwargs):
            pass
    class FakeLiveListener:
        def __init__(self, **kw):
            captured['on_clip'] = kw['on_clip']
            self.running = False
        def start(self):
            self.running = True
        def stop(self):
            self.running = False
            return True
    async def process(clip, **_):
        received.append(clip.sequence)
    monkeypatch.setattr(listener_module, 'YamnetMusicGate', FakeGate)
    monkeypatch.setattr(module, 'LiveListener', FakeLiveListener)
    plugin._process_live_clip = process
    model_file = tmp_path / 'yamnet.tflite'
    model_file.write_bytes(b'fake-model')
    plugin._yamnet_model = lambda: model_file

    async def startup():
        await plugin.startup()
        plugin._auto_consented = True
    asyncio.run(startup())

    async def new_action():
        result = await plugin._start_live()
        assert result['ok']
    asyncio.run(new_action())

    # Both SDK loops have closed here; our worker's loop is still alive.
    captured['on_clip'](FinishedClip(np.zeros(SAMPLE_RATE), 'live_progress', 1, sequence=4))
    for _ in range(100):
        if received:
            break
        time.sleep(.02)
    assert received == [4]
    asyncio.run(plugin.stop_auto_listen())
    assert not plugin._background_worker.is_alive()


def test_audio_queue_backpressure_and_revocation(plugin_fixture):
    plugin, *_ = plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._auto_consented = True
        for n in range(8):
            assert plugin._enqueue_live_clip_threadsafe(
                FinishedClip(np.zeros(400), 'live_progress', 1, sequence=n))
        assert plugin._background_queue.qsize() == 4
        assert plugin._live_dropped_segments == 4
        await plugin.stop_auto_listen()
        assert plugin._background_queue.empty()
        assert not plugin._enqueue_live_clip_threadsafe(
            FinishedClip(np.zeros(400), 'live_progress', 1))
        await plugin.shutdown()
    asyncio.run(scenario())


def test_history_spectra_excludes_previous_track(plugin_fixture):
    plugin, *_ = plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._auto_consented = True
        plugin._ai_delivery = True
        p = {'levels_time_by_frequency_0_100': [[10]*48]*24,
             'energy_envelope_0_100': [10]*24}
        a = {'rhythm_estimate': {'bpm':115}}
        song_a = ai_auditory_context(p,a,{'music_sequence':1,'part_index':1})
        song_b = ai_auditory_context(p,a,{'music_sequence':2,'part_index':1})
        song_a['musical_event_timeline']={'beat_times_s':[0.5], 'schema':'music_events_v2'}
        song_b['musical_event_timeline']={'beat_times_s':[0.5], 'schema':'music_events_v2'}
        plugin._auditory_history = [song_a, song_b]
        plugin._recent_auditory_context = song_b
        response = (await plugin.get_ai_auditory_context(include_recent_history=True)).value
        assert response['available']
        assert len(response['recent_music_parts']) == 1
        assert response['recent_music_parts'][0]['music_capture']['music_sequence'] == 2
        assert 'auditory_observation' not in response['recent_music_parts'][0]
        await plugin.shutdown()
    asyncio.run(scenario())


def test_compact_history_retains_duration_and_resolution():
    from music_perception.spectral_context import condensed_history_item
    sample = np.ones(10*SAMPLE_RATE, dtype=np.float32)*.01
    src = ai_auditory_context(spectrum_for_ai(sample),
        {'rhythm_estimate':{'bpm':None}},
        {'music_sequence':7, 'part_index':2,'offset_s':10,'end_s':20})
    compact = condensed_history_item(src)
    assert len(compact['time_frequency_thumbnail_6x12_0_100']) == 6
    assert all(len(r)==12 for r in compact['time_frequency_thumbnail_6x12_0_100'])
    assert compact['music_capture']['music_sequence'] == 7


def test_ai_tool_cannot_bypass_separate_sharing_consent(plugin_fixture):
    plugin, *_ = plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._auto_consented = True
        plugin._ai_delivery = False  # Explicit opt-out still blocks the AI API under default-on policy.
        plugin._recent_auditory_context = {"auditory_observation": {"levels_time_by_frequency_0_100": [[90]]},
                                          "musical_event_timeline": {"schema":"music_events_v2","beat_times_s":[1.0]}}
        denied = (await plugin.get_ai_auditory_context()).value
        assert denied['available'] is False and denied['ok'] is False
        assert '授权' in denied['reason']
        plugin._ai_delivery = True
        allowed = (await plugin.get_ai_auditory_context()).value
        assert allowed['available'] is True
        await plugin.shutdown()
    asyncio.run(scenario())

def test_failed_sdk_delivery_is_not_marked_success(plugin_fixture):
    plugin, _, _, pushes = plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._auto_consented = True
        plugin._ai_delivery = True
        plugin._auditory_history = []
        context = ai_auditory_context(spectrum_for_ai(np.zeros(10*SAMPLE_RATE)),
            {'rhythm_estimate': {'bpm': None}},
            {'music_sequence': 1, 'part_index':1, 'offset_s': 0, 'end_s': 10}, music_events={'schema':'music_events_v1','timeline_basis':'music_only_elapsed','events':[{'type':'beat','at_s':1.0}],'beat_times_s':[1.0],'downbeat_times_s':[],'downbeat_status':'unavailable_without_trained_model','mood_sections':[],'sound_layers':{'status':'estimated'},'lyrics':{'status':'not_available'}})
        context['musical_event_timeline'] = {'schema':'music_events_v2', 'beat_times_s':[],
            'downbeat_status':'unavailable_without_trained_model'}
        plugin.push_message = lambda **kwargs: {'submitted': False, 'reason': 'backpressure'}
        result = plugin._deliver_auditory_context(context,
            FinishedClip(np.zeros(10*SAMPLE_RATE), 'live_progress', 3), 10)
        assert result is False
        assert plugin._last_auditory_submit_error is not None
        assert 'backpressure' in plugin._last_auditory_submit_error
        await plugin.shutdown()
    asyncio.run(scenario())

def test_minimal_events_arrive_even_if_auxiliary_analysis_fails(plugin_fixture, monkeypatch):
    plugin, _, _, pushed = plugin_fixture
    def broken_feature_extractor(*args, **kwargs):
        raise RuntimeError('failed optional melody analyzer')
    async def scenario():
        await plugin.startup()
        plugin._analyze_live_audio_isolated = broken_feature_extractor
        plugin._auto_consented = True
        plugin._ai_delivery = True
        plugin._live_session_config = {'segment_seconds': 10, 'max_seconds': 90,
             'emotion_mode': 'fast', 'emotion_models_dir':''}
        sample = np.zeros(SAMPLE_RATE*10, dtype=np.float32)
        await plugin._process_live_clip(FinishedClip(sample, 'live_progress', 4, sequence=1, part_index=1))
        assert plugin._auditory_context_count == 1
        assert plugin._recent_auditory_context['auditory_observation']['primary_evidence']
        assert len(pushed) == 1
        assert pushed[0]['ai_behavior'] == 'read' and pushed[0]['visibility'] == []
        assert 'musical_event_timeline' in pushed[0]['parts'][0]['text']
        assert plugin._live_last_error and 'optional melody analyzer' in plugin._live_last_error
        await plugin.shutdown()
    asyncio.run(scenario())
