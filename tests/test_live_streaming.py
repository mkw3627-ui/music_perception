"""Regression: live music must reach analysis WHILE still playing."""
from __future__ import annotations

import asyncio

import numpy as np
from music_perception.auto_listener import SAMPLE_RATE, FinishedClip, MusicClipCollector


def wave(seconds=2, amplitude=.04):
    return np.full(int(SAMPLE_RATE * seconds), amplitude, dtype=np.float32)


def test_sustained_music_emits_first_chunk_without_waiting_for_silence():
    c = MusicClipCollector(progress_seconds=20, first_progress_seconds=10)
    clips = [c.feed(wave(), True) for _ in range(5)]
    assert clips[:4] == [None] * 4
    first = clips[-1]
    assert first is not None and first.end_reason == 'live_progress'
    assert first.part_index == 1
    assert first.sequence == 1
    assert first.offset_s == 0.0
    assert len(first.samples) == 10 * SAMPLE_RATE
    assert c.recording is True  # The song is STILL playing.
    assert c.buffered_seconds == 0
    assert c.progress_target_seconds == 20
    assert c.progress_percent == 0
    assert all(c.feed(wave(), True) is None for _ in range(9))
    second = c.feed(wave(), True)
    assert second is not None and second.end_reason == 'live_progress'
    assert second.part_index == 2 and second.offset_s == 10
    assert len(second.samples) == 20 * SAMPLE_RATE
    assert c.recording is True


def test_stops_on_non_music_without_forwarding_speech_or_silence():
    c = MusicClipCollector(first_progress_seconds=10)
    assert [c.feed(wave(), False) for _ in range(40)] == [None]*40
    assert not c.recording
    for _ in range(5):
        clip = c.feed(wave(), True)
    assert clip is not None and clip.end_reason == 'live_progress'
    assert c.feed(wave(), True) is None
    assert c.feed(wave(), True) is None
    assert c.feed(wave(), True) is None
    assert c.feed(wave(), True) is None
    assert c.feed(wave(), False) is None
    assert c.feed(wave(), False) is None
    tail = c.feed(wave(), False)
    assert tail is not None and tail.end_reason == 'non_music_boundary'
    assert tail.offset_s == 10
    assert len(tail.samples) == 8 * SAMPLE_RATE  # the rejected speech is absent
    assert not c.recording
    assert all(c.feed(wave(), False) is None for _ in range(20))
    assert not c.recording
    assert c.feed(wave(), True) is None
    assert c.feed(wave(), True) is None
    # New detected music is assigned a different sequence.
    assert c._sequence == 2


def test_session_cap_applies_across_progress_chunks():
    c = MusicClipCollector(max_seconds=34, first_progress_seconds=10, progress_seconds=20)
    results = [c.feed(wave(), True) for _ in range(30)]
    clips = [item for item in results if item is not None]
    assert [item.end_reason for item in clips] == ['live_progress', 'live_progress', 'live_progress']
    # The first session reached its cap after 34s; its 4s tail was below
    # minimum length and discarded, then a second live session began without
    # waiting for silence and already emitted its first 10 seconds.
    assert [item.sequence for item in clips] == [1, 1, 2]
    assert c.recording is True
    assert c._total_samples == SAMPLE_RATE * 26
    assert c._sequence == 2


def test_plugin_sends_music_to_analyzer_while_playing(plugin_fixture, monkeypatch):
    plugin, music, other, pushed = plugin_fixture
    events = []

    def fake_analyzer(samples, cfg):
        events.append(len(samples))
        return {
            'schema_version': 1, 'file_name': '电脑系统正在播放的音乐（实时分段）',
            'duration_s': len(samples)/SAMPLE_RATE,
            'rhythm': {'bpm_estimate': 125, 'beat_times_s': [], 'beat_count': 0},
            'harmony': {'estimated_key': {'name': '不确定'}},
            'melody': {'points_every_2s': []}, 'emotion': {'label': '气氛不确定', 'method': 'fast'},
            'segments': [], 'spectrum': {'rows_low_to_high': [[0,0] for _ in range(32)]},
            'music_events': {'schema':'music_events_v2','timeline_basis':'file_relative_seconds',
               'beat_times_s': [1.0,2.0], 'downbeat_times_s':[],
               'downbeat_status':'unavailable_without_trained_model',
               'sound_layers':{'method':'hpss_spectral_proxy'}, 'events':[],
               'mood_sections':[], 'lyrics':{'status':'not_available'}},
            'limitations': [], 'elapsed_seconds': 0.01,
        }
    plugin._analyze_live_audio_isolated = fake_analyzer

    async def scenario():
        await plugin.startup()
        plugin._auto_consented = True
        plugin._ai_delivery = True
        c = MusicClipCollector(first_progress_seconds=10, progress_seconds=20)
        for _ in range(5):
            result = c.feed(wave(), True)
        assert result is not None and c.recording
        plugin._on_live_clip(result)
        await plugin._live_analysis_task
        assert len(events) == 1 and len(pushed) == 1
        assert plugin._live_reports_completed == 1
        latest = (await plugin.get_last_report()).value['report']
        assert latest['source'] == 'speaker_loopback_after_yamnet_music_gate'
        assert latest['processed_entire_song'] is False
        assert latest['capture']['end_reason'] == 'live_progress'
        assert latest['capture']['offset_s'] == 0
        assert latest['capture']['part_index'] == 1
        # Second chunk from SAME song is automatically processed without
        # spamming a second chat or waiting for a silence boundary.
        for _ in range(10):
            second = c.feed(wave(), True)
        assert second is not None and c.recording
        plugin._on_live_clip(second)
        await plugin._live_analysis_task
        assert len(events) == 2 and len(pushed) == 2
        assert all(item['visibility'] == [] and item['ai_behavior'] == 'read' for item in pushed)
        assert all(item['coalesce_key'] == 'music_perception_latest_auditory_scene' for item in pushed)
        assert '"musical_event_timeline"' in pushed[0]['parts'][0]['text']
        assert '"rhythm_estimate"' in pushed[-1]['parts'][0]['text']
        assert '"auditory_observation"' not in pushed[-1]['parts'][0]['text']
        assert plugin._live_reports_completed == 2
        live = (await plugin.get_status()).value
        assert live['live_recent_segments'][-1]['start_s'] == 10
        assert live['live_recent_segments'][-1]['end_s'] == 30
        # Nonmusic never produces a report. Revoked consent rejects queued PCM.
        for _ in range(10):
            assert c.feed(wave(), False) is None
        assert plugin._live_reports_completed == 2
        plugin._auto_consented = False
        plugin._on_live_clip(FinishedClip(wave(10), 'live_progress', 5))
        assert plugin._live_analysis_queue.empty()
        await plugin.shutdown()

    asyncio.run(scenario())


def test_intermittent_non_music_never_reaches_incremental_analysis():
    """Even if the hysteresis keeps a music session open, a speech/noise PCM
    buffer may NEVER be included in the emitted music waveform.
    """
    c = MusicClipCollector(first_progress_seconds=10, progress_seconds=20)
    for i in range(4):
        assert c.feed(wave(amplitude=.05), True) is None
    assert c.feed(wave(amplitude=.75), False) is None  # rejected loud speech
    chunk = c.feed(wave(amplitude=.05), True)
    assert chunk is not None and chunk.end_reason == 'live_progress'
    assert len(chunk.samples) == 10 * SAMPLE_RATE
    assert np.all(chunk.samples == np.float32(.05))
    assert c.recording  # brief rejection does not restart the music session


def test_steady_music_does_not_repeat_already_analyzed_audio():
    c = MusicClipCollector(first_progress_seconds=10, progress_seconds=20)
    clips = [c.feed(wave(amplitude=.05), True) for _ in range(35)]
    clips = [x for x in clips if x is not None]
    assert [len(x.samples)//SAMPLE_RATE for x in clips] == [10,20,20,20]
    assert [x.offset_s for x in clips] == [0,10,30,50]
    assert [x.part_index for x in clips] == [1,2,3,4]
    assert all(x.sequence == 1 for x in clips)
    assert c.buffered_seconds == 0
