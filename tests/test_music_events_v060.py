"""Timing and consent-safe semantics tests, no remote models or copyrighted audio."""
from __future__ import annotations

import json

import numpy as np
from music_perception.lyrics_sync import load_adjacent_lrc, parse_lrc
from music_perception.music_events import (
    compact_music_context,
    enrich_with_user_lrc,
    offset_event_timeline,
    predict_beat_this,
    timed_music_events,
)


def _mock_report():
    return {'duration_s':6.0, 'rhythm':{'beat_times_s':[0.5,1.0,1.5,2.0,2.5,3.0]},
            'melody':{'points_every_2s':[{'t':0,'midi':60},
                                            {'t':2,'midi':64},{'t':4,'midi':60}]},
            'segments':[{'start_s':0,'end_s':3,'mood_cue':'可能偏欢快'},
                        {'start_s':3,'end_s':6,'mood_cue':'情绪不确定'}]}


def test_beat_timestamp_and_no_made_up_downbeat():
    timeline=timed_music_events(_mock_report())
    assert timeline['beat_times_s']==[0.5,1,1.5,2,2.5,3]
    assert timeline['downbeat_times_s']==[]
    assert timeline['downbeat_status']=='unavailable_without_trained_model'
    assert '声源分离' in timeline['sound_layer_warning']
    assert timeline['mood_sections'][0]['candidate']=='可能偏欢快'
    assert timeline['lyrics']['status']=='not_available'
    assert {e['motion'] for e in timeline['events'] if e['type']=='dominant_pitch_motion'}=={'rising','falling'}
    json.dumps(timeline, allow_nan=False, ensure_ascii=False)


def test_beat_this_mock_returns_downbeat_only_if_trained_checkpoint_exists(tmp_path):
    ckpt=tmp_path/'beat_this.ckpt'
    ckpt.write_bytes(b'stub')
    observed=[]
    def fake_predictor(audio, sr):
        observed.append((len(audio),sr))
        return [0.5,1,1.5,2], [0.5,2]
    b,d=predict_beat_this(np.zeros(3*16000,dtype=np.float32),
                         checkpoint=ckpt, predictor=fake_predictor)
    assert (b,d)==([0.5,1,1.5,2],[0.5,2])
    assert observed==[(48000,16000)]
    timeline=timed_music_events(_mock_report(),waveform=np.zeros(6*16000,np.float32),
                                beat_this_checkpoint=ckpt)
    # Real pretrained backend missing in test runtime: do NOT fake downbeats.
    assert timeline['downbeat_status'].startswith('model_unavailable:')


def test_offset_is_not_player_position():
    timeline=timed_music_events(_mock_report())
    shifted=offset_event_timeline(timeline,15.0)
    assert shifted['beat_times_s'][0]==15.5
    assert any(x.get('type')=='beat' and x.get('at_s')==15.5 for x in shifted['events'])
    assert shifted['mood_sections'][0]['from_s']==15
    assert timeline['beat_times_s'][0]==.5
    assert shifted['timeline_basis']=='confirmed_music_elapsed'
    assert '不是播放器时间' in shifted['time_warning']
    minimal=compact_music_context(shifted)
    assert 'levels_time_by_frequency' not in str(minimal)


def test_hpss_layer_is_not_vocals_or_true_separated_stems():
    t=np.arange(2*16000)/16000
    x=np.sin(2*np.pi*220*t)*.15
    x[::8000]+=.8
    timeline=timed_music_events({'duration_s':2.0,'rhythm':{'beat_times_s':[]}},
                                waveform=x.astype(np.float32))
    assert timeline['sound_layers']['method']=='hpss_spectral_proxy'
    assert timeline['sound_layers']['not_true_stems'] is True
    assert timeline['sound_layers']['cannot_isolate_voice'] is True
    assert timeline['downbeat_times_s']==[]


def test_lrc_parser_offsets_multitags_and_local_authorization(tmp_path):
    raw='''[ar:Someone]\n[offset:+100]\n[00:01.23][00:03.450]first line\n[00:04.2]second line\n[00:90.4]invalid\n'''
    lyrics=parse_lrc(raw)
    assert [x['at_s'] for x in lyrics]==[1.33,3.55,4.3]
    music=tmp_path/'allowed'
    music.mkdir()
    song=music/'song.wav'
    song.write_bytes(b'synthetic')
    song.with_suffix('.lrc').write_text(raw,'utf-8')
    assert load_adjacent_lrc(song,allowed_roots=[music])==lyrics
    import pytest
    with pytest.raises(PermissionError):
        load_adjacent_lrc(song,allowed_roots=[tmp_path/'other'])
    timeline=enrich_with_user_lrc(timed_music_events(_mock_report()),lyrics)
    assert len(timeline['lyrics']['timed_cues'])==3
    assert timeline['lyrics']['status']=='user_supplied_lrc_aligned_to_file_start'


def test_live_lrc_without_player_clock_is_not_falsely_synced():
    lines=[{'at_s':1.2,'text':'do not present as a real lyric'}]
    timeline=timed_music_events(_mock_report(),local_lyrics=lines,
                                lyrics_aligned=False,time_basis='confirmed_music_elapsed')
    assert timeline['lyrics']['status']=='exists_but_media_clock_unaligned'
    assert not timeline['lyrics']['timed_cues']


def test_human_notes_are_grounded_not_generic_numbers():
    from music_perception.music_events import human_music_evidence
    report = _mock_report()
    timeline = timed_music_events(report)
    notes = human_music_evidence(timeline)
    assert any('0.5' in line and '节拍' in line for line in notes)
    assert any('未安装' in line or '没有可验证' in line for line in notes)
    assert any('候选' in line and '情绪' in line for line in notes)
    assert any('不能根据频谱推断唱词' in line for line in notes)
    assert not timeline['downbeat_times_s']


def test_compact_long_track_keeps_end_not_just_first_36_beats():
    report = {'duration_s':120.0,'rhythm':{'beat_times_s':[i*.5 for i in range(240)]},
              'melody':{'points_every_2s':[]},'segments':[]}
    timeline = timed_music_events(report)
    context = compact_music_context(timeline,max_beats=36)
    assert context['beat_times_s'][0] == 0
    assert context['beat_times_s'][-1] == 119.5
    assert len(context['beat_times_s']) == 36
    assert context['beat_times_sampled']
    assert len(context['listening_notes']) >= 2


def test_optional_beat_this_predictions_are_normal_beats_then_downbeats(tmp_path):
    ckpt = tmp_path/'trained.ckpt'
    ckpt.write_bytes(b'only used by mock')
    # The actual CPJKU Postprocessor returns (beats, downbeats), not swapped.
    def fake_prediction(audio, sr):
        return np.array([.25,.75,1.25,1.75]), np.array([.25,1.75])
    beats, downbeats = predict_beat_this(np.zeros(3*16000,dtype=np.float32),
                         checkpoint=ckpt, predictor=fake_prediction)
    assert beats == [.25,.75,1.25,1.75]
    assert downbeats == [.25,1.75]


def test_model_unavailable_does_not_fabricate_lyrics_or_downbeats(tmp_path):
    original = timed_music_events(_mock_report(),beat_this_checkpoint=tmp_path/'missing.ckpt',
                                  waveform=np.zeros(6*16000,dtype=np.float32))
    assert not original['downbeat_times_s']
    assert original['downbeat_status'].startswith('model_unavailable')
    assert original['lyrics']['status'] == 'not_available'
    assert original['source_separation']['status'] == 'not_available'
    ctx = compact_music_context(original)
    assert 'source_separation' in ctx and ctx['source_separation']['vocals'] == 'unknown'


def test_complete_15_minute_song_preserves_beat_end_not_cut_at_400():
    report = {'duration_s': 900.0,
              'rhythm': {'beat_times_s': [i * .5 for i in range(1800)]}}
    timeline = timed_music_events(report)
    assert len(timeline['beat_times_s']) == 1800
    assert timeline['beat_times_s'][-1] == 899.5
    preview = compact_music_context(timeline)
    assert preview['beat_times_s'][-1] == 899.5
    assert preview['beat_times_sampled'] is True


def test_long_local_lyrics_samples_whole_song_without_dumping_all_lines():
    lyrics = [{'at_s':float(i * 5), 'text':f'line_{i}'} for i in range(150)]
    timeline = enrich_with_user_lrc(timed_music_events({'duration_s':900.0,
                                  'rhythm':{'beat_times_s':[]}}), lyrics)
    assert len(timeline['lyrics']['timed_cues']) == 150
    compact = compact_music_context(timeline)
    assert compact['lyrics']['cues_sampled'] is True
    assert len(compact['lyrics']['timed_cues']) == 12
    assert compact['lyrics']['timed_cues'][0]['at_s'] == 0
    assert compact['lyrics']['timed_cues'][-1]['at_s'] == 745
