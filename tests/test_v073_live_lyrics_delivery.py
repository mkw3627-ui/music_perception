"""0.7.3: distinguish captured music from finished ASR, and show text quickly.

Synthetic data only; these tests never start the Windows sound device or
contact model repositories.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from music_perception.auto_listener import FinishedClip
from music_perception.lyrics_backend import transcribe_singing_candidate


def run(coro):
    return asyncio.run(coro)


def synthetic_clip(seconds=6, sequence=8, offset=0):
    return FinishedClip(np.ones(16000 * seconds, dtype=np.float32) * .04,
                        'early_lyrics_preview', seconds//2,
                        offset_s=float(offset),sequence=sequence,part_index=1)


def test_captured_clip_is_visible_to_ui_while_asr_has_not_responded(plugin_fixture):
    plugin,_,_,pushes = plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=False
        plugin._listener_session=5
        plugin._on_track_boundary('track_started',8)
        assert plugin._on_early_lyrics_clip(synthetic_clip())
        payload=(await plugin.get_live_lyrics()).value
        assert payload['ok'] and payload['songs']
        song=payload['songs'][-1]
        assert song['current']
        assert not song['cues']  # No false textual transcription yet.
        progress=song['recognition_progress']
        assert progress['pending'] is True
        assert progress['state']=='queued'
        assert progress['captured_until_s']==6
        assert progress['completed_until_s']==0
        assert payload['queue_depth']==1
        assert not pushes  # Local panel does not send chat messages.
        await plugin.shutdown()
    run(scenario())


def test_successful_first_segment_is_visible_immediately_without_rhythm_wait(plugin_fixture):
    plugin,_,_,pushes=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._listener_session=5
        plugin._on_track_boundary('track_started',8)
        assert plugin._on_early_lyrics_clip(synthetic_clip())
        job=plugin._lyrics_queue.get_nowait()
        plugin._live_session_config={'lyrics_model':'base','lyrics_language':'zh'}
        plugin._transcribe_singing_live_audio_isolated=lambda *_: {
            'status':'asr_hypothesis','language_candidate':'zh',
            'timed_cues':[{'at_s':.4,'end_s':2.1,'text_preview':'自造测试歌句'}]}
        clip, ctx, session, generation=job
        plugin._process_one_lyrics_task(clip,ctx,session,session,plugin._lyrics_queue,
                                        quality_generation=generation)
        payload=(await plugin.get_live_lyrics()).value
        progress=payload['songs'][-1]['recognition_progress']
        assert progress['state']=='asr_hypothesis'
        assert progress['completed_until_s']==6
        assert progress['pending'] is False
        assert payload['songs'][-1]['cues'][0]['text_preview']=='自造测试歌句'
        assert all(x.get('visibility')==[] and x.get('ai_behavior')=='read' for x in pushes)  # Private AI only, no chat.
        await plugin.shutdown()
    run(scenario())


def test_empty_transcription_has_completed_status_not_infinite_wait(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._listener_session=5
        plugin._on_track_boundary('track_started',8)
        assert plugin._on_early_lyrics_clip(synthetic_clip())
        clip, ctx, session, generation=plugin._lyrics_queue.get_nowait()
        plugin._live_session_config={'lyrics_model':'base','lyrics_language':'zh'}
        plugin._transcribe_singing_live_audio_isolated=lambda *_: {
            'status':'no_reliable_lyrics','language_candidate':'zh',
            'timed_cues':[]}
        plugin._process_one_lyrics_task(clip,ctx,session,session,plugin._lyrics_queue,
                                        quality_generation=generation)
        info=(await plugin.get_live_lyrics()).value
        assert info['songs'][-1]['recognition_progress']['state']=='no_reliable_lyrics'
        assert info['songs'][-1]['recognition_progress']['pending'] is False
        assert info['songs'][-1]['cues']==[]
        await plugin.shutdown()
    run(scenario())


def test_switch_prevents_old_lyric_progress_from_being_current(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._listener_session=5
        plugin._on_track_boundary('track_started',8)
        assert plugin._on_early_lyrics_clip(synthetic_clip(sequence=8))
        previous=plugin._song_current_id
        plugin._on_track_boundary('track_ended',8)
        plugin._on_track_boundary('track_started',9)
        state=(await plugin.get_live_lyrics()).value
        assert state['current_song_id'] != previous
        old=next(s for s in state['songs'] if s['id']==previous)
        new=next(s for s in state['songs'] if s['id']==state['current_song_id'])
        assert old['recognition_progress']['state']=='interrupted_by_song_switch'
        assert not new['recognition_progress']['pending']
        assert new['recognition_progress']['state']=='collecting_audio'
        assert len(plugin._lyrics_queue.queue)==0
        await plugin.shutdown()
    run(scenario())


def test_fast_whisper_line_mode_still_returns_timestamps_without_word_alignment(tmp_path):
    settings=[]
    class Model:
        def transcribe(self,pcm,**kwargs):
            settings.append(kwargs)
            from types import SimpleNamespace as S
            return [S(start=.1,end=1.2,text='原创合成短句',avg_logprob=-.3,
                      no_speech_prob=.01,compression_ratio=1.,words=None)],S(language='zh')
    result=transcribe_singing_candidate(np.ones(96000,dtype=np.float32)*.01,
        model_dir=tmp_path,model_factory=lambda *a,**k:Model(),
        device_policy='cpu',language_hint='zh')
    assert settings[0]['word_timestamps'] is False
    assert result['timed_cues'][0]['at_s']==.1
    assert result['timed_cues'][0]['end_s']==1.2
    assert result['timed_cues'][0]['words']==[]


def test_ui_explains_capture_and_processing_without_inventing_lyrics():
    html=Path(__file__).resolve().parents[1].joinpath('static/index.html').read_text('utf-8')
    assert 'recognition_progress' in html
    assert '已抓到音频片段，正在识别文字' in html
    assert '这段识别结束，但没有得到足够可靠的唱词' in html
    assert 'lyrics.textContent=String(cue.text_preview||\'\')' in html
    assert 'innerHTML=cue.text_preview' not in html
