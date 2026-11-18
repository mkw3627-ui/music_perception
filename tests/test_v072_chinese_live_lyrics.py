"""New local lyric presentation: no network, no real songs or copyrighted data."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from music_perception.auto_listener import FinishedClip
from music_perception.lyrics_backend import transcribe_singing_candidate
from music_perception.song_memory import RecentSongMemory


def run(coroutine):
    return asyncio.run(coroutine)


def test_multilingual_chinese_lock_is_sent_to_actual_asr_decoder():
    observed=[]
    class FakeModel:
        def transcribe(self, pcm, **kwargs):
            observed.append(kwargs)
            seg=SimpleNamespace(text='测试中文歌声',start=.2,end=2.1,
                avg_logprob=-.2,no_speech_prob=.01,words=[])
            return [seg],SimpleNamespace(language='zh')
    def factory(name,**kwargs):
        return FakeModel()
    pcm=np.sin(np.arange(16000*6)*np.pi/60).astype(np.float32)*.05
    result=transcribe_singing_candidate(pcm,model_dir=Path('/tmp/unused-music-model'),
        model_factory=factory,device_policy='cpu',language_hint='zh')
    assert observed[0]['language']=='zh'
    assert observed[0]['task']=='transcribe'
    assert observed[0]['multilingual'] is False
    assert result['language_mode']=='zh'
    assert result['language_candidate']=='zh'
    assert result['timed_cues'][0]['language_candidate']=='zh'
    assert result['timed_cues'][0]['text_preview']=='测试中文歌声'


def test_auto_language_does_not_force_english_translation():
    observed=[]
    class FakeModel:
        def transcribe(self, pcm, **kwargs):
            observed.append(kwargs)
            return [],SimpleNamespace(language='zh')
    transcribe_singing_candidate(np.ones(16000,dtype=np.float32)*.05,
        model_dir=Path('/tmp/no-download'),model_factory=lambda *a,**k:FakeModel(),
        device_policy='cpu',language_hint='auto')
    assert observed[0]['language'] is None
    assert observed[0]['multilingual'] is True
    assert observed[0]['task']=='transcribe'


def test_two_interrupted_songs_have_independent_display_and_oldest_is_evicted(tmp_path):
    mem=RecentSongMemory(tmp_path/'recent_songs.json')
    for i in range(3):
        song_id=f'session:{i}'
        mem.begin(song_id,title=f'试听 {i}')
        assert mem.observe(song_id,lyrics=[{'at_s':1.1,'text_preview':f'自造示例 {i}',
            'language_candidate':'zh'}],lyrics_quality='base|zh')
        mem.end(song_id,reason='metadata_switch',position_s=15.1,duration_s=200.)
    assert [song['id'] for song in mem.recent()]==['session:1','session:2']
    assert mem.lyric_display('session:0') is None
    assert mem.lyric_display('session:1')['cues'][0]['text_preview']=='自造示例 1'
    assert mem.lyric_display('session:2')['cues'][0]['text_preview']=='自造示例 2'
    assert mem.recent_summary()[-1]['interrupted_early'] is True
    assert mem.recent_summary()[-1]['end_position_s']==15.1


def test_local_preview_no_ai_share_and_ui_gets_lyrics(plugin_fixture):
    plugin,_,_,pushes=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=False
        plugin._listener_session=3
        plugin._on_track_boundary('track_started',8)
        clip=FinishedClip(np.ones(96000,dtype=np.float32)*.03,
            'early_lyrics_preview',3,offset_s=0,sequence=8,part_index=1)
        assert plugin._on_early_lyrics_clip(clip)
        payload,ctx,_,_=plugin._lyrics_queue.get_nowait()
        plugin._lyrics_queue.task_done()
        assert ctx['music_capture']['music_sequence']==8
        plugin._live_session_config={'lyrics_model':'base','lyrics_language':'zh'}
        plugin._transcribe_singing_live_audio_isolated=lambda a,cfg:{
            'status':'asr_hypothesis','language_candidate':'zh',
            'timed_cues':[{'at_s':.4,'end_s':2.,'text_preview':'这是测试语句'}]}
        queue=plugin._lyrics_queue
        queue.put((payload,ctx,3,plugin._lyrics_quality_generation))
        clip,ctx,session,generation=queue.get_nowait()
        plugin._process_one_lyrics_task(clip,ctx,3,3,queue,quality_generation=generation)
        feed=(await plugin.get_live_lyrics()).value
        assert feed['ok'] and feed['ai_share_allowed'] is False
        assert feed['songs'][0]['cues'][0]['text_preview']=='这是测试语句'
        assert feed['recognized_language']=='zh'
        assert not pushes
        await plugin.shutdown()
    run(scenario())


def test_language_selection_is_saved_and_separated_from_model_choice(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def scenario():
        await plugin.startup()
        assert plugin._lyrics_language=='auto'
        quality=plugin._lyrics_profile
        result=(await plugin.set_lyrics_language('zh')).value
        assert result['ok'] and plugin._lyrics_language=='zh'
        assert plugin._lyrics_profile==quality
        assert plugin._lyrics_session is None
        assert not (await plugin.set_lyrics_language('bad')).value['ok']
        assert plugin._lyrics_cache_quality({'lyrics_model':'base','lyrics_language':'zh'})=='singing-unavailable|zh'
        assert 'zh' in plugin._preferences_path().read_text(encoding='utf-8')
        await plugin.shutdown()
    run(scenario())


def test_ui_has_real_lyrics_panel_and_safe_plain_text():
    html=Path(__file__).resolve().parents[1].joinpath('static/index.html').read_text(encoding='utf8')
    for marker in ('id="live-lyrics-card"','id="lyrics-song-tabs"',
                   'id="live-lyrics-lines"','id="lyrics-language"',
                   "call('get_live_lyrics'", "call('set_lyrics_language'",
                   "lyrics.textContent=String(cue.text_preview||'')"):
        assert marker in html
    assert 'lyricLanguageLabel' in html
    assert 'setInterval(()=>{if(!document.hidden)safely(refreshLiveLyrics)},2000)' in html


def test_reuse_resets_old_interruption_marker(tmp_path):
    mem=RecentSongMemory(tmp_path/'memory.json')
    identity='media:0123456789abcdef'
    mem.begin('play-1',title='原创示例',identity=identity)
    mem.end('play-1',reason='metadata_switch',position_s=10,duration_s=150)
    assert mem.recent_summary()[-1]['interrupted_early']
    again=mem.begin('play-2',title='原创示例',identity=identity)
    assert again['reused']
    assert not mem.recent_summary()[-1]['interrupted_early']
    assert mem.recent_summary()[-1]['end_reason'] is None
