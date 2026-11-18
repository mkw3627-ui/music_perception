"""Offline lyric-track integration tests. No pretrained weights downloaded."""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
from music_perception.auto_listener import FinishedClip
from music_perception.lyric_timing import align_lyrics_to_music
from music_perception.lyrics_backend import transcribe_singing_candidate


def pcm(seconds=10):
    t=np.arange(16000*seconds,dtype=np.float32)/16000
    return (.08*np.sin(2*np.pi*220*t)).astype(np.float32)


class FakeSegment:
    def __init__(self,text,start,end):
        self.text=text
        self.start=start
        self.end=end
        self.avg_logprob=-0.15
        self.no_speech_prob=0.05
        self.words=[SimpleNamespace(word=text[:2],start=start)]


class FakeModel:
    def transcribe(self,*args,**kwargs):
        assert kwargs['condition_on_previous_text'] is False
        assert kwargs['word_timestamps'] is False
        return ([FakeSegment('你好',.35,1.1),
                 FakeSegment('こんにちは',1.2,2.5),
                 FakeSegment('hello',3.0,4.0)], SimpleNamespace(language='zh'))


def test_multilingual_lyrics_fallback_to_cpu_and_timestamp_hints():
    seen=[]
    def factory(name,**kw):
        seen.append((name,kw['device'],kw['compute_type']))
        if kw['device']=='cuda':
            raise RuntimeError('CUDA unavailable')
        return FakeModel()
    result=transcribe_singing_candidate(pcm(),model_dir='/tmp/test-lyrics-model-dir',model_factory=factory)
    assert seen[:2]==[('base','cuda','float16'),('base','cpu','int8')]
    assert result['status']=='asr_hypothesis' and result['device']=='cpu'
    assert len(result['timed_cues'])==3
    assert [r['text_preview'] for r in result['timed_cues']]==['你好','こんにちは','hello']
    assert all(r['certainty']=='unverified_singing_transcription' for r in result['timed_cues'])


def test_timestamps_follow_music_events_without_quantizing_song_vocals():
    base={'timed_cues':[{'at_s':.35,'end_s':1.2,'text_preview':'候选',
                         'words':[{'at_s':.45,'word_preview':'hello'}]}]}
    result=align_lyrics_to_music(base,offset_s=20,beats=[20,20.5,21,21.5])
    cue=result['timed_cues'][0]
    assert cue['at_s']==20.35 and cue['end_s']==21.2
    assert cue['nearest_beat_at_s']==20.5 and cue['offset_from_beat_s']==-.15
    assert cue['words'][0]['at_s']==20.45
    assert result['time_basis']=='confirmed_music_elapsed_not_media_time'
    assert base['timed_cues'][0]['at_s']==.35


def test_stopping_previous_song_immediately_clears_context_and_stale_queue(plugin_fixture):
    plugin,_,_,_ = plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._listener_session=1
        plugin._recent_auditory_context={'musical_event_timeline':{'lyrics':{'timed_cues':[{'text_preview':'old'}]}}}
        plugin._auditory_history=[plugin._recent_auditory_context]
        plugin._on_track_boundary('track_started',1)
        assert plugin._clip_is_current(FinishedClip(pcm(),'live_progress',5,sequence=1),1)
        plugin._recent_auditory_context={'old_lyrics':True}
        plugin._on_track_boundary('track_ended',1)
        assert plugin._recent_auditory_context is None
        assert plugin._auditory_history==[]
        assert not plugin._clip_is_current(FinishedClip(pcm(),'live_progress',5,sequence=1),1)
        plugin._on_track_boundary('track_started',2)
        assert plugin._clip_is_current(FinishedClip(pcm(),'live_progress',5,sequence=2),1)
        await plugin.shutdown()
    asyncio.run(run())


def test_lyrics_task_uses_same_song_beat_timeline_and_passive_ai(plugin_fixture):
    plugin,_,_,pushed=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._listener_session=2
        plugin._song_boundary_enforced=True
        plugin._active_track_sequence=4
        plugin._live_session_config={'lyrics_model':'tiny'}
        context={
          'music_capture':{'music_sequence':4,'offset_s':20,'end_s':30},
          'musical_event_timeline':{'beat_times_s':[20.0,20.5,21.0], 'events':[]},
          'auxiliary_feature_estimates':{},
        }
        plugin._recent_auditory_context=context
        plugin._transcribe_singing_live_audio_isolated=lambda a,cfg:{
           'status':'asr_hypothesis','timed_cues':[{'at_s':.42,'end_s':1.1,'text_preview':'歌词候选'}]}
        q=plugin._lyrics_queue
        stop=threading.Event()
        thread=threading.Thread(target=plugin._background_lyrics_main,args=(2,q,stop),daemon=True)
        thread.start()
        clip=FinishedClip(pcm(),'live_progress',5,offset_s=20,sequence=4,part_index=2)
        q.put((clip,context,2))
        for _ in range(120):
            if plugin._lyrics_completed:
                break
            await asyncio.sleep(.02)
        stop.set()
        thread.join(1)
        assert plugin._lyrics_completed==1
        lyrics=context['musical_event_timeline']['lyrics']
        assert lyrics['timed_cues'][0]['at_s']==20.42
        assert lyrics['timed_cues'][0]['nearest_beat_at_s']==20.5
        assert len(pushed)==1 and pushed[0]['visibility']==[]
        assert pushed[0]['ai_behavior']=='read'
        assert '歌词候选' in pushed[0]['parts'][0]['text']
        await plugin.shutdown()
    asyncio.run(run())


def test_metadata_switch_invokes_collector_reset(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    class FakeMedia:
        def __init__(self):
            self.title='第一首'
            self.index=0
        async def call_entry(self,*args,**kw):
            return {'scheduler_running':True,'current_title':self.title,
                    'current_artist':'示例歌手','current_index':self.index,
                    'total_sec':120,'remaining_sec':100}
    media=FakeMedia()
    class Listener:
        def __init__(self):
            self.calls=0
        def mark_track_changed(self):
            self.calls+=1
        def stop(self):
            return True
    listener=Listener()
    async def run():
        await plugin.startup()
        plugin.ctx=SimpleNamespace(plugins=media)
        plugin._auto_consented=True
        plugin._listener=listener
        plugin._track_metadata_polled_at=-9999
        await plugin._refresh_player_track()
        assert listener.calls==0
        plugin._song_boundary_enforced=True
        plugin._active_track_sequence=1
        plugin._recent_auditory_context={'old':True}
        media.title='第二首'
        media.index=1
        plugin._track_metadata_polled_at=-9999
        await plugin._refresh_player_track()
        assert listener.calls==1
        assert plugin._recent_auditory_context is None
        await plugin.shutdown()
    asyncio.run(run())


def test_slow_lyrics_are_merged_into_latest_interval_of_same_song(plugin_fixture):
    plugin,_,_,pushed=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._listener_session=2
        plugin._song_boundary_enforced=True
        plugin._active_track_sequence=7
        plugin._live_session_config={'lyrics_model':'tiny'}
        earlier={'music_capture':{'music_sequence':7,'offset_s':20,'end_s':30},
                 'musical_event_timeline':{'beat_times_s':[20,20.5,21],'lyrics':{'status':'pending'}}}
        later={'music_capture':{'music_sequence':7,'offset_s':30,'end_s':40},
               'musical_event_timeline':{'beat_times_s':[30,30.5],'lyrics':{'status':'pending'}}}
        plugin._recent_auditory_context=earlier
        plugin._transcribe_singing_live_audio_isolated=lambda a,cfg:{
           'status':'asr_hypothesis','model':'whisper-tiny',
           'timed_cues':[{'at_s':.33,'end_s':1.7,'text_preview':'新唱词候选'}]}
        q=plugin._lyrics_queue
        stop=threading.Event()
        # A later music interval arrives WHILE this transcript is queued.
        q.put((FinishedClip(pcm(),'live_progress',5,offset_s=20,sequence=7),earlier,2))
        plugin._recent_auditory_context=later
        thread=threading.Thread(target=plugin._background_lyrics_main,args=(2,q,stop),daemon=True)
        thread.start()
        for _ in range(120):
            if plugin._lyrics_completed:
                break
            await asyncio.sleep(.02)
        stop.set()
        thread.join(1)
        assert plugin._lyrics_completed==1
        assert later['musical_event_timeline']['lyrics']['timed_cues'][0]['at_s']==20.33
        assert len(plugin._track_lyrics_history)==1
        assert pushed and pushed[-1]['visibility']==[]
        assert pushed[-1]['ai_behavior']=='read'
        plugin._on_track_boundary('track_ended',7)
        assert plugin._track_lyrics_history==[]
        assert '上一首歌曲' in pushed[-1]['parts'][0]['text']
        assert pushed[-1]['coalesce_key']=='music_perception_latest_auditory_scene'
        await plugin.shutdown()
    asyncio.run(run())


def test_same_song_lyrics_do_not_appear_after_switch(plugin_fixture):
    plugin,_,_,pushed=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._listener_session=2
        plugin._song_boundary_enforced=True
        plugin._active_track_sequence=8
        plugin._track_lyrics_history=[{'at_s':1.2,'text_preview':'旧歌的候选'}]
        plugin._on_track_boundary('track_ended',8)
        plugin._on_track_boundary('track_started',9)
        assert plugin._active_track_sequence==9
        assert plugin._track_lyrics_history==[]
        assert plugin._recent_auditory_context is None
        assert all('旧歌的候选' not in x['parts'][0]['text'] for x in pushed)
        await plugin.shutdown()
    asyncio.run(run())


def test_private_subprocess_lyrics_json_handles_cjk_utf8(tmp_path):
    import sys

    from music_perception.lyrics_bridge import recognize_pcm_isolated
    vendor=tmp_path/'vendor'
    vendor.mkdir()
    worker=tmp_path/'fake_lyrics_worker.py'
    worker.write_text('''import json,sys
sys.stdin.buffer.read()
ans={'version':1,'lyrics':{'status':'asr_hypothesis','timed_cues':[{'text_preview':'你好こんにちは♪'}]}}
sys.stdout.buffer.write(json.dumps(ans,ensure_ascii=False).encode('utf-8'))
''',encoding='utf-8')
    payload=recognize_pcm_isolated(pcm(1),python_exe=sys.executable,vendor=vendor,
                                    worker_script=worker,model_dir=tmp_path/'models',timeout=5)
    assert payload['timed_cues'][0]['text_preview']=='你好こんにちは♪'
