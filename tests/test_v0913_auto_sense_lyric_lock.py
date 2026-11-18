"""v0.9.14: regression tests for live monitoring and song/UI isolation.

These are deterministic local tests; they don't attest native Windows process
loopback or NVIDIA GPU correctness on the user's PC.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
from music_perception import MusicPerceptionPlugin
from music_perception.auto_listener import SAMPLE_RATE, Decision, LiveListener
from music_perception.song_memory import RecentSongMemory


def test_startup_autosenses_and_private_ai_context(plugin_fixture,monkeypatch):
    p,*_=plugin_fixture
    seen=[]
    monkeypatch.setattr(MusicPerceptionPlugin,'_start_audible_scanner',lambda self:seen.append('scanner'))
    async def task():
        result=(await p.startup()).value
        assert result['listening_enabled'] is True
        assert p._ai_delivery is True
        assert seen==['scanner']
        assert p._auto_health['state']=='running'
        await p.shutdown()
    asyncio.run(task())


def test_lightweight_auto_diagnostics_never_preloads_gpu(plugin_fixture,monkeypatch):
    p,*_=plugin_fixture
    async def task():
        await p.startup()
        async def bad_probe():
            raise AssertionError('GPU deep probe must not run on plugin startup')
        p.run_lyrics_diagnostics=bad_probe
        await p._automatic_audio_check_task(lightweight=True)
        assert p._auto_health['state']=='limited'
        assert p._auto_health.get('heavy_model_loaded') is False
        await p.shutdown()
    asyncio.run(task())


def test_only_external_reviewed_app_allows_heavy_music_asr(plugin_fixture):
    p,*_=plugin_fixture
    async def task():
        await p.startup()
        p._auto_consented=True
        p._selected_application={'pid':711,'created':'1234567890123456789','name':'Player.exe','mode':'music'}
        p._capture_mode='application'
        d=Decision(True,.9,.02)
        raw=p._decide_scene(d)
        assert raw['kind']=='unknown' and not raw['verified']
        p._audio_scene=raw
        assert not p._asr_source_reviewed()
        app={**p._selected_application,'audible':True,'host_ai':False,'tag':'music'}
        p._app_catalog_cache=[app]
        p._audible_scan_at=time.time()
        source_id='app:711:1234567890123456789'
        assert (await p.review_audio_source_label(source_id,'music','ai_window_review')).value['ok']
        reviewed=p._decide_scene(d)
        assert reviewed['kind']=='music'
        assert reviewed['tag_status']=='external_review'
        p._audio_scene=reviewed
        assert p._asr_source_reviewed()
        # Clear review: no ASR even though the source emits loud music.
        assert (await p.review_audio_source_label(source_id,'unknown','ai_window_review')).value['ok']
        assert p._decide_scene(d)['kind']=='unknown'
        await p.shutdown()
    asyncio.run(task())


def test_default_system_mix_without_ai_review_does_not_allocate_asr(plugin_fixture):
    p,*_=plugin_fixture
    async def task():
        await p.startup()
        p._auto_consented=True
        p._capture_mode='system'
        p._media_source_snapshot={'sessions':[{'app':'Player.exe','playing':True,'playback_type':'music'}]}
        p._media_source_updated=time.monotonic()
        p._audible_scan_at=time.time()
        p._app_catalog_cache=[{'pid':711,'created':'1234567890123456789',
                                 'name':'Player.exe','host_ai':False,'audible':True,'tag':'music'}]
        d=Decision(True,.9,.02)
        assert p._decide_scene(d)['kind']=='unknown'
        assert (await p.review_audio_source_label('app:711:1234567890123456789',
                                                    'music','window_plugin')).value['ok']
        assert p._decide_scene(d)['kind']=='music'
        await p.shutdown()
    asyncio.run(task())


def test_yamnet_model_is_lazy_until_first_non_silent_pcm(tmp_path):
    calls=[]
    class Tap:
        def __init__(self):
            self.reads=0
        def __enter__(self):
            return self
        def __exit__(self,*args):
            pass
        def read(self):
            self.reads+=1
            # Three silent reads must not allocate the YAMNet checkpoint.
            if self.reads<4:
                return np.zeros(2*SAMPLE_RATE,np.float32)
            listener._stop.set()
            return np.full(2*SAMPLE_RATE,.08,np.float32)
    class Gate:
        def __init__(self,*a,**k):
            calls.append('allocated')
        def classify(self,data):
            return Decision(False,.1,.4)
    listener=LiveListener(model_file=tmp_path/'mock',max_seconds=60,threshold=.2,
        on_clip=lambda _:None,source_factory=Tap,gate_factory=Gate,
        scene_decider=lambda d:{'kind':'unknown','verified':False,
                                'reason':'no_label','source_count':0})
    listener._run_once()
    assert calls==[]  # stop requested before first live frame was processed
    listener._stop.clear()
    class ShortTap(Tap):
        def read(self):
            self.reads+=1
            if self.reads<4:
                return np.zeros(2*SAMPLE_RATE,np.float32)
            if self.reads==4:
                return np.full(2*SAMPLE_RATE,.08,np.float32)
            listener._stop.set()
            return np.zeros(2*SAMPLE_RATE,np.float32)
    listener._source_factory=ShortTap
    listener._run_once()
    assert calls==['allocated']


def test_replayed_song_cannot_show_previous_lyrics_until_fresh_asr(tmp_path):
    mem=RecentSongMemory(tmp_path/'songs.json')
    mem.begin('play-1',title='Same song',identity='identity-A')
    mem.observe('play-1',lyrics=[{'at_s':1.,'text_preview':'OLDER WORDS'}],lyrics_quality='base')
    mem.end('play-1')
    replay=mem.begin('play-2',title='Same song',identity='identity-A')
    assert replay['reused'] is True
    assert mem.lyric_display('play-2')['cues']==[]
    # Old ASR is retained in fingerprint cache for matching, but not visualised.
    mem.observe('play-2',lyrics=[{'at_s':2.,'text_preview':'CURRENT WORDS'}],lyrics_quality='base')
    assert [line['text_preview'] for line in mem.lyric_display('play-2')['cues']]==['CURRENT WORDS']


def test_ai_focus_rejects_unverified_or_ai_program_before_stopping(plugin_fixture,monkeypatch):
    p,*_=plugin_fixture
    async def task():
        await p.startup()
        p._auto_consented=True
        p._ai_delivery=True
        monkeypatch.setattr(p._process_capture_install,'installed',lambda:True)
        monkeypatch.setattr('music_perception.verify_selected_process',lambda pid,created:True)
        p._audible_scan_at=time.time()
        p._app_catalog_cache=[{'pid':1,'created':'123','name':'N.E.K.O.exe',
             'host_ai':True,'audible':True}]
        assert not (await p.ai_focus_audio_application('app:1:123','auto')).value['ok']
        assert p._capture_mode=='system'
        p._app_catalog_cache=[{'pid':701,'created':'1234567890123456789',
             'name':'player.exe','host_ai':False,'audible':True,'mode':'auto'}]
        result=(await p.ai_focus_audio_application('app:701:1234567890123456789','auto')).value
        assert result['ok'] and result['selected']['pid']==701
        assert result['listening'] is False
        await p.shutdown()
    asyncio.run(task())


def test_frontend_current_song_never_falls_back_to_recent_song():
    jsx=(Path(__file__).resolve().parents[1]/'ui'/'panel.tsx').read_text('utf8')
    assert 'songs[songs.length-1]' not in jsx
    assert 'current_song_id || ""' in jsx
    assert 'setHistorySongId("")' in jsx
    assert 'focus_live_audio_source' in jsx
    assert 'auto_sense' not in jsx or '0.9.14' in jsx
    assert 'onChange={(value:boolean)=>perform("set_ai_delivery"' not in jsx


def test_video_preview_does_not_allocate_whisper_before_source_review(plugin_fixture):
    from music_perception.auto_listener import FinishedClip
    p,*_=plugin_fixture
    async def task():
        await p.startup()
        p._auto_consented=True
        p._video_session_id='video-session'
        p._audio_scene={'kind':'video','verified':True,'tag_status':'candidate'}
        activated=[]
        p._ensure_lyrics_consumer_alive=lambda:activated.append('heavy') or True
        clip=FinishedClip(np.ones(SAMPLE_RATE*6,np.float32)*.1,'video_preview',1)
        assert p._on_video_preview(clip) is False
        assert activated==[]
        assert p._lyrics_queue.empty()
        p._audio_scene={'kind':'video','verified':True,'tag_status':'external_review'}
        assert p._on_video_preview(clip) is True
        assert activated==['heavy']
        await p.shutdown()
    asyncio.run(task())


def test_inflight_lyrics_requires_review_when_capture_is_live(plugin_fixture):
    from music_perception.auto_listener import FinishedClip
    p,*_=plugin_fixture
    async def task():
        await p.startup()
        p._auto_consented=True
        p._source_arbiter_live=True
        p._listener_session=3
        p._active_track_sequence=10
        p._song_boundary_enforced=True
        clip=FinishedClip(np.ones(16000,np.float32), 'early_lyrics_preview',10,sequence=10)
        p._audio_scene={'kind':'music','verified':True,'tag_status':'candidate'}
        assert p._clip_is_current(clip,3) is False
        p._audio_scene={'kind':'music','verified':True,'tag_status':'external_review'}
        assert p._clip_is_current(clip,3) is True
        p._audio_scene={'kind':'unknown','verified':False}
        assert p._clip_is_current(clip,3) is False
        await p.shutdown()
    asyncio.run(task())


def test_live_app_focus_stops_old_source_and_clears_identity(plugin_fixture,monkeypatch):
    p,*_=plugin_fixture
    async def task():
        await p.startup()
        p._auto_consented=True
        monkeypatch.setattr(p._process_capture_install,'installed',lambda:True)
        monkeypatch.setattr('music_perception.verify_selected_process',lambda pid,created:True)
        p._audible_scan_at=time.time()
        p._app_catalog_cache=[{'pid':718,'created':'1234567890123456789',
             'name':'NewApp.exe','host_ai':False,'audible':True,'mode':'auto'}]
        p._capture_mode='application'
        p._selected_application={'pid':717,'created':'1234567890123456789',
                                 'name':'OldApp.exe','mode':'auto'}
        stopped=[]
        started=[]
        class OldListener:
            running=True
            def stop(self):
                stopped.append('stopped')
                self.running=False
                return True
        p._listener=OldListener()
        monkeypatch.setattr(p,'_stop_background_processor',lambda:stopped.append('processor'))
        async def fake_start_live():
            started.append('new')
            return {'ok':True}
        monkeypatch.setattr(p,'_start_live',fake_start_live)
        result=(await p.focus_live_audio_source(718,'1234567890123456789','auto',True)).value
        assert result['ok'] and result['selected']['pid']==718
        assert stopped==['stopped','processor'] and started==['new']
        assert p._audio_scene['kind']=='unknown'
        p._listener=None
        await p.shutdown()
    asyncio.run(task())


def test_even_lightweight_music_classifier_waits_for_external_review(tmp_path):
    reviews={'allowed':False}
    counts=[]
    class Tap:
        def __enter__(self):
            return self
        def __exit__(self,*_):
            pass
        def __init__(self):
            self.n=0
        def read(self):
            self.n+=1
            if self.n==3:
                reviews['allowed']=True
            if self.n==5:
                reviews['allowed']=False
            if self.n==6:
                reviews['allowed']=True
            if self.n==7:
                listener._stop.set()
            return np.ones(SAMPLE_RATE,np.float32)*.08
    class Gate:
        def __init__(self,*args,**kwargs):
            counts.append('loaded')
        def classify(self,pcm):
            return Decision(False,.0,.8,reason='speech')
    listener=LiveListener(model_file=tmp_path/'fake-yamnet', max_seconds=30,
        threshold=.2,on_clip=lambda _:None,source_factory=Tap,gate_factory=Gate,
        classifier_allowed=lambda:reviews['allowed'],
        scene_decider=lambda d:{'kind':'unknown','verified':False,'reason':d.reason,'source_count':0})
    listener._run_once()
    assert counts==['loaded','loaded'], 'review revocation frees classifier; re-approval reloads'
    assert listener.status()['status'] != 'error'


def test_host_audio_prevents_global_classifier_even_if_another_app_reviewed(plugin_fixture):
    p,*_=plugin_fixture
    async def task():
        await p.startup()
        p._auto_consented=True
        p._capture_mode='system'
        p._audible_scan_at=time.time()
        p._app_catalog_cache=[
            {'pid':333,'created':'1','name':'Player.exe','audible':True,
             'tag':'music','tag_status':'external_review','host_ai':False},
            {'pid':334,'created':'2','name':'N.E.K.O.exe','audible':True,'host_ai':True},
        ]
        assert not p._classifier_reviewed_source()
        p._app_catalog_cache=p._app_catalog_cache[:1]
        assert p._classifier_reviewed_source()
        await p.shutdown()
    asyncio.run(task())
