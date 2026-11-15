"""v0.8.7 audio provenance, silent AI, source routing and video ASR tests.

No soundcard, network, model weights, host TTS system or CUDA used.
"""
from __future__ import annotations
import asyncio
import json
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from music_perception.audio_scene import classify_scene
from music_perception.auto_listener import Decision, FinishedClip, LiveListener, SAMPLE_RATE
from music_perception import MusicPerceptionPlugin
from test_plugin import plugin_fixture
from _version import ui_build, panel_ui_build, plugin_version  # noqa: E402


def decision(music=True, speech=0.1):
    return Decision(music, .89 if music else .02, speech)


def snap(*sources):
    return {'sessions':[dict(app=a, playing=True, playback_type=t) for a,t in sources]}


def test_music_is_not_a_video_without_source_evidence():
    item=classify_scene(decision(True),snapshot=snap())
    assert item.kind=='music' and not item.verified
    speech=classify_scene(decision(False,.9),snapshot=snap())
    assert speech.kind=='unknown'  # speech != verified video


def test_automatic_video_from_os_playback_type():
    scene=classify_scene(decision(False,.85),snapshot=snap(('browser','video')))
    assert scene.kind=='video' and scene.verified
    assert scene.reason=='verified_video_media_session'


def test_video_can_be_user_marked_when_browser_cannot_report_type():
    sessions=snap(('chrome','unknown'))
    assert classify_scene(decision(False,.8),snapshot=sessions).kind=='unknown'
    out=classify_scene(decision(False,.8),snapshot=sessions,source_roles={'chrome':'video'})
    assert out.kind=='video' and out.verified


@pytest.mark.parametrize('items', [
    [('music.exe','music'),('video.exe','video')],
    [('video_a','video'),('video_b','video')],
    [('music.exe','music'),('unknown','unknown')],
])
def test_multiple_simultaneous_sources_are_blocked_even_when_music_loud(items):
    scene=classify_scene(decision(True,.1),snapshot=snap(*items))
    assert scene.kind=='mixed' and not scene.verified


def test_own_tts_mixed_is_blocked_until_actually_routed_elsewhere():
    mus=snap(('music.exe','music'))
    assert classify_scene(decision(True),snapshot=mus,ai_speaking=True).kind=='ai_output'
    assert classify_scene(decision(True),snapshot=mus,ai_speaking=True,
                          ai_routed_elsewhere=True).kind=='music'
    with_tts=snap(('music.exe','music'),('assistant.exe','unknown'))
    assert classify_scene(decision(True),snapshot=with_tts,
                          source_roles={'assistant.exe':'ai'}).kind=='ai_output'
    assert classify_scene(decision(True),snapshot=with_tts,
                          source_roles={'assistant.exe':'ai'},
                          ai_routed_elsewhere=True).kind=='music'


def test_paused_media_session_does_not_count_as_second_active():
    sessions=snap(('player','music'))
    sessions['sessions'].append(dict(app='video',playing=False,playback_type='video'))
    assert classify_scene(decision(True),snapshot=sessions).kind=='music'


def test_listener_never_emits_mixed_samples_to_music_or_video(tmp_path):
    events=[]
    music=[]
    video=[]
    changes=[]
    windows=6
    class Source:
        def __enter__(self):self.i=0;return self
        def __exit__(self,*a):pass
        def read(self):
            if self.i>=windows: time.sleep(.02)
            self.i+=1
            return np.full(2*SAMPLE_RATE,.09,np.float32)
    class Gate:
        def __init__(self,*a,**kw):pass
        def classify(self,w):return decision(True)
    def scene(d):
        counter=source.i
        if counter<=4:return dict(kind='video',verified=True,reason='test_single_video',source_count=1)
        return dict(kind='mixed',verified=False,reason='two_sources',source_count=2)
    # The scene closure shares the same source instance with the listener.
    source=Source()
    listener=LiveListener(model_file=tmp_path/'fake', max_seconds=900,
        threshold=.24, on_clip=music.append,
        source_factory=lambda:source, gate_factory=Gate,
        scene_decider=scene,on_video_preview=video.append,
        on_scene_change=changes.append)
    listener.start()
    for _ in range(150):
        if listener.status().get('scene_kind')=='mixed':break
        time.sleep(.01)
    assert listener.status()['scene_kind']=='mixed'
    assert listener.status()['scene_blocked_windows']>=1
    assert len(music)==0
    assert len(video)==1  # first 6 seconds of verified video
    assert changes[0]['kind']=='video' and changes[-1]['kind']=='mixed'
    assert listener.stop()


def test_plugin_verifies_scene_and_blocks_ai_playback(plugin_fixture):
    plugin,*_=plugin_fixture
    async def check():
        await plugin.startup()
        plugin._media_source_snapshot=snap(('video.exe','video'))
        plugin._media_source_updated=time.monotonic()
        assert plugin._decide_scene(decision(False,.8))['kind']=='unknown'  # No reviewed unique app yet.
        await plugin.set_ai_voice_playback(active=True)
        assert plugin._decide_scene(decision(True))['kind']=='ai_output'
        await plugin.set_ai_voice_playback(active=False)
        plugin._media_source_snapshot=snap(('music.exe','music'),('video.exe','video'))
        assert plugin._decide_scene(decision(True))['kind']=='mixed'
        await plugin.shutdown()
    asyncio.run(check())


def test_source_role_and_output_setting_saved_with_explicit_consent(plugin_fixture):
    plugin,*_=plugin_fixture
    async def check():
        await plugin.startup()
        assert (await plugin.set_media_source_role(app='Chrome',role='video')).value['ok']
        assert plugin._source_role_overrides['chrome']=='video'
        assert not (await plugin.set_capture_output(output_id='speaker-id', confirmed=False)).value['ok']
        done=(await plugin.set_capture_output(output_id='speaker-id', ai_separate=True,confirmed=True)).value
        assert done['ok']
        data=json.loads(plugin._preferences_path().read_text())
        assert data['media_source_roles']['chrome']=='video'
        assert data['capture_output_id']=='speaker-id' and data['ai_output_separate_device'] is True
        await plugin.shutdown()
    asyncio.run(check())


def test_video_speech_only_shared_privately_and_only_when_consented(plugin_fixture):
    plugin,_,_,pushed=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._lyrics_profile='medium'
        plugin._video_session_id='test_video_session'
        plugin._audio_scene={'kind':'video','verified':True,'tag_status':'external_review'}
        plugin._transcribe_live_audio_isolated=lambda *_: {
            'language_candidate':'zh',
            'timed_cues':[{'at_s':0.5,'text_preview':'这是测试句子'}]}
        plugin.push_message=lambda **kwargs: pushed.append(kwargs) or {'submitted':True}
        clip=FinishedClip(np.full(6*SAMPLE_RATE,.15,np.float32),'video_speech_preview',3)
        import queue
        q=queue.Queue();q.put(1)
        plugin._process_video_speech(clip, plugin._listener_session,'test_video_session',q)
        assert q.unfinished_tasks == 0
        assert plugin._video_transcript['cues'][0]['text_candidate']=='这是测试句子'
        assert pushed and all(msg['visibility']==[] and msg['ai_behavior']=='read' for msg in pushed)
        assert plugin._video_transcript['audio_capture_only'] is True
        assert 'samples' not in plugin._video_transcript  # no raw PCM
        plugin._ai_delivery=False
        q.put(1)
        plugin._process_video_speech(clip, plugin._listener_session,'test_video_session',q)
        assert len(pushed)==1
        # When a second source arrives, outstanding results are invalidated.
        plugin._on_scene_change({'kind':'mixed','verified':False,'source_count':2})
        assert plugin._video_transcript is None
        assert (await plugin.get_ai_auditory_context()).value['available'] is False
        await plugin.shutdown()
    asyncio.run(run())


def test_ui_and_manifest_declares_new_source_actions():
    root=Path(__file__).resolve().parents[1]
    text=(root/'ui/panel.tsx').read_text(encoding='utf8')
    source=(root/'__init__.py').read_text(encoding='utf8')
    for action in ('set_ai_voice_playback','set_media_source_role','list_audio_outputs','set_capture_output'):
        assert f'"{action}"' in text
        assert f'@ui.action(id=\'{action}\'' in source
    assert ui_build() in text and ui_build() in source


def test_all_smtc_sessions_and_video_playback_type_can_be_read(monkeypatch):
    from music_perception.media_session_worker import get_media_metadata
    class Props:
        def __init__(self, kind, title):
            self.playback_type = types.SimpleNamespace(name=kind)
            self.title=title
            self.artist=''
            self.album_title=''
    class Session:
        def __init__(self, app, kind):
            self.source_app_user_model_id=app
            self.kind=kind
        def get_playback_info(self):
            return types.SimpleNamespace(playback_status=types.SimpleNamespace(name='Playing'))
        async def try_get_media_properties_async(self):
            return Props(self.kind,'fake title')
        def get_timeline_properties(self):return None
    music=Session('music.exe','Music')
    video=Session('video.exe','Video')
    class Manager:
        def get_current_session(self):return video
        def get_sessions(self):return [music,video]
    class FakeManager:
        @staticmethod
        async def request_async():return Manager()
    # Real pywinrt module name; no network nor Windows runtime used.
    module=types.ModuleType('winrt.windows.media.control')
    module.GlobalSystemMediaTransportControlsSessionManager=FakeManager
    monkeypatch.setitem(sys.modules,'winrt',types.ModuleType('winrt'))
    monkeypatch.setitem(sys.modules,'winrt.windows',types.ModuleType('winrt.windows'))
    monkeypatch.setitem(sys.modules,'winrt.windows.media',types.ModuleType('winrt.windows.media'))
    monkeypatch.setitem(sys.modules,'winrt.windows.media.control',module)
    result=asyncio.run(get_media_metadata())
    assert len(result['sessions'])==2
    assert {s['playback_type'] for s in result['sessions']}=={'music','video'}
    assert classify_scene(decision(True),snapshot=result).kind=='mixed'


def test_separate_output_cannot_silently_fall_back_to_wrong_speaker(monkeypatch):
    from music_perception.auto_listener import SoundCardLoopback,ListenerError
    class Speaker:
        id='physical-default'
    module=types.ModuleType('soundcard')
    module.default_speaker=lambda:Speaker()
    module.all_speakers=lambda:[Speaker()]
    monkeypatch.setitem(sys.modules,'soundcard',module)
    with pytest.raises(ListenerError,match='设备已不存在'):
        with SoundCardLoopback(output_device_id='unavailable-target'):pass


def test_old_video_asr_result_does_not_reappear_after_mixed_scene(plugin_fixture):
    plugin, *_ = plugin_fixture
    async def check():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._lyrics_profile='medium'
        plugin._video_session_id='v1'
        plugin._audio_scene={'kind':'video','verified':True}
        plugin._on_scene_change({'kind':'mixed','verified':False,'source_count':2})
        plugin._transcribe_live_audio_isolated=lambda *a:{'timed_cues':[{'at_s':0,'text_preview':'stale'}]}
        q=__import__('queue').Queue();q.put(1)
        clip=FinishedClip(np.zeros(6*SAMPLE_RATE,np.float32),'video_speech_preview',3)
        plugin._process_video_speech(clip,plugin._listener_session,'v1',q)
        assert plugin._video_transcript is None
        await plugin.shutdown()
    asyncio.run(check())


def test_two_browser_tabs_share_app_but_still_count_as_mixed():
    scene=classify_scene(decision(True),snapshot=snap(('chrome.exe','music'),('chrome.exe','video')))
    assert scene.kind=='mixed' and scene.source_count==2


def test_smtc_uses_playback_info_for_type_and_numeric_playing():
    # Windows SMTC PlaybackType belongs to playback info, not media props.
    from music_perception.media_session_worker import _session_info
    class Media:
        title='Video clip'
        artist=''
        album_title=''
        playback_type='music'  # Deliberately misleading / unused.
    class Info:
        playback_status=4
        playback_type=2
    class Session:
        source_app_user_model_id='browser'
        async def try_get_media_properties_async(self): return Media()
        def get_playback_info(self): return Info()
        def get_timeline_properties(self): return None
    value=asyncio.run(_session_info(Session()))
    assert value['playing'] is True and value['playback_type']=='video'


def test_conservative_mode_abstains_without_single_source_confirmation():
    dec=decision(True)
    assert classify_scene(dec, snapshot=snap(), require_verified_source=True).kind == 'unknown'
    assert classify_scene(dec, snapshot=snap(('unknown-app','unknown')),
                          require_verified_source=True).kind == 'unknown'
    assert classify_scene(dec, snapshot=snap(('player','music')),
                          require_verified_source=True).kind == 'music'


def test_two_windows_media_sessions_from_same_app_still_count_as_mixed():
    scene=classify_scene(decision(True), snapshot=snap(('browser','video'),('browser','music')),
                         require_verified_source=True)
    assert scene.kind=='mixed' and scene.source_count==2


def test_source_detection_exception_fails_closed(tmp_path):
    sounds=[]
    class Source:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def read(self):return np.full(2*SAMPLE_RATE,.12,np.float32)
    class Gate:
        def __init__(self,*args,**kwargs):pass
        def classify(self,wave):return decision(True)
    inst=LiveListener(model_file=tmp_path/'none',max_seconds=10,
        threshold=.24,on_clip=sounds.append,
        source_factory=Source, gate_factory=Gate,
        scene_decider=lambda _:(_ for _ in ()).throw(RuntimeError('media helper unavailable')))
    inst.start()
    for _ in range(100):
        if inst.status().get('scene_kind')=='unknown' and inst.status().get('frames_received',0)>0:
            break
        time.sleep(.005)
    assert inst.status()['scene_kind']=='unknown'
    assert not sounds
    assert inst.stop()


def test_tts_hook_immediately_invalidates_music_context(plugin_fixture):
    plugin,*_=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._audio_scene={'kind':'music','verified':True}
        plugin._recent_auditory_context={'musical_event_timeline':{'lyrics':{'timed_cues':['old']}},
                                         'music_capture':{'music_sequence':1}}
        result=(await plugin.set_ai_voice_playback(active=True)).value
        assert result['ok'] is True
        assert plugin._audio_scene['kind']=='ai_output'
        assert plugin._recent_auditory_context is None
        result=(await plugin.set_ai_voice_playback(active=False)).value
        assert result['ai_voice_active'] is False
        assert plugin._audio_scene['kind']=='unknown'
        await plugin.shutdown()
    asyncio.run(run())
