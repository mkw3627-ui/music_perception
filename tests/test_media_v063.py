"""Windows metadata protocol, privacy, isolated track ownership. No network."""
from __future__ import annotations
import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from music_perception.media_session_bridge import read_system_media
from music_perception.lyric_timing import align_lyrics_to_music
from music_perception import MusicPerceptionPlugin
from test_plugin import plugin_fixture


def test_gbk_independent_metadata_protocol_uses_utf8_json(tmp_path, monkeypatch):
    import music_perception.media_session_bridge as bridge
    (tmp_path/'vendor').mkdir()
    (tmp_path/'media_session_worker.py').write_text('',encoding='utf8')
    calls=[]
    def fake_run(args,**kw):
        calls.append((args,kw))
        return SimpleNamespace(returncode=0,stdout=json.dumps(
            {'available':True,'playing':True,'title':'银河边界','artist':'咲子',
             'source':'windows_smtc','position_s':16.2},ensure_ascii=False).encode('utf-8'))
    monkeypatch.setattr(bridge.subprocess,'run',fake_run)
    data=read_system_media(python_exe=tmp_path/'python.exe',vendor=tmp_path/'vendor',
                           worker_script=tmp_path/'media_session_worker.py')
    assert data['title']=='银河边界' and data['artist']=='咲子'
    assert '-I' in calls[0][0] and '-S' in calls[0][0]
    assert calls[0][1]['timeout'] == 4.0


def test_system_media_not_available_is_not_a_song(tmp_path,monkeypatch):
    import music_perception.media_session_bridge as bridge
    (tmp_path/'vendor').mkdir()
    (tmp_path/'media_session_worker.py').write_text('',encoding='utf8')
    def timeout(*args,**kwargs):
        raise subprocess.TimeoutExpired(['fake'],4)
    monkeypatch.setattr(bridge.subprocess,'run',timeout)
    data=read_system_media(python_exe=tmp_path/'python.exe',vendor=tmp_path/'vendor',
                           worker_script=tmp_path/'media_session_worker.py')
    assert not data['available']


def test_song_switch_clears_lyrics_and_silently_invalidates_last_ai_scene(plugin_fixture):
    plugin,_,_,pushed=plugin_fixture
    async def check():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._listener_session=3
        plugin._song_boundary_enforced=True
        plugin._active_track_sequence=2
        plugin._track_lyrics_history=[{'text_preview':'上一首内容','at_s':1.5}]
        plugin._recent_auditory_context={'musical_event_timeline':{'lyrics':{'timed_cues':[1]}}}
        plugin._on_track_boundary('track_ended',2)
        assert plugin._track_lyrics_history==[]
        assert plugin._recent_auditory_context is None
        assert len(pushed)==1
        assert pushed[-1]['ai_behavior']=='read'
        assert pushed[-1]['visibility']==[]
        assert pushed[-1]['coalesce_key']=='music_perception_latest_auditory_scene'
        assert '上一首内容' not in pushed[-1]['parts'][0]['text']
        plugin._on_track_boundary('track_started',3)
        assert plugin._active_track_sequence==3
        await plugin.shutdown()
    asyncio.run(check())


def test_lyric_timeline_never_quantizes_to_beat_but_adds_clock_hint():
    result=align_lyrics_to_music(
        {'status':'asr_hypothesis','timed_cues':[{'at_s':0.16,'end_s':1.3,'text_preview':'示例'}]},
        offset_s=8.0,beats=[8.0,8.5,9.0],media_time_anchor_s=32.0)
    cue=result['timed_cues'][0]
    assert cue['at_s']==8.16
    assert cue['nearest_beat_at_s']==8.0
    assert cue['player_at_s_estimated']==32.16
    assert result['time_basis']=='confirmed_music_elapsed_not_media_time'
    assert result['player_time_basis']=='estimated_player_position_not_verified'


def test_windows_media_session_seamless_switch_without_silence(plugin_fixture,monkeypatch):
    """SMTC changes detect consecutive tracks even while YAMNet stays music."""
    import music_perception as module
    monkeypatch.setattr(module,'sys',SimpleNamespace(platform='win32'))
    info={'available':True,'playing':True,'source':'windows_smtc',
          'player':'SamplePlayer.exe','title':'第一首','artist':'测试','position_s':55.0}
    monkeypatch.setattr(module,'read_system_media',lambda **kw:dict(info))
    class FakeRuntime:
        def snapshot(self):return {'ready':True}
        def worker_runtime(self):return '/test/python.exe','/test/vendor'
        def stop(self):pass
    class Listener:
        def __init__(self):self.changes=0
        def mark_track_changed(self):self.changes+=1
        def stop(self):pass
    plugin,_,_,_=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._environment=FakeRuntime()
        listener=Listener()
        plugin._listener=listener
        await plugin._refresh_player_track()
        assert plugin._track_display_name=='第一首'
        assert listener.changes==0
        plugin._active_track_sequence=1
        plugin._song_boundary_enforced=True
        plugin._recent_auditory_context={'music_capture':{'music_sequence':1},'auditory_observation':{'not':'retained'}}
        plugin._track_lyrics_history=[{'text_preview':'older lyrics'}]
        info['title']='第二首'
        info['position_s']=0.25
        plugin._track_metadata_polled_at=-9999
        await plugin._refresh_player_track()
        assert listener.changes==1
        assert plugin._recent_auditory_context is None
        assert not plugin._track_lyrics_history
        assert plugin._track_display_name=='第二首'
        await plugin.shutdown()
    asyncio.run(run())


def test_legacy_last_report_does_not_expose_previous_song(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def run():
        await plugin.startup()
        report={'schema_version':1,'source':'speaker_loopback_after_yamnet_music_gate',
                'capture':{'music_sequence':5,'listener_session':10}}
        plugin._write_last(report)
        plugin._listener_session=10
        plugin._active_track_sequence=5
        assert plugin._read_last()['capture']['music_sequence']==5
        plugin._active_track_sequence=None
        plugin._song_boundary_enforced=True
        result=await plugin.get_last_report()
        assert not result.value['ok']
        plugin._active_track_sequence=5
        plugin._listener_session=11
        result=await plugin.get_last_report()
        assert not result.value['ok']
        await plugin.shutdown()
    asyncio.run(run())
