"""No-model active Windows session labels + fixed N.E.K.O AI audio exclusion."""
from __future__ import annotations
import asyncio
import json
import math
from pathlib import Path

from music_perception import app_channels
from music_perception import MusicPerceptionPlugin
from music_perception.windows_audio_sessions import active_session_rows,level_dbfs
from test_plugin import plugin_fixture


def test_session_peak_filters_silent_shared_and_invalid():
    rows=active_session_rows([
        {'pid':200,'peak':0.0}, {'pid':201,'peak':0.008},
        {'pid':201,'peak':0.02}, {'pid':202,'peak':float('nan')},
        {'pid':203,'peak':0.95,'shared_pid':True},
        {'pid':204,'peak':0.002}, {'pid':205,'peak':.09},
        {'pid':4,'peak':.9}, {'pid':206,'peak':.9,'system_audio':True}])
    assert [r['pid'] for r in rows]==[205,201]
    assert rows[1]['sample_count']==2
    assert level_dbfs(0)==-120.0
    assert level_dbfs(.01)<0


def test_fixed_neko_ai_tree_label_without_models():
    sample=[
       {'pid':100,'parent_pid':0,'name':'N.E.K.O.exe','created':'10000000000000'},
       {'pid':105,'parent_pid':100,'name':'python.exe','created':'10000000000100'},
       {'pid':110,'parent_pid':105,'name':'other.exe','created':'10000000000300'},
       {'pid':200,'parent_pid':0,'name':'Spotify.exe','created':'10000000000000'},
       {'pid':300,'parent_pid':0,'name':'chrome.exe','created':'10000000000000'}]
    ai, roots=app_channels.identify_host_tree(sample)
    assert ai == {100,105,110} and roots=={100}
    assert app_channels.tag_application({**sample[1],'host_ai':True})['tag']=='ai'
    assert app_channels.tag_application(sample[3])['tag']=='music'
    assert app_channels.tag_application(sample[4])['tag']=='unknown'
    assert app_channels.tag_application({'name':'N.E.K.O-evil.exe','host_ai':False})['tag']=='unknown'


def test_scanner_only_returns_peak_meter_pids(monkeypatch,plugin_fixture):
    plugin,*_=plugin_fixture
    seen={}
    def fake_scan():
        return [{'pid':500,'peak':.55,'level_dbfs':-5.2},
                {'pid':501,'peak':.01,'level_dbfs':-40.0}]
    def fake_proc(*,limit=250,pid_allowlist=None,host_exe_path=""):
        seen['allow']=pid_allowlist
        return [{'pid':500,'created':'1234567890123456789','name':'Spotify.exe','parent_pid':0,'host_ai':False},
                {'pid':501,'created':'1234567890123456790','name':'N.E.K.O.exe','parent_pid':0,'host_ai':True}]
    monkeypatch.setattr(MusicPerceptionPlugin,'_run_audible_sessions_worker',lambda self:fake_scan())
    monkeypatch.setattr('music_perception.list_app_processes',fake_proc)
    async def run():
        await plugin.startup()
        apps=plugin._refresh_audible_cache(publish=False)
        assert seen['allow']=={500,501}
        assert {a['tag'] for a in apps}=={'music','ai'}
        assert all(a['audible'] for a in apps)
        assert apps[0]['host_ai']  # pinned AI first in control console
        assert plugin._app_catalog_cache==apps
        await plugin.shutdown()
    asyncio.run(run())


def test_unauthorized_ai_pid_selection_blocked(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def run():
        await plugin.startup()
        monkeypatch.setattr(plugin._process_capture_install,'installed',lambda:True)
        monkeypatch.setattr('music_perception.verify_selected_process',lambda p,t:True)
        plugin._app_catalog_cache=[{'pid':500,'created':'1234567890123456789',
                                    'name':'N.E.K.O.exe','host_ai':True,'audible':True}]
        result=(await plugin.select_app_capture(500,'1234567890123456789',mode='music',confirmed=True)).value
        assert not result['ok']
        assert plugin._capture_mode=='system' and plugin._selected_application is None
        await plugin.shutdown()
    asyncio.run(run())


def test_private_tags_read_only_and_changed_only(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    emissions=[]
    def fake_scan():
        return [{'pid':500,'peak':.55,'level_dbfs':-5.2}]
    def fake_proc(**_):
        return [{'pid':500,'created':'1234567890123456789','name':'foobar2000.exe','host_ai':False}]
    monkeypatch.setattr(MusicPerceptionPlugin,'_run_audible_sessions_worker',lambda self:fake_scan())
    monkeypatch.setattr('music_perception.list_app_processes',fake_proc)
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        # startup() may already have delivered the same snapshot through the
        # real push; reset the dedup bus so this test observes its own emission.
        plugin._private_audio_bus=None
        monkeypatch.setattr(plugin,'push_message',lambda **kw: emissions.append(kw))
        plugin._refresh_audible_cache()
        plugin._refresh_audible_cache()
        assert len(emissions)==1
        first=emissions[0]
        assert first['visibility']==[] and first['ai_behavior']=='read'
        data=first['parts'][0]['text']
        assert 'private_active_application_sound_tags' in data
        assert 'requests_response' in data
        await plugin.shutdown()
    asyncio.run(run())


def test_temporary_dialog_scroll_and_audio_ui():
    ui=(Path(__file__).resolve().parents[1]/'ui/panel.tsx').read_text('utf-8')
    for item in ['overscrollBehaviorY','onWheel','maxHeight:"min(56dvh, 470px)"',
                 'audible_sources','host_ai','立即扫描有声应用']:
        assert item in ui


def test_custom_ai_main_path_pins_restarted_pid_tree():
    samples=[
       {'pid':310,'parent_pid':0,'name':'NekoRelease64.exe',
        'created':'10000000000000','path':r'E:\Games\N.E.K.O\NekoRelease64.exe'},
       {'pid':315,'parent_pid':310,'name':'tts_audio.exe',
        'created':'10000000000100','path':r'E:\Games\N.E.K.O\tts_audio.exe'},
       {'pid':500,'parent_pid':0,'name':'NekoRelease64.exe',
        'created':'10000000000000','path':r'C:\Fake\NekoRelease64.exe'}]
    ai,roots=app_channels.identify_host_tree(samples,host_exe_path=r'e:\games\n.e.k.o\nekorelease64.exe')
    assert roots=={310}
    assert ai=={310,315}
    assert 500 not in ai


def test_user_can_pin_ai_host_only_from_real_recent_audible_process(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def run():
        await plugin.startup()
        monkeypatch.setattr('music_perception.verify_selected_process',lambda p,t:p==250)
        monkeypatch.setattr('music_perception.process_image_path',lambda pid:r'E:\STEAM\NekoRelease64.exe')
        plugin._app_catalog_cache=[{'pid':250,'created':'1234567890123456789',
                                    'name':'NekoRelease64.exe','audible':True}]
        no=(await plugin.pin_ai_main_process(250,'1234567890123456789',confirmed=False)).value
        assert not no['ok']
        yes=(await plugin.pin_ai_main_process(250,'1234567890123456789',confirmed=True)).value
        assert yes['ok'] and yes['pinned']
        assert plugin._fixed_ai_host_path==r'E:\STEAM\NekoRelease64.exe'
        assert plugin._app_catalog_cache==[]
        assert (await plugin.get_audio_source_control()).value['ai_host_pinned'] is True
        await plugin.shutdown()
    asyncio.run(run())
