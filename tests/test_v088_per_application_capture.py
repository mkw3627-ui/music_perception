"""Source-switch security, selection consent and actual process PCM dispatch.

Offline doubles only; real Windows WASAPI and proc-tap wheel still require
Windows application integration testing.
"""
from __future__ import annotations
import asyncio
import io
import os
from pathlib import Path
import numpy as np
import pytest

from music_perception import MusicPerceptionPlugin
from music_perception import app_channels
from music_perception.auto_listener import Decision
from music_perception.application_loopback_worker import _stream
from music_perception.isolated_loopback import IsolatedApplicationLoopback, IsolatedWindowsLoopback
from music_perception.process_capture_installer import ProcessCaptureInstaller
from test_plugin import plugin_fixture
from _version import ui_build, panel_ui_build, plugin_version  # noqa: E402


def d(music=False, score=0.0, speech=0.0):
    return Decision(music, score, speech)

@pytest.mark.parametrize('mode,voice,expected',[('monitor',False,'unknown'),
    ('ai',True,'ai_output'),('music',True,'music'),('music',False,'unknown'),
    ('video',False,'video'),('auto',True,'music')])
def test_single_pid_policy_only_selected_app(mode,voice,expected):
    app={'pid':22001,'created':'1234567890123456789','mode':mode}
    result=app_channels.selected_scene(d(voice,.9 if voice else .02,.8),app)
    assert result['kind']==expected
    assert result['source_count']==1
    assert result['source_pid']==22001
    assert result['source_key'].startswith('app:22001:')
    assert result['capture_provenance']=='wasapi_process_tree'


def test_video_music_speech_overlap_stays_blocked():
    app={'pid':22001,'created':'1234567890123456789','mode':'video'}
    result=app_channels.selected_scene(d(True,.9,.7),app)
    assert result['kind']=='mixed' and not result['verified']


def test_ai_voices_cannot_enter_recognition():
    app={'pid':22001,'created':'1234567890123456789','mode':'ai'}
    assert app_channels.selected_scene(d(True,.9,.9),app)['kind']=='ai_output'
    app['mode']='music';app['might_contain_ai_tts']=True
    assert app_channels.selected_scene(d(True,.9,.9),app,ai_voice_active=True)['kind']=='ai_output'


def test_unverified_pid_fails_closed(monkeypatch):
    monkeypatch.setattr(app_channels, 'process_start_token',lambda pid:'1000000000123456789' if pid==500 else '')
    assert app_channels.verify_selected_process(500,'1000000000123456789')
    assert not app_channels.verify_selected_process(500,'1000000000123456790')
    assert not app_channels.verify_selected_process(100,'')
    assert not app_channels.verify_selected_process(600,'1000000000123456789')


def test_select_requires_consent_installer_and_verified_identity(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def run():
        await plugin.startup()
        monkeypatch.setattr(plugin._process_capture_install,'installed',lambda:True)
        monkeypatch.setattr('music_perception.verify_selected_process',lambda p,c:int(p)==500 and c=='1234567890123456789')
        plugin._app_catalog_cache=[{'pid':500,'created':'1234567890123456789','name':'player.exe'}]
        result=(await plugin.select_app_capture(500,'1234567890123456789','music',confirmed=False)).value
        assert result['ok'] is False
        result=(await plugin.select_app_capture(502,'1234567890123456789','music',confirmed=True)).value
        assert result['ok'] is False
        result=(await plugin.select_app_capture(500,'1234567890123456789','music',confirmed=True)).value
        assert result['ok'] is True
        assert plugin._capture_mode=='application'
        assert plugin._selected_application['name']=='player.exe'
        scene=plugin._decide_scene(d(True,.9,.05))
        assert scene['kind']=='unknown'  # Wait for an independently reviewed source label.
        assert scene['source_count']==1
        assert 'PCM' not in str((await plugin.get_audio_source_control()).value)
        await plugin.shutdown()
    asyncio.run(run())


def test_capture_never_silent_fallback_to_default_speaker(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._capture_mode='application'
        plugin._selected_application={'pid':500,'created':'1234567890123456789','mode':'music','name':'player.exe'}
        out=(await plugin._start_live()).get('error','')
        assert '环境' in out or '模型' in out or '应用' in out or 'Python' in out
        await plugin.shutdown()
    asyncio.run(run())


def test_switch_rejected_if_capture_active(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def run():
        await plugin.startup()
        monkeypatch.setattr(plugin._process_capture_install,'installed',lambda:True)
        plugin._listener=type('FakeListener',(),{'running':True})()
        out=(await plugin.select_app_capture(500,'1234567890123456789','music',confirmed=True)).value
        assert not out['ok'] and plugin._capture_mode=='system'
        # Avoid shutdown calling fake listener.stop (no real source).
        plugin._listener=None
        await plugin.shutdown()
    asyncio.run(run())


def test_process_installer_never_installs_without_consent(tmp_path):
    install=ProcessCaptureInstaller(tmp_path)
    out=install.start(confirmed=False,env=None)
    assert not out['ok'] and not list(tmp_path.iterdir())
    assert not install.installed()


def test_application_backend_requires_specific_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(IsolatedWindowsLoopback,'_validate_paths',lambda self:None)
    monkeypatch.setattr('music_perception.app_channels.verify_selected_process',lambda p,t:True)
    addon=tmp_path/'addon';addon.mkdir()
    sample=IsolatedApplicationLoopback(
      python='/fake/python.exe',vendor=tmp_path,app_vendor=addon,pid=500,
      created='1234567890123456789',worker_script=tmp_path/'application_loopback_worker.py',
      diagnostics=tmp_path/'diag.log',capture_stereo=True)
    sample._validate_paths()
    argv=sample._launch_args()
    assert '500' in argv and '1234567890123456789' in argv and argv[-1]=='stereo'
    sample.script=tmp_path/'loopback_worker.py'
    with pytest.raises(Exception,match='必须|禁止'):
        sample._validate_paths()


def test_selected_process_samples_not_mixed_with_other_process(monkeypatch):
    import music_perception.application_loopback_worker as w
    monkeypatch.setattr(app_channels,'verify_selected_process',lambda p,t:p==500)
    import sys
    monkeypatch.setitem(sys.modules,'app_channels',app_channels)
    class FakeTap:
        is_running = True
        def read(self,timeout=0.3):
            # Synthetic user-selected process 500 frames only; no system PCM.
            return np.full((96_000,2),.25,np.float32).tobytes()
    class End(Exception):pass
    captured=[]
    def sender(tag,payload):
        captured.append((tag,payload))
        if tag==b'D':raise End()
    monkeypatch.setattr(w,'_send',sender)
    with pytest.raises(End):
        w._stream(FakeTap(),500,'1234567890123456789',stereo=True)
    assert [x[0] for x in captured]==[b'H',b'D']
    mono=np.frombuffer(captured[1][1],dtype='<f4')
    assert mono.shape==(32_000,)
    assert np.allclose(mono[300:-300],.25,atol=.001)


def test_ui_exposes_routing_and_plugin_manifest_version():
    root=Path(__file__).resolve().parents[1]
    html=(root/'ui/panel.tsx').read_text(encoding='utf-8')
    assert 'appCaptureInstaller' in html
    for action in ['install_app_capture','list_application_sources',
                   'focus_live_audio_source','select_system_capture']:
        assert action in html
    assert ui_build() in html
    assert f'version = "{plugin_version()}"' in (root/'plugin.toml').read_text(encoding='utf-8')


def test_app_selection_uses_live_switch_only_while_listening():
    root=Path(__file__).resolve().parents[1]
    html=(root/'ui/panel.tsx').read_text(encoding='utf-8')
    assert 'if (listening) perform("focus_live_audio_source",args' in html
    assert 'else perform("select_app_capture",args' in html


def test_inter_plugin_export_fails_closed_unless_user_allows(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def run():
        await plugin.startup()
        monkeypatch.setattr('music_perception.verify_selected_process',lambda p,c:p==505 and c=='1234567890123456789')
        app={'pid':505,'created':'1234567890123456789','mode':'video','name':'browser.exe'}
        plugin._capture_mode='application'
        plugin._selected_application=app
        plugin._auto_consented=True
        plugin._listener=type('FakeListener',(),{'running':True})()
        plugin._audio_scene=app_channels.selected_scene(d(False,.05,.8),app)
        plugin._video_transcript={'cues':[{'at_s':.5,'text_candidate':'视频测试'}]}
        blocked=(await plugin.get_selected_app_outputs()).value
        assert not blocked['ok']
        result=(await plugin.set_external_channel_share(enabled=True,confirmed=False)).value
        assert result['ok'] is False
        assert (await plugin.set_external_channel_share(enabled=True,confirmed=True)).value['ok'] is True
        exported=(await plugin.get_selected_app_outputs()).value
        assert exported['ok'] and exported['available']
        assert exported['text_candidates'][0]['text_candidate']=='视频测试'
        assert exported['background_music'] is None
        # A mixed or unknown source must never leak the previous video's text.
        plugin._on_scene_change({'kind':'mixed','verified':False,'source_count':2})
        exported=(await plugin.get_selected_app_outputs()).value
        assert exported['available'] is False
        assert (await plugin.set_external_channel_share(enabled=False,confirmed=True)).value['ok'] is True
        assert (await plugin.get_selected_app_outputs()).value['ok'] is False
        plugin._listener=None
        await plugin.shutdown()
    asyncio.run(run())


def test_consent_switch_persisted_no_raw_audio_in_console(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._app_catalog_cache=[{'pid':10,'created':'1000000000000','name':'first.exe'},
                                   {'pid':11,'created':'1000000000001','name':'second.exe'}]
        plugin._capture_mode='application'
        plugin._selected_application={'pid':10,'created':'1000000000000','name':'first.exe','mode':'music'}
        plugin._external_channel_share=True
        plugin._save_preferences()
        payload=(await plugin.get_audio_source_control()).value
        assert len(payload['channels'])==2
        assert payload['channels'][0]['is_selected'] is True
        assert payload['channels'][1]['pcm_capture'] is False
        assert 'PCM' not in str(payload) and 'raw_audio' not in str(payload)
        import json
        prefs=json.loads(plugin._preferences_path().read_text(encoding='utf-8'))
        assert prefs['external_channel_share'] is True
        await plugin.shutdown()
    asyncio.run(run())


def test_process_tap_context_manager_explicit_native_ready_guard(monkeypatch):
    import music_perception.application_loopback_worker as w
    assert 'with tap:' in __import__('inspect').getsource(w.main)
    class CrashedTap:
        is_running=False
        def read(self,timeout=.3):
            raise AssertionError('should not attempt to read terminated native capture')
    monkeypatch.setattr(app_channels,'verify_selected_process',lambda p,t:True)
    import sys
    monkeypatch.setitem(sys.modules,'app_channels',app_channels)
    with pytest.raises(RuntimeError,match='selected_process_native_capture_stopped'):
        w._stream(CrashedTap(),500,'1234567890123456789',stereo=False)


def test_cross_plugin_music_output_is_source_scoped_and_audio_free(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def run():
        await plugin.startup()
        app={'pid':500,'created':'1234567890123456789','mode':'music','name':'MusicApp.exe'}
        monkeypatch.setattr('music_perception.verify_selected_process',lambda p,c:p==500)
        plugin._capture_mode='application'
        plugin._selected_application=app
        plugin._auto_consented=True
        plugin._external_channel_share=True
        plugin._listener=type('FakeListener',(),{'running':True})()
        plugin._audio_scene=app_channels.selected_scene(d(True,.87,.01),app)
        plugin._song_current_id='current_song'
        plugin._recent_auditory_context={
            'music_capture':{'song_id':'current_song'},
            'musical_event_timeline':{'bpm_estimate':124,
                'lyrics':{'timed_cues':[{'at_s':1.,'text_preview':'测试唱词'}]},
                'chord_progression':{'changes':[{'at_s':1,'name':'C'}]}},
            'source_separation':{'state':'separated','device':'cpu'},
            'auditory_observation':{'raw_spectrum':'should_never_pass'},
        }
        out=(await plugin.get_selected_app_outputs()).value
        assert out['ok'] and out['available'] and out['source_kind']=='music'
        assert out['text_candidates'][0]['text_preview']=='测试唱词'
        assert out['background_music']['timeline']['bpm_estimate']==124
        assert 'should_never_pass' not in str(out)
        plugin._song_current_id='next_song'
        assert (await plugin.get_selected_app_outputs()).value['available'] is False
        plugin._listener=None
        await plugin.shutdown()
    asyncio.run(run())


def test_compiled_wheel_hashes_for_supported_windows_python_are_pinned():
    from music_perception.process_capture_installer import WIN_NATIVE_WHEEL_SHA256
    assert set(WIN_NATIVE_WHEEL_SHA256)=={'cp311','cp312','cp313'}
    assert all(len(value)==64 and set(value)<=set('0123456789abcdef') for value in WIN_NATIVE_WHEEL_SHA256.values())


def test_installer_marker_without_windows_native_extension_not_ready(tmp_path):
    import json
    root=tmp_path/'app_capture';root.mkdir()
    directory=root/'proc-tap-1.1.1-fake';directory.mkdir()
    (directory/'.neko-process-capture-ready').write_text('1.1.1')
    (root/'process_capture_active.json').write_text(json.dumps({'dir':str(directory),'version':'1.1.1'}))
    assert not ProcessCaptureInstaller(root).installed()
