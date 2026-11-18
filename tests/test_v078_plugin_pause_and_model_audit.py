"""Manual PLUGIN pause, not music playback pause; no data or model deletion."""
import asyncio
import json
import subprocess
import sys


def test_pause_plugin_closes_asr_process_but_retains_consent_memory(plugin_fixture):
    plugin, _, _, _ = plugin_fixture
    class FakeLyrics:
        running = True
        def close(self):
            self.running = False
    class FakeListener:
        running = True
        def stop(self):
            self.running = False
            return True
    async def run():
        await plugin.startup()
        plugin._auto_consented = True
        plugin._ai_delivery = True
        fake = FakeLyrics()
        plugin._lyrics_session = fake
        listener = FakeListener()
        plugin._listener = listener
        plugin._song_current_id = 'some-song'
        before = plugin._song_current_id
        response = (await plugin.pause_plugin()).value
        assert response['ok'] and response['paused']
        assert response['release_status'] == 'subprocess_exit_confirmed'
        assert not fake.running and not listener.running
        assert plugin._lyrics_session is None
        assert plugin._lyrics_runtime_device is None
        assert plugin._auto_consented and plugin._ai_delivery
        assert plugin._song_current_id == before
        data=json.loads(plugin._preferences_path().read_text(encoding='utf-8'))
        assert data['plugin_paused'] is True and data['speaker_loopback_enabled'] is True
        await plugin.shutdown()
    asyncio.run(run())


def test_pause_does_not_trigger_on_music_pause_or_change_model_files(plugin_fixture, tmp_path):
    plugin, _, _, _ = plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented = True
        folder=plugin.data_path('models','whisper_multilingual','base')
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'model.bin').write_bytes(b'preserve-existing-model-cache')
        await plugin.pause_plugin()
        assert (folder/'model.bin').read_bytes()==b'preserve-existing-model-cache'
        # The user explicitly pauses the PLUGIN: no media-session pause hook.
        assert not hasattr(plugin,'_pause_lyrics_for_playback')
        assert plugin._plugin_paused
        assert (await plugin._start_live())['ok'] is False
        await plugin.shutdown()
    asyncio.run(run())


def test_resume_plugin_keeps_existing_settings_and_reopens_only_on_demand(plugin_fixture,monkeypatch):
    plugin, _, _, _ = plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented = True
        await plugin.pause_plugin()
        starts=[]
        async def start():
            starts.append('start')
            return {'ok':True}
        monkeypatch.setattr(plugin,'_start_live',start)
        monkeypatch.setattr(plugin,'_schedule_lyrics_model_prepare',lambda: starts.append('prepare'))
        plugin._lyrics_profile='medium'
        answer=(await plugin.resume_plugin()).value
        assert answer['ok'] and not answer['paused']
        assert starts==['start']  # v0.8.6 never silently downloads a missing model
        data=json.loads(plugin._preferences_path().read_text('utf-8'))
        assert data['plugin_paused'] is False and data['speaker_loopback_enabled'] is True
        await plugin.shutdown()
    asyncio.run(run())


def test_stop_revokes_consent_and_resets_pause_state(plugin_fixture):
    plugin, _, _, _ = plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented = True
        await plugin.pause_plugin()
        response=(await plugin.stop_auto_listen()).value
        assert response['ok']
        assert not plugin._auto_consented and not plugin._plugin_paused
        assert plugin._ai_delivery  # AI delivery preference remains default-on; no capture when stopped.
        await plugin.shutdown()
    asyncio.run(run())


def test_pause_kills_plugin_owned_child_processes(plugin_fixture):
    plugin, _, _, _ = plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        child = subprocess.Popen([sys.executable, '-c','import time; time.sleep(15)'])
        try:
            plugin._analysis_children.add(child)
            report=(await plugin.pause_plugin()).value
            assert report['ok'] and report['release_status']=='subprocess_exit_confirmed'
            assert child.poll() is not None
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=2)
        await plugin.shutdown()
    asyncio.run(run())


def test_paused_preferences_prevent_automatic_listener_restart(plugin_fixture):
    plugin, _, _, _ = plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        await plugin.pause_plugin()
        # The on-disk state must survive process restart.
        saved=plugin._preferences_path().read_text('utf8')
        assert json.loads(saved)['plugin_paused'] is True
        plugin._plugin_paused=False
        plugin._load_preferences()
        assert plugin._plugin_paused is True
        assert (await plugin._start_live())['ok'] is False
        await plugin.shutdown()
    asyncio.run(run())


def test_paused_plugin_rejects_late_music_clip_and_media_poll(plugin_fixture, monkeypatch):
    # A listener callback already in flight must not revive work after pause.
    plugin, _, _, _ = plugin_fixture
    from types import SimpleNamespace
    async def run():
        await plugin.startup()
        plugin._auto_consented = True
        plugin._plugin_paused = True
        plugin._active_track_sequence = 17
        clip = SimpleNamespace(sequence=17)
        assert not plugin._clip_is_current(clip)
        # The media session should not be polled while this plugin is paused.
        before = plugin._track_metadata_polled_at
        await plugin._refresh_player_track()
        assert plugin._track_metadata_polled_at == before
        await plugin.shutdown()
    asyncio.run(run())


def test_pause_does_not_reopen_gpu_via_diagnostics(plugin_fixture):
    plugin, _, _, _ = plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented = True
        await plugin.pause_plugin()
        answer = (await plugin.run_lyrics_diagnostics()).value
        assert answer['ok'] is False and '暂停' in answer['error']
        assert plugin._lyrics_session is None
        await plugin.shutdown()
    asyncio.run(run())
