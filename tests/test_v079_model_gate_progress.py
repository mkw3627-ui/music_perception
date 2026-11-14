"""v0.7.9: no false model readiness, staged progress and verified activation.

Simulations deliberately use tiny fake file sizes and NEVER fetch model weights.
"""
from __future__ import annotations
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import pytest
from music_perception import lyrics_model_store as store
from music_perception import lyrics_prepare_worker as worker
from music_perception import MusicPerceptionPlugin
from types import SimpleNamespace

@pytest.fixture
def plugin_fixture(tmp_path):
    plugin=object.__new__(MusicPerceptionPlugin)
    plugin.data_path=lambda *parts: tmp_path.joinpath('data',*parts)
    plugin._default_roots=lambda: []
    plugin.ctx=SimpleNamespace(plugins=SimpleNamespace())
    plugin.push_message=lambda **kw: None
    plugin.logger=SimpleNamespace(info=lambda *a,**k:None,
                                  warning=lambda *a,**k:None,
                                  exception=lambda *a,**k:None)
    plugin.config=SimpleNamespace(dump=lambda **kw: {'music_perception':{}})
    yield (plugin,None,None,None)


def test_four_model_quick_inventory_never_claims_sha(tmp_path,monkeypatch):
    monkeypatch.setitem(store.MODELS,'tiny',dict(store.MODELS['tiny'],min_bytes=10))
    root=tmp_path/'whisper'
    assert store.inspect_model_files('tiny',root)['state']=='missing'
    item=root/'tiny';item.mkdir(parents=True)
    (item/'model.bin').write_bytes(b'partial')
    assert store.inspect_model_files('tiny',root)['state']=='incomplete'
    (item/'model.bin').write_bytes(b'0'*40)
    (item/'config.json').write_text('{"metadata": "checked file"}')
    (item/'tokenizer.json').write_text('{"metadata": "checked file"}')
    # Full CT2 readiness requires its plain-text vocabulary too; the old
    # inventory test accidentally omitted this runtime-essential sidecar.
    (item/'vocabulary.txt').write_text('SYNTHETIC VOCABULARY FILE FOR TEST')
    assert store.inspect_model_files('tiny',root)['sha256_verified'] is False
    # UI must distinguish 'file exists' from 'cryptographic verification'.
    assert store.inspect_model_files('tiny',root)['state']=='present_unverified'


def test_progress_measurement_uses_partial_file_and_never_invents_network_bytes(tmp_path):
    root=tmp_path/'models'
    assert store.observed_download_bytes('base',root)==0
    cache=root/'base'/'.cache'/'huggingface'/'download';cache.mkdir(parents=True)
    (cache/'model.bin.hash.incomplete').write_bytes(b'a'*1234)
    assert store.observed_download_bytes('base',root)==1234


def test_staged_worker_reports_real_partial_bytes_and_final_verification(tmp_path,monkeypatch):
    model=tmp_path/'models';cache=model/'base'/'.cache'/'huggingface'/'download'
    cache.mkdir(parents=True)
    dest=tmp_path/'prepare.json';vendor=tmp_path/'vendor';vendor.mkdir()
    states=[]
    def simulated_download(model_name, root, **kwargs):
        (cache/'model.bin.partial.incomplete').write_bytes(b'1'*100)
        time.sleep(1.3)   # let independent byte monitor observe REAL partial file
        states.append(json.loads(dest.read_text('utf-8')))
        return Path(root)/model_name
    monkeypatch.setattr(store,'ensure_local_model',simulated_download)
    monkeypatch.setattr(sys,'argv',['lyrics_prepare_worker.py',str(vendor),str(model),'base',str(dest),'1'])
    assert worker.main()==0
    assert states and states[0]['received_bytes']==100
    assert states[0]['state']=='running'
    final=json.loads(dest.read_text('utf-8'))
    assert final['state']=='ready' and final['progress_pct']==100
    assert final['stage']=='sha256_verified'


def test_prepare_new_choice_does_not_overwrite_active_model(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._lyrics_profile='medium'
        plugin._environment.snapshot=lambda:{'ready':True}
        calls=[]
        monkeypatch.setattr(plugin,'_schedule_lyrics_model_prepare',
                            lambda model_override=None: (calls.append(model_override) or {'ok':True,'started':True}))
        out=(await plugin.stage_lyrics_model('ultra',confirmed=True)).value
        assert out['started'] and calls==['large-v3-turbo']
        assert plugin._lyrics_profile=='medium'
        await plugin.shutdown()
    asyncio.run(scenario())


def test_backend_cannot_activate_missing_weights(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._environment.snapshot=lambda:{'ready':True}
        monkeypatch.setattr(store,'find_local_model',lambda *args: None)
        out=(await plugin.activate_verified_lyrics_model('ultra','cuda')).value
        assert out['ok'] is False and out['next_action']=='stage_lyrics_model'
        assert plugin._lyrics_profile is None
        await plugin.shutdown()
    asyncio.run(scenario())


def test_backend_starts_load_only_after_full_verification(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._environment.snapshot=lambda:{'ready':True}
        monkeypatch.setattr(store,'find_local_model',lambda *args: Path('/verified/read-only/model'))
        started=[]
        monkeypatch.setattr(plugin,'_begin_lyrics_switch',
                      lambda *, quality,device: (started.append((quality,device)) or
                                                   {'ok':True,'switch_status':'preparing'}))
        out=(await plugin.activate_verified_lyrics_model('high','cuda')).value
        assert out['ok'] and started==[('high','cuda')]
        assert plugin._lyrics_profile is None  # not saved before CUDA preflight
        await plugin.shutdown()
    asyncio.run(scenario())


def test_model_catalog_has_four_independent_statuses_and_no_audio(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        with plugin._lyrics_catalog_lock:
            plugin._lyrics_catalog['base']={'state':'verified','sha256_verified':True}
            plugin._lyrics_catalog['large-v3-turbo']={'state':'missing','sha256_verified':False}
        r=(await plugin.get_lyrics_model_catalog()).value
        assert len(r['models'])==4
        # An in-memory hint is not a cryptographic proof when disk files are missing.
        assert r['models']['base']['sha256_verified'] is False
        assert r['models']['base']['state'] == 'missing'
        assert r['models']['large-v3-turbo']['sha256_verified'] is False
        assert plugin._listener is None
        await plugin.shutdown()
    asyncio.run(scenario())


def test_ui_requires_verified_model_before_enabling_apply_button():
    html=(Path(__file__).resolve().parents[1]/'static'/'index.html').read_text('utf-8')
    assert 'id="lyrics-model-download-progress"' in html
    assert "!verified||!!response.paused" in html
    assert "activate_verified_lyrics_model" in html
    assert "stage_lyrics_model" in html


def test_first_activation_is_preflighted_and_frees_gpu_if_not_listening(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._environment.snapshot=lambda:{'ready':True}
        plugin._environment.worker_runtime=lambda:(Path(sys.executable),plugin.data_path('environment'))
        monkeypatch.setattr(store,'find_local_model',lambda *args: Path('/verified/local/model'))
        import music_perception as package
        closed=[]
        class Probe:
            def __init__(self,**kwargs):self.device=kwargs['device']
            def prepare(self):return {'ready':True,'device':self.device}
            def close(self):closed.append(True)
        monkeypatch.setattr(package,'PersistentLyricsSession',Probe)
        result=(await plugin.activate_verified_lyrics_model('high','cpu')).value
        assert result['switch_status']=='preparing'
        plugin._lyrics_switch_thread.join(timeout=3)
        assert not plugin._lyrics_switch_thread.is_alive()
        assert plugin._lyrics_switch_state['status']=='ready'
        assert plugin._lyrics_profile=='high'
        assert plugin._lyrics_session is None
        assert closed==[True]  # preflight must not leave unused model in memory
        await plugin.shutdown()
    asyncio.run(scenario())


def test_failed_preflight_never_saves_selected_model(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._environment.snapshot=lambda:{'ready':True}
        plugin._environment.worker_runtime=lambda:(Path(sys.executable),plugin.data_path('environment'))
        monkeypatch.setattr(store,'find_local_model',lambda *args: Path('/verified/local/model'))
        import music_perception as package
        closed=[]
        class Probe:
            def __init__(self,**kwargs):pass
            def prepare(self):raise RuntimeError('CUDA library unavailable')
            def close(self):closed.append(True)
        monkeypatch.setattr(package,'PersistentLyricsSession',Probe)
        result=(await plugin.activate_verified_lyrics_model('ultra','cuda')).value
        assert result['switch_status']=='preparing'
        plugin._lyrics_switch_thread.join(timeout=3)
        assert plugin._lyrics_switch_state['status']=='failed'
        assert plugin._lyrics_profile is None and closed
        await plugin.shutdown()
    asyncio.run(scenario())
