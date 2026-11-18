"""v0.8.4 regression: old-code detection, ASR offline gating, Hosted UI responsiveness.

No network or actual model weights. Windows/GPU live acceptance is separate.
"""
import asyncio
from pathlib import Path

import pytest
from _version import panel_ui_build, plugin_version, ui_build  # noqa: E402
from music_perception import UI_PROTOCOL_BUILD, lyrics_backend
from music_perception import lyrics_model_store as store


def test_live_asr_refuses_network_until_explicit_verified_model_ready(tmp_path,monkeypatch):
    attempts=[]
    monkeypatch.setattr(store,'find_local_model',lambda *a,**kw:None)
    monkeypatch.setattr(lyrics_backend,'find_local_model',lambda *a,**kw:None)
    def forbid(*args,**kwargs):
        attempts.append(args)
        raise AssertionError('A live Whisper child must NEVER fetch a model')
    monkeypatch.setattr(store.urllib.request,'urlopen',forbid)
    steps=[]
    with pytest.raises(store.LyricsModelDownloadError,match='不会自动联网'):
        lyrics_backend._make_model('large-v3-turbo', tmp_path,
                                   device_policy='cuda',stage_hook=steps.append)
    assert steps==['checking_model']
    assert attempts==[]


def test_quick_inventory_does_not_parse_json_or_claim_sha(tmp_path,monkeypatch):
    model='large-v3-turbo'
    monkeypatch.setitem(store.MODELS,model,dict(store.MODELS[model],min_bytes=30))
    folder=tmp_path/model
    folder.mkdir()
    (folder/'model.bin').write_bytes(b'w'*31)
    for filename in store.REQUIRED_FILES[model]:
        (folder/filename).write_bytes(b'x'*32)
    def no_json_parse(*a,**kw):
        raise AssertionError('Hosted UI must not JSON-parse huge sidecars')
    monkeypatch.setattr(store,'validate_sidecar_file',no_json_parse)
    quick=store.inspect_model_files(model,tmp_path,deep=False)
    assert quick['state']=='present_unverified'
    assert quick['sha256_verified'] is False
    with pytest.raises(AssertionError,match='must not JSON-parse'):
        store.inspect_model_files(model,tmp_path,deep=True)


def test_hosted_dashboard_never_awaits_full_config_dump_or_media_probe(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def forbid(*a,**kw):
        raise AssertionError('Hosted UI must not call synchronous slow host services')
    async def scenario():
        await plugin.startup()
        monkeypatch.setattr(plugin,'_roots',forbid)
        monkeypatch.setattr(plugin,'_refresh_player_track',forbid)
        plugin._auto_consented=True
        snapshots=await plugin.music_dashboard()
        assert snapshots['status']['ok']
        assert snapshots['catalog']['ok']
        assert snapshots['status']['ui_protocol_build']==UI_PROTOCOL_BUILD
        assert snapshots['catalog']['ui_protocol_build']==UI_PROTOCOL_BUILD
        await plugin.shutdown()
    asyncio.run(scenario())


def test_cpu_is_not_reported_as_actual_without_successful_preflight(plugin_fixture):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._lyrics_runtime_device='cpu'
        class Unready:
            running=True
            _model_ready=False
        plugin._lyrics_session=Unready()
        status=(await plugin.get_status(_for_ui=True)).value
        assert status['lyrics_runtime_device'] is None
        plugin._lyrics_session._model_ready=True
        status=(await plugin.get_status(_for_ui=True)).value
        assert status['lyrics_runtime_device']=='cpu'
        plugin._lyrics_session=None
        await plugin.shutdown()
    asyncio.run(scenario())


def test_download_sources_try_direct_before_hub(tmp_path,monkeypatch):
    calls=[]
    target=tmp_path/'large-v3-turbo'
    target.mkdir()
    monkeypatch.setattr(store,'find_local_model',lambda *args: None)
    def direct(model,root,source,cb):
        calls.append(('direct',source))
        if source==store.PUBLIC_SOURCES[-1]:
            return target
        raise store.LyricsModelDownloadError('vocabulary.json 不是 JSON；镜像文件被拒绝')
    monkeypatch.setattr(store,'_download_direct_model',direct)
    assert store.ensure_local_model('large-v3-turbo',tmp_path)==target
    assert calls==[('direct',s) for s in store.PUBLIC_SOURCES]


def test_build_signature_in_back_and_front_and_manifest():
    root=Path(__file__).resolve().parents[1]
    assert UI_PROTOCOL_BUILD==ui_build()
    panel=(root/'ui/panel.tsx').read_text('utf-8')
    assert f'const UI_PROTOCOL_BUILD = "{panel_ui_build()}"' in panel or panel_ui_build() in panel
    assert 'mismatchedBuild' in panel and '版本未同步' in panel
    assert '运行设备：' in panel and '尚未验证' in panel
    assert '实际 CUDA：' in panel and '试推理通过' in panel  # Actual CUDA requires verified inference
    assert 'get_status(self, _for_ui: bool = False' in (root/'__init__.py').read_text('utf-8')
    manifest=(root/'plugin.toml').read_text('utf-8')
    assert f'version = "{plugin_version()}"' in manifest


def test_healthy_already_downloaded_model_still_reused_offline(tmp_path,monkeypatch):
    folder=tmp_path/'base'
    folder.mkdir()
    payload=b'good'*200
    import hashlib
    monkeypatch.setitem(store.MODELS,'base',dict(store.MODELS['base'],min_bytes=80,
                                              sha256=hashlib.sha256(payload).hexdigest()))
    (folder/'model.bin').write_bytes(payload)
    (folder/'config.json').write_text('{"kind":"valid", "description":"synthetic config test"}',encoding='utf8')
    (folder/'tokenizer.json').write_text('{"kind":"valid", "description":"synthetic config test"}',encoding='utf8')
    (folder/'vocabulary.txt').write_text('test tokens sufficient for nonempty vocabulary',encoding='utf8')
    monkeypatch.setattr(store.urllib.request,'urlopen',
                        lambda *a,**k:pytest.fail('A verified cache must not be downloaded'))
    assert store.ensure_local_model('base',tmp_path)==folder
