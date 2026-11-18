"""v0.8.6: explicit model install/uninstall authorization + safe offline fallbacks.

All tests are isolated: no real audio, no network, and no real model weights.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

from music_perception import MusicPerceptionPlugin
from music_perception import model_assets as assets
from music_perception import stem_prepare_worker as downloader
from music_perception import stem_mdx
from music_perception.lyrics_model_store import MODELS
from test_plugin import plugin_fixture


def test_download_plans_explain_disk_and_suggested_gpu_capacity():
    for model in ('tiny','base','small','large-v3-turbo','stems'):
        info=assets.model_plan(model)
        assert info['download_bytes_estimate']>50_000_000
        assert info['disk_required_bytes_estimate']>info['download_bytes_estimate']
        assert info['suggested_free_vram_mib']>=1024
        assert info['cpu_supported'] is True
        assert info['disclaimer']
        assert info['download_source']=='mirror'
    assert assets.model_plan('large-v3-turbo')['suggested_free_vram_mib']>=6144
    with pytest.raises(ValueError):assets.model_plan('unknown')
    with pytest.raises(ValueError):assets.model_plan('stems',source='file:///untrusted')


def test_stem_source_is_pinned_and_prioritized():
    official=downloader.source_urls('official')
    mirror=downloader.source_urls('mirror')
    assert official[0][0].endswith('官方')
    assert mirror[0][0].startswith('魔搭') and mirror[1][0].startswith('HF-Mirror')
    assert mirror[-1]==official[0]
    for title,url in official + mirror:
        assert url.startswith('https://')
        assert 'modelscope' in url or '495f5b4625eb092fe1fbe336f0b67c847566e52e' in url
        assert stem_mdx.MODEL_FILENAME in url
    assert stem_mdx.MODEL_SHA256=='317554b07fe1ea5279a77f2b1520a41ea4b93432560c4ffd08792c30fddf9adc'


def test_reject_download_without_explicit_confirm(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        requested=[]
        monkeypatch.setattr(plugin,'_schedule_lyrics_model_prepare',
                            lambda *a,**k:requested.append(k))
        out=(await plugin.stage_lyrics_model('ultra',confirmed=False)).value
        assert not out['ok'] and not requested
        assert plugin._lyrics_profile is None
        out=(await plugin.start_stem_download(confirmed=False)).value
        assert not out['ok']
        assert plugin._stem_download_thread is None
        out=(await plugin.set_download_source(source='official',confirmed=False)).value
        assert not out['ok'] and plugin._download_source_preference=='mirror'
        out=(await plugin.set_stem_device(device='cuda',confirmed=False)).value
        assert not out['ok'] and plugin._stem_device_preference=='cpu'
        await plugin.shutdown()
    asyncio.run(scenario())


def test_confirmed_download_preserves_active_model(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._lyrics_profile='high'
        plugin._lyrics_device_preference='cuda'
        plugin._environment.snapshot=lambda:{'ready':True}
        started=[]
        monkeypatch.setattr(plugin,'_schedule_lyrics_model_prepare',
                            lambda model_override=None:(started.append(model_override) or {'ok':True,'started':True}))
        res=(await plugin.stage_lyrics_model('ultra','official',confirmed=True)).value
        assert res['ok'] and res['started']
        assert started==['large-v3-turbo']
        assert plugin._lyrics_profile=='high'
        assert plugin._lyrics_device_preference=='cuda'
        assert plugin._download_source_preference=='official'
        await plugin.shutdown()
    asyncio.run(scenario())


def test_lyrics_uninstall_requires_confirmation_and_paused_for_active(plugin_fixture,tmp_path):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._lyrics_profile='high'
        model=plugin.data_path('models')/'whisper_multilingual'/'small'
        model.mkdir(parents=True)
        (model/'model.bin').write_bytes(b'synthetic fake model, no inference')
        (model.parent/'base').mkdir()
        (model.parent/'base'/'model.bin').write_bytes(b'keep me')
        assert not (await plugin.uninstall_lyrics_model('high',confirmed=False)).value['ok']
        assert not (await plugin.uninstall_lyrics_model('high',confirmed=True)).value['ok']
        assert model.exists() and (model.parent/'base'/'model.bin').exists()
        plugin._plugin_paused=True
        r=(await plugin.uninstall_lyrics_model('high',confirmed=True)).value
        assert r['ok'] and not model.exists()
        assert (model.parent/'base'/'model.bin').read_bytes()==b'keep me'
        assert plugin._lyrics_profile is None
        await plugin.shutdown()
    asyncio.run(scenario())


def test_uninstall_rejects_symlink_and_nonmodel_path(tmp_path):
    root=tmp_path/'models';root.mkdir()
    other=tmp_path/'private';other.mkdir()
    (other/'secret.txt').write_text('keep',encoding='utf8')
    try:
        (root/'base').symlink_to(other,target_is_directory=True)
    except OSError:
        pytest.skip('symlinks unavailable')
    with pytest.raises(OSError):assets.remove_plugin_model_dir(root,'base')
    assert (other/'secret.txt').read_text('utf8')=='keep'
    with pytest.raises(ValueError):assets.remove_plugin_model_dir(root,'../../private')


def test_stem_uninstall_does_not_touch_lyrics_or_other_model_files(plugin_fixture):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        root=plugin.data_path('models')/'stems';root.mkdir(parents=True)
        (root/stem_mdx.MODEL_FILENAME).write_bytes(b'synthetic model')
        (root/'other-model.onnx').write_bytes(b'do not touch')
        assert not (await plugin.uninstall_stem_model(confirmed=False)).value['ok']
        plugin._stem_enabled=True
        assert not (await plugin.uninstall_stem_model(confirmed=True)).value['ok']
        plugin._stem_enabled=False
        r=(await plugin.uninstall_stem_model(confirmed=True)).value
        assert r['ok'] and not (root/stem_mdx.MODEL_FILENAME).exists()
        assert (root/'other-model.onnx').read_bytes()==b'do not touch'
        await plugin.shutdown()
    asyncio.run(scenario())


def _fake_dependencies(monkeypatch,*,success_after=1):
    """Fake BOTH modules imported by the -I -S downloader; simulate CDN responses."""
    probe=types.ModuleType('stem_mdx')
    probe.MODEL_SHA256=stem_mdx.MODEL_SHA256
    probe.MODEL_FILENAME=stem_mdx.MODEL_FILENAME
    probe.verify_model=lambda path:Path(path).is_file() and Path(path).read_bytes()==b'verified fake ONNX'
    monkeypatch.setitem(sys.modules,'stem_mdx',probe)
    from music_perception import stem_catalog
    safe_catalog=types.ModuleType('stem_catalog')
    safe_catalog.spec=stem_catalog.spec
    safe_catalog.urls_for=stem_catalog.urls_for
    safe_catalog.verify_local_model=lambda path,quality: Path(path).is_file() and Path(path).read_bytes()==b'verified fake ONNX'
    monkeypatch.setitem(sys.modules,'stem_catalog',safe_catalog)
    transfer=types.ModuleType('lyrics_model_store')
    calls=[]
    def stream(url,destination,*,transfer_progress=None,verification_progress=None,**_):
        calls.append(url)
        if len(calls)<success_after:raise OSError('simulated CDN outage')
        if transfer_progress:transfer_progress(8,17,False)
        Path(destination).write_bytes(b'verified fake ONNX')
        if verification_progress:verification_progress(17,17)
    transfer._stream_to_file=stream
    monkeypatch.setitem(sys.modules,'lyrics_model_store',transfer)
    return calls


def test_stem_worker_fails_over_and_ends_only_after_verification(monkeypatch,tmp_path):
    calls=_fake_dependencies(monkeypatch,success_after=2)
    destination=tmp_path/'stems';status=tmp_path/'progress.json'
    assert downloader.main([str(destination),str(status),'mirror'])==0
    assert len(calls)==2 and calls[0].startswith('https://www.modelscope.cn')
    assert calls[1].startswith('https://hf-mirror.com')
    result=json.loads(status.read_text('utf8'))
    assert result['state']=='ready' and result['sha256_verified'] is True
    assert result['pct']==100
    assert (destination/stem_mdx.MODEL_FILENAME).read_bytes()==b'verified fake ONNX'


def test_stem_worker_uses_verified_disk_file_without_network(monkeypatch,tmp_path):
    calls=_fake_dependencies(monkeypatch)
    destination=tmp_path/'stems';destination.mkdir()
    (destination/stem_mdx.MODEL_FILENAME).write_bytes(b'verified fake ONNX')
    status=tmp_path/'progress.json'
    assert downloader.main([str(destination),str(status),'official'])==0
    assert not calls
    assert json.loads(status.read_text('utf8'))['source']=='本地缓存'


def test_stem_worker_failure_is_visible_and_never_claims_success(monkeypatch,tmp_path):
    calls=_fake_dependencies(monkeypatch,success_after=20)
    destination=tmp_path/'stems';status=tmp_path/'progress.json'
    assert downloader.main([str(destination),str(status),'official'])==1
    assert len(calls)==4
    record=json.loads(status.read_text('utf8'))
    assert record['state']=='failed' and record['sha256_verified'] is False
    assert not (destination/stem_mdx.MODEL_FILENAME).exists()
    assert 'https://' not in json.dumps(record)


def test_hosted_ui_contains_model_install_uninstall_and_explanation():
    panel=(Path(__file__).resolve().parents[1]/'ui/panel.tsx').read_text('utf8')
    for value in ['lyrics-download','lyrics-uninstall','stem-download',
                  'stem-uninstall','set_stem_device','set_download_source',
                  'suggested_free_vram_mib','download_bytes_estimate',
                  'disk_required_bytes_estimate','<Modal open={true}',
                  'userInitiated:true','确认并下载']:
        assert value in panel, value
    assert '已确认下载' not in panel  # must not fake a completed network call
    assert '请将 UVR-MDX-NET' not in panel  # no forced manual installation


def test_persistent_download_settings_are_reused(plugin_fixture):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        await plugin.set_download_source('official',confirmed=True)
        await plugin.set_stem_device('cpu',confirmed=True)
        assert plugin._download_source_preference=='official'
        assert plugin._stem_device_preference=='cpu'
        plugin._download_source_preference='mirror'
        plugin._load_preferences()
        assert plugin._download_source_preference=='official'
        await plugin.shutdown()
    asyncio.run(scenario())
