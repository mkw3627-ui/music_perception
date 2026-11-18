"""v0.8.6: confirm-first model manager, strict file scopes and download protection."""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import types
from pathlib import Path

from music_perception import model_assets, stem_prepare_worker


def test_model_plans_are_explanations_not_hard_gpu_requirements():
    for model in ('tiny','base','small','large-v3-turbo','stems'):
        x=model_assets.model_plan(model)
        assert x['download_bytes_estimate']>60_000_000
        assert x['disk_required_bytes_estimate']>x['download_bytes_estimate']
        assert x['suggested_free_vram_mib']>0
        assert x['cpu_supported']
        assert '估计' in x['disclaimer'] or '估算' in x['disclaimer']


def test_stem_downloads_reference_same_pinned_file_on_every_host():
    for priority in ('mirror','official'):
        sources=stem_prepare_worker.source_urls(priority)
        assert len(sources)==4 and len(set(url for name,url in sources))==4
        for name,url in sources:
            assert url.startswith('https://')
            if 'modelscope' in url:
                assert url.endswith('/UVR-MDX-NET-Inst_HQ_3.onnx')
                continue
            assert '495f5b4625eb092fe1fbe336f0b67c847566e52e' in url
            assert url.endswith('/UVR-MDX-NET-Inst_HQ_3.onnx?download=true')
    assert stem_prepare_worker.source_urls('mirror')[0][0].startswith('魔搭')
    assert stem_prepare_worker.source_urls('official')[0][0].startswith('Hugging Face')


def test_stem_worker_tracks_real_progress_and_hash_without_download(tmp_path,monkeypatch):
    """Tests user-approved preparation with a fake response: zero HTTP."""
    payload=b'portable-verified-onnx-fixture'
    model=types.SimpleNamespace(MODEL_FILENAME='UVR-MDX-NET-Inst_HQ_3.onnx',
        MODEL_SHA256=hashlib.sha256(payload).hexdigest(),
        verify_model=lambda p: Path(p).is_file() and Path(p).read_bytes()==payload)
    def stream(url,destination,**kwargs):
        assert kwargs['expected_sha256']==__import__('music_perception.stem_catalog',fromlist=['spec']).spec('ultra')['sha256']
        assert '/Politrees/UVR_resources/' in url
        assert url.startswith('https://')
        kwargs['transfer_progress'](len(payload),len(payload),False)
        kwargs['verification_progress'](len(payload),len(payload))
        Path(destination).write_bytes(payload)
    monkeypatch.setitem(sys.modules,'stem_mdx',model)
    from music_perception import stem_catalog
    safe_catalog=types.ModuleType('stem_catalog')
    safe_catalog.spec=stem_catalog.spec
    safe_catalog.urls_for=stem_catalog.urls_for
    safe_catalog.verify_local_model=lambda path,quality: Path(path).is_file() and Path(path).read_bytes()==payload
    monkeypatch.setitem(sys.modules,'stem_catalog',safe_catalog)
    monkeypatch.setitem(sys.modules,'lyrics_model_store',types.SimpleNamespace(_stream_to_file=stream))
    folder=tmp_path/'stems'
    status=tmp_path/'status.json'
    assert stem_prepare_worker.main([str(folder),str(status),'mirror'])==0
    data=json.loads(status.read_text('utf-8'))
    assert data['state']=='ready' and data['sha256_verified']
    assert data['pct']==100 and data['model_bytes']==len(payload)
    assert (folder/model.MODEL_FILENAME).read_bytes()==payload
    # Second confirmation reuses local verified data, with no new stream.
    def fail(*args,**kwargs):
        raise AssertionError('model unexpectedly re-downloaded')
    monkeypatch.setitem(sys.modules,'lyrics_model_store',types.SimpleNamespace(_stream_to_file=fail))
    assert stem_prepare_worker.main([str(folder),str(status),'official'])==0


def test_stem_download_requires_consent(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        calls=[]
        monkeypatch.setattr(plugin,'_stem_download_background',lambda source:calls.append(source))
        # Even with model absent, UI refresh / startup does not download it.
        before=(await plugin.get_status(_for_ui=True)).value
        assert before['stem_separation']['download']['running'] is False
        unconfirmed=(await plugin.start_stem_download(confirmed=False)).value
        assert unconfirmed['ok'] is False and not calls
        plugin._environment.snapshot=lambda:{'ready':True}
        monkeypatch.setattr('music_perception.verify_stem_model',lambda _:False)
        monkeypatch.setattr('music_perception.free_disk_check',lambda *args:(True,30_000_000_000))
        confirmed=(await plugin.start_stem_download(confirmed=True,download_source='official')).value
        assert confirmed['started'] is True
        plugin._stem_download_thread.join(timeout=3)
        assert calls==['official']
        await plugin.shutdown()
    asyncio.run(scenario())


def test_model_uninstall_precise_scope(tmp_path):
    root=tmp_path/'models'/'whisper_multilingual'
    (root/'tiny').mkdir(parents=True)
    (root/'base').mkdir()
    (root/'tiny'/'model.bin').write_bytes(b'fake')
    (root/'base'/'model.bin').write_bytes(b'keep')
    env=tmp_path/'models'/'environment'
    env.mkdir()
    (env/'python.exe').write_bytes(b'keep')
    r=model_assets.remove_plugin_model_dir(root,'tiny')
    assert r['ok'] and r['bytes_removed']==4
    assert not (root/'tiny').exists()
    assert (root/'base'/'model.bin').read_bytes()==b'keep'
    assert (env/'python.exe').read_bytes()==b'keep'


def test_no_shared_or_symlink_delete(tmp_path):
    root=tmp_path/'models'
    external=tmp_path/'elsewhere'
    external.mkdir()
    (external/'model.bin').write_bytes(b'private')
    root.mkdir()
    try:
        (root/'tiny').symlink_to(external,target_is_directory=True)
    except (OSError,NotImplementedError):
        return
    try:
        model_assets.remove_plugin_model_dir(root,'tiny')
    except OSError:
        pass
    else:
        raise AssertionError('symlink was followed')
    assert (external/'model.bin').read_bytes()==b'private'


def test_stem_uninstall_removes_only_allowlisted_files(tmp_path):
    directory=tmp_path/'stems'
    directory.mkdir()
    filename='UVR-MDX-NET-Inst_HQ_3.onnx'
    (directory/filename).write_bytes(b'123')
    (directory/(filename+'.partial')).write_bytes(b'partial')
    (directory/'other_model.onnx').write_bytes(b'keep')
    result=model_assets.remove_stem_model(directory)
    assert result['bytes_removed']==10
    assert not (directory/filename).exists()
    assert (directory/'other_model.onnx').read_bytes()==b'keep'


def test_lyrics_uninstall_guards_pause_and_requires_consent(plugin_fixture):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        root=plugin.data_path('models')/'whisper_multilingual'
        (root/'base').mkdir(parents=True)
        (root/'base'/'model.bin').write_bytes(b'dummy')
        plugin._lyrics_profile='medium'
        assert not (await plugin.uninstall_lyrics_model('medium',confirmed=False)).value['ok']
        assert not (await plugin.uninstall_lyrics_model('medium',confirmed=True)).value['ok']
        assert (root/'base'/'model.bin').exists()
        plugin._plugin_paused=True
        result=(await plugin.uninstall_lyrics_model('medium',confirmed=True)).value
        assert result['ok'] and plugin._lyrics_profile is None
        assert not (root/'base').exists()
        await plugin.shutdown()
    asyncio.run(scenario())


def test_stem_uninstall_guards_active_and_preserves_whisper(plugin_fixture):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        path=plugin._stem_model_path()
        path.parent.mkdir(parents=True)
        path.write_bytes(b'fixture')
        whisper=plugin.data_path('models')/'whisper_multilingual'/'base'/'model.bin'
        whisper.parent.mkdir(parents=True)
        whisper.write_bytes(b'keep')
        plugin._stem_enabled=True
        assert not (await plugin.uninstall_stem_model(confirmed=True)).value['ok']
        assert path.exists()
        plugin._stem_enabled=False
        assert (await plugin.uninstall_stem_model(confirmed=True)).value['ok']
        assert not path.exists()
        assert whisper.read_bytes()==b'keep'
        await plugin.shutdown()
    asyncio.run(scenario())


def test_hosted_panel_requires_explicit_modal_before_mutation():
    script=(Path(__file__).resolve().parents[1]/'ui/panel.tsx').read_text('utf-8')
    assert '<Modal open={true} title={dialogTitle}' in script
    assert 'closeOnBackdrop={false}' in script
    assert '</Page>\n  {dialog?<Modal' in script
    assert 'confirmed:true' in script
    for action in ('start_stem_download','uninstall_stem_model','uninstall_lyrics_model',
                   'stage_lyrics_model','set_download_source','set_stem_device'):
        assert action in script
    assert '建议空闲显存' in script
    assert '正在下载分离模型' not in script or 'stemProgressActive' in script
