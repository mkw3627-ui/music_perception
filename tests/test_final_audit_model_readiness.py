"""Final delivery audits: no stale 'verified' UI and no false model activation.

These tests use tiny synthetic files and never download weights or run CUDA.
"""
import asyncio
import json
from pathlib import Path

from music_perception import lyrics_model_store as store
from music_perception.lyrics_prepare_worker import report
from test_plugin import plugin_fixture


def _synthetic_base(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    weight=folder/'model.bin'
    weight.write_bytes(b'not-a-real-model: synthetic testing only')
    for name in store.REQUIRED_FILES['base']:
        if name.endswith('.json'):
            (folder/name).write_text('{"synthetic": "valid test sidecar"}',encoding='utf-8')
        else:
            (folder/name).write_text('sufficiently long synthetic vocabulary data',encoding='utf-8')
    return weight


def test_previous_sha_verdict_never_uses_stale_sidecars_in_hosted_ui(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        root=plugin.data_path('models')/'whisper_multilingual'
        folder=root/'base'
        weight=_synthetic_base(folder)
        monkeypatch.setitem(store.MODELS,'base',dict(store.MODELS['base'], min_bytes=10))
        stat=weight.stat()
        status=plugin.data_path('lyrics_model_prepare_base.json')
        status.write_text(json.dumps({'schema':'music_model_prepare_v1', 'model':'base',
            'state':'ready','weight_bytes':stat.st_size,
            'weight_mtime_ns':stat.st_mtime_ns}),encoding='utf-8')
        # Old status records omit sidecar identity, and cannot justify a green
        # badge on every UI heartbeat without an expensive complete re-check.
        data=(await plugin.get_lyrics_model_catalog(_for_ui=True)).value['models']['base']
        assert data['state']=='present_unverified' and not data['sha256_verified']
        # New complete record includes a sidecar signature from the actual
        # model preparation's verified snapshot.
        signed=json.loads(status.read_text(encoding='utf8'))
        signed['sidecar_signature']=store.sidecar_stat_signature(folder,'base')
        status.write_text(json.dumps(signed),encoding='utf-8')
        data=(await plugin.get_lyrics_model_catalog(_for_ui=True)).value['models']['base']
        assert data['state']=='verified' and data['sha256_verified']
        # Even if model.bin is unchanged, replaced tokenizer or vocabulary
        # invalidates the cached 'verified' claim.
        (folder/'tokenizer.json').write_text('{"synthetic": "changed and unverified"}',encoding='utf-8')
        data=(await plugin.get_lyrics_model_catalog(_for_ui=True)).value['models']['base']
        assert data['state']!='verified' and not data['sha256_verified']
        await plugin.shutdown()
    asyncio.run(scenario())


def test_in_memory_verified_flag_does_not_override_local_files(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        folder=plugin.data_path('models')/'whisper_multilingual'/'base'
        _synthetic_base(folder)
        monkeypatch.setitem(store.MODELS,'base',dict(store.MODELS['base'], min_bytes=10))
        with plugin._lyrics_catalog_lock:
            plugin._lyrics_catalog['base']={'state':'verified','sha256_verified':True}
        data=(await plugin.get_lyrics_model_catalog(_for_ui=True)).value['models']['base']
        assert data['state']=='present_unverified' and not data['sha256_verified']
        await plugin.shutdown()
    asyncio.run(scenario())


def test_preparation_snapshot_persists_all_sidecar_stat_identities(tmp_path):
    path=tmp_path/'model.json'
    sig=[{'file':'tokenizer.json','size':123,'mtime_ns':987}]
    written=report(path,'base','sha256_verified','ready',weight_bytes=1234,
                   weight_mtime_ns=4567,sidecar_signature=sig)
    assert written['sidecar_signature']==sig
    assert json.loads(path.read_text(encoding='utf-8'))['sidecar_signature']==sig
