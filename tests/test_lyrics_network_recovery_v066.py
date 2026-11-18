"""No network or large model downloads: deterministic v0.6.7 regression tests."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import pytest
from music_perception import lyrics_model_store as store
from music_perception.auto_listener import FinishedClip
from music_perception.lyrics_backend import transcribe_singing_candidate
from music_perception.lyrics_bridge import LyricsModelDownloadError as BridgeDownloadError
from test_lyrics_v063 import pcm


@pytest.fixture
def stub_model(monkeypatch):
    binary = b'only-testing-weights-' * 18
    monkeypatch.setitem(store.MODELS, 'base', dict(
        repo='Systran/faster-whisper-base', min_bytes=len(binary),
        sha256=hashlib.sha256(binary).hexdigest()))
    return binary


def _create_local_model(root: Path, payload: bytes):
    root.mkdir(parents=True, exist_ok=True)
    (root/'model.bin').write_bytes(payload)
    (root/'config.json').write_text(json.dumps({'type': 'whisper','ok':True}),encoding='utf-8')
    (root/'tokenizer.json').write_text(json.dumps({'model': {'type':'BPE'}, 'padding':'example'*8}),encoding='utf-8')
    (root/'vocabulary.txt').write_text('SYNTHETIC WHISPER VOCABULARY TEST DATA',encoding='utf-8')


def test_complete_local_model_requires_no_http(tmp_path,stub_model):
    _create_local_model(tmp_path/'base', stub_model)
    def no_network(**kwargs):
        pytest.fail('offline model must not download again')
    assert store.ensure_local_model('base',tmp_path,downloader=no_network)==tmp_path/'base'


def test_corrupt_model_never_claimed_ready(tmp_path,stub_model):
    folder=tmp_path/'base'
    _create_local_model(folder,b'B'*len(stub_model))
    valid,reason=store.check_local_model(folder,'base')
    assert not valid and 'SHA-256' in reason
    assert store.find_local_model('base',tmp_path) is None


def test_official_timeout_mirror_succeeds_with_checksum(tmp_path,stub_model):
    visited=[]
    def fake_downloader(**args):
        visited.append(args)
        assert args['repo_id']=='Systran/faster-whisper-base'
        assert args['token'] is False, 'Never send saved private HF token to a mirror'
        assert args['max_workers'] <= 2
        if args['endpoint']=='https://huggingface.co':
            raise TimeoutError('network timeout')
        _create_local_model(Path(args['local_dir']),stub_model)
        return args['local_dir']
    path=store.ensure_local_model('base',tmp_path,downloader=fake_downloader)
    assert path==tmp_path/'base'
    assert [x['endpoint'] for x in visited] == ['https://huggingface.co','https://hf-mirror.com']
    assert store.ensure_local_model('base',tmp_path,downloader=lambda **k: pytest.fail('should use cache'))==path


def test_disable_mirror_uses_only_official(tmp_path,stub_model):
    hits=[]
    def offline(**kwargs):
        hits.append(kwargs['endpoint'])
        raise ConnectionError('unreachable')
    with pytest.raises(store.LyricsModelDownloadError,match='权重未下载完成'):
        store.ensure_local_model('base',tmp_path,downloader=offline,allow_public_mirror=False)
    assert hits==['https://huggingface.co']


def test_both_sources_fail_with_short_actionable_status(tmp_path,stub_model):
    def offline(**kwargs):
        raise TimeoutError('C:\\Users\\private\\secrets.txt username password')
    with pytest.raises(store.LyricsModelDownloadError) as failure:
        store.ensure_local_model('base',tmp_path,downloader=offline)
    msg=str(failure.value)
    assert 'Whisper' not in msg or 'Systran' in msg
    assert 'models/whisper_multilingual/base/' in msg
    assert 'private' not in msg and 'password' not in msg
    assert len(msg)<600


def test_model_root_can_reuse_old_hf_snapshot_offline(tmp_path,stub_model):
    old=tmp_path/'models--Systran--faster-whisper-base'/'snapshots'/'abc123'
    _create_local_model(old,stub_model)
    assert store.ensure_local_model('base',tmp_path,downloader=lambda **k: pytest.fail('should reuse existing'))==old


def test_local_model_fake_factory_no_hub_or_gpu_when_cpu(tmp_path):
    seen=[]
    class EmptyASR:
        def transcribe(self,*a,**k):
            from types import SimpleNamespace
            return [],SimpleNamespace(language='zh')
    def factory(model_ref,**kwargs):
        seen.append((model_ref,kwargs))
        return EmptyASR()
    report=transcribe_singing_candidate(pcm(2),model_dir=tmp_path,model_factory=factory,
                                         device_policy='cpu',cpu_threads=2)
    assert report['status']=='no_reliable_lyrics'
    assert seen[0][0]=='base'
    assert seen[0][1]['local_files_only'] is True
    assert seen[0][1]['device']=='cpu'


def test_song_lyrics_download_failure_backoff_without_stopping_music(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._listener_session=1
        plugin._song_boundary_enforced=True
        plugin._active_track_sequence=4
        plugin._live_session_config={'lyrics_model':'base'}
        plugin._transcribe_singing_live_audio_isolated=lambda *a,**k: (_ for _ in ()).throw(
            BridgeDownloadError('歌词模型权重未下载完成'))
        clip=FinishedClip(pcm(2),'live_progress',2,sequence=4)
        scene={'music_capture':{'music_sequence':4,'offset_s':0,'end_s':2},
               'musical_event_timeline':{'events':[]}}
        plugin._lyrics_queue.put((clip,scene,1))
        item=plugin._lyrics_queue.get_nowait()
        plugin._process_one_lyrics_task(*item,session=1,q=plugin._lyrics_queue)
        assert plugin._lyrics_retry_after > time.monotonic()
        assert plugin._lyrics_status=='model_download_waiting_network'
        assert not plugin._enqueue_lyrics(clip,scene,1)
        assert plugin._lyrics_queue.empty()
        assert plugin._auto_consented
        await plugin.shutdown()
    asyncio.run(run())


def test_resume_sharing_allows_immediate_model_retry(plugin_fixture):
    plugin,_,_,_ = plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=False
        plugin._lyrics_retry_after=time.monotonic()+300
        response=await plugin.set_ai_delivery(enabled=True)
        assert response.value['ok']
        assert plugin._lyrics_retry_after == 0.0
        await plugin.shutdown()
    asyncio.run(run())


def test_config_mirror_switch_exposed(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def run():
        await plugin.startup()
        settings=await plugin._settings()
        assert settings['lyrics_mirror_fallback'] is True
        await plugin.shutdown()
    asyncio.run(run())
