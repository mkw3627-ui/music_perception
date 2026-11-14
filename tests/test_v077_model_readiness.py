"""Regression tests for misleading model-ready status and reused verified caches."""
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from music_perception import lyrics_model_store as store
from music_perception.lyrics_health_probe import run_probe
from music_perception.lyrics_prepare_worker import main as prepare_main
from music_perception.auto_listener import FinishedClip
from test_lyrics_v063 import pcm
from test_plugin import plugin_fixture


def _model(root, data):
    root.mkdir(parents=True)
    (root/'model.bin').write_bytes(data)
    (root/'config.json').write_text('{"model":"whisper","description":"sample"}',encoding='utf-8')
    (root/'tokenizer.json').write_text('{"model":"test","description":"sample"}',encoding='utf-8')
    (root/'vocabulary.txt').write_text('SYNTHETIC WHISPER VOCABULARY TEST DATA',encoding='utf-8')


def test_verified_global_hf_cache_is_reused_without_redownload(monkeypatch,tmp_path):
    payload=b'fake-model-offline-cache'*15
    spec=dict(repo='Systran/faster-whisper-base',sha256=hashlib.sha256(payload).hexdigest(),min_bytes=len(payload))
    monkeypatch.setitem(store.MODELS,'base',spec)
    global_cache=tmp_path/'shared_hf_cache'
    monkeypatch.setenv('HF_HUB_CACHE',str(global_cache))
    old=global_cache/'models--Systran--faster-whisper-base'/'snapshots'/'old_download'
    _model(old,payload)
    root=tmp_path/'plugin-data'/'models'
    valid=store.find_local_model('base',root)
    assert valid==old
    assert store.ensure_local_model('base',root,downloader=lambda **kw:pytest.fail('no network permitted'))==old
    report=run_probe(tmp_path,root,'base','cpu',tmp_path/'report.json',deep=False,
                     module_loader=lambda name:object())
    assert report['weight_files_present'] is True
    assert report['model_cache_origin']=='huggingface_cache'


def test_corrupt_global_cache_is_not_treated_as_verified(monkeypatch,tmp_path):
    payload=b'one'*300
    monkeypatch.setitem(store.MODELS,'base',dict(repo='Systran/faster-whisper-base',min_bytes=10,
                     sha256=hashlib.sha256(payload).hexdigest()))
    cache=tmp_path/'hub'
    monkeypatch.setenv('HF_HUB_CACHE',str(cache))
    model=cache/'models--Systran--faster-whisper-base'/'snapshots'/'bad'
    _model(model,b'corrupt'*200)
    assert store.find_local_model('base',tmp_path/'local') is None


def test_live_unready_child_does_not_block_model_prepare(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    class Env:
        def snapshot(self):return {'ready':True}
        def stop(self):pass
    class ASR:
        running=True
        _model_ready=False
        closed=False
        def close(self):self.closed=True;self.running=False
    async def scenario():
        await plugin.startup()
        plugin._environment=Env()
        plugin._lyrics_profile='medium'
        prior=ASR();plugin._lyrics_session=prior
        jobs=[]
        monkeypatch.setattr(plugin,'_prepare_lyrics_model_background',lambda name:jobs.append(name))
        result=plugin._schedule_lyrics_model_prepare()
        assert result['started'] and prior.closed
        assert plugin._lyrics_session is None
        plugin._lyrics_prepare_thread.join(timeout=2)
        assert jobs==['base']
        await plugin.shutdown()
    asyncio.run(scenario())


def test_loaded_child_is_not_needlessly_restarted(plugin_fixture):
    plugin,*_=plugin_fixture
    class Env:
        def snapshot(self):return {'ready':True}
        def stop(self):pass
    class ASR:
        running=True
        _model_ready=True
        closed=False
        def close(self):self.closed=True;self.running=False
    async def scenario():
        await plugin.startup()
        plugin._environment=Env();plugin._lyrics_profile='medium'
        previous=ASR();plugin._lyrics_session=previous
        answer=plugin._schedule_lyrics_model_prepare()
        assert answer['model_loaded'] is True and not previous.closed
        assert plugin._lyrics_prepare_thread is None
        await plugin.shutdown()
    asyncio.run(scenario())


def test_new_music_continues_during_separate_download(plugin_fixture):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._listener_session=1
        plugin._song_boundary_enforced=True
        plugin._active_track_sequence=3
        plugin._singing_prepare_state={'state':'running','quality':'medium'}
        plugin._transcribe_singing_live_audio_isolated=lambda *a,**k: pytest.fail('never launch duplicate model downloader')
        clip=FinishedClip(pcm(7),'live_progress',7,sequence=3)
        plugin._lyrics_queue.put((clip,{},1))
        task=plugin._lyrics_queue.get_nowait()
        plugin._process_one_lyrics_task(*task,session=1,q=plugin._lyrics_queue)
        assert plugin._lyrics_status=='waiting_model_preparation'
        assert plugin._lyrics_queue.unfinished_tasks==0
        await plugin.shutdown()
    asyncio.run(scenario())


def test_model_prepare_error_tells_user_network_without_leaking_paths(monkeypatch,tmp_path):
    def fail(*a,**k):
        raise store.LyricsModelDownloadError('歌词模型权重未下载完成；官方源及可用镜像均无法取得已校验文件。')
    monkeypatch.setattr(store,'ensure_local_model',fail)
    vendor=tmp_path/'vendor';vendor.mkdir()
    report=tmp_path/'status.json'
    monkeypatch.setattr(sys,'argv',['lyrics_prepare_worker.py',str(vendor),str(tmp_path/'models'),
                         'base',str(report),'1'])
    assert prepare_main()==1
    data=json.loads(report.read_text('utf-8'))
    assert data['state']=='failed' and '未下载完成' in data['message']
    assert 'tmp' not in data['message']
