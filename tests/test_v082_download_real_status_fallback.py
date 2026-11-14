"""Regression cases: no phantom 99%, resumable HTTP and persistent HF cache.

All tests are offline and use synthetic responses/files. They are not a claim
that an actual 1.62GB model can be downloaded from this environment.
"""
import asyncio
import hashlib
import io
import json
from pathlib import Path

import pytest
from music_perception import lyrics_model_store as store
from test_plugin import plugin_fixture


def test_model_download_eof_does_not_promote_then_range_resumes(tmp_path, monkeypatch):
    good = (b'file-contents-' * 400) + b'end'
    expected = hashlib.sha256(good).hexdigest()
    interrupted = good[:1100]
    url = 'https://huggingface.co/a/b/resolve/revision/model.bin'
    events = []

    class Response:
        def __init__(self, content, code, headers):
            self.data = io.BytesIO(content)
            self.status = code
            self.headers = headers
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, n): return self.data.read(n)
        def geturl(self): return url
        def getcode(self): return self.status

    def first(_req, **_kw):
        return Response(interrupted, 200, {'Content-Length': str(len(good))})
    monkeypatch.setattr(store.urllib.request, 'urlopen', first)
    target = tmp_path/'model.bin'
    with pytest.raises(store.LyricsModelDownloadError, match='提前结束'):
        store._stream_to_file(url, target, min_bytes=100, expected_sha256=expected,
                              transfer_progress=lambda *x:events.append(x))
    assert not target.exists()
    assert (tmp_path/'model.bin.partial').read_bytes()==interrupted

    def second(request, **_kw):
        assert request.get_header('Range') == f'bytes={len(interrupted)}-'
        return Response(good[len(interrupted):], 206,
                        {'Content-Range':f'bytes {len(interrupted)}-{len(good)-1}/{len(good)}',
                         'Content-Length':str(len(good)-len(interrupted))})
    monkeypatch.setattr(store.urllib.request,'urlopen',second)
    store._stream_to_file(url,target,min_bytes=100,expected_sha256=expected,
                          transfer_progress=lambda *x:events.append(x))
    assert target.read_bytes()==good
    assert not (tmp_path/'model.bin.partial').exists()
    assert any(bytes_written==len(good) for bytes_written,_,_ in events)


def test_verified_hf_cache_not_ignored_when_plugin_model_dir_is_empty(plugin_fixture,
                                                                       tmp_path,monkeypatch):
    plugin,*_ = plugin_fixture
    hf_cache = tmp_path/'huggingface'/'snapshot'
    hf_cache.mkdir(parents=True)
    weight=hf_cache/'model.bin'; weight.write_bytes(b'already-verified-by-worker')
    (hf_cache/'config.json').write_text('{"description":"synthetic test config"}')
    (hf_cache/'tokenizer.json').write_text('{"description":"synthetic test tokenizer"}')
    (hf_cache/'vocabulary.txt').write_text('SYNTHETIC CT2 VOCABULARY FOR TEST')
    monkeypatch.setitem(store.MODELS,'base',dict(store.MODELS['base'],min_bytes=10))
    monkeypatch.setattr(store,'candidate_model_directories',
                        lambda model, root:[(hf_cache,'huggingface_cache')] if model=='base' else [])

    async def scenario():
        await plugin.startup()
        state=plugin.data_path('lyrics_model_prepare_base.json')
        stat=weight.stat()
        state.write_text(json.dumps({'schema':'music_model_prepare_v1',
                          'model':'base','state':'ready',
                          'weight_bytes':stat.st_size,
                          'weight_mtime_ns':stat.st_mtime_ns}),encoding='utf8')
        catalog=(await plugin.get_lyrics_model_catalog()).value['models']['base']
        assert catalog['state']=='verified'
        assert catalog['sha256_verified']
        assert catalog['origin']=='huggingface_cache'
        # Changed weights must not be activated from an old SHA verdict.
        weight.write_bytes(b'changed-after-verification')
        catalog=(await plugin.get_lyrics_model_catalog()).value['models']['base']
        assert catalog['state']!='verified'
        await plugin.shutdown()
    asyncio.run(scenario())


def test_preparation_file_audit_has_no_fake_download_progress(plugin_fixture):
    plugin,*_ = plugin_fixture
    class LiveWorker:
        def is_alive(self):return True
    async def scenario():
        await plugin.startup()
        plugin._lyrics_prepare_thread=LiveWorker()
        plugin._lyrics_prepare_state={'model':'large-v3-turbo','state':'running'}
        state=plugin.data_path('lyrics_model_prepare_large-v3-turbo.json')
        state.write_text(json.dumps({'schema':'music_model_prepare_v1',
                       'model':'large-v3-turbo','state':'running',
                       'phase':'checking_existing','received_bytes':0}),encoding='utf8')
        catalog=(await plugin.get_lyrics_model_catalog()).value['models']['large-v3-turbo']
        assert catalog['state']=='checking'
        assert not catalog['download_active']
        plugin._lyrics_prepare_thread=None
        await plugin.shutdown()
    asyncio.run(scenario())
