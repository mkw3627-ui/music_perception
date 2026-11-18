"""v0.8.0: pinned official/mirrors, real-bytes-only progress, resume and private UI."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from music_perception import lyrics_model_store as store
from music_perception import runtime_environment as runtime


def test_all_multilingual_tiers_have_pinned_model_and_three_allowlisted_sources(monkeypatch):
    assert set(store.MODELS)=={'tiny','base','small','large-v3-turbo'}
    assert set(store.PUBLIC_SOURCES)=={
        'https://huggingface.co','https://hf-mirror.com','https://hf-mirror.net'}
    for name,spec in store.MODELS.items():
        assert not spec['repo'].endswith('.en')
        assert len(spec['sha256'])==64
        for endpoint in store.PUBLIC_SOURCES+(store.MODELSCOPE,):
            url=store._direct_file_url(endpoint,name,'model.bin')
            assert '/resolve/' in url and '/model.bin' in url
            assert url.startswith(endpoint+'/')
    monkeypatch.setenv('NEKO_LYRICS_DOWNLOAD_PRIORITY','mirror')
    assert store._safe_sources(True)==(
        store.MODELSCOPE,'https://hf-mirror.com','https://hf-mirror.net','https://huggingface.co')
    monkeypatch.setenv('NEKO_LYRICS_DOWNLOAD_PRIORITY','official')
    assert store._safe_sources(True)[0]=='https://huggingface.co'
    assert store._safe_sources(True)[-1]==store.MODELSCOPE
    assert store._safe_sources(False)==('https://huggingface.co',)


def test_download_stream_resumes_using_server_confirmed_range(tmp_path,monkeypatch):
    result=tmp_path/'model.bin'
    result.with_name('model.bin.partial').write_bytes(b'1234567')
    events=[]
    class Response:
        status=206
        headers={'Content-Range':'bytes 7-12/13'}
        def __init__(self):
            self.buf=io.BytesIO(b'890123')
        def __enter__(self):
            return self
        def __exit__(self,*args):
            pass
        def read(self,n):
            return self.buf.read(n)
        def geturl(self):
            return 'https://trusted.example/content'
        def getcode(self):
            return self.status
    def urlopen(req,timeout):
        events.append((req.headers,timeout))
        return Response()
    monkeypatch.setattr(store.urllib.request,'urlopen',urlopen)
    store._stream_to_file('https://hf-mirror.com/test',result,min_bytes=10)
    assert result.read_bytes()==b'1234567890123'
    assert 'Range' in events[0][0] and events[0][0]['Range']=='bytes=7-'


def test_range_ignored_restarts_without_appending_corrupted_bytes(tmp_path,monkeypatch):
    result=tmp_path/'model.bin'
    result.with_name('model.bin.partial').write_bytes(b'BADPART')
    class Response:
        status=200
        headers={}
        def __init__(self):
            self.buf=io.BytesIO(b'CLEAN COMPLETE')
        def __enter__(self):
            return self
        def __exit__(self,*a):
            pass
        def read(self,n):
            return self.buf.read(n)
        def geturl(self):
            return 'https://hf-mirror.com/valid'
        def getcode(self):
            return 200
    monkeypatch.setattr(store.urllib.request,'urlopen',lambda *a,**kw:Response())
    store._stream_to_file('https://hf-mirror.com/valid',result,min_bytes=10)
    assert result.read_bytes()==b'CLEAN COMPLETE'


def test_direct_downloader_rejects_unsigned_weight_and_has_correct_file_names(tmp_path,monkeypatch):
    payload=b'complete weight file with enough bytes'*4
    original=store.MODELS['tiny']
    monkeypatch.setitem(store.MODELS,'tiny',dict(original,min_bytes=35,
                     sha256=hashlib.sha256(payload).hexdigest()))
    records=[]
    class Response:
        status=200
        headers={}
        def __init__(self,payload):
            self.stream=io.BytesIO(payload)
        def __enter__(self):
            return self
        def __exit__(self,*a):
            pass
        def read(self,n):
            return self.stream.read(n)
        def geturl(self):
            return 'https://cdn.example/allowed'
        def getcode(self):
            return 200
    def urlopen(req,timeout):
        url=req.full_url
        records.append(url)
        if 'model.bin' in url:
            return Response(payload)
        if '.json' in url:
            return Response(json.dumps({'metadata':'good data'*4}).encode())
        if '.txt' in url:
            return Response(b'TOKENIZER CONTENT LONG ENOUGH')
        raise AssertionError(url)
    monkeypatch.setattr(store.urllib.request,'urlopen',urlopen)
    folder=store._download_direct_model('tiny',tmp_path,'https://hf-mirror.com')
    assert store.check_local_model(folder,'tiny')[0]
    assert all('/resolve/' in u for u in records)
    assert len(records)==4
    # Corrupted replacement must never be considered safe just because a mirror served it.
    (folder/'model.bin').write_bytes(b'WRONG'*30)
    assert not store.check_local_model(folder,'tiny')[0]


def test_yamnet_has_hash_pinned_mirror_backup_after_google():
    assert len(runtime.YAMNET_SOURCES)>=4
    for url,digest in runtime.YAMNET_SOURCES[2:]:
        assert digest==runtime.YAMNET_SHA256
        assert '/resolve/'+runtime.YAMNET_PINNED_REVISION+'/' in url
    assert runtime.YAMNET_SOURCES[0][1] is None


def test_ui_hides_transfer_when_not_actively_downloading():
    html=(Path(__file__).parents[1]/'static/index.html').read_text('utf8')
    assert 'id="lyrics-model-transfer-panel" class="note model-transfer" hidden' in html
    assert "panel.hidden=!(active||failed)" in html
    assert "模型下载未完成 · 已保存进度" in html
    assert 'bar.value=0;' in html  # no fake half-filled animated bar at zero bytes
    assert 'id="lyrics-download-source"' in html
    assert "download_source:$('lyrics-download-source').value" in html
