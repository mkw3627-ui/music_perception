"""0.8.1: no loss of 99%-complete downloads; keep error and SHA states visible.
All HTTP responses are fake; no Internet, GPU, or model weight downloads.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.error import HTTPError

import pytest
from music_perception import lyrics_model_store as store


class FakeResponse:
    def __init__(self, payload, *, status=200, headers=None):
        self.content = io.BytesIO(payload)
        self.status = status
        self.headers = headers or {}
    def __enter__(self):
        return self
    def __exit__(self,*_):
        return False
    def read(self,n):
        return self.content.read(n)
    def geturl(self):
        return 'https://hf-mirror.com/trusted/weights'
    def getcode(self):
        return self.status


def test_truncated_99_percent_keeps_partial_and_next_request_resumes(tmp_path,monkeypatch):
    final=tmp_path/'model.bin'
    payload=b'a'*4096+b'z'*32
    sha=hashlib.sha256(payload).hexdigest()
    calls=[]
    def get(req,timeout=25):
        calls.append(req.headers.copy())
        if len(calls)==1:
            # The CDN promised more bytes than it really sent.
            return FakeResponse(payload[:-32],headers={'Content-Length':str(len(payload))})
        assert req.headers['Range']=='bytes=4096-'
        return FakeResponse(payload[-32:],status=206,
                headers={'Content-Range':f'bytes 4096-{len(payload)-1}/{len(payload)}',
                         'Content-Length':'32'})
    monkeypatch.setattr(store.urllib.request,'urlopen',get)
    with pytest.raises(store.LyricsModelDownloadError,match='提前结束'):
        store._stream_to_file('https://hf-mirror.com/model',final,min_bytes=20,expected_sha256=sha)
    part=tmp_path/'model.bin.partial'
    assert part.stat().st_size==4096
    assert not final.exists()
    store._stream_to_file('https://hf-mirror.com/model',final,min_bytes=20,expected_sha256=sha)
    assert final.read_bytes()==payload and not part.exists()
    assert len(calls)==2


def test_http_range_ignored_and_restart_fails_retains_original_partial(tmp_path,monkeypatch):
    path=tmp_path/'model.bin'
    old=b'P'*700
    (tmp_path/'model.bin.partial').write_bytes(old)
    def get(req,timeout=25):
        assert req.headers.get('Range')=='bytes=700-'
        return FakeResponse(b'Q'*200,headers={'Content-Length':'300'})
    monkeypatch.setattr(store.urllib.request,'urlopen',get)
    with pytest.raises(store.LyricsModelDownloadError,match='提前结束'):
        store._stream_to_file('https://hf-mirror.com/file',path,min_bytes=20)
    assert (tmp_path/'model.bin.partial').read_bytes()==old
    assert (tmp_path/'model.bin.restart').stat().st_size==200
    assert not path.exists()


def test_http_416_invalid_partial_falls_back_to_fresh_verified_response(tmp_path,monkeypatch):
    path=tmp_path/'model.bin'
    invalid=b'BAD'*200
    correct=b'NEW-SIGNED'*90
    sha=hashlib.sha256(correct).hexdigest()
    (tmp_path/'model.bin.partial').write_bytes(invalid)
    called=[]
    def get(req,timeout=25):
        called.append(req.headers.copy())
        if len(called)==1:
            raise HTTPError(req.full_url,416,'Requested Range Not Satisfiable',{},None)
        assert 'Range' not in req.headers
        return FakeResponse(correct,headers={'Content-Length':str(len(correct))})
    monkeypatch.setattr(store.urllib.request,'urlopen',get)
    store._stream_to_file('https://hf-mirror.com/model',path,min_bytes=20,expected_sha256=sha)
    assert path.read_bytes()==correct and len(called)==2


def test_sha_failure_does_not_delete_near_complete_weight(tmp_path,monkeypatch):
    path=tmp_path/'model.bin'
    wrong=b'WRONG-WEIGHT'*300
    monkeypatch.setattr(store.urllib.request,'urlopen',lambda *args,**kw:FakeResponse(wrong,
                          headers={'Content-Length':str(len(wrong))}))
    with pytest.raises(store.LyricsModelDownloadError,match='SHA-256 不匹配'):
        store._stream_to_file('https://hf-mirror.com/model',path,
                              min_bytes=20,expected_sha256=hashlib.sha256(b'CORRECT').hexdigest())
    assert not path.exists()
    assert (tmp_path/'model.bin.partial').read_bytes()==wrong


def test_validated_full_partial_needs_no_network(tmp_path,monkeypatch):
    path=tmp_path/'model.bin'
    data=b'DONE-WEIGHT'*100
    (tmp_path/'model.bin.partial').write_bytes(data)
    monkeypatch.setattr(store.urllib.request,'urlopen',lambda *a,**kw:(_ for _ in ()).throw(
        AssertionError('must not download a complete validated partial')))
    store._stream_to_file('https://huggingface.co/model',path,min_bytes=20,
                          expected_sha256=hashlib.sha256(data).hexdigest())
    assert path.read_bytes()==data


def test_ui_exposes_pending_sha_and_failed_retry_instead_of_disappearing():
    html=(Path(__file__).parents[1]/'static'/'index.html').read_text('utf-8')
    assert "panel.hidden=!(active||failed)" in html
    assert "rec.state==='verifying'" in html
    assert '模型下载未完成 · 已保存进度' in html
    assert '继续下载所选模型' in html
    assert '模型权重 SHA-256' in (Path(__file__).parents[1]/'lyrics_model_store.py').read_text('utf-8')


def test_previous_build_unverified_model_bin_is_reused_as_range_partial(tmp_path,monkeypatch):
    payload=b'FIRST'*200+b'LAST'*30
    cutoff=len(payload)-60
    original=store.MODELS['tiny']
    monkeypatch.setitem(store.MODELS,'tiny',dict(original,min_bytes=80,sha256=hashlib.sha256(payload).hexdigest()))
    folder=tmp_path/'tiny'
    folder.mkdir()
    (folder/'model.bin').write_bytes(payload[:cutoff])
    for filename in store.REQUIRED_FILES['tiny']:
        if filename.endswith('.json'):
            (folder/filename).write_text('{"metadata":"safe bundled test"}')
        else:
            (folder/filename).write_bytes(b'TOKENIZER FILE LONG ENOUGH')
    requests=[]
    def fetch(req,timeout=25):
        requests.append(req.headers.get('Range'))
        assert requests[-1]==f'bytes={cutoff}-'
        return FakeResponse(payload[cutoff:],status=206,
            headers={'Content-Range':f'bytes {cutoff}-{len(payload)-1}/{len(payload)}',
                     'Content-Length':str(len(payload)-cutoff)})
    monkeypatch.setattr(store.urllib.request,'urlopen',fetch)
    result=store._download_direct_model('tiny',tmp_path,'https://hf-mirror.com')
    assert result==folder
    assert (folder/'model.bin').read_bytes()==payload
    assert len(requests)==1
