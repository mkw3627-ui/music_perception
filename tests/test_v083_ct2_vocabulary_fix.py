"""v0.8.3 regression: CT2 vocabulary.json is an ARRAY, not a JSON object.

These are offline, synthetic tests: no model download, Windows or GPU access.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from music_perception import lyrics_model_store as store


def ct2_vocab() -> bytes:
    # CTranslate2 conversion serializes List[str], not Dict[str, int].
    return json.dumps([f"token_{i}" for i in range(1200)]).encode("utf-8")


def make_turbo(tmp_path, monkeypatch):
    payload = b"SYNTHETIC-WEIGHTS" * 100
    original = store.MODELS['large-v3-turbo']
    monkeypatch.setitem(store.MODELS, 'large-v3-turbo', dict(
        original, min_bytes=400, sha256=hashlib.sha256(payload).hexdigest()))
    folder = tmp_path / 'large-v3-turbo'
    folder.mkdir()
    (folder/'model.bin').write_bytes(payload)
    (folder/'config.json').write_text('{"n_mels": 128, "test": true}')
    (folder/'tokenizer.json').write_text('{"model": {"type": "BPE"}}')
    (folder/'preprocessor_config.json').write_text('{"feature_size": 128}')
    (folder/'vocabulary.json').write_bytes(ct2_vocab())
    return folder


def test_valid_ct2_token_list_verifies_offline_and_never_refetches(tmp_path, monkeypatch):
    folder = make_turbo(tmp_path, monkeypatch)
    valid, reason = store.check_local_model(folder, 'large-v3-turbo')
    assert (valid, reason) == (True, 'verified')
    assert store.inspect_model_files('large-v3-turbo', tmp_path)['state'] == 'present_unverified'
    def forbid_network(*args, **kwargs):
        raise AssertionError('A verified model must not be re-downloaded')
    monkeypatch.setattr(store.urllib.request, 'urlopen', forbid_network)
    assert store.ensure_local_model('large-v3-turbo', tmp_path) == folder


def test_existing_valid_json_array_is_not_falsely_rejected_by_direct_downloader(tmp_path, monkeypatch):
    folder = make_turbo(tmp_path, monkeypatch)
    calls = []
    def forbid_stream(*args, **kwargs):
        calls.append(args)
        raise AssertionError('No sidecar or weight should be re-downloaded')
    monkeypatch.setattr(store, '_stream_to_file', forbid_stream)
    assert store._download_direct_model('large-v3-turbo', tmp_path, 'https://hf-mirror.com') == folder
    assert not calls


class FakeResponse:
    def __init__(self, payload, content_type='application/json'):
        self.status = 200
        self.headers = {'Content-Length':str(len(payload)), 'Content-Type':content_type}
        self.stream = io.BytesIO(payload)
    def __enter__(self):return self
    def __exit__(self,*_):return False
    def read(self,n):return self.stream.read(n)
    def geturl(self):return 'https://hf-mirror.com/example'
    def getcode(self):return self.status


def test_mirror_html_error_cannot_overwrite_existing_sidecar(tmp_path,monkeypatch):
    file=tmp_path/'vocabulary.json'
    original=ct2_vocab()
    file.write_bytes(original)
    monkeypatch.setattr(store.urllib.request, 'urlopen',
        lambda *_a,**_kw:FakeResponse(b'<html><h1>cloudflare error</h1></html>', 'text/html'))
    with pytest.raises(store.LyricsModelDownloadError, match='错误页面'):
        store._stream_to_file('https://hf-mirror.com/pinned/vocabulary.json',file,
            file_validator=lambda p:store.validate_sidecar_file(p,'vocabulary.json'))
    assert file.read_bytes()==original


def test_mirror_error_json_is_rejected_before_promotion_and_no_poisoned_resume(tmp_path,monkeypatch):
    file=tmp_path/'vocabulary.json'
    malicious=b'{"error": "temporary upstream failure, please retry this download"}'
    calls=[]
    valid=ct2_vocab()
    def network(request,**_):
        calls.append(request.headers.copy())
        return FakeResponse(malicious if len(calls)==1 else valid)
    monkeypatch.setattr(store.urllib.request,'urlopen',network)
    with pytest.raises(store.LyricsModelDownloadError,match='JSON 词元数组'):
        store._stream_to_file('https://hf-mirror.com/vocabulary.json',file,
            file_validator=lambda p:store.validate_sidecar_file(p,'vocabulary.json'))
    assert not file.exists()
    assert (tmp_path/'vocabulary.json.rejected').read_bytes()==malicious
    assert not (tmp_path/'vocabulary.json.partial').exists()
    store._stream_to_file('https://huggingface.co/vocabulary.json',file,
        file_validator=lambda p:store.validate_sidecar_file(p,'vocabulary.json'))
    assert file.read_bytes()==valid
    assert len(calls)==2
    assert 'Range' not in calls[1]


def test_empty_array_or_object_not_accepted_as_real_vocabulary(tmp_path):
    file=tmp_path/'vocabulary.json'
    for payload in ('{}', '[]', '{"error":"rate limit"}', '["only one token"]',
                    '<html>network error</html>'):
        file.write_text(payload,encoding='utf-8')
        ok,why=store.validate_sidecar_file(file,'vocabulary.json')
        assert not ok, (payload,why)


def test_invalid_tokenizer_document_stays_unverified(tmp_path,monkeypatch):
    folder=make_turbo(tmp_path,monkeypatch)
    (folder/'tokenizer.json').write_text('<html>proxy access denied</html>')
    assert store.check_local_model(folder,'large-v3-turbo')[0] is False
    assert store.inspect_model_files('large-v3-turbo',tmp_path)['state']=='incomplete'


def test_url_repository_pinned_turbo_revision_is_still_the_same():
    url=store._direct_file_url('https://huggingface.co','large-v3-turbo','vocabulary.json')
    assert '/resolve/44cbbd1adefe7387c83df88963a6d9ac4c9adea5/vocabulary.json' in url
