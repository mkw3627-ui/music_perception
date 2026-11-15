"""Regressions for WinError 32 on ORT CUDA diagnostic profile cleanup.

These isolate the Windows handle lifecycle without needing Windows, GPUs, or
large weights; hardware CUDA inference remains an explicit real-machine test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pytest

from music_perception import stem_gpu_probe_worker as probe
from music_perception import stem_mdx


def _trace(root):
    path=root/'stem_cuda_probe_fake.json'
    path.write_text(json.dumps([{'cat':'Node','name':'Conv_1_kernel_time',
                 'args':{'provider':'CUDAExecutionProvider'}}]), encoding='utf-8')
    return path


def test_cuda_profile_json_lock_retries_and_reads(monkeypatch, tmp_path):
    path=_trace(tmp_path)
    original=Path.read_text
    attempts=[]
    def reader(self,*args,**kw):
        if self == path:
            attempts.append(1)
            if len(attempts)<3:
                raise PermissionError(32, 'WinError32: file being used')
        return original(self,*args,**kw)
    monkeypatch.setattr(Path,'read_text',reader)
    monkeypatch.setattr(probe.time,'sleep',lambda _:None)
    assert probe.cuda_profile_evidence(path)['cuda_nodes']==1
    assert len(attempts)==3


def test_cuda_profile_json_permanent_lock_returns_evidence_error(monkeypatch,tmp_path):
    path=_trace(tmp_path)
    original=Path.read_text
    def reader(self,*args,**kw):
        if self==path:raise PermissionError(32,'WinError32')
        return original(self,*args,**kw)
    monkeypatch.setattr(Path,'read_text',reader)
    monkeypatch.setattr(probe.time,'sleep',lambda _:None)
    result=probe.cuda_profile_evidence(path)
    assert result['cuda_nodes']==0
    assert 'locked' in result['error']


def test_profile_cleanup_retries_windows_file_lock(monkeypatch,tmp_path):
    root=tmp_path/'owned';root.mkdir()
    (root/'sample.json').write_text('{}')
    actual=probe.shutil.rmtree
    attempts=[]
    def locked_rmtree(path):
        attempts.append(1)
        if len(attempts)<3:
            raise PermissionError(32,'file in use')
        return actual(path)
    monkeypatch.setattr(probe.shutil,'rmtree',locked_rmtree)
    monkeypatch.setattr(probe.time,'sleep',lambda _:None)
    assert probe._remove_profile_dir(root)
    assert not root.exists()
    assert len(attempts)==3


def test_permanent_profile_lock_is_not_gpu_failure(monkeypatch,tmp_path):
    original=probe.shutil.rmtree
    def always_locked(path):
        raise PermissionError(32,'file in use')
    monkeypatch.setattr(probe.shutil,'rmtree',always_locked)
    monkeypatch.setattr(probe.time,'sleep',lambda _:None)
    assert probe._remove_profile_dir(tmp_path,retries=2) is False
    monkeypatch.setattr(probe.shutil,'rmtree',original)


def test_gpu_session_is_released_before_deleting_its_profile(monkeypatch,tmp_path):
    released=[]
    class FakeORTSession:
        def __init__(self, root):self.root=root
        def end_profiling(self):return str(_trace(self.root))
    class FakeMDX:
        def __init__(self, model_path, request_gpu=False, profile_dir=None):
            assert request_gpu
            self.session=FakeORTSession(profile_dir)
        def separate(self, *args, **kwargs):return {'device':'cuda'}
        def __del__(self):released.append(True)
    monkeypatch.setattr(stem_mdx,'TwoStemMDX',FakeMDX)
    monkeypatch.setattr(probe,'_clean_old_profile_dirs',lambda:None)
    original=probe._remove_profile_dir
    def removal(path):
        assert released, 'ONNX session still referenced during profile deletion'
        return original(path)
    monkeypatch.setattr(probe,'_remove_profile_dir',removal)
    result={'ok':False,'real_inference':False,'stage':'start'}
    probe.run_real_cuda_probe(tmp_path/'dummy.onnx',result)
    assert result['ok'] is True and result['real_inference'] is True
    assert result['profile_evidence']['cuda_nodes']==1
    assert 'profile_cleanup_warning' not in result


def test_gpu_profile_cleanup_lock_keeps_verified_inference(monkeypatch,tmp_path):
    class FakeORTSession:
        def __init__(self,root):self.root=root
        def end_profiling(self):return str(_trace(self.root))
    class FakeMDX:
        def __init__(self,*a,profile_dir=None,**kw):self.session=FakeORTSession(profile_dir)
        def separate(self,*a,**kw):return {'device':'cuda'}
    monkeypatch.setattr(stem_mdx,'TwoStemMDX',FakeMDX)
    monkeypatch.setattr(probe,'_clean_old_profile_dirs',lambda:None)
    monkeypatch.setattr(probe,'_remove_profile_dir',lambda _path:False)
    result={'ok':False,'real_inference':False,'stage':'start'}
    probe.run_real_cuda_probe(tmp_path/'dummy.onnx',result)
    assert result['ok'] is True
    assert result['stage']=='complete'
    assert result['real_inference'] is True
    assert 'Windows' in result['profile_cleanup_warning']


def test_cleanup_old_profiles_does_not_touch_current_probe(tmp_path):
    recent=tmp_path/(probe.PROFILE_PREFIX+'recent')
    recent.mkdir();_trace(recent)
    old=tmp_path/(probe.PROFILE_PREFIX+'old')
    old.mkdir();_trace(old)
    timestamp=time.time()-7200
    os.utime(old,(timestamp,timestamp))
    other=tmp_path/'completely_unrelated_folder'
    other.mkdir()
    probe._clean_old_profile_dirs(tmp_path,min_age_s=3600)
    assert recent.exists()
    assert not old.exists()
    assert other.exists()


def test_invalid_profile_never_counts_as_cuda(tmp_path):
    path=tmp_path/'bad.json'
    path.write_text('<html>blocked</html>')
    result=probe.cuda_profile_evidence(path)
    assert result['cuda_nodes']==0
    assert result.get('error')
