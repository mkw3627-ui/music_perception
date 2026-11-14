"""Model file acquisition must not wait for music, never touch private audio."""
from __future__ import annotations
import asyncio
import json
import sys
import threading
import types
from pathlib import Path

import numpy as np
from music_perception.auto_listener import YamnetMusicGate, SAMPLE_RATE
from music_perception.lyrics_prepare_worker import main as prepare_main
from test_plugin import plugin_fixture


class FakeInterpreter:
    def __init__(self, **kwargs):
        self.scores=np.zeros((1,521),dtype=np.float32)
        self.scores[0,132]=.91
    def get_input_details(self):return [{'shape':np.array([15600]),'index':0}]
    def get_output_details(self):return [{'shape':np.array([1,521]),'index':1}]
    def allocate_tensors(self):pass
    def set_tensor(self, index, value):pass
    def invoke(self):pass
    def get_tensor(self, index):return self.scores


def test_gate_reports_quiet_signal_distinct_from_nonmusic(monkeypatch,tmp_path):
    wrapper=types.ModuleType('ai_edge_litert.interpreter')
    wrapper.Interpreter=lambda **kw:FakeInterpreter()
    package=types.ModuleType('ai_edge_litert');package.interpreter=wrapper
    monkeypatch.setitem(sys.modules,'ai_edge_litert',package)
    monkeypatch.setitem(sys.modules,'ai_edge_litert.interpreter',wrapper)
    f=tmp_path/'yamnet.tflite';f.write_bytes(b'fake')
    gate=YamnetMusicGate(f)
    quiet=gate.classify(np.zeros(SAMPLE_RATE*2,dtype=np.float32))
    assert not quiet.music and quiet.reason=='input_too_quiet'
    # A -66 dBFS signal was silently rejected before v0.7.6 without running
    # classification. A strong genuine YAMNet music score must now be tested.
    accepted=gate.classify(np.ones(SAMPLE_RATE*2,dtype=np.float32)*.0005)
    assert accepted.music and accepted.reason=='confirmed_music'
    assert accepted.top_class_index==132 and accepted.top_class_score>.9
    bad=gate.classify(np.ones(SAMPLE_RATE,dtype=np.float32)*float('nan'))
    assert bad.reason in {'invalid_pcm','insufficient_samples'} and not bad.music


def test_prepare_worker_does_not_import_sdk_or_audio(monkeypatch,tmp_path):
    from music_perception import lyrics_model_store
    calls=[]
    monkeypatch.setattr(lyrics_model_store,'ensure_local_model',
        lambda model,root,allow_public_mirror=True:(calls.append((model,root)) or root/model))
    dest=tmp_path/'progress.json';vendor=tmp_path/'vendor';vendor.mkdir()
    monkeypatch.setattr(sys,'argv',['lyrics_prepare_worker.py',str(vendor),
        str(tmp_path/'models'),'base',str(dest),'1'])
    assert prepare_main()==0
    state=json.loads(dest.read_text('utf-8'))
    assert state['state']=='ready' and state['model']=='base'
    assert calls[0][0]=='base'
    assert 'lyrics' not in state and 'audio' not in state


def test_prepare_worker_reports_safe_download_error(monkeypatch,tmp_path):
    from music_perception import lyrics_model_store
    def fail(*args,**kwargs):
        raise ValueError('SECRET_TOKEN=https://bad/very-private-path')
    monkeypatch.setattr(lyrics_model_store,'ensure_local_model',fail)
    dest=tmp_path/'progress.json';vendor=tmp_path/'vendor';vendor.mkdir()
    monkeypatch.setattr(sys,'argv',['lyrics_prepare_worker.py',str(vendor),
        str(tmp_path/'models'),'base',str(dest),'1'])
    assert prepare_main()==1
    payload=dest.read_text('utf-8')
    assert 'SECRET_TOKEN' not in payload
    assert json.loads(payload)['state']=='failed'


def test_model_prepare_can_start_without_music_event(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    class Env:
        def snapshot(self): return {'ready':True}
        def worker_runtime(self): return Path(sys.executable), Path('/tmp')
        def stop(self): pass
    class CompletedChild:
        def __init__(self,*a,**kw):pass
        def wait(self,timeout):return 0
        def poll(self):return 0
    import subprocess
    monkeypatch.setattr(subprocess,'Popen',CompletedChild)
    async def scenario():
        await plugin.startup()
        plugin._environment=Env()
        plugin._lyrics_profile='ultra'
        plugin._lyrics_staged_quality='ultra'
        result=(await plugin.prepare_lyrics_model(confirmed=True)).value
        assert result.get('started')
        thread=plugin._lyrics_prepare_thread;thread.join(timeout=2)
        assert not thread.is_alive()
        assert plugin._lyrics_prepare_state['state']=='ready'
        # Neither YAMNet classification nor speaker access is invoked.
        assert plugin._listener is None
        await plugin.shutdown()
    asyncio.run(scenario())


def test_diagnostics_now_include_meter_and_gate_reason(plugin_fixture):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        from types import SimpleNamespace
        plugin._listener=SimpleNamespace(stop=lambda:True,status=lambda:{
           'status':'listening','device':'ToDesk Virtual Audio',
           'last_music_score':0.0,'last_gate_reason':'input_too_quiet',
           'input_level_dbfs':-92.0})
        r=(await plugin.get_lyrics_diagnostics()).value
        assert r['audio_pipeline']['gate_reason']=='input_too_quiet'
        assert r['audio_pipeline']['input_level_dbfs']==-92.0
        assert 'model_preparation' in r
        await plugin.shutdown()
    asyncio.run(scenario())


def test_quality_switch_cancels_old_model_preparation(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    class Env:
        def snapshot(self):return {'ready':True}
        def worker_runtime(self):return Path(sys.executable), Path('/tmp')
        def stop(self):pass
    class Child:
        def __init__(self, argv,**kw):
            self.argv=argv
            self.exit=threading.Event()
            self.cancelled=False
        def wait(self,timeout):
            if self.argv[-3]=='base':
                self.exit.wait(timeout=2)
                return -15 if self.cancelled else 0
            return 0
        def poll(self):return None if not self.exit.is_set() else -15
        def terminate(self):self.cancelled=True;self.exit.set()
    children=[]
    def factory(args,**kw):
        # Only count model-preparation workers. Other Popen users in the same
        # process (for example GPU telemetry via nvidia-smi) are unrelated.
        if not any('lyrics_prepare_worker' in str(a) for a in args):
            return real_popen(args,**kw)
        child=Child(args,**kw);children.append(child)
        return child
    import subprocess
    real_popen=subprocess.Popen
    monkeypatch.setattr(subprocess,'Popen',factory)
    async def scenario():
        await plugin.startup();plugin._environment=Env()
        plugin._lyrics_profile='medium';plugin._auto_consented=True
        assert plugin._schedule_lyrics_model_prepare()['started']
        for _ in range(200):
            if children:break
            await asyncio.sleep(.001)
        assert children and children[0].argv[-3]=='base'
        plugin._lyrics_profile='low'
        assert plugin._schedule_lyrics_model_prepare()['queued_for_switch']
        for _ in range(200):
            if len(children)>=2 and plugin._lyrics_prepare_state['state']=='ready':break
            await asyncio.sleep(.001)
        assert len(children)==2 and children[0].cancelled
        assert children[1].argv[-3]=='tiny'
        assert plugin._lyrics_prepare_state['model']=='tiny'
        plugin._auto_consented=False
        await plugin.shutdown()
    asyncio.run(scenario())
