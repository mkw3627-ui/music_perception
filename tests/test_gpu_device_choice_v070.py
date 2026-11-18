"""v0.7.0: user-selected CPU / CUDA; no weights, sound capture, or network."""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import numpy as np
from music_perception.lyrics_backend import transcribe_singing_candidate
from music_perception.lyrics_bridge import PersistentLyricsSession
from music_perception.lyrics_device_capabilities import probe_cuda_isolated
from music_perception.lyrics_device_probe import inspect_cuda
from test_low_resource_lyrics_v064 import _make_fake_stream_worker


def audio():
    t=np.arange(16000,dtype=np.float32) / 16000
    return (0.06*np.sin(2*np.pi*220*t)).astype(np.float32)


class FakeModel:
    def transcribe(self,*args,**kwargs):
        piece=SimpleNamespace(text='识别候选',start=.2,end=.7,avg_logprob=-.1,
                              no_speech_prob=.05,compression_ratio=1.0,words=[])
        return [piece],SimpleNamespace(language='zh')


def test_explicit_cpu_only_never_initializes_cuda(tmp_path):
    calls=[]
    def make(name,**opts):
        calls.append(opts)
        assert opts['device']=='cpu' and opts['compute_type']=='int8'
        return FakeModel()
    report=transcribe_singing_candidate(audio(),model_dir=tmp_path,
            model_factory=make,device_policy='cpu')
    assert report['device']=='cpu' and report['requested_device']=='cpu'
    assert not report['gpu_fallback'] and report['device_note'] is None
    assert len(calls)==1


def test_explicit_cuda_fp16_success_and_model_cache_reuse(tmp_path):
    calls=[]
    cache={}
    def make(name,**opts):
        calls.append(opts)
        assert opts['device']=='cuda' and opts['compute_type']=='float16'
        return FakeModel()
    for _ in range(2):
        report=transcribe_singing_candidate(audio(),model_dir=tmp_path,
            model_factory=make,device_policy='cuda',model_cache=cache)
        assert report['device']=='cuda' and not report['gpu_fallback']
        assert report['requested_device']=='cuda'
    assert len(calls)==1, 'Do not reload GPU weights per lyric chunk'


def test_unavailable_cuda_falls_back_with_visible_warning_once(tmp_path):
    calls=[]
    cache={}
    def make(name,**opts):
        calls.append(opts['device'])
        if opts['device']=='cuda':
            raise RuntimeError('missing CUDA library')
        return FakeModel()
    for _ in range(2):
        report=transcribe_singing_candidate(audio(),model_dir=tmp_path,
            model_factory=make,device_policy='cuda',model_cache=cache)
        assert report['device']=='cpu' and report['gpu_fallback']
        assert '回退' in report['device_note']
    assert calls==['cuda','cpu'], 'broken GPU must not retry on next lyric segment'


def test_cuda_inference_failure_also_falls_back_and_is_cached(tmp_path):
    calls=[]
    cache={}
    class BrokenGPU:
        def transcribe(self,*args,**kwargs):
            raise RuntimeError('cudnn failed')
    def make(name,**opts):
        calls.append(opts['device'])
        return BrokenGPU() if opts['device']=='cuda' else FakeModel()
    for _ in range(2):
        report=transcribe_singing_candidate(audio(),model_dir=tmp_path,
            model_factory=make,device_policy='cuda',model_cache=cache)
        assert report['device']=='cpu' and report['gpu_fallback']
        assert '推理' in report['device_note']
    assert calls==['cuda','cpu']


def test_gpu_capability_probe_read_only(tmp_path):
    assert not inspect_cuda(SimpleNamespace(get_cuda_device_count=lambda:0))['cuda_candidate']
    ok=SimpleNamespace(get_cuda_device_count=lambda:1,
                       get_supported_compute_types=lambda dev: {'int8_float16','float16'})
    report=inspect_cuda(ok)
    assert report['cuda_candidate'] and report['gpu_count']==1
    fail=SimpleNamespace(get_cuda_device_count=lambda:1,
                         get_supported_compute_types=lambda dev: (_ for _ in ()).throw(OSError('cuDNN')))
    assert not inspect_cuda(fail)['cuda_candidate']


def test_isolated_gpu_probe_reads_only_plugin_vendor(tmp_path):
    vendor=tmp_path/'vendor'
    vendor.mkdir()
    source=tmp_path/'worker.py'
    from music_perception.lyrics_device_probe import __file__ as src
    source.write_bytes(__import__('pathlib').Path(src).read_bytes())
    (vendor/'ctranslate2.py').write_text('def get_cuda_device_count(): return 1\ndef get_supported_compute_types(device): return {"float16", "int8"}\n')
    output=probe_cuda_isolated(python_exe=sys.executable,vendor=vendor,script=source,timeout=5)
    assert output['cuda_candidate'] and output['gpu_count']==1
    # Missing DLLs/gpu are not an error to the rest of the plugin.
    (vendor/'ctranslate2.py').write_text('def get_cuda_device_count(): return 0\n')
    output=probe_cuda_isolated(python_exe=sys.executable,vendor=vendor,script=source,timeout=5)
    assert not output['cuda_candidate']


def test_persistent_worker_accepts_explicit_cuda_and_sends_choice(tmp_path):
    vendor=tmp_path/'vendor'
    vendor.mkdir()
    worker=tmp_path/'stream.py'
    _make_fake_stream_worker(worker)
    session=PersistentLyricsSession(python_exe=sys.executable,vendor=vendor,
                                    worker_script=worker,model_dir=tmp_path/'models',
                                    device='cuda',cpu_threads=2,timeout=5)
    try:
        result=session.transcribe(audio())
        assert result['device']=='cuda'
    finally:
        session.close()


def test_unverified_gpu_switch_is_rejected_without_interrupting_music(plugin_fixture):
    """v0.8.2 forbids changing device before validated weight + preflight."""
    plugin,_,_,_=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._lyrics_profile='medium'
        plugin._live_session_config={'lyrics_model':'base','lyrics_device':'cpu'}
        keep_dsp=SimpleNamespace(stop=lambda:None)
        plugin._listener=keep_dsp
        plugin._song_current_id='playing:1'
        class FakeSession:
            closed=False
            def close(self):
                self.closed=True
        old=FakeSession()
        plugin._lyrics_session=old
        before=plugin._lyrics_quality_generation
        previous_device=plugin._lyrics_device_preference
        reply=(await plugin.set_lyrics_device('cuda')).value
        assert not reply['ok'] and 'SHA' in reply['error']
        assert plugin._lyrics_device_preference==previous_device
        assert not old.closed and plugin._lyrics_session is old
        assert plugin._lyrics_quality_generation==before
        assert plugin._listener is keep_dsp and plugin._song_current_id=='playing:1'
        assert plugin._live_session_config['lyrics_device']=='cpu'
        assert (await plugin.set_lyrics_device('no-such-gpu')).value['ok'] is False
        await plugin.shutdown()
    asyncio.run(run())


def test_probe_action_uses_worker_runtime_not_neko_executable(plugin_fixture,monkeypatch):
    plugin,_,_,_=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._environment.worker_runtime=lambda: (sys.executable, plugin.data_path("environment"))
        module=sys.modules[plugin.__class__.__module__]
        seen=[]
        def fake_probe(**kw):
            seen.append(kw)
            return {'cuda_candidate':True, 'gpu_count':1,'message':'GPU preflight only'}
        monkeypatch.setattr(module,'probe_cuda_isolated',fake_probe)
        report=(await plugin.check_lyrics_gpu()).value
        assert report['ok'] and report['cuda_candidate']
        assert seen and seen[0]['script'].name=='lyrics_device_probe.py'
        assert (await plugin.get_status()).value['lyrics_gpu_probe']['cuda_candidate']
        await plugin.shutdown()
    asyncio.run(run())


def test_unverified_model_choice_does_not_change_saved_quality(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def run():
        await plugin.startup()
        before=plugin._lyrics_profile
        no_weight=(await plugin.set_lyrics_quality('high')).value
        assert not no_weight['ok']
        assert plugin._lyrics_profile==before
        assert (await plugin.set_lyrics_device('cuda')).value['ok'] is False
        cfg=await plugin._settings()
        assert cfg['lyrics_model']=='base' and cfg['lyrics_device']=='cpu'
        await plugin.shutdown()
    asyncio.run(run())


def test_native_cuda_worker_crash_retries_once_on_cpu_without_stream_restart(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._environment.worker_runtime=lambda: (sys.executable,plugin.data_path('environment'))
        plugin._live_session_config={'lyrics_model':'base','lyrics_device':'cuda'}
        # This legacy crash/retry scenario explicitly authorizes RAM/CPU
        # fallback. v0.9.7 never silently falls back when consent is absent.
        plugin._resource_cpu_consent['video']=True
        plugin._resource_choice=lambda domain: {
            'state':'ready','quality':'medium','device':'cuda',
            'preferred_quality':'medium','reason':'mock_verified_local_model'}
        gpu_cpu=[]
        class FakeLyricsSession:
            def __init__(self,**kw):
                self.device=kw['device']
                gpu_cpu.append(self.device)
            def transcribe(self,samples,*,language='auto'):
                assert language=='auto'  # v0.7.2 adds per-clip language to the worker protocol
                if self.device=='cuda':
                    from music_perception.lyrics_bridge import LyricsWorkerError
                    raise LyricsWorkerError('native CUDA DLL terminated worker')
                return {'status':'no_reliable_lyrics','timed_cues':[],
                        'device':'cpu','requested_device':'cpu'}
            def close(self):
                pass
        from unittest.mock import patch
        module=sys.modules[plugin.__class__.__module__]
        with patch.object(module,'PersistentLyricsSession',FakeLyricsSession):
            result=plugin._transcribe_live_audio_isolated(audio(),plugin._live_session_config)
            again=plugin._transcribe_live_audio_isolated(audio(),plugin._live_session_config)
        assert gpu_cpu==['cuda','cpu']
        assert result['device']=='cpu' and result['gpu_fallback']
        assert 'CPU' in result['device_note']
        assert again['device']=='cpu'
        assert plugin._live_session_config['lyrics_device']=='cuda'
        await plugin.shutdown()
    asyncio.run(run())
