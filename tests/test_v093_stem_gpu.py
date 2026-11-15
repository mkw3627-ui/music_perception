"""v0.9.3 regressions: ONNX GPU provider, fallback, private install and consent.

No real CUDA or network is needed; on a user's Windows PC a real model inference
must still run before reporting GPU verified.
"""
from __future__ import annotations
import asyncio
import hashlib
import io
import json
from pathlib import Path
import sys
import types
import zipfile

import numpy as np
import pytest

from music_perception import stem_gpu_runtime as gpu
from music_perception import stem_mdx
from test_plugin import plugin_fixture


def _fake_ort(monkeypatch, *, available=None, active=None, crash=False):
    providers=available if available is not None else ['CPUExecutionProvider']
    active=active if active is not None else providers
    class Options:
        intra_op_num_threads=0
        inter_op_num_threads=0
        graph_optimization_level=None
    class Inp:
        name='input'
        shape=[1,4,3072,256]
    class Session:
        def __init__(self,*a,**k):self.disabled=False
        def get_providers(self):return active
        def get_inputs(self):return [Inp()]
        def disable_fallback(self):self.disabled=True
        def run(self,names,feed):
            if crash:raise RuntimeError('native cudnn model failure')
            return [feed['input']]
    module=types.ModuleType('onnxruntime')
    module.__version__='1.26.0'
    module.get_available_providers=lambda:providers
    module.InferenceSession=Session
    module.SessionOptions=Options
    module.GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL='all')
    module.preload_dlls=lambda:None
    monkeypatch.setitem(sys.modules,'onnxruntime',module)
    monkeypatch.setattr(stem_mdx,'verify_model',lambda path:True)
    return module


def test_gpu_missing_provider_is_never_silent_cpu(monkeypatch,tmp_path):
    _fake_ort(monkeypatch,available=['CPUExecutionProvider'])
    with pytest.raises(stem_mdx.StemGpuUnavailable,match='CUDAExecutionProvider'):
        stem_mdx.TwoStemMDX(tmp_path/'UVR-MDX-NET-Inst_HQ_3.onnx',request_gpu=True)
    cpu=stem_mdx.TwoStemMDX(tmp_path/'UVR-MDX-NET-Inst_HQ_3.onnx',request_gpu=False)
    assert cpu.device=='cpu'


def test_gpu_provider_cannot_claim_cuda_if_session_reverted_cpu(monkeypatch,tmp_path):
    _fake_ort(monkeypatch,available=['CUDAExecutionProvider','CPUExecutionProvider'],active=['CPUExecutionProvider'])
    with pytest.raises(stem_mdx.StemGpuUnavailable,match='回退 CPU'):
        stem_mdx.TwoStemMDX(tmp_path/'UVR-MDX-NET-Inst_HQ_3.onnx',request_gpu=True)


def test_gpu_session_disables_silent_ort_fallback(monkeypatch,tmp_path):
    _fake_ort(monkeypatch,available=['CUDAExecutionProvider','CPUExecutionProvider'])
    model=stem_mdx.TwoStemMDX(tmp_path/'UVR-MDX-NET-Inst_HQ_3.onnx',request_gpu=True)
    assert model.session.disabled
    assert model.device=='cuda'


def test_real_cuda_inference_failure_is_reported(monkeypatch,tmp_path):
    _fake_ort(monkeypatch,available=['CUDAExecutionProvider','CPUExecutionProvider'],crash=True)
    model=stem_mdx.TwoStemMDX(tmp_path/'UVR-MDX-NET-Inst_HQ_3.onnx',request_gpu=True)
    with pytest.raises(stem_mdx.StemGpuUnavailable,match='实际推理失败'):
        model.separate(np.zeros((44100,2),dtype=np.float32),44100,strength='off')


def test_overlay_inactive_without_verification_marker(tmp_path):
    folder=gpu.gpu_root(tmp_path)/gpu.GPU_ACTIVE_NAME
    (folder/'onnxruntime'/'capi').mkdir(parents=True)
    (folder/'onnxruntime'/'__init__.py').write_text('')
    (folder/'onnxruntime'/'capi'/'onnxruntime_providers_cuda.dll').write_bytes(b'fake')
    assert gpu.overlay_path(tmp_path) is None
    (folder/gpu.MARKER).write_text(json.dumps({'wheel':gpu.GPU_WHEEL,'sha256':'not expected'}))
    assert gpu.overlay_path(tmp_path) is None


def _make_mock_gpu_wheel(tmp_path):
    whl=tmp_path/'fake_verified.whl'
    with zipfile.ZipFile(whl,'w') as arc:
        arc.writestr('onnxruntime/__init__.py','__version__="fake"')
        arc.writestr('onnxruntime/capi/onnxruntime_providers_cuda.dll','placeholder')
        arc.writestr('onnxruntime_gpu-1.26.0.dist-info/METADATA','Name: onnxruntime-gpu')
    return whl


def test_gpu_wheel_overlay_never_changes_audio_vendor(monkeypatch,tmp_path):
    whl=_make_mock_gpu_wheel(tmp_path)
    digest=hashlib.sha256(whl.read_bytes()).hexdigest()
    monkeypatch.setattr(gpu,'GPU_WHEEL_SHA256',digest)
    monkeypatch.setattr(gpu,'_check_platform',lambda:None)
    from music_perception import stem_cuda_bundle as private_cuda
    def install_simulated_cuda(base, status, priority):
        target=private_cuda._folder(base)
        target.mkdir(parents=True,exist_ok=True)
        for spec in private_cuda.CUDA_WHEELS:
            folder=target/'nvidia'/spec['family']/'bin'
            folder.mkdir(parents=True,exist_ok=True)
            for dll in spec['dlls']:
                (folder/dll).write_bytes(b'v' * 2048)
        (target/private_cuda.CUDA_MARKER).write_text(json.dumps({
            'version':private_cuda.CUDA_BUNDLE_VERSION,
            'wheels':{private_cuda.wheel_name(w):w['sha256'] for w in private_cuda.CUDA_WHEELS}}))
        return True
    monkeypatch.setattr(private_cuda,'install_cuda_runtime',install_simulated_cuda)
    monkeypatch.setattr(gpu,'_index_urls',lambda src:[('test','https://example.test/index')])
    monkeypatch.setattr(gpu,'_fetch_source',lambda url:'https://example.test/wheel')
    from music_perception import lyrics_model_store
    monkeypatch.setattr(lyrics_model_store,'_stream_to_file',lambda url, dest, **kw:
                        dest.write_bytes(whl.read_bytes()))
    base=tmp_path/'environment'
    vendor=base/'vendor';vendor.mkdir(parents=True)
    (vendor/'onnxruntime_cpu_marker.txt').write_text('do not modify')
    status=base/'gpu-status.json'
    assert gpu.install_gpu(base,status,'mirror')
    overlay=gpu.overlay_path(base)
    assert overlay and (overlay/'onnxruntime'/'__init__.py').is_file()
    assert (vendor/'onnxruntime_cpu_marker.txt').read_text()=='do not modify'
    assert json.loads(status.read_text(encoding='utf-8'))['state']=='ready'
    # A second install always reuses the already verified wheel and runtime.
    assert gpu.install_gpu(base,status,'mirror')


def test_wheel_safe_extract_rejects_traversal(tmp_path):
    wheel=tmp_path/'malicious.whl'
    with zipfile.ZipFile(wheel,'w') as archive:
        archive.writestr('../escape.txt','bad')
    with pytest.raises(gpu.StemGpuInstallError,match='unsafe'):
        gpu._install_wheel(wheel,tmp_path/'install')
    assert not (tmp_path/'escape.txt').exists()


def test_gpu_installer_requires_user_confirmation(plugin_fixture):
    plugin,*_=plugin_fixture
    async def case():
        await plugin.startup()
        result=(await plugin.install_stem_gpu(confirmed=False)).value
        assert result['ok'] is False
        assert plugin._stem_gpu_install_thread is None
        await plugin.shutdown()
    asyncio.run(case())


def test_status_exposes_gpu_package_and_probe_without_running_network(plugin_fixture):
    plugin,*_=plugin_fixture
    async def case():
        await plugin.startup()
        gpu_status=(await plugin.get_status(_for_ui=True)).value['stem_separation']['gpu_runtime']
        assert gpu_status['installed'] is False
        assert gpu_status['probe']['state']=='idle'
        assert gpu_status['wheel_bytes_estimate']>=200_000_000
        await plugin.shutdown()
    asyncio.run(case())


def test_GPU_probe_requires_managed_environment(plugin_fixture):
    plugin,*_=plugin_fixture
    def _not_ready():
        raise RuntimeError('managed environment not ready')
    async def case():
        await plugin.startup()
        # Simulate an unprovisioned managed environment for this case only.
        plugin._environment.worker_runtime=_not_ready
        result=(await plugin.check_stem_gpu()).value
        assert result['ok'] is False
        assert '环境' in result['error']
        await plugin.shutdown()
    asyncio.run(case())


def test_gpu_source_priority_is_known_and_not_arbitrary():
    mirror=gpu._index_urls('mirror')
    official=gpu._index_urls('official')
    assert mirror[0][0]=='清华镜像'
    assert official[0][0]=='PyPI 官方'
    with pytest.raises(gpu.StemGpuInstallError):gpu._index_urls('other')


def test_stem_worker_gpu_flag_is_passed_to_child_and_device_reported(tmp_path):
    source=Path(__file__).parents[1]
    worker=(source/'stem_worker.py').read_text('utf8')
    assert 'loaded_for_gpu != wants_gpu' in worker  # no stale CPU session
    assert 'gpu_fallback_reason' in worker
    assert 'cuda' in (source/'stem_bridge.py').read_text('utf8')


def test_gpu_profiler_must_exhibit_cuda_nodes_to_claim_real_gpu(tmp_path):
    from music_perception.stem_gpu_probe_worker import cuda_profile_evidence
    trace=tmp_path/'profile.json'
    events=[{'cat':'Node','name':'Conv_1_kernel_time','args':{'provider':'CPUExecutionProvider'}},
            {'cat':'Session','name':'session','args':{'provider':'CUDAExecutionProvider'}}]
    trace.write_text(json.dumps(events))
    assert cuda_profile_evidence(trace)=={'cuda_nodes':0,'cpu_nodes':1}
    events.append({'cat':'Node','name':'Conv_3_kernel_time','args':{'provider':'CUDAExecutionProvider'}})
    trace.write_text(json.dumps(events))
    assert cuda_profile_evidence(trace)=={'cuda_nodes':1,'cpu_nodes':1}
    trace.write_text(json.dumps({'not':'an onnx runtime event list'}))
    assert cuda_profile_evidence(trace)['cuda_nodes']==0


def test_gpu_probe_is_actual_inference_only_with_model_and_cuda_profile():
    probe=(Path(__file__).parents[1]/'stem_gpu_probe_worker.py').read_text('utf8')
    assert 'separator.session.end_profiling()' in probe
    assert "evidence['cuda_nodes'] > 0" in probe
    assert 'run_real_cuda_probe(' in probe
    assert 'gc.collect()' in probe
    assert 'tempfile.mkdtemp(' in probe


def test_interrupted_gpu_wheel_download_is_not_shown_as_running(plugin_fixture):
    plugin,*_=plugin_fixture
    async def case():
        await plugin.startup()
        state=plugin.data_path('environment')/'stem_gpu_install_status.json'
        state.parent.mkdir(parents=True,exist_ok=True)
        state.write_text(json.dumps({'schema':'stem_gpu_install_v1','state':'running','stage':'downloading'}),encoding='utf8')
        snap=plugin._stem_gpu_install_status()
        assert snap['running'] is False
        assert snap['install']['state']=='interrupted'
        await plugin.shutdown()
    asyncio.run(case())
