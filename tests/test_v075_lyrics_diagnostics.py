"""v0.7.5 runtime diagnostics must be bounded, private, and non-blocking."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from music_perception.lyrics_diagnostics import publish_stage, read_stage, diagnose_stall
from music_perception.lyrics_backend import transcribe_singing_candidate
from music_perception.lyrics_health_probe import run_probe
from test_plugin import plugin_fixture


def test_diagnostics_stage_and_privacy(tmp_path):
    f=tmp_path/'lyrics_worker_status.json'
    publish_stage(f,'loading_cuda',job=4,model='base',requested_device='cuda',
                  error_type='some_private_message_do_not_log')
    state=read_stage(f)
    assert state['stage']=='loading_cuda' and state['job']==4
    assert state['requested_device']=='cuda' and state['actual_device'] is None
    assert state['error_type'] is None
    lines=(tmp_path/'lyrics_diagnostics.log').read_text('utf-8').splitlines()
    assert len(lines)==1
    assert 'some_private' not in lines[0]
    assert 'text_preview' not in lines[0] and 'pcm' not in lines[0].lower()
    assert read_stage(f,max_age=-1) is None


def test_stall_detection_does_not_assert_gpu_failure():
    warning=diagnose_stall(stage='loading_cuda',stage_age_seconds=65,
                           queue_depth=2,completed_count=0,child_running=True)
    assert '可能' in warning and '超过45秒' in warning
    assert 'GPU' not in warning or 'CUDA' in warning
    warning=diagnose_stall(stage='decoding',stage_age_seconds=45,
                           queue_depth=2,completed_count=0,child_running=True)
    assert '推理' in warning
    assert '未检测到' in diagnose_stall(stage=None,stage_age_seconds=0,
                           queue_depth=2,completed_count=0,child_running=False)


def test_whisper_hook_records_model_and_decoding(tmp_path):
    steps=[]
    class Model:
        def transcribe(self, audio, **k):
            return ([SimpleNamespace(text='合成句',start=0,end=1,avg_logprob=-.1,
                         no_speech_prob=.05,compression_ratio=1.0,words=[])],
                    SimpleNamespace(language='zh'))
    result=transcribe_singing_candidate(np.ones(16000,dtype=np.float32)*.05,
            model_dir=tmp_path, model_factory=lambda *a, **kw:Model(),
            device_policy='cpu',stage_hook=steps.append)
    assert result['status']=='asr_hypothesis'
    assert steps==['checking_model','model_ready','loading_cpu','decoding']


def test_health_probe_missing_model_is_offline_and_no_inference(tmp_path, monkeypatch):
    vendor=tmp_path/'vendor';vendor.mkdir()
    report=tmp_path/'report.json'
    # Provide fake version metadata. No actual downloads, runtime probing or sound.
    import music_perception.lyrics_health_probe as module
    monkeypatch.setattr(module.importlib.metadata,'version',lambda name: '1.0')
    fake=SimpleNamespace(get_cuda_device_count=lambda:1,
         get_supported_compute_types=lambda device:['float16'])
    fake_run=SimpleNamespace(returncode=0,stdout='Fake Nvidia 5070, 555.01, 9096\n')
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:fake_run)
    result=run_probe(vendor,tmp_path/'models','base','cuda',report,deep=True,
                     module_loader=lambda name:fake)
    assert result['cuda_candidate']
    assert result['gpu_name']=='Fake Nvidia 5070'
    assert result['weight_files_present'] is False
    assert result['state']=='complete'
    assert any(row['stage']=='synthetic_inference' and row['status']=='skipped'
               for row in result['steps'])
    assert '音频' not in report.read_text('utf-8')


def test_plugin_diagnostics_reports_idle_without_model_download(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def scenario():
        await plugin.startup()
        report=(await plugin.get_lyrics_diagnostics()).value
        assert report['ok']
        assert report['runtime']['child_running'] is False
        assert report['self_check']=={}
        assert not (Path(report['report_path'])).exists()
        await plugin.shutdown()
    asyncio.run(scenario())


def test_ui_has_one_click_action_and_safe_dom():
    html=Path(__file__).resolve().parents[1].joinpath('static/index.html').read_text('utf8')
    assert 'run_lyrics_diagnostics' in html
    assert 'get_lyrics_diagnostics' in html
    assert 'lyrics-worker-health' in html
    assert "$('lyrics-diagnostics-report').textContent=lines.join" in html
    assert 'innerHTML=lines.join' not in html


def test_health_report_does_not_claim_success_when_deps_or_weights_missing(tmp_path, monkeypatch):
    vendor=tmp_path/'vendor';vendor.mkdir()
    import music_perception.lyrics_health_probe as module
    def version(name):
        if name == 'faster-whisper':
            raise module.importlib.metadata.PackageNotFoundError(name)
        return '1.0'
    monkeypatch.setattr(module.importlib.metadata,'version',version)
    out=run_probe(vendor,tmp_path/'models','base','cpu',tmp_path/'diag.json',
                  deep=False,module_loader=lambda _:object())
    assert out['state']=='complete'
    assert out['health']=='needs_attention'
    assert 'faster-whisper' in out['missing_dependencies']


def test_gpu_health_is_verified_only_after_real_synthetic_inference(tmp_path, monkeypatch):
    vendor=tmp_path/'vendor';vendor.mkdir()
    model=tmp_path/'models'/'base';model.mkdir(parents=True)
    (model/'model.bin').write_bytes(b'123456789')
    (model/'config.json').write_text('{}',encoding='utf-8')
    (model/'tokenizer.json').write_text('{}',encoding='utf-8')
    import music_perception.lyrics_health_probe as module
    import music_perception.lyrics_model_store as store
    monkeypatch.setitem(store.MODELS,'base',dict(store.MODELS['base'],min_bytes=5))
    monkeypatch.setattr(module.importlib.metadata,'version',lambda name:'1.0')
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout='GPU, 600.11, 10000\n'))
    class Model:
        def __init__(self,*args,**kwargs):
            assert kwargs.get('device')=='cuda' and kwargs.get('local_files_only') is True
        def transcribe(self,audio,**kwargs):
            return iter([]),None
    import sys
    monkeypatch.setitem(sys.modules,'faster_whisper',SimpleNamespace(WhisperModel=Model))
    fake_ct2=SimpleNamespace(get_cuda_device_count=lambda:1,
        get_supported_compute_types=lambda d:['float16'])
    result=run_probe(vendor,tmp_path/'models','base','cuda',tmp_path/'report.json',deep=True,
                     module_loader=lambda _:fake_ct2)
    assert result['health']=='verified_cuda' and result['actual_test_device']=='cuda'
    # In-use self-check must never launch a second heavy model.
    result=run_probe(vendor,tmp_path/'models','base','cuda',tmp_path/'report.json',deep=False,
                     module_loader=lambda _:fake_ct2)
    assert result['health']=='not_fully_tested'


def test_diagnostic_button_is_single_flight(plugin_fixture, tmp_path, monkeypatch):
    plugin,_,_,_=plugin_fixture
    import threading,sys
    started=threading.Event(); finish=threading.Event()
    async def scenario():
        await plugin.startup()
        monkeypatch.setattr(plugin._environment,'snapshot',lambda:{'ready':True})
        monkeypatch.setattr(plugin._environment,'worker_runtime',lambda:(sys.executable,tmp_path),raising=False)
        def fake_check(*a):
            started.set()
            assert finish.wait(4)
        monkeypatch.setattr(plugin,'_run_lyrics_check_target',fake_check)
        first=(await plugin.run_lyrics_diagnostics()).value
        assert first['started']
        assert await asyncio.to_thread(started.wait,2)
        second=(await plugin.run_lyrics_diagnostics()).value
        assert second['already_running'] is True
        finish.set()
        for _ in range(40):
            if not plugin._lyrics_diagnostic_lock.locked():
                break
            await asyncio.sleep(.025)
        assert not plugin._lyrics_diagnostic_lock.locked()
        await plugin.shutdown()
    try:
        asyncio.run(scenario())
    finally:
        finish.set()
