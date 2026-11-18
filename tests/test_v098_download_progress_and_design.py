"""v0.9.8: real transfer telemetry and a consistent task dashboard, no network."""
from __future__ import annotations

import io
import json
import sys
import time
import types
from pathlib import Path

import pytest
from _version import ui_build  # noqa: E402
from music_perception import MusicPerceptionPlugin
from music_perception import runtime_environment as runtime
from music_perception import singing_catalog as singer
from music_perception.download_telemetry import TransferMeter, snapshot_bytes
from music_perception.runtime_environment import MusicEnvironment


def test_progress_speed_eta_stall_resume_and_no_premature_100():
    moment=[0.0]
    def clock():
        return moment[0]
    tracker=TransferMeter(stall_after=12,clock=clock)
    initial=tracker.update(0,100,estimated=False)
    assert initial['progress_pct']==0 and initial['speed_bytes_s'] is None
    moment[0]=2.0
    result=tracker.update(50,100,estimated=False)
    assert result['progress_pct']==50 and result['speed_bytes_s']==25.0
    assert result['eta_seconds']==2
    moment[0]=15
    assert tracker.update(50,100)['stalled'] is True
    # A range-restart cannot turn stale 99% into a completed file.
    moment[0]=16
    reset=tracker.update(0,100,restarted=True)
    assert reset['progress_pct']==0 and reset['speed_bytes_s'] is None
    moment[0]=20
    assert tracker.update(100,100)['progress_pct']==99  # SHA yet to run
    assert tracker.update(1,None)['progress_pct'] is None


def test_snapshot_meter_observes_partial_and_completed_bytes_without_duplication(tmp_path):
    stage=tmp_path/'snapshot'
    stage.mkdir()
    assert snapshot_bytes(stage)==0
    cache=stage/'.cache/huggingface/download'
    cache.mkdir(parents=True)
    (cache/'model.bin.XYZ.incomplete').write_bytes(b'partial-byte-stream')
    assert snapshot_bytes(stage)==len(b'partial-byte-stream')
    (stage/'model.bin').write_bytes(b'verified')
    assert snapshot_bytes(stage)==len(b'verified')
    (stage/'tokenizer.json').write_bytes(b'{}')
    assert snapshot_bytes(stage)==len(b'verified')+2


def test_singer_worker_progress_emits_estimate_and_isolation(monkeypatch,tmp_path):
    class MockApi:
        def __init__(self,endpoint):
            pass
        def model_info(self,*_args,**_kwargs):
            return types.SimpleNamespace(sha=singer.PINNED_REVISION['low'],siblings=[])
    def abort_download(**kwargs):
        folder=Path(kwargs['local_dir'])
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'model.bin').write_bytes(b'synthetic-only')
        raise RuntimeError('offline expected')
    monkeypatch.setitem(sys.modules,'huggingface_hub',types.SimpleNamespace(
        HfApi=MockApi,snapshot_download=abort_download))
    monkeypatch.setattr(singer,'_download_modelscope',
                        lambda stage,quality,**_:abort_download(local_dir=stage))
    events=[]
    with pytest.raises(singer.SingingModelError):
        singer.acquire_model(tmp_path,'low',source='official',
                             on_progress=lambda stage,event:events.append((stage,event)))
    assert events
    assert any(stage=='downloading' and value.get('progress_estimated') is True
               for stage,value in events)
    assert not (tmp_path/singer.PROFILES['low']['id']/singer.MANIFEST_NAME).exists()


def test_singing_catalog_reads_progress_file_without_loading_weights(monkeypatch,tmp_path):
    modelroot=tmp_path/'models'/'singing_specialist'
    modelroot.mkdir(parents=True)
    starting=time.time()-10
    (modelroot/'singing_download_medium.json').write_text(json.dumps({
        'state':'running','stage':'downloading','updated_at':time.time(),
        'bytes_received':123_000_000,'total_bytes':1_900_000_000,
        'progress_pct':6.5,'progress_estimated':True,'speed_bytes_s':2_000_000,
        'eta_seconds':890,'last_progress_at':time.time(),
    }),encoding='utf8')
    dummy=types.SimpleNamespace(
        data_path=lambda leaf:tmp_path/leaf,
        _singing_prepare_state={'quality':'medium','state':'running','started_at':starting},
        _resource_effective={'singing':{}},
        _singing_profile='medium',_singing_device_preference='cpu',_singing_runtime_device=None,
        _singing_activation_state={'state':'idle'},
    )
    result=MusicPerceptionPlugin._singing_catalog_snapshot(dummy)
    assert result['preparation']['speed_bytes_s']==2_000_000
    assert result['preparation']['progress_pct']==6.5
    assert result['models']['medium']['download']['eta_seconds']==890
    assert result['models']['low']['state']!='running'


def test_yamnet_bootstrap_reports_real_bytes_not_fake_progress(monkeypatch,tmp_path):
    model=bytearray(b'\x00\x00\x00\x00TFL3')+bytearray(1_000_100)
    class Response:
        def __init__(self):
            self.io=io.BytesIO(model)
            self.headers={'Content-Length':str(len(model))}
        def __enter__(self):
            return self
        def __exit__(self,*_):
            pass
        def read(self,n):
            return self.io.read(n)
        def geturl(self):
            return 'https://storage.googleapis.com/test-model'
    monkeypatch.setattr(runtime,'YAMNET_SOURCES',(('https://storage.googleapis.com/test-model',None),))
    monkeypatch.setattr(runtime.urllib.request,'urlopen',lambda *_a,**_kw:Response())
    env=MusicEnvironment(source=tmp_path,data=tmp_path/'runtime',models=tmp_path/'models')
    done=env._download_model()
    assert done.is_file()
    state=env.snapshot()['model_download']
    assert state['progress_pct']==100 and state['bytes_received']==len(model)
    assert state['source_name']=='Google 官方'


def test_natively_hosted_ui_has_real_metrics_and_matching_model_styles():
    root=Path(__file__).resolve().parents[1]
    ui=(root/'ui/panel.tsx').read_text('utf8')
    for fragment in ('function DownloadProgress', 'function ModelSpotlight',
                     'function ModelTiles','实时速度','预计剩余','可能卡住',
                     '下载任务','role="progressbar"','ModelSpotlight kind="VIDEO SPEECH"',
                     'ModelSpotlight kind="SINGING RECOGNITION"',
                     'ModelSpotlight kind="VOCAL SEPARATION"',
                     '1800:5000', f'"{ui_build()}"'):
        assert fragment in ui
    # Rendering merely reads task states, no implicit network or installation.
    assert 'userInitiated:true' in ui
    assert 'confirmed:true' in ui
    assert 'props.api.call("install_singing_model"' not in ui
    assert (root/'download_telemetry.py').is_file()
