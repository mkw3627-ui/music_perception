"""0.8.5 optional full-band local stem separation regression."""
from __future__ import annotations
import asyncio
import math
from pathlib import Path

import numpy as np
import pytest

from music_perception import stem_mdx
from music_perception.auto_listener import FinishedClip, MusicClipCollector, RapidLyricsSampler
from music_perception.lyrics_continuity import join_song_audio
from test_plugin import plugin_fixture


def test_stft_istft_roundtrip_44k_stereo():
    sr=stem_mdx.RATE
    t=np.arange(stem_mdx.CHUNK)/sr
    x=np.stack([.14*np.sin(2*np.pi*230*t),.12*np.sin(2*np.pi*420*t)],axis=1).astype('float32')
    spec=stem_mdx.stft_channels(x)
    assert spec.shape==(1,4,3072,256)
    reconstructed=stem_mdx.istft_channels(spec)
    assert np.max(np.abs(x-reconstructed)) < 0.0001


def test_model_signature_rejects_short_files_and_html(tmp_path):
    p=tmp_path/stem_mdx.MODEL_FILENAME
    p.write_text('<html>Access denied</html>',encoding='utf8')
    assert stem_mdx.verify_model(p) is False
    p.unlink()
    assert stem_mdx.verify_model(p) is False


def test_mdx_splits_both_stems_without_using_torch(monkeypatch,tmp_path):
    monkeypatch.setattr(stem_mdx,'verify_model',lambda _:True)
    class Input:
        name='input'
        shape=[1,4,3072,256]
    class IdentityInstrument:
        def get_inputs(self):return [Input()]
        def get_providers(self):return ['CPUExecutionProvider']
        def run(self,names,feed):return [feed['input']]
    model=stem_mdx.TwoStemMDX(tmp_path/'fake.onnx',session_factory=lambda *_:IdentityInstrument())
    sr=44100
    t=np.arange(sr*2)/sr
    x=np.stack([0.05*np.sin(2*np.pi*300*t),0.05*np.sin(2*np.pi*500*t)],axis=1).astype('float32')
    result=model.separate(x,sample_rate=sr)
    assert result['device']=='cpu'
    assert abs(len(result['vocals'])-32000)<=1
    assert len(result['vocals'])==len(result['instrumental'])
    assert np.max(abs(result['vocals'])) < .002
    assert np.std(result['instrumental'])>.015


def test_fullband_preview_and_10second_clip_buffers():
    collector=MusicClipCollector(first_progress_seconds=10,progress_seconds=20)
    sampler=RapidLyricsSampler()
    frames=np.ones(32000,dtype='float32')*.03
    stereo=np.ones((96000,2),dtype='float32')*.03
    sequence=None; preview=None; clip=None
    full_candidates=[]
    for i in range(5):
        clip=collector.feed(frames,True,full_band=stereo)
        if i<2:full_candidates.append(stereo)
        if collector.recording:
            high=list(full_candidates) if i==1 else [stereo]
            low=collector._music if i==1 else [frames]
            preview=sampler.feed(collector._sequence, low, full_band_frames=high) or preview
    assert clip is not None and clip.full_band_samples.shape==(480000,2)
    assert preview is not None and preview.full_band_samples.shape==(288000,2)
    assert preview.full_band_rate==48000
    # Track boundary clears both microphone-free buffers.
    collector.discard();sampler.reset()
    assert sampler._full_frames==[]


def test_overlap_keeps_highband_alignment_and_rejects_weak_prefix():
    a=FinishedClip(np.zeros(96000,dtype=np.float32),'early_lyrics_preview',3,
                   sequence=12,part_index=1,full_band_samples=np.zeros((288000,2),dtype=np.float32))
    joined,tail,key=join_song_audio(a,previous_tail=None,previous_sequence=None)
    assert joined.full_band_samples.shape==(288000,2)
    b=FinishedClip(np.zeros(160000,dtype=np.float32),'early_lyrics_preview',5,
                   sequence=12,part_index=2,offset_s=6,
                   full_band_samples=np.ones((480000,2),dtype=np.float32))
    high_tail=a.full_band_samples[-4*48000:]
    joined,_,_=join_song_audio(b,previous_tail=tail,previous_sequence=key,
                                previous_full_tail=high_tail,previous_full_sequence=key)
    assert joined.offset_s==2.0
    assert len(joined.samples)==224000
    assert joined.full_band_samples.shape==(672000,2)
    wrong,_,_=join_song_audio(b,previous_tail=tail,previous_sequence=key,
                               previous_full_tail=high_tail,previous_full_sequence=(11,1))
    assert wrong.full_band_samples is None


def test_optin_requires_verified_model_and_never_downloads(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    monkeypatch.setattr('music_perception.verify_stem_model',lambda path:False)
    async def scenario():
        await plugin.startup()
        info=(await plugin.set_stem_separation(enabled=True)).value
        assert not info['ok']
        assert 'onnx' in info['error']
        assert plugin._stem_enabled is False
        snap=(await plugin.get_status(_for_ui=True)).value
        assert snap['stem_separation']['enabled'] is False
        await plugin.shutdown()
    asyncio.run(scenario())


def test_split_failure_falls_back_and_mutes_retries(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._stem_enabled=True
        # Stub an already installed / SHA-verified CPU stem model. The test
        # exercises worker lifecycle rather than filesystem integrity.
        plugin._resource_choice=lambda domain: {
            'state':'ready','quality':'ultra','device':'cpu',
            'preferred_quality':'ultra','reason':'mock_verified_local_stem'}
        plugin._stem_faulted=False
        from music_perception import PersistentStemSession
        def fail(*a,**kw):raise RuntimeError('mocked corrupted model')
        monkeypatch.setattr(PersistentStemSession,'__init__',fail)
        class Env:
            def worker_runtime(self):return ('python.exe','vendor')
            def stop(self):pass
        plugin._environment=Env()
        pcm=np.zeros((48000*2,2),dtype=np.float32)
        clip=FinishedClip(np.zeros(32000,dtype=np.float32),'live_progress',1,
                          full_band_samples=pcm)
        assert plugin._separate_live_stems(clip) is None
        assert plugin._stem_faulted is True
        assert 'mocked' in plugin._stem_state['error']
        assert plugin._separate_live_stems(clip) is None
        await plugin.shutdown()
    asyncio.run(scenario())


def test_isolated_pipe_can_transfer_two_second_full_band_stereo(monkeypatch,tmp_path):
    import music_perception.isolated_loopback as iso
    from test_isolated_loopback import prepare,pack
    stereo=np.arange(2*48000*2,dtype='<f4')/500000
    mono=np.ones(32000,dtype='<f4')*.04
    source,procs=prepare(monkeypatch,tmp_path,
        pack(b'R',{'device':'speaker','capture_rate':48000})+
        pack(b'H',stereo.tobytes())+pack(b'D',mono.tobytes()))
    source.capture_stereo=True
    with source as reader:
        actual=reader.read()
        assert np.array_equal(actual,mono)
        assert reader.last_stereo.shape==(96000,2)
        assert reader.last_stereo[0,0]==stereo[0]
        assert procs[0].argv[-1]=='stereo'


def test_stem_bridge_real_local_pipe_mock_worker(tmp_path):
    import sys
    from music_perception.stem_bridge import PersistentStemSession
    vendor=tmp_path/'vendor';vendor.mkdir()
    worker=tmp_path/'mock_worker.py'
    worker.write_text('''import sys,struct,json,array
L=struct.Struct('!I')
def read(n):
 b=bytearray()
 while len(b)<n:
  p=sys.stdin.buffer.read(n-len(b))
  if not p:return None
  b.extend(p)
 return bytes(b)
while True:
 size=read(4)
 if not size:break
 meta=json.loads(read(L.unpack(size)[0]))
 data=read(meta['frames']*8)
 if data is None:break
 n=int(meta['frames']*16000/meta['rate'])
 first=array.array('f',[0.03])*n
 second=array.array('f',[0.02])*n
 info={'id':meta['id'],'ok':True,'voice_frames':n,'instrument_frames':n,'device':'cpu'}
 head=json.dumps(info).encode()
 sys.stdout.buffer.write(L.pack(len(head))+head+first.tobytes()+second.tobytes())
 sys.stdout.buffer.flush()
''',encoding='utf8')
    ses=PersistentStemSession(python=sys.executable,vendor=vendor,script=worker,
                              weights=tmp_path/'unused.onnx',timeout=9)
    try:
        wav=np.zeros((16000,2),dtype=np.float32)
        first=ses.separate(wav,rate=16000)
        second=ses.separate(wav,rate=16000)
        assert np.isclose(first['vocals'].mean(),.03)
        assert np.isclose(first['instrumental'].mean(),.02)
        assert np.array_equal(first['vocals'],second['vocals'])
        assert ses.running
    finally:
        ses.close()
    assert not ses.running


def test_fullband_44100_rate_is_passed_to_clip():
    collector=MusicClipCollector(first_progress_seconds=10,progress_seconds=20)
    mono=np.ones(32000,dtype=np.float32)*.04
    stereo=np.ones((44100*2,2),dtype=np.float32)*.04
    result=None
    for _ in range(5):
        result=collector.feed(mono, True, full_band=stereo, full_band_rate=44100)
    assert result is not None
    assert result.full_band_rate==44100
    assert result.full_band_samples.shape==(441000,2)


def test_parallel_song_analysis_does_not_block_fast_lyrics(plugin_fixture):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._stem_enabled=True
        plugin._stem_faulted=False
        clip=FinishedClip(np.zeros(32000,dtype='float32'),'early_lyrics_preview',1,
                          full_band_samples=np.zeros((96000,2),dtype='float32'))
        ready=plugin._stem_session_lock.acquire(blocking=False)
        assert ready
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future=pool.submit(plugin._separate_live_stems,clip)
                assert future.result(timeout=1) is None
            assert not plugin._stem_faulted
        finally:
            plugin._stem_session_lock.release()
        await plugin.shutdown()
    asyncio.run(scenario())


def test_pipe_sender_handles_short_writes(tmp_path):
    """Windows raw pipes may accept fewer bytes than passed to write()."""
    import json, struct, sys
    from music_perception.stem_bridge import PersistentStemSession
    class PartialPipe:
        def __init__(self):self.data=bytearray()
        def write(self,payload):
            count=min(len(payload),7)
            self.data.extend(payload[:count]);return count
        def flush(self):pass
        def close(self):pass
    class FakeChild:
        def __init__(self):
            self.stdin=PartialPipe()
            self.stdout=PartialPipe()
        def poll(self):return None
        def terminate(self):pass
        def wait(self,timeout=None):return 0
    model=PersistentStemSession(python=sys.executable,vendor=tmp_path,
                                script=__file__,weights=tmp_path/'none.onnx')
    model._child=FakeChild()
    frames=16000
    model._replies.put(({'id':1,'ok':True,'voice_frames':frames,
                         'instrument_frames':frames,'device':'cpu'},
                        np.zeros(frames*2,dtype='<f4').tobytes()))
    out=model.separate(np.zeros((frames,2),dtype='<f4'),rate=16000)
    raw=model._child.stdin.data
    h=struct.unpack('!I',raw[:4])[0]
    meta=json.loads(raw[4:4+h])
    assert meta['frames']==frames
    assert len(raw)==4+h+frames*2*4
    assert len(out['vocals'])==frames
    model.close()


def test_stem_state_reports_real_device_not_requested_device(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    async def scenario():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._stem_enabled=True
        # Stub an already installed / SHA-verified CPU stem model. The test
        # exercises worker lifecycle rather than filesystem integrity.
        plugin._resource_choice=lambda domain: {
            'state':'ready','quality':'ultra','device':'cpu',
            'preferred_quality':'ultra','reason':'mock_verified_local_stem'}
        class FakeSession:
            def separate(self,*args,**kwargs):
                return {'vocals':np.zeros(32000,dtype='float32'),
                        'instrumental':np.zeros(32000,dtype='float32'),
                        'device':'cpu','warning':'estimate only'}
            def close(self):pass
        plugin._stem_session=FakeSession()
        plugin._stem_effective_session=('ultra','cpu')
        clip=FinishedClip(np.zeros(32000,dtype='float32'),'live_progress',1,
                          full_band_samples=np.zeros((96000,2),dtype='float32'))
        out=plugin._separate_live_stems(clip)
        assert out is not None
        assert plugin._stem_state['actual_device']=='cpu'
        assert plugin._stem_state['state']=='active'
        await plugin.shutdown()
    asyncio.run(scenario())
