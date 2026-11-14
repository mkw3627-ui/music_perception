"""Low-load ASR model reuse, optional GPU, and bounded child IPC. No download."""
from __future__ import annotations
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import numpy as np
import pytest

from music_perception.lyrics_backend import transcribe_singing_candidate
from music_perception.lyrics_bridge import PersistentLyricsSession, LyricsWorkerError
from test_lyrics_v063 import pcm, FakeSegment, plugin_fixture


class TinyFake:
    def transcribe(self, *args, **kwargs):
        assert kwargs['beam_size'] == 2
        assert kwargs['condition_on_previous_text'] is False
        return [FakeSegment('歌声候选',.2,.9)],SimpleNamespace(language='zh')


def test_model_reuse_and_cpu_int8_only(tmp_path):
    created=[]
    def factory(name, **kwargs):
        created.append(kwargs)
        return TinyFake()
    cache={}
    for i in range(3):
        report=transcribe_singing_candidate(
            pcm(2), model_dir=tmp_path, model_factory=factory,
            model_cache=cache, device_policy='cpu', cpu_threads=2)
        assert report['device']=='cpu'
    assert len(created)==1
    assert created[0]['device']=='cpu'
    assert created[0]['compute_type']=='int8'
    assert created[0]['cpu_threads']==2
    assert created[0]['num_workers']==1


def test_cuda_opt_in_failure_caches_cpu_fallback(tmp_path):
    attempts=[]
    def factory(name, **kwargs):
        attempts.append(kwargs['device'])
        if kwargs['device']=='cuda':
            raise RuntimeError('no CUDA on this system')
        return TinyFake()
    cache={}
    for _ in range(2):
        transcribe_singing_candidate(pcm(2), model_dir=tmp_path, model_factory=factory,
                                     model_cache=cache, device_policy='auto')
    assert attempts==['cuda','cpu']


def _make_fake_stream_worker(path:Path, halt=False):
    path.write_text('''import json,os,sys,time
inp=sys.stdin.buffer
n=0
while True:
    head=inp.readline()
    if not head:break
    record=json.loads(head)
    remaining=record['samples']*4
    while remaining:
        part=inp.read(min(remaining,65536))
        if not part:sys.exit(3)
        remaining-=len(part)
    n+=1
    if '''+str(halt)+''':time.sleep(3)
    res={'version':2,'id':record['id'], 'lyrics':{'status':'asr_hypothesis',
           'timed_cues':[{'text_preview':'测试用片段','part':n}],
           'worker_pid':os.getpid(), 'count':n, 'threads':record['cpu_threads'],
           'device':record['device']}}
    sys.stdout.buffer.write((json.dumps(res,ensure_ascii=False)+'\\n').encode('utf-8'))
    sys.stdout.buffer.flush()
''',encoding='utf-8')


def test_persistent_worker_reuses_one_process_across_chunks(tmp_path):
    vendor=tmp_path/'vendor';vendor.mkdir()
    fake=tmp_path/'fake_stream.py';_make_fake_stream_worker(fake)
    tracked=[]
    session=PersistentLyricsSession(
        python_exe=sys.executable,vendor=vendor,worker_script=fake,
        model_dir=tmp_path/'weights',model_name='base',device='cpu',
        cpu_threads=2,timeout=5,on_process=tracked.append)
    try:
        one=session.transcribe(pcm(1))
        two=session.transcribe(pcm(1))
        assert one['worker_pid']==two['worker_pid']
        assert one['count']==1 and two['count']==2
        assert two['threads']==2 and two['device']=='cpu'
        assert session.running
        assert len([p for p in tracked if p is not None])==1
    finally:
        session.close()
    assert not session.running
    assert tracked[-1] is None


def test_hung_asr_worker_killed_before_next_song(tmp_path):
    vendor=tmp_path/'vendor';vendor.mkdir()
    fake=tmp_path/'hang.py';_make_fake_stream_worker(fake,halt=True)
    session=PersistentLyricsSession(python_exe=sys.executable,vendor=vendor,
        worker_script=fake, model_dir=tmp_path/'weights',timeout=.2)
    with pytest.raises(LyricsWorkerError,match='超时'):
        session.transcribe(pcm(1))
    assert not session.running
    session.close()


@pytest.mark.asyncio
async def test_default_settings_use_cpu_base_two_threads(plugin_fixture):
    plugin,_,_,_ = plugin_fixture
    await plugin.startup()
    settings=await plugin._settings()
    assert settings['lyrics_model']=='base'
    assert settings['lyrics_device']=='cpu'
    assert settings['lyrics_cpu_threads']==2
    assert plugin._bounded_asr_threads(100)==4
    await plugin.shutdown()


def test_local_lyrics_continue_without_ai_sharing(plugin_fixture):
    from music_perception.auto_listener import FinishedClip
    plugin,_,_,_=plugin_fixture
    import asyncio
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=False
        clip=FinishedClip(pcm(8),'live_progress',4,sequence=1)
        assert plugin._enqueue_lyrics(clip,{},plugin._listener_session)
        assert plugin._lyrics_status=='queued'
        assert plugin._lyrics_queue.qsize()==1
        await plugin.shutdown()
    asyncio.run(run())


def test_revoked_ai_delivery_keeps_local_lyrics_but_sends_no_chat(plugin_fixture):
    import asyncio,threading
    from music_perception.auto_listener import FinishedClip
    plugin,_,_,pushed=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._listener_session=4
        plugin._song_boundary_enforced=True
        plugin._active_track_sequence=3
        plugin._live_session_config={'lyrics_model':'base'}
        scene={'music_capture':{'music_sequence':3,'offset_s':0,'end_s':10},
               'musical_event_timeline':{'events':[]}}
        plugin._recent_auditory_context=scene
        plugin._transcribe_singing_live_audio_isolated=lambda audio,cfg: (
            setattr(plugin,'_ai_delivery',False) or
            {'status':'asr_hypothesis','timed_cues':[{'at_s':0.1,'text_preview':'不应泄露'}]})
        q=plugin._lyrics_queue
        q.put((FinishedClip(pcm(8),'live_progress',4,sequence=3),scene,4))
        item=q.get_nowait()
        clip,context,_ = item
        plugin._process_one_lyrics_task(clip,context,4,4,q)
        assert plugin._track_lyrics_history
        assert 'lyrics_understanding' in scene
        assert not pushed
        await plugin.shutdown()
    asyncio.run(run())
