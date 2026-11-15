"""v0.9.12 regressions: skip -> next song without toggling monitoring.

Runs with synthetic local PCM and stub recognizers, not a claim of Windows
WASAPI or CUDA hardware validation.
"""
import asyncio
import queue
import threading
import time

import numpy as np

from music_perception import MusicPerceptionPlugin
from music_perception.auto_listener import (LiveListener, FinishedClip,
                                             Decision, SAMPLE_RATE)
from test_plugin import plugin_fixture


def test_capture_error_reopens_same_source_and_new_sequence(tmp_path):
    events = []
    uses = []
    class Source:
        def __init__(self, listener, index):
            self.listener = listener
            self.index = index
            self.frames = 0
        def __enter__(self):
            return self
        def __exit__(self,*args):
            return False
        def read(self):
            self.frames += 1
            if self.index == 1 and self.frames == 4:
                raise OSError('test: render endpoint disconnected on skip')
            if self.index == 2 and self.frames == 5:
                self.listener._stop.set()
            return np.full(2*SAMPLE_RATE,.08,np.float32)
    class Gate:
        def __init__(self,*a,**k): pass
        def classify(self,pcm): return Decision(True,.95,.04)
    def source_factory():
        source = Source(listener,len(uses)+1)
        uses.append(source)
        return source
    listener = LiveListener(model_file=tmp_path/'no_model_needed',
        threshold=.2, max_seconds=900, on_clip=lambda c:None,
        gate_factory=Gate, source_factory=source_factory,
        scene_decider=lambda d:{'kind':'music','verified':True,'reason':'test_music','source_count':1},
        on_track_boundary=lambda name,seq:events.append((name,seq)))
    listener._run()  # no scheduling, deterministic frames and one automatic retry
    assert len(uses)==2
    assert listener.status()['capture_retries']==1
    assert events[:3]==[('track_started',1),('track_ended',1),('track_started',1)]
    assert listener.status()['status']=='stopped'


def test_skip_interrupts_only_inflight_old_singing_child(plugin_fixture):
    p, *_ = plugin_fixture
    async def start(): await p.startup()
    asyncio.run(start())
    p._auto_consented = True
    p._listener_session = 3
    p._active_track_sequence = 7
    p._song_current_id = 'song-old'
    p._song_boundary_enforced = True
    p._lyrics_inflight = (3,7)
    p._lyrics_progress_by_song = {'song-old':{'state':'transcribing'}}
    calls=[]
    class OldModel:
        def close(self): calls.append('closed_stale')
    old=OldModel()
    p._singing_session=old
    p._on_track_boundary('track_ended',7)
    assert calls == ['closed_stale']
    assert p._active_track_sequence is None
    p._active_track_sequence=8
    p._song_current_id = 'song-new'
    p._lyrics_inflight = None
    p._on_track_boundary('track_ended',8)
    assert calls == ['closed_stale']  # no idle-model churn
    p._singing_session=None
    asyncio.run(p.shutdown())


def test_track_change_during_stems_does_not_load_singing_model(plugin_fixture):
    p,*_ = plugin_fixture
    asyncio.run(p.startup())
    p._auto_consented=True
    p._listener_session=4
    p._active_track_sequence=3
    p._song_boundary_enforced=True
    p._song_current_id='song-3'
    current=[True]
    p._clip_is_current=lambda *a,**k:current[0]
    p._note_lyrics_progress=lambda *a,**k:None
    p._reap_retired_lyrics_sessions=lambda:None
    def separate(clip):
        current[0]=False  # a new song started while GPU stem separated old song
        return {'vocals':clip.samples, 'enhanced_vocals':clip.samples, 'device':'cpu'}
    p._separate_live_stems=separate
    calls=[]
    p._transcribe_singing_live_audio_isolated=lambda *a,**k:calls.append('BAD')
    clip=FinishedClip(np.full(6*SAMPLE_RATE,.08,np.float32),
                      'early_lyrics_preview',3,offset_s=0,sequence=3,part_index=1)
    q=queue.Queue();q.put('work')
    p._process_one_lyrics_task(clip,{'music_capture':{}},4,4,q,
                               quality_generation=p._lyrics_quality_generation)
    assert calls==[]
    assert p._lyrics_inflight is None
    assert q.unfinished_tasks==0
    asyncio.run(p.shutdown())


def test_dead_asr_consumer_restart_is_bounded_and_lazy(plugin_fixture):
    p,*_=plugin_fixture
    asyncio.run(p.startup())
    p._auto_consented=True
    p._lyrics_stop=threading.Event()
    p._listener_session=4
    p._lyrics_queue=queue.Queue(maxsize=2)
    p._lyrics_queue.put(('obsolete',))
    class Dead:
        def is_alive(self):return False
    p._lyrics_worker_thread=Dead()
    seen=[]
    def consumer(session,q,stop):
        seen.append(session)
        stop.wait(1)
    p._background_lyrics_main=consumer
    assert p._ensure_lyrics_consumer_alive() is True
    assert seen == [4] or p._lyrics_worker_thread.is_alive()
    assert p._lyrics_queue.qsize() == 0
    assert p._lyrics_worker_restart_count == 1
    assert p._ensure_lyrics_consumer_alive() is True
    p._lyrics_stop.set()
    p._lyrics_worker_thread.join(timeout=2)
    asyncio.run(p.shutdown())


def test_pipeline_status_explains_gpu_idle_on_unknown_scene(plugin_fixture):
    p,*_=plugin_fixture
    asyncio.run(p.startup())
    p._auto_consented=True
    p._audio_scene={'kind':'unknown','reason':'music_session_unconfirmed'}
    p._lyrics_worker_thread=threading.current_thread()
    p._listener=None
    result=p._singing_pipeline_status()
    assert result['model_loaded_and_tested'] is False
    assert result['actual_device'] is None
    assert '待' in result['reason'] or '等待' in result['reason']
    p._lyrics_worker_thread=None
    asyncio.run(p.shutdown())


def test_stem_error_has_cooldown_and_new_song_resets_fault(plugin_fixture):
    p,*_=plugin_fixture
    asyncio.run(p.startup())
    p._auto_consented=True
    p._stem_enabled=True
    p._stem_faulted=True
    p._stem_fault_retries=1
    p._stem_faulted_at=time.monotonic()
    clip=FinishedClip(np.ones(6*SAMPLE_RATE,dtype=np.float32)*.03,
                      'early_lyrics_preview',3,sequence=1,part_index=1)
    assert p._separate_live_stems(clip) is None
    assert p._stem_faulted is True  # no hot-loop retry
    p._on_track_boundary('track_started',1)
    assert p._stem_faulted is False
    assert p._stem_fault_retries == 0
    asyncio.run(p.shutdown())
