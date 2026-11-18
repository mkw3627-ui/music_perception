"""v0.7.1: early lyrics, optional turbo, private context, make-before-break swaps."""
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

# Since v0.8.2 all model switches REQUIRE verified local weights. These
# older hot-swap unit tests intentionally simulate already-validated files:
# they exercise post-verification lifecycle, not SHA/download integrity.
import pytest
from music_perception.auto_listener import SAMPLE_RATE, FinishedClip, RapidLyricsSampler
from music_perception.lyrics_backend import prepare_singing_model
from music_perception.lyrics_bridge import LyricsWorkerError, PersistentLyricsSession
from music_perception.lyrics_model_store import MODELS
from music_perception.song_memory import QUALITY_MODEL


@pytest.fixture(autouse=True)
def _preverified_test_model_for_legacy_swap(request, monkeypatch):
    selected = {
        'test_hot_swap_only_commits_after_new_model_ready',
        'test_failed_cuda_warmup_keeps_cpu_model_and_saved_preference',
        'test_user_can_cancel_long_model_warmup_without_changing_preferences',
        'test_windows_locked_settings_file_prevents_switch_and_keeps_old_model',
        'test_profile_hot_switch_preference_write_failure_rolls_back',
    }
    if request.node.name in selected:
        from music_perception import lyrics_model_store
        monkeypatch.setattr(lyrics_model_store,'find_local_model',
            lambda model, root: Path(root)/model)

def wave(seconds=2):
    t = np.arange(int(seconds * SAMPLE_RATE),dtype=np.float32)/SAMPLE_RATE
    return 0.02*np.sin(2*np.pi*220*t)


def test_preview_comes_after_six_seconds_while_dsp_uses_ten():
    sampler=RapidLyricsSampler()
    assert sampler.feed(8, [wave(2)]) is None
    assert sampler.feed(8, [wave(2)]) is None
    first=sampler.feed(8, [wave(2)])
    assert first.end_reason=='early_lyrics_preview'
    assert first.sequence==8 and len(first.samples)==6*SAMPLE_RATE
    assert first.offset_s==0 and first.part_index==1
    for _ in range(4):
        assert sampler.feed(8,[wave(2)]) is None
    second=sampler.feed(8,[wave(2)])
    assert second.offset_s==6.0 and len(second.samples)==10*SAMPLE_RATE
    assert second.sequence==8 and second.part_index==2
    assert sampler.feed(9,[wave(2)]) is None
    assert sampler.feed(9,[wave(2)]) is None
    changed=sampler.feed(9,[wave(2)])
    assert changed.sequence==9 and changed.offset_s==0
    assert not np.array_equal(changed.samples[:100],np.zeros(100))


def test_only_preview_callback_queues_music_without_chat(plugin_fixture):
    plugin,_,_,pushes=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._listener_session=7
        clip=FinishedClip(wave(6),'early_lyrics_preview',3,sequence=1)
        assert plugin._on_early_lyrics_clip(clip)
        assert plugin._lyrics_queue.qsize()==1
        queued=plugin._lyrics_queue.get_nowait()
        assert queued[0].sequence==1
        assert queued[1]['music_capture']['end_reason']=='early_lyrics_preview'
        assert queued[1]['musical_event_timeline']['lyrics']['status']=='recognizing'
        plugin._lyrics_queue.task_done()
        assert not pushes, 'Early buffered audio must not print in chat'
        await plugin.shutdown()
    asyncio.run(run())


def test_ultra_model_is_multilingual_turbo_with_pinned_integrity():
    assert QUALITY_MODEL['ultra']=='large-v3-turbo'
    spec=MODELS['large-v3-turbo']
    assert spec['repo']=='deepdml/faster-whisper-large-v3-turbo-ct2'
    assert len(spec['sha256'])==64 and spec['min_bytes']>1_500_000_000


def test_standby_warmup_uses_synthetic_audio_and_keeps_weights(tmp_path):
    calls=[]
    class FakeModel:
        def transcribe(self, samples, **kw):
            calls.append(len(samples))
            return [],SimpleNamespace(language='en')
    from music_perception import lyrics_backend
    def factory(*args, **kwargs):
        return FakeModel(),'cpu'
    cache={}
    with patch.object(lyrics_backend,'_make_model',factory):
        first=prepare_singing_model(model_dir=tmp_path,model_name='base',
                                    model_cache=cache,device_policy='cpu')
        second=prepare_singing_model(model_dir=tmp_path,model_name='base',
                                    model_cache=cache,device_policy='cpu')
    assert first['ready'] and second['device']=='cpu'
    assert calls==[16000,16000]
    assert cache['instance'] is not None


def _make_prepare_worker(path):
    path.write_text('''import sys,json,os
while True:
    line=sys.stdin.buffer.readline()
    if not line:break
    job=json.loads(line)
    remaining=job['samples']*4
    while remaining:
        data=sys.stdin.buffer.read(remaining)
        if not data:sys.exit(2)
        remaining-=len(data)
    response={'version':2,'id':job['id'],'device':job['device'],'ready':True,
              'lyrics':{'status':'asr_hypothesis','timed_cues':[]},'pid':os.getpid()}
    sys.stdout.write(json.dumps(response)+'\\n');sys.stdout.flush()
''',encoding='utf-8')


def test_preflight_session_process_reused_for_real_lyrics(tmp_path):
    vendor=tmp_path/'vendor'
    vendor.mkdir()
    worker=tmp_path/'fake.py'
    _make_prepare_worker(worker)
    session=PersistentLyricsSession(python_exe=sys.executable,vendor=vendor,
                worker_script=worker,model_dir=tmp_path/'models',device='cpu',timeout=4)
    try:
        assert session.prepare()['device']=='cpu'
        assert session.transcribe(wave(2))['status']=='asr_hypothesis'
        assert session.running
    finally:
        session.close()


def _await_status(plugin,status,timeout=4):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if plugin._lyrics_switch_state.get('status')==status:
            return
        time.sleep(.03)
    raise AssertionError(f'Not {status}: {plugin._lyrics_switch_state}')


def test_hot_swap_only_commits_after_new_model_ready(plugin_fixture,monkeypatch):
    plugin,_,_,_=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._lyrics_profile='medium'
        plugin._lyrics_device_preference='cpu'
        plugin._environment.worker_runtime=lambda:(sys.executable,plugin.data_path('environment'))
        plugin._live_session_config={'lyrics_model':'base','lyrics_device':'cpu'}
        release=[]
        ready=[]
        class Session:
            def __init__(self,**kwargs):
                self.name=kwargs['model_name']
                self.device=kwargs['device']
                ready.append(self.name)
            def prepare(self):
                time.sleep(.05)
                return {'ready':True,'device':self.device}
            def close(self):
                release.append(self.name)
        class Current:
            def close(self):
                release.append('old')
        old=Current()
        plugin._lyrics_session=old
        module=sys.modules[plugin.__class__.__module__]
        monkeypatch.setattr(module,'PersistentLyricsSession',Session)
        pending=(await plugin.set_lyrics_quality('ultra')).value
        assert pending['switch_status']=='preparing'
        assert plugin._lyrics_profile=='medium' and plugin._lyrics_session is old
        _await_status(plugin,'ready')
        assert plugin._lyrics_profile=='ultra'
        assert plugin._lyrics_session is not old
        assert plugin._lyrics_device_preference=='cpu'
        assert json.loads(plugin._preferences_path().read_text())['lyrics_quality']=='ultra'
        assert 'old' not in release  # in-flight clip may still be using it
        plugin._reap_retired_lyrics_sessions()
        assert 'old' in release
        await plugin.shutdown()
    asyncio.run(run())


def test_failed_cuda_warmup_keeps_cpu_model_and_saved_preference(plugin_fixture,monkeypatch):
    plugin,_,_,_=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._lyrics_profile='medium'
        plugin._lyrics_device_preference='cpu'
        plugin._environment.worker_runtime=lambda:(sys.executable,plugin.data_path('environment'))
        plugin._live_session_config={'lyrics_model':'base','lyrics_device':'cpu'}
        class Current:
            def __init__(self):
                self.closed=False
            def close(self):
                self.closed=True
        old=Current()
        plugin._lyrics_session=old
        class BadGPU:
            def __init__(self,**kw):
                pass
            def prepare(self):
                raise LyricsWorkerError('CUDA DLL unavailable')
            def close(self):
                pass
        module=sys.modules[plugin.__class__.__module__]
        monkeypatch.setattr(module,'PersistentLyricsSession',BadGPU)
        pending=(await plugin.set_lyrics_device('cuda')).value
        assert pending['switch_status']=='preparing'
        _await_status(plugin,'failed')
        assert plugin._lyrics_session is old and not old.closed
        assert plugin._lyrics_device_preference=='cpu'
        assert plugin._live_session_config['lyrics_device']=='cpu'
        assert '保留旧模型' in plugin._lyrics_switch_state['message']
        await plugin.shutdown()
    asyncio.run(run())


def test_rapid_path_does_not_delay_rhythm_or_accept_speech():
    import threading

    from music_perception.auto_listener import Decision, LiveListener
    previews=[]
    rhythm=[]
    ends=[]
    stop=threading.Event()
    class FakeSource:
        device='test-loopback-speaker'
        def __enter__(self):
            return self
        def __exit__(self,*args):
            return False
        def read(self):
            if getattr(self,'index',0)>=5:
                stop.set()
                raise RuntimeError('completed synthetic stream')
            self.index=getattr(self,'index',0)+1
            return wave(2)
    class FakeGate:
        def __init__(self,*args,**kwargs):
            pass
        def classify(self,frame):
            return Decision(True,.92,.01)
    listener=LiveListener(model_file=Path('unused'),max_seconds=120,
            threshold=.25,on_clip=rhythm.append,
            source_factory=FakeSource,gate_factory=FakeGate,
            on_lyrics_preview=previews.append,
            on_track_boundary=lambda typ,seq:ends.append((typ,seq)))
    listener._audio_boundary_detector.feed=lambda *a,**kw:None
    listener.start()
    assert stop.wait(3)
    listener.stop(timeout=3)
    assert len(rhythm)==1 and len(previews)==1
    assert rhythm[0].sequence==previews[0].sequence
    assert len(previews[0].samples)==6*SAMPLE_RATE
    assert len(rhythm[0].samples)==10*SAMPLE_RATE
    assert previews[0].end_reason=='early_lyrics_preview'


def test_user_can_cancel_long_model_warmup_without_changing_preferences(plugin_fixture,monkeypatch):
    plugin,_,_,_=plugin_fixture
    import threading
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._lyrics_profile='medium'
        plugin._lyrics_device_preference='cpu'
        plugin._environment.worker_runtime=lambda:(sys.executable,plugin.data_path('environment'))
        plugin._live_session_config={'lyrics_model':'base','lyrics_device':'cpu'}
        reached=threading.Event()
        cancelled=threading.Event()
        class Old:
            def close(self):
                pass
        old=Old()
        plugin._lyrics_session=old
        class Slow:
            def __init__(self,**kw):
                pass
            def prepare(self):
                reached.set()
                assert cancelled.wait(3)
                raise LyricsWorkerError('cancelled by user')
            def close(self):
                cancelled.set()
        module=sys.modules[plugin.__class__.__module__]
        monkeypatch.setattr(module,'PersistentLyricsSession',Slow)
        start=(await plugin.set_lyrics_quality('ultra')).value
        assert start['switch_status']=='preparing'
        assert reached.wait(1)
        stopped=(await plugin.set_lyrics_quality('medium')).value
        assert stopped['switch_status']=='cancelled'
        assert plugin._lyrics_profile=='medium' and plugin._lyrics_session is old
        assert plugin._lyrics_switch_state['status']=='cancelled'
        assert plugin._lyrics_device_preference=='cpu'
        await plugin.shutdown()
    asyncio.run(run())


def test_private_message_is_sensor_evidence_not_a_request_to_speak(plugin_fixture):
    plugin,_,_,pushes=plugin_fixture
    from music_perception.auto_listener import FinishedClip
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        clip=FinishedClip(wave(2),'live_progress',1,sequence=1)
        context={'music_capture':{'music_sequence':1},
                 'musical_event_timeline':{'events':[{'at_s':1.2,'type':'beat_candidate'}]}}
        assert plugin._deliver_auditory_context(context,clip,2.0)
        pushed=pushes[0]
        assert pushed['visibility']==[] and pushed['ai_behavior']=='read'
        text=pushed['parts'][0]['text']
        assert '不构成发言请求' in text
        assert '"requests_response":false' in text
        assert '"display_in_chat":false' in text
        await plugin.shutdown()
    asyncio.run(run())


def test_windows_locked_settings_file_prevents_switch_and_keeps_old_model(plugin_fixture,monkeypatch):
    plugin,_,_,_=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._lyrics_profile='medium'
        plugin._lyrics_device_preference='cpu'
        plugin._environment.worker_runtime=lambda:(sys.executable,plugin.data_path('environment'))
        plugin._live_session_config={'lyrics_model':'base','lyrics_device':'cpu'}
        class Old:
            closed=False
            def close(self):
                self.closed=True
        old=Old()
        plugin._lyrics_session=old
        class New:
            def __init__(self,**kw):
                pass
            def prepare(self):
                return {'ready':True,'device':'cpu'}
            def close(self):
                pass
        module=sys.modules[plugin.__class__.__module__]
        monkeypatch.setattr(module,'PersistentLyricsSession',New)
        def cannot_save():
            raise PermissionError('[WinError 5] access denied')
        plugin._save_preferences=cannot_save
        msg=(await plugin.set_lyrics_quality('ultra')).value
        assert msg['switch_status']=='preparing'
        _await_status(plugin,'failed')
        assert plugin._lyrics_profile=='medium'
        assert plugin._lyrics_session is old and not old.closed
        assert plugin._live_session_config['lyrics_model']=='base'
        await plugin.shutdown()
    asyncio.run(run())


def test_preview_has_song_identity_for_latest_ai_query(plugin_fixture, monkeypatch):
    plugin, _, _, pushes = plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented = True
        plugin._ai_delivery = True
        plugin._listener_session = 1
        plugin._on_track_boundary('track_started', 23)
        current_id = plugin._song_current_id
        preview = FinishedClip(wave(6), 'early_lyrics_preview', 3,
                               offset_s=0., sequence=23, part_index=1)
        assert plugin._on_early_lyrics_clip(preview)
        job = plugin._lyrics_queue.get_nowait()
        assert job[1]['music_capture']['song_id'] == current_id
        def fake_asr(pcm, cfg):
            return {'status': 'asr_hypothesis','timed_cues': [
                {'at_s': 0.8, 'end_s': 2.2, 'text_preview': '测试歌声',
                 'provenance': 'local_whisper_multilingual_asr'}],
                'model': 'whisper-base', 'device': 'cpu'}
        monkeypatch.setattr(plugin, '_transcribe_singing_live_audio_isolated', fake_asr)
        plugin._process_one_lyrics_task(job[0], job[1], job[2], 1, plugin._lyrics_queue,
                                       quality_generation=job[3])
        assert plugin._recent_auditory_context['music_capture']['song_id'] == current_id
        assert pushes, 'Only a private, silent observation should have been queued'
        assert all(x['visibility'] == [] and x['ai_behavior'] == 'read' for x in pushes)
        monkeypatch.setattr(plugin, '_refresh_player_track', lambda: asyncio.sleep(0))
        latest = (await plugin.get_ai_auditory_context()).value
        assert latest['available']
        assert latest['song_scope'] == 'current'
        # New song must never re-use a stale lyrics scene, even if it is newer
        # than the older spectrum/event batches.
        plugin._on_track_boundary('track_ended', 23)
        plugin._on_track_boundary('track_started', 24)
        newer = (await plugin.get_ai_auditory_context()).value
        assert not newer['available']
        await plugin.shutdown()
    asyncio.run(run())


def test_profile_hot_switch_preference_write_failure_rolls_back(plugin_fixture, monkeypatch):
    plugin, _, _, _ = plugin_fixture
    async def run():
        await plugin.startup()
        plugin._auto_consented = True
        plugin._ai_delivery = True
        plugin._lyrics_profile = 'medium'
        plugin._lyrics_device_preference = 'cpu'
        plugin._live_session_config = {'lyrics_model': 'base', 'lyrics_device': 'cpu'}
        plugin._environment.worker_runtime = lambda: (sys.executable, plugin.data_path('environment'))
        class Old:
            def __init__(self):
                self.closed=False
            def close(self):
                self.closed=True
        old=Old()
        plugin._lyrics_session=old
        class New:
            def __init__(self, **kw):
                self.closed=False
            def prepare(self):
                return {'ready':True,'device':'cpu'}
            def close(self):
                self.closed=True
        module=sys.modules[plugin.__class__.__module__]
        monkeypatch.setattr(module,'PersistentLyricsSession',New)
        monkeypatch.setattr(plugin,'_save_preferences',lambda:(_ for _ in ()).throw(PermissionError('locked')))
        answer=(await plugin.set_lyrics_quality('ultra')).value
        assert answer['switch_status']=='preparing'
        _await_status(plugin,'failed')
        assert plugin._lyrics_profile=='medium'
        assert plugin._lyrics_device_preference=='cpu'
        assert plugin._lyrics_session is old and not old.closed
        assert plugin._live_session_config['lyrics_model']=='base'
        await plugin.shutdown()
    asyncio.run(run())


def test_turbo_download_includes_vocabulary_and_preprocessor():
    from music_perception.lyrics_model_store import ALLOWED_FILES
    assert 'vocabulary.json' in ALLOWED_FILES
    assert 'preprocessor_config.json' in ALLOWED_FILES
