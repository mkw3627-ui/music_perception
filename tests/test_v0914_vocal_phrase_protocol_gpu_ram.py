"""Song lyric lines, chosen stem, bounded IPC and GPU-first RAM offload policy."""
from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import numpy as np
import pytest

from music_perception.adaptive_resources import select_model
from music_perception.speech_timeline import build_music_lines, music_phrases
from music_perception.lyrics_bridge import PersistentLyricsSession, LyricsWorkerError
from test_plugin import plugin_fixture


def test_song_sentence_not_split_by_beats_or_asr_chunks():
    parts=[{'at_s':1.0,'end_s':2.1,'text_preview':'我的心是午夜城'},
           {'at_s':2.15,'end_s':2.7,'text_preview':'因为沉待无人听了。'},
           {'at_s':3.1,'end_s':4.2,'text_preview':'安宁将被驱离。'}]
    beats=[i*.45 for i in range(30)]
    out=music_phrases(parts,beats)
    assert out['cues'][0]['phrase_index']==out['cues'][1]['phrase_index']
    assert out['cues'][2]['phrase_index'] !=out['cues'][0]['phrase_index']
    lines=build_music_lines(parts,beats)
    assert len(lines)==2 and lines[0]['source_cues']==2
    assert lines[0]['text_preview']=='我的心是午夜城因为沉待无人听了。'
    assert parts[0]=={'at_s':1.0,'end_s':2.1,'text_preview':'我的心是午夜城'}


def test_pause_and_sentence_markers_also_break_lines():
    lines=build_music_lines([
        {'at_s':0,'end_s':1,'text_preview':'I can'},
        {'at_s':1.02,'end_s':1.5,'text_preview':'hear you'},
        {'at_s':4.3,'end_s':5.4,'text_preview':'tonight'},
        {'at_s':5.42,'end_s':6,'text_preview':'!'},
        {'at_s':6.05,'end_s':8,'text_preview':'你好。'}])
    assert len(lines)==3
    assert lines[0]['text_preview']=='I can hear you'
    assert lines[1]['text_preview']=='tonight!'
    assert lines[2]['text_preview']=='你好。'


def test_missing_gpu_does_not_guess_shared_ram():
    out=select_model(preferred_tier='ultra',preferred_device='cuda',
       installed={'ultra':True},requirements={'ultra':9000},free_mib=None,
       allow_hybrid=True,hybrid_tiers=('ultra',),free_ram_mib=30000)
    assert out['device']=='cuda' and out['reason']=='gpu_unmeasured'


def test_qwen_gpu_then_hybrid_then_explicit_cpu():
    args=dict(preferred_tier='ultra',preferred_device='cuda',
              installed={'ultra':True},requirements={'ultra':9000},
              free_mib=4300,reserve_mib=2048,
              allow_hybrid=True,hybrid_tiers=('ultra',))
    gpu=select_model(**args,free_ram_mib=10000,allow_cpu=True)
    assert gpu['device']=='cuda' and gpu['placement']=='gpu_ram_weights'
    assert gpu['gpu_budget_mib']<9000 and gpu['cpu_budget_mib']<=8000
    blocked=select_model(**args,free_ram_mib=1000,allow_cpu=False)
    assert blocked['state']=='insufficient_vram' and blocked['requires_cpu_consent']
    fallback=select_model(**args,free_ram_mib=1000,allow_cpu=True)
    assert fallback['device']=='cpu'


def test_verified_lower_gpu_tier_before_hybrid():
    out=select_model(preferred_tier='ultra',preferred_device='cuda',
        installed={'ultra':True,'low':True},requirements={'ultra':9000,'low':1100},
        free_mib=4000,reserve_mib=2048,allow_hybrid=True,
        hybrid_tiers=('ultra',),free_ram_mib=20000)
    assert out['quality']=='low' and out['device']=='cuda'
    assert out.get('placement')!='gpu_ram_weights'


def _fake_worker(tmp_path, script_body):
    script=tmp_path/'worker.py';script.write_text(textwrap.dedent(script_body),encoding='utf-8')
    (tmp_path/'private_vendor').mkdir()
    (tmp_path/'model').mkdir()
    return PersistentLyricsSession(python_exe=sys.executable,vendor=tmp_path/'private_vendor',
        worker_script=script,model_dir=tmp_path/'model',device='cpu')


def test_noise_stdout_is_ignored_until_json_reply(tmp_path):
    session=_fake_worker(tmp_path, '''
        import sys,json
        while True:
            line=sys.stdin.buffer.readline()
            if not line:break
            request=json.loads(line)
            n=request['samples']
            if n: sys.stdin.buffer.read(n*4)
            sys.stdout.buffer.write(b'progress log from a model backend\\n')
            sys.stdout.buffer.write(b'\\n')
            data={'version':2,'id':request['id'],'ready':True,'device':'cpu'} if not n else \\
                {'version':2,'id':request['id'],'lyrics':{'device':'cpu','timed_cues':[]}}
            sys.stdout.buffer.write(json.dumps(data).encode()+b'\\n');sys.stdout.buffer.flush()
    ''')
    try:
        assert session.prepare()['device']=='cpu'
        assert session.transcribe(np.zeros(16000,np.float32))['device']=='cpu'
    finally:session.close()


def test_worker_oversized_reply_returns_specific_reason(tmp_path):
    session=_fake_worker(tmp_path, '''
        import sys
        sys.stdin.buffer.readline()
        sys.stdout.buffer.write(b'A' * 1048578 + b'\\n');sys.stdout.buffer.flush()
    ''')
    try:
        with pytest.raises(LyricsWorkerError,match='1MiB'):
            session.prepare()
    finally:session.close()


def test_placement_not_available_to_whisper_or_cpu(tmp_path):
    from music_perception.lyrics_bridge import PersistentLyricsSession
    script=tmp_path/'worker.py';script.write_text('pass')
    (tmp_path/'private_vendor').mkdir();(tmp_path/'model').mkdir()
    for model,device in [('base','cuda'),('sing-qwen-1.7b','cpu')]:
        with pytest.raises(LyricsWorkerError,match='仅适用于'):
            PersistentLyricsSession(python_exe=sys.executable,vendor=tmp_path/'private_vendor',
                worker_script=script,model_dir=tmp_path/'model',
                model_name=model,device=device,placement='gpu_ram_weights',
                gpu_budget_mib=2000,cpu_budget_mib=6000)


def test_selected_stem_input_is_saved_and_invalid_rejected(plugin_fixture):
    plugin,*_=plugin_fixture
    async def run():
        await plugin.startup()
        x=(await plugin.set_stem_vocal_input('raw')).value
        assert x['ok'] and plugin._stem_vocal_input=='raw'
        plugin._track_lyrics_history=[{'text_preview':'旧输入'}]
        prior=plugin._lyrics_quality_generation
        y=(await plugin.set_stem_vocal_input('enhanced')).value
        assert y['ok'] and plugin._stem_vocal_input=='enhanced'
        assert plugin._track_lyrics_history==[] and plugin._lyrics_quality_generation==prior+1
        failed=(await plugin.set_stem_vocal_input('raw_mixture')).value
        assert not failed['ok'] and plugin._stem_vocal_input=='enhanced'
        assert 'enhanced' in plugin._preferences_path().read_text('utf-8')
        await plugin.shutdown()
    asyncio.run(run())


def test_ui_exposes_both_vocal_choices_and_hybrid_toggle():
    from pathlib import Path
    root=Path(__file__).parents[1]
    front=(root/'ui/panel.tsx').read_text('utf8')
    backend=(root/'__init__.py').read_text('utf8')
    assert 'set_stem_vocal_input' in front and 'set_stem_vocal_input' in backend
    assert 'display_lines' in front and 'build_music_lines' in backend
    assert 'gpu_ram_offload' in front and 'gpu_ram_offload' in backend


def test_qwen_long_unpunctuated_song_never_invents_phrases():
    from music_perception.singing_backend import _timed_qwen_lines
    plain='今夜的风里还有一个故事' * 30
    cues=_timed_qwen_lines(plain,30,'Chinese')
    assert len(cues)==1
    assert cues[0]['time_alignment']=='estimated_phrase_span_not_word_aligned'
    assert cues[0]['at_s']==0
    assert cues[0]['end_s']==30


def test_comma_with_real_pause_breaks_and_beats_do_not():
    rows=[{'at_s':0.0,'end_s':1.6,'text_preview':'我走在街头，'},
          {'at_s':2.3,'end_s':4.2,'text_preview':'你却不在身边'},
          {'at_s':4.23,'end_s':5.3,'text_preview':'等待明天'}]
    output=build_music_lines(rows,[i*.5 for i in range(30)])
    assert len(output)==2
    assert output[0]['text_preview']=='我走在街头，'
    assert output[1]['text_preview']=='你却不在身边等待明天'


def test_crashed_asr_worker_reports_exit_code(tmp_path):
    session=_fake_worker(tmp_path, """
        import sys
        sys.stdin.buffer.readline()
        sys.exit(27)
    """)
    try:
        with pytest.raises(LyricsWorkerError, match='进程返回码：27'):
            session.prepare()
    finally:
        session.close()


def test_player_relative_time_is_kept_on_display_line():
    cues=[{'at_s':1.2,'end_s':1.8,'player_at_s_estimated':92.5,'text_preview':'走向远方'},
          {'at_s':1.85,'end_s':2.2,'player_at_s_estimated':93.15,'text_preview':'的路'}]
    display=build_music_lines(cues)
    assert len(display)==1
    assert display[0]['player_at_s_estimated']==92.5
    assert display[0]['text_preview']=='走向远方的路'


def test_native_worker_stderr_is_private_and_error_is_classified(tmp_path):
    worker=tmp_path/'crash_worker.py'
    worker.write_text('''import sys
sys.stdin.buffer.readline()
print("torch.cuda.OutOfMemoryError: CUDA out of memory",file=sys.stderr,flush=True)
sys.exit(14)
''',encoding='utf8')
    vendor=tmp_path/'vendor';vendor.mkdir()
    root=tmp_path/'model';root.mkdir()
    diag=tmp_path/'diagnostics.json'
    session=PersistentLyricsSession(python_exe=sys.executable,vendor=vendor,
            worker_script=worker,model_dir=root,model_name='base',device='cpu',
            diagnostics_file=diag)
    try:
        with pytest.raises(LyricsWorkerError) as err:
            session.prepare()
        message=str(err.value)
        assert '进程返回码：14' in message and 'CUDA 显存不足' in message
        assert 'OutOfMemoryError' not in message  # never echo raw stack traces to UI
        log=diag.with_name('diagnostics_stderr.log')
        assert 'OutOfMemoryError' in log.read_text('utf8')
    finally:session.close()


def test_qwen_weight_split_uses_transformers_gpu_and_cpu_budget(monkeypatch,tmp_path):
    import types
    import music_perception.singing_backend as backend
    inputs={}
    class FakeQwen:
        @classmethod
        def from_pretrained(cls,folder,**kwargs):
            inputs.update(kwargs)
            return types.SimpleNamespace(model=types.SimpleNamespace(hf_device_map={'encoder':0,'decoder':'cpu'}))
    fake_torch=types.SimpleNamespace(float16='float16',float32='float32',
        cuda=types.SimpleNamespace(is_available=lambda:True))
    monkeypatch.setitem(sys.modules,'qwen_asr',types.SimpleNamespace(Qwen3ASRModel=FakeQwen))
    monkeypatch.setitem(sys.modules,'torch',fake_torch)
    monkeypatch.setitem(sys.modules,'accelerate',types.SimpleNamespace())
    monkeypatch.setattr(backend,'verified_folder',lambda *args,**kwargs:tmp_path)
    obj,device=backend._open_model('sing-qwen-1.7b',tmp_path,'cuda',2,
        placement='gpu_ram_weights',gpu_budget_mib=2500,cpu_budget_mib=8000)
    assert device=='cuda' and inputs['device_map']=='auto'
    assert inputs['max_memory']=={0:'2500MiB','cpu':'8000MiB'}


def test_qwen_weight_split_rejects_false_gpu_residency(monkeypatch,tmp_path):
    import types
    import music_perception.singing_backend as backend
    class FakeQwen:
        @classmethod
        def from_pretrained(cls,folder,**kwargs):
            return types.SimpleNamespace(model=types.SimpleNamespace(hf_device_map={'all':'cpu'}))
    fake_torch=types.SimpleNamespace(float16='float16',float32='float32',
        cuda=types.SimpleNamespace(is_available=lambda:True))
    monkeypatch.setitem(sys.modules,'qwen_asr',types.SimpleNamespace(Qwen3ASRModel=FakeQwen))
    monkeypatch.setitem(sys.modules,'torch',fake_torch)
    monkeypatch.setitem(sys.modules,'accelerate',types.SimpleNamespace())
    monkeypatch.setattr(backend,'verified_folder',lambda *args,**kwargs:tmp_path)
    with pytest.raises(RuntimeError,match='拒绝将纯CPU误报GPU'):
        backend._open_model('sing-qwen-0.6b',tmp_path,'cuda',2,
            placement='gpu_ram_weights',gpu_budget_mib=2500,cpu_budget_mib=8000)
