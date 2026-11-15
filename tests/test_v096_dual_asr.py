"""Offline v0.9.6 dual engine, safe checkpoint management and accuracy gates."""
import hashlib
import importlib
import json
from pathlib import Path
import types

import numpy as np
import pytest

from music_perception import singing_backend as sb
from music_perception import singing_catalog as catalog
from music_perception import singing_quality_eval as evals
from music_perception import singing_install as install
from _version import ui_build, panel_ui_build, plugin_version  # noqa: E402


def _model_files(root, monkeypatch, quality='low'):
    spec=catalog.PROFILES[quality]
    monkeypatch.setitem(spec, 'minimum_weight_bytes', 8)
    folder=catalog.model_folder(root, quality)
    folder.mkdir(parents=True)
    blobs={'config.json':b'{"kind":"fake-test"}',
           'tokenizer.json':b'{"name":"test"}',
           'model.bin':b'123456789abcdefghijklmnop'}
    for name,data in blobs.items():
        (folder/name).write_bytes(data)
    monkeypatch.setitem(catalog.PINNED_WEIGHT_HASHES[quality],
                        'model.bin',hashlib.sha256(blobs['model.bin']).hexdigest())
    return folder


def test_four_song_tiers_are_distinct_and_video_preserved():
    assert set(catalog.PROFILES)=={'low','medium','high','ultra'}
    assert len({v['id'] for v in catalog.PROFILES.values()})==4
    assert set(catalog.MODEL_BY_ID) == {v['id'] for v in catalog.PROFILES.values()}
    assert catalog.PROFILES['ultra']['engine']=='qwen_asr'
    init=Path(__file__).parents[1]/'__init__.py'
    text=init.read_text('utf8')
    video=text.split('def _process_video_speech(',1)[1].split('def ',1)[0]
    music=text.split('def _process_one_lyrics_task(',1)[1].split('def ',1)[0]
    assert 'self._transcribe_live_audio_isolated(clip.samples,cfg)' in video
    assert 'self._transcribe_singing_live_audio_isolated(asr_audio,cfg)' in music
    assert '_transcribe_live_audio_isolated(asr_audio,cfg)' not in music
    assert 'result=self._transcribe_live_audio_isolated(clip.samples,cfg)' in video


def test_manually_installed_song_weights_get_offline_verified(monkeypatch,tmp_path):
    folder=_model_files(tmp_path,monkeypatch)
    assert catalog.verified_folder(tmp_path,'low') is None
    state=catalog.verify_manual_model(tmp_path,'low')
    assert state['ok'] and state['state']=='verified'
    assert catalog.verified_folder(tmp_path,'low',deep=True)==folder
    assert (folder/catalog.MANIFEST_NAME).exists()
    assert catalog.inventory(tmp_path)['low']['state']=='verified'


def test_corrupt_manual_model_is_rejected(monkeypatch,tmp_path):
    folder=_model_files(tmp_path,monkeypatch)
    (folder/'model.bin').write_bytes(b'CORRUPT_WEIGHT_'*2)
    assert catalog.verify_manual_model(tmp_path,'low')['ok'] is False
    assert catalog.verified_folder(tmp_path,'low') is None
    assert not (folder/catalog.MANIFEST_NAME).exists()


def test_manual_model_rejects_other_folders_and_symlinks(monkeypatch,tmp_path):
    other=tmp_path/'different'
    _model_files(other,monkeypatch)
    if (tmp_path/'sing-whisper-small').exists():
        pytest.skip('unexpected filesystem state')
    try:
        (tmp_path/'sing-whisper-small').symlink_to(other/'sing-whisper-small',target_is_directory=True)
    except (OSError,NotImplementedError):
        pytest.skip('symlinks are not enabled')
    assert catalog.verify_manual_model(tmp_path,'low')['ok'] is False
    assert catalog.uninstall_model(tmp_path,'low')['ok'] is False
    assert (other/'sing-whisper-small'/'model.bin').is_file()


def test_tampering_after_verified_check_detected(monkeypatch,tmp_path):
    folder=_model_files(tmp_path,monkeypatch)
    assert catalog.verify_manual_model(tmp_path,'low')['ok']
    weight=folder/'model.bin'
    weight.write_bytes(b'tamper'+weight.read_bytes())
    assert catalog.verified_folder(tmp_path,'low') is None
    assert catalog.verified_folder(tmp_path,'low',deep=True) is None


def test_uninstall_exact_tier_preserves_unrelated_files(monkeypatch,tmp_path):
    folder=_model_files(tmp_path,monkeypatch)
    other=tmp_path/'keep_me'
    other.mkdir()
    (other/'video_model.bin').write_bytes(b'video')
    assert catalog.uninstall_model(tmp_path,'low')['ok']
    assert not folder.exists()
    assert (other/'video_model.bin').read_bytes()==b'video'


def test_lyrics_offline_worker_validates_pcm_and_timestamps(monkeypatch):
    class Result: 
        text='这是歌唱的歌词。'
        language='Chinese'
    class FakeQwen:
        def transcribe(self,**kwargs):
            assert kwargs['audio'][1]==16000
            assert kwargs['language']=='Chinese'
            return [Result()]
    props=catalog.PROFILES['ultra']
    cache={'key':(props['id'],'C:/songs','cuda',2),
           'instance':FakeQwen(),'device':'cuda'}
    raw=np.full(16000*2,0.08,dtype=np.float32)
    out=sb.transcribe_singing_candidate(raw,model_dir='C:/songs',
           model_name=props['id'],model_cache=cache,device_policy='cuda',
           language_hint='zh')
    assert out['status']=='asr_hypothesis'
    assert out['engine']=='qwen_asr' and out['device']=='cuda'
    assert out['source_kind']=='separated_singing_vocals'
    assert out['timed_cues'][0]['time_alignment']=='estimated_phrase_span_not_word_aligned'
    assert out['warning'] and '5%' in out['warning']


def test_quiet_audio_does_not_hallucinate_lyrics():
    out=sb.transcribe_singing_candidate(np.zeros(16000,np.float32),
           model_dir='/unused',model_name='sing-qwen-0.6b')
    assert out['status']=='no_singing_detected'
    assert not out['timed_cues']
    with pytest.raises(ValueError):
        sb.transcribe_singing_candidate(np.full(16000,np.nan,dtype=np.float32),
              model_dir='/unused',model_name='sing-qwen-0.6b')


def test_video_lyrics_worker_keeps_old_whisper_names():
    work=(Path(__file__).parents[1]/'lyrics_worker.py').read_text('utf8')
    assert "if model_name.startswith('sing-')" in work
    assert 'music_perception.lyrics_backend' in work
    for name in ('tiny','base','small','large-v3-turbo'):
        assert f"'{name}'" in work
    assert "model_name.startswith('sing-')" in work


def test_private_dependency_paths_do_not_touch_system_packages(tmp_path):
    root,python,vendor=install.paths(tmp_path/'environment')
    assert root.parent == tmp_path/'environment'
    assert root.name=='singing_asr_venv'
    assert root in python.parents and root in vendor.parents
    assert 'site-packages' in str(vendor)
    assert install.SINGING_QWEN_PACKAGES[0]=='qwen-asr==0.0.6'


def test_target_metric_multilingual_and_threshold():
    rows=[{'source':'separated_vocals','language':'zh',
           'reference':'我喜欢这样的音乐','predicted':'我喜欢这样的音乐'},
          {'source':'separated_vocals','language':'ja',
           'reference':'桜が咲いた','predicted':'桜が咲いた'},
          {'source':'separated_vocals','language':'en',
           'reference':'I hear the sound','predicted':'I hear the sound'}]
    result=evals.measure(rows)
    # A single perfect phrase per language cannot validate the 5% goal.
    assert not result['validated'] and not result['passed']
    complete=[]
    for row in rows:
        for i in range(10):
            complete.append({**row,'reference':row['reference']*32,
                             'predicted':row['reference']*32})
    covered=evals.measure(complete)
    assert covered['validated'] and covered['passed']
    complete[-1]['predicted']='I hear'
    assert evals.measure(complete)['passed'] is False
    rows[0]['source']='whole_song_mix'
    with pytest.raises(ValueError):evals.measure(rows)


def test_incomplete_benchmark_cannot_falsely_pass():
    only_zh=[{'source':'separated_vocals','language':'zh',
              'reference':'听见了歌曲','predicted':'听见了歌曲'}]
    report=evals.measure(only_zh)
    assert not report['validated'] and not report['passed']


def test_ui_actions_registered_and_native_tsx():
    root=Path(__file__).parents[1]
    source=(root/'__init__.py').read_text('utf8')
    panel=(root/'ui/panel.tsx').read_text('utf8')
    for action in ('install_singing_model','activate_singing_model',
                   'uninstall_singing_model','verify_singing_import'):
        assert f'@ui.action(id=\'{action}\'' in source
        assert f'"{action}"' in panel
    assert "@ui.action(id='verify_singing_local_files'" in source  # backward compatible
    assert ui_build() in panel
    assert ui_build() in source
    assert '原模型保留' in panel
    assert '5%' in panel
