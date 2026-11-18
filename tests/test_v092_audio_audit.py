"""Audit-specific regressions: four authentic model presets, privacy-bounded
preview, independent music/video timelines, unload safety and repeatable health
failure alerts. Tests use synthetic PCM and mocked workers; no web/GPU required.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import pytest
from music_perception import stem_catalog, stem_mdx, stem_postprocess, speech_timeline
from music_perception.model_assets import remove_stem_model,remove_plugin_model_dir,model_plan
from test_plugin import plugin_fixture


def test_four_models_have_distinct_pinned_weights_and_specific_shapes():
    expected={
        'low':('UVR_MDXNET_KARA_2.onnx',5120,2048,'instrumental',1.035),
        'medium':('UVR-MDX-NET-Inst_HQ_2.onnx',6144,3072,'instrumental',1.033),
        'high':('UVR-MDX-NET-Voc_FT.onnx',7680,3072,'vocals',1.021),
        'ultra':('UVR-MDX-NET-Inst_HQ_3.onnx',6144,3072,'instrumental',1.022),
    }
    for name,(file,fft,dim,primary,comp) in expected.items():
        info=stem_catalog.spec(name)
        assert info['file']==file and info['fft']==info['n_fft']==fft
        assert info['dim_f']==dim and info['output']==primary
        assert abs(info['compensation']-comp)<1e-8
        assert len(info['sha256'])==64
        assert info['bytes_estimate']>50_000_000
        assert len(stem_catalog.urls_for(name,'official'))==3
        assert stem_catalog.urls_for(name,'mirror')[0][1].startswith(stem_catalog.MODELSCOPE_STEM_BASE)
        assert stem_catalog.urls_for(name,'official')[0][0].startswith('Hugging Face')
        assert '/'+info['repo']+'/resolve/main/'+file in stem_catalog.urls_for(name)[1][1]
        assert model_plan('stems:'+name)['sha256']==info['sha256']


@pytest.mark.parametrize('fft,dim',[(5120,2048),(6144,3072),(7680,3072)])
def test_mdx_stft_istft_uses_correct_per_model_parameters(fft,dim):
    t=np.arange(stem_mdx.CHUNK, dtype=np.float32)/44100
    mix=np.stack([.12*np.sin(2*np.pi*330*t),.11*np.sin(2*np.pi*770*t)],axis=1).astype('float32')
    bins=stem_mdx.stft_channels(mix,fft=fft,dim_f=dim)
    assert bins.shape==(1,4,dim,256)
    inverse=stem_mdx.istft_channels(bins,fft=fft,dim_f=dim)
    # Cropping high frequencies loses only negligible energy for low-band tests.
    assert np.max(np.abs(inverse-mix)) < .002


@pytest.mark.parametrize('quality', ['low','medium','high','ultra'])
def test_four_profiles_construct_model_onnx_input_without_mismatch(tmp_path, monkeypatch,quality):
    item=stem_catalog.spec(quality)
    monkeypatch.setattr(stem_mdx,'verify_model',lambda _:True)
    class Input:
        name='input'
        shape=[1,4,item['dim_f'],256]
    class FakeSession:
        def get_inputs(self):return [Input()]
        def get_providers(self):return ['CPUExecutionProvider']
        def run(self,names,feed):return [feed['input']]
    engine=stem_mdx.TwoStemMDX(tmp_path/item['file'],session_factory=lambda *_:FakeSession())
    assert engine.fft==item['fft']
    times=np.arange(44100,dtype=np.float32)/44100
    audio=np.stack([.07*np.sin(2*np.pi*210*times),.04*np.sin(2*np.pi*480*times)],axis=1).astype('float32')
    result=engine.separate(audio,44100,strength='off')
    assert result['vocals'].shape==(16000,)
    assert result['instrumental'].shape==(16000,)
    assert result['enhanced_vocals'].shape==(16000,)
    assert np.isfinite(result['vocals']).all()
    assert np.isfinite(result['instrumental']).all()
    assert np.max(abs(result['vocals']))<.5


def test_spectral_enhancement_is_bounded_and_wet_dry_differs():
    sr=16000
    t=np.arange(sr*2,dtype=np.float32)/sr
    vocals=.12*np.sin(2*np.pi*200*t)+.015*np.sin(2*np.pi*1200*t)
    backing=.08*np.sin(2*np.pi*500*t)
    output=stem_postprocess.enhance_vocals(vocals,backing,strength='balanced')
    assert len(output)==len(vocals) and np.isfinite(output).all()
    assert np.max(abs(output))<=1.0
    assert np.array_equal(stem_postprocess.enhance_vocals(vocals,backing,strength='off'),vocals)
    assert float(np.mean(abs(output-vocals)))>0.000001
    assert np.max(abs(stem_postprocess.enhance_vocals(np.zeros(sr,dtype=np.float32),
                        np.zeros(sr,dtype=np.float32),strength='strong')))==0


def test_lyrics_and_video_have_separate_pause_rules():
    cues=[{'at_s':1.0,'end_s':2.0,'text_candidate':'甲'},
          {'at_s':3.2,'end_s':4.0,'text_candidate':'乙'},
          {'at_s':4.4,'end_s':4.8,'text_candidate':'丙'}]
    beats=[0.0+i*.4 for i in range(30)]
    m=speech_timeline.music_phrases(cues,beats)
    v=speech_timeline.video_sentences(cues,pause_split_s=1.8)
    assert m['mode']=='music_semantic_pause_phrase' and v['mode']=='video_speech_pauses'
    assert all('beat_index_estimated' in x for x in m['cues'])
    assert all('beat_index_estimated' not in x for x in v['cues'])
    assert v['sections_count']==1
    assert m['groups_count']==2 # 1.2 second gap splits musical phrases
    assert cues[1] == {'at_s':3.2,'end_s':4.0,'text_candidate':'乙'}


def test_model_uninstall_does_not_touch_other_tiers(tmp_path):
    root=tmp_path/'stems';root.mkdir()
    for label in ('low','medium','high','ultra'):
        (root/stem_catalog.spec(label)['file']).write_bytes(label.encode())
    assert remove_stem_model(root,'high')['ok']
    assert not (root/stem_catalog.spec('high')['file']).exists()
    for label in ('low','medium','ultra'):
        assert (root/stem_catalog.spec(label)['file']).read_bytes()==label.encode()
    with pytest.raises(ValueError):remove_stem_model(root,'../../../hi')


def test_uninstall_reports_locked_directory_and_restores_when_possible(tmp_path, monkeypatch):
    import shutil
    root=tmp_path/'whisper';directory=root/'small';directory.mkdir(parents=True)
    (directory/'model.bin').write_bytes(b'fixed')
    old=shutil.rmtree
    def deny(path,*args,**kwargs):raise PermissionError('simulated Windows file lock')
    monkeypatch.setattr(shutil,'rmtree',deny)
    result=remove_plugin_model_dir(root,'small')
    assert result['ok'] is False and result['bytes_removed']==0
    assert (root/'small'/'model.bin').read_bytes()==b'fixed'


def test_preview_three_tracks_never_appears_without_consent(plugin_fixture):
    plugin,*_=plugin_fixture
    async def task():
        await plugin.startup()
        from music_perception.auto_listener import FinishedClip
        clip=FinishedClip(np.zeros(16000,dtype='float32'),'live_progress',1,sequence=5)
        tone=np.sin(np.linspace(0,2*np.pi*400,16000)).astype('float32')*.16
        sep={'vocals':tone,'enhanced_vocals':tone*.7,'instrumental':tone*.2,'model':'audit.onnx'}
        plugin._auto_consented=False  # Deliberate explicit opt-out in a default-on application.
        plugin._remember_stem_preview(sep,clip)
        assert (await plugin.get_stem_preview(stage='raw')).value['ok'] is False
        plugin._auto_consented=True
        plugin._remember_stem_preview(sep,clip)
        outputs=[]
        for name in ('raw','enhanced','instrumental'):
            got=(await plugin.get_stem_preview(stage=name)).value
            assert got['ok'] and got['preview_only'] is True
            wav_bytes=base64.b64decode(got['data_uri'].split(',',1)[1])
            with wave.open(io.BytesIO(wav_bytes),'rb') as wav:
                assert wav.getframerate()==16000 and wav.getnchannels()==1
                assert wav.getnframes()==16000
            outputs.append(wav_bytes)
        assert len(set(outputs))==3
        plugin._clear_stem_preview()
        assert (await plugin.get_stem_preview(stage='raw')).value['ok'] is False
        assert not list(plugin.data_path().rglob('*.wav'))
        await plugin.shutdown()
    asyncio.run(task())


def test_preview_clip_is_capped_to_seven_seconds(plugin_fixture):
    plugin,*_=plugin_fixture
    async def task():
        await plugin.startup()
        from music_perception.auto_listener import FinishedClip
        plugin._auto_consented=True
        samples=np.ones(16000*14,dtype='float32')*.02
        clip=FinishedClip(np.zeros(16000,dtype='float32'),'live_progress',1,sequence=1)
        plugin._remember_stem_preview({'vocals':samples,'enhanced_vocals':samples,'instrumental':samples},clip)
        got=(await plugin.get_stem_preview()).value
        assert got['ok'] and got['duration_s']==7.
        with wave.open(io.BytesIO(base64.b64decode(got['data_uri'].split(',',1)[1])),'rb') as wav:
            assert wav.getnframes()==112000
        await plugin.shutdown()
    asyncio.run(task())


def test_auto_health_run_is_distinct_even_if_error_repeats(plugin_fixture):
    plugin,*_=plugin_fixture
    async def task():
        await plugin.startup()
        for index in (1,2):
            outcome=(await plugin.automatic_audio_check()).value
            assert outcome['started'] is True
            await plugin._auto_health_task
            status=(await plugin.get_status(_for_ui=True)).value
            health=status['automatic_health']
            assert health['run_id']==index
            assert health['state'] in {'limited','failed'}
            assert health.get('message') or health.get('error')
        await plugin.shutdown()
    asyncio.run(task())


def test_frontend_has_real_preview_modal_auto_repair_and_scroll():
    ui=(Path(__file__).resolve().parents[1]/'ui/panel.tsx').read_text('utf8')
    backend=(Path(__file__).resolve().parents[1]/'__init__.py').read_text('utf8')
    for action in ('get_stem_preview','uninstall_stem_model','uninstall_lyrics_model',
                   'automatic_audio_check','set_stem_enhancement','select_stem_quality'):
        assert ('"'+action+'"') in ui
        assert 'id=\''+action+'\'' in backend
    for token in ('data:audio/wav;base64,','<audio','overscrollBehavior','maxHeight:"min(56dvh, 470px)"',
                  'health.run_id','确认安全卸载'):
        assert token in ui
