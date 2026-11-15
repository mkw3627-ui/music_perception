"""Resource-pressure, genuine device choice, ASR mutual exclusion and UI policies."""
from __future__ import annotations
import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from music_perception import MusicPerceptionPlugin
from music_perception.adaptive_resources import (select_model, gpu_memory,
    PressureTracker, DOMAINS, DEFAULT_RESERVE_MIB)

REQUIREMENTS={'low':1100,'medium':1600,'high':3000,'ultra':6000}
INSTALLED={t:True for t in REQUIREMENTS}


def policy(**k):
    return select_model(preferred_tier='ultra',preferred_device='cuda',
                        installed=dict(INSTALLED), requirements=dict(REQUIREMENTS),
                        free_mib=4200,reserve_mib=2048,**k)


def test_prefer_highest_verified_fitting_gpu_not_reserved_for_game():
    out=policy()
    assert out['state']=='ready' and out['quality']=='medium'
    assert out['reason']=='vram_pressure_downshift'
    assert out['device']=='cuda'


def test_free_gpu_does_not_install_missing_checkpoint():
    out=select_model(preferred_tier='ultra',preferred_device='cuda',
                     installed={**INSTALLED,'ultra':False},requirements=REQUIREMENTS,
                     free_mib=6000)
    assert out['state']=='model_missing'
    no_lower=select_model(preferred_tier='ultra',preferred_device='cuda',
                     installed={'ultra':True},requirements=REQUIREMENTS,
                     free_mib=4000,reserve_mib=2048)
    assert no_lower['state']=='insufficient_vram' and no_lower['requires_cpu_consent']


def test_user_consent_controls_cpu_ram_fallback():
    none=select_model(preferred_tier='ultra',preferred_device='cuda',
                      installed={'ultra':True},requirements=REQUIREMENTS,
                      free_mib=1000,reserve_mib=1024,allow_cpu=False)
    assert none['device'] is None and none['requires_cpu_consent']
    allowed=select_model(preferred_tier='ultra',preferred_device='cuda',
                      installed={'ultra':True},requirements=REQUIREMENTS,
                      free_mib=1000,reserve_mib=1024,allow_cpu=True)
    assert allowed['device']=='cpu' and allowed['quality']=='ultra'
    selected_cpu=select_model(preferred_tier='ultra',preferred_device='cpu',
                      installed={'ultra':True},requirements=REQUIREMENTS,
                      free_mib=0,reserve_mib=4096)
    assert selected_cpu['device']=='cpu'


def test_gpu_probe_good_bad_and_timeout():
    def ok(cmd,**kw):return SimpleNamespace(stdout='9000, 16384\n',returncode=0)
    out=gpu_memory(runner=ok,executable='nvidia-smi')
    assert out['available'] and out['free_mib']==9000 and out['total_mib']==16384
    def error(cmd,**kw):return SimpleNamespace(stdout='ERROR: driver',returncode=1)
    assert gpu_memory(runner=error,executable='nvidia-smi')['available'] is False
    def invalid(cmd,**kw):return SimpleNamespace(stdout='bad, data',returncode=0)
    assert gpu_memory(runner=invalid,executable='nvidia-smi')['available'] is False
    assert gpu_memory(executable='')['reason']=='nvidia-smi_not_found'


def test_unknown_gpu_is_never_fake_free_or_force_downshift():
    out=select_model(preferred_tier='ultra',preferred_device='cuda',
            installed={'ultra':True},requirements=REQUIREMENTS,free_mib=None)
    assert out['quality']=='ultra' and out['reason']=='gpu_unmeasured'


def test_gpu_release_credit_only_when_existing_model_is_stopped():
    blocked=policy()
    credited=policy(released_vram_mib=5800)
    assert blocked['quality']=='medium' and credited['quality']=='ultra'


def test_vram_pressure_requires_stable_samples_and_recovers():
    tracker=PressureTracker(2048)
    low={'available':True,'free_mib':500,'updated_at':time.time()}
    high={'available':True,'free_mib':10000,'updated_at':time.time()}
    assert tracker.update(low,now=10) is False
    assert tracker.update(low,now=16) is True
    assert tracker.update(high,now=30) is True
    assert tracker.pressured(now=80) is False


def test_model_domains_explicit_and_no_gpu_gpu_swap_semantics():
    assert set(DOMAINS)=={'video','singing','stem'}
    text=(Path(__file__).parents[1]/'adaptive_resources.py').read_text('utf8')
    assert 'unload' in text.lower()
    assert 'CPU' in text
    assert 'CUDA allocations directly' in text


class FakeChild:
    def __init__(self,name='child'):
        self.name=name;self.closed=False
    def close(self):self.closed=True


def make_shell():
    obj=object.__new__(MusicPerceptionPlugin)
    obj._asr_residency_lock=threading.RLock()
    obj._resource_cpu_consent={'video':False,'singing':False,'stem':False}
    obj._resource_effective={'video':{},'singing':{},'stem':{}}
    obj._lyrics_session=None
    obj._singing_session=None
    obj._singing_runtime_device=None
    obj._lyrics_runtime_device=None
    obj._lyrics_switch_lock=threading.RLock()
    obj._lyrics_switch_token=0
    obj._lyrics_preparing_session=None
    obj._lyrics_switch_state={}
    obj._lyrics_retired_sessions=[]
    obj._lyrics_model_loaded_at=None
    obj._lyrics_device_warning=None
    return obj


def test_video_releases_singing_before_loading_video(monkeypatch):
    shell=make_shell()
    singer=FakeChild('singing');shell._singing_session=singer
    shell._resource_choice=lambda domain:{'state':'ready','quality':'high',
       'device':'cuda','preferred_quality':'ultra','reason':'vram_pressure_downshift'}
    order=[]
    shell._transcribe_video_core=lambda samples,cfg: (order.append(cfg),{'device':'cuda'})[1]
    result=shell._transcribe_live_audio_isolated([0.1], {'lyrics_language':'zh'})
    assert singer.closed and shell._singing_session is None
    assert order[0]['lyrics_model']=='small' and order[0]['lyrics_device']=='cuda'
    assert result['adaptive_quality']=='high'


def test_video_no_cpu_fallback_without_user_permission():
    shell=make_shell()
    shell._resource_choice=lambda domain:{'state':'ready','quality':'ultra',
       'device':'cuda','preferred_quality':'ultra','reason':'gpu_unmeasured'}
    shell._transcribe_video_core=lambda samples,cfg:{'device':'cpu'}
    with pytest.raises(RuntimeError,match='CPU|CUDA'):
        shell._transcribe_live_audio_isolated([0.1],{})
    assert shell._resource_effective['video']['requires_cpu_consent']


def test_music_route_never_calls_video_asr(monkeypatch):
    shell=make_shell()
    prev=FakeChild('video');shell._lyrics_session=prev
    shell._resource_choice=lambda domain:{'state':'ready','quality':'medium',
       'device':'cpu','preferred_quality':'ultra','reason':'vram_pressure_user_approved_cpu'}
    seen=[]
    shell._transcribe_singing_core=lambda samples,cfg,plan:(seen.append(plan),{'device':'cpu'})[1]
    result=shell._transcribe_singing_live_audio_isolated([0.3],{})
    assert prev.closed and shell._lyrics_session is None
    assert seen[0]['quality']=='medium' and result['adaptive_quality']=='medium'


@pytest.mark.asyncio
async def test_consent_action_rejects_false_confirmation_and_persists(monkeypatch):
    shell=make_shell()
    shell._resource_auto_enabled=True
    shell._resource_reserve_mib=2048
    shell._resource_cpu_consent={'video':False,'singing':False,'stem':False}
    shell._resource_lock=threading.RLock()
    shell._resource_probe=PressureTracker(2048)
    saved=[];shell._save_preferences=lambda:saved.append(True)
    denied=await shell.set_adaptive_resources(domain='singing',cpu_fallback=True,confirmed=False)
    assert not denied.value['ok'] and shell._resource_cpu_consent['singing'] is False
    allowed=await shell.set_adaptive_resources(domain='singing',cpu_fallback=True,
                                           reserve_mib=4096,confirmed=True)
    assert allowed.value['ok'] and saved
    assert shell._resource_cpu_consent['singing'] is True
    assert shell._resource_reserve_mib==4096
    assert not (await shell.set_adaptive_resources(domain='stem',reserve_mib=5000,
                                                  confirmed=True)).value['ok']


def test_ui_contains_symmetric_cpu_gpu_model_sections_and_confirmations():
    source=(Path(__file__).parents[1]/'ui/panel.tsx').read_text('utf8')
    backend=(Path(__file__).parents[1]/'__init__.py').read_text('utf8')
    for text in ('专业歌声歌词 · 中日英','视频人物说话 · Whisper',
                 '人声 / 伴奏分离','显存 · 自适应调度','gpu-shortage',
                 '改用内存（CPU）','仍尝试显存（GPU）','set_adaptive_resources'):
        assert text in source
    for text in ('_asr_residency_lock','_resource_choice(\'stem\')',
                 "_resource_choice('video')","_resource_choice('singing')"):
        assert text in backend
