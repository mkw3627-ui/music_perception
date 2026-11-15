"""v0.9.1: private audio bus, visual source reviews, fixed AI exclusion and UI.

These are offline safety/contract tests; NOT proof of real Windows WASAPI.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from music_perception import MusicPerceptionPlugin
from music_perception.audio_dispatch import PrivateAudioBus, SourceLabels, source_id
from music_perception.auto_listener import Decision
from test_plugin import plugin_fixture


def src(pid=501,*,ai=False,name='vlc.exe'):
    return {'pid':pid, 'created':'132000000000000000', 'name':name,
            'host_ai':ai,'audible':True,'tag':'unknown','label':'待确认'}


def test_private_bus_sends_once_on_change_and_retries_host_rejection():
    messages=[]
    responses=[{'submitted':False,'reason':'backpressure'}, {'submitted':True}]
    def push(**kwargs):
        messages.append(kwargs)
        return responses.pop(0) if responses else {'submitted':True}
    bus=PrivateAudioBus(push)
    assert not bus.publish('sources',{'a':1})
    assert bus.last_rejection=='backpressure'
    assert bus.publish('sources',{'a':1})
    assert not bus.publish('sources',{'a':1})
    assert bus.publish('sources',{'a':2})
    assert len(messages)==3
    assert all(x['visibility']==[] and x['ai_behavior']=='read' for x in messages)
    assert all(x['coalesce_key']=='music_perception_private_sources' for x in messages)
    for msg in messages:
        text=msg['parts'][0]['text']
        assert '"display_in_chat":false' in text
        assert '"requests_response":false' in text
        assert '"assistant_decides_whether_and_how_to_refer_to_data":true' in text


def test_private_bus_block_large_content_and_risky_topic():
    bus=PrivateAudioBus(lambda **kw: None)
    import pytest
    with pytest.raises(ValueError):bus.publish('../bad',{'small':True})
    with pytest.raises(ValueError):bus.publish('music',{'lyrics':'x'*49000})


def test_external_label_review_does_not_claim_native_video_capture():
    now=[0.0]
    labels=SourceLabels(clock=lambda: now[0],ttl_s=20)
    app=src()
    sid=source_id(app)
    assert labels.apply(app)['tag_status']=='candidate'
    out=labels.update(sid,'video',evidence='window_plugin',current_sources=[app])
    assert out['changed'] and out['state']=='external_review'
    info=labels.apply(app)
    assert info['tag']=='video' and info['tag_origin']=='window_plugin'
    assert info['confirmed'] is False
    assert labels.update(sid,'video',evidence='window_plugin',current_sources=[app])['changed'] is False
    # Short silent periods do not delete process-identity-labelled evidence.
    labels.drop_disappeared(set())
    assert labels.apply(app)['tag']=='video'
    # A new PID creation token does not inherit old visual review.
    new={**app,'created':'132000000000000001'}
    assert labels.apply(new)['tag_status']=='candidate'
    now[0]=25
    assert labels.apply(app)['tag_status']=='candidate'
    assert labels.update(sid,'music',evidence='ai_window_review',current_sources=[app])['changed']
    assert labels.update(sid,'unknown',evidence='ai_window_review',current_sources=[app])['changed']
    assert labels.apply(app)['tag_status']=='candidate'


def test_fixed_ai_tag_never_changes():
    labels=SourceLabels()
    ai=src(510,ai=True,name='N.E.K.O.exe')
    info=labels.apply(ai)
    assert info['tag']=='ai' and info['tag_status']=='fixed'
    import pytest
    with pytest.raises(ValueError,match='fixed_ai_host'):
        labels.update(source_id(ai),'video',evidence='window_plugin',current_sources=[ai])


def test_review_label_with_consent_checks_and_changed_once(plugin_fixture):
    plugin, *_=plugin_fixture
    messages=[]
    async def run():
        await plugin.startup()
        candidate=src()
        plugin._auto_consented=False   # Explicitly test revocation under v0.9.14 default-on policy.
        plugin._audible_scan_at=time.time()
        plugin._app_catalog_cache=[candidate]
        assert not (await plugin.review_audio_source_label(source_id(candidate),'music','window_plugin')).value['ok']
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin.push_message=lambda **kw: messages.append(kw) or {'submitted':True}
        first=(await plugin.review_audio_source_label(source_id(candidate),'video','window_plugin')).value
        assert first['ok'] and first['changed']
        assert plugin._app_catalog_cache[0]['tag']=='video'
        assert plugin._app_catalog_cache[0]['tag_status']=='external_review'
        second=(await plugin.review_audio_source_label(source_id(candidate),'video','window_plugin')).value
        assert second['ok'] and not second['changed']
        assert len(messages)==1
        assert 'private_active_application_sound_tags' in messages[0]['parts'][0]['text']
        plugin._selected_application={**candidate,'mode':'auto'}
        plugin._capture_mode='application'
        decision=plugin._decide_scene(Decision(False,.02,.55))
        assert decision['kind']=='video' and decision['tag']=='video'
        reset=(await plugin.review_audio_source_label(source_id(candidate),'unknown','window_plugin')).value
        assert reset['ok'] and reset['changed']
        assert plugin._app_catalog_cache[0]['tag']=='unknown'
        plugin._app_catalog_cache=[src(510,ai=True,name='N.E.K.O.exe')]
        blocked=(await plugin.review_audio_source_label(source_id(plugin._app_catalog_cache[0]),'music','window_plugin')).value
        assert not blocked['ok']
        await plugin.shutdown()
    asyncio.run(run())


def test_system_mix_blocks_when_fixed_ai_sound_audible(plugin_fixture):
    plugin,*_=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._capture_mode='system'
        plugin._app_catalog_cache=[src(510,ai=True,name='N.E.K.O.exe')]
        plugin._audible_scan_at=time.time()
        result=plugin._decide_scene(Decision(True,.95,.15))
        assert result['kind']=='ai_output'
        assert result['reason']=='fixed_ai_host_audible_on_mixed_output'
        plugin._capture_mode='application'
        plugin._selected_application={**src(),'mode':'music'}
        verified=plugin._decide_scene(Decision(True,.95,.15))
        assert verified['kind']=='unknown'  # A manual mode does not count as AI-reviewed label.
        await plugin.shutdown()
    asyncio.run(run())


def test_private_ai_source_listing_and_audio_panel_have_no_side_effects(plugin_fixture):
    plugin,*_=plugin_fixture
    async def run():
        await plugin.startup()
        plugin._app_catalog_cache=[src()]
        plugin._auto_consented=True;plugin._ai_delivery=True
        one=(await plugin.get_current_audio_sources_for_ai()).value
        assert one['available'] and len(one['channels'])==1
        assert one['channels'][0]['tag_status']=='candidate' or one['channels'][0].get('tag_status')=='candidate'
        panel=(await plugin.get_live_audio_content()).value
        assert panel['scene_kind']=='unknown'
        assert panel['video_speech_candidates']==[]
        assert panel['audio_frames_to_ai']=='not_supported_by_current_text_only_sdk'
        plugin._audio_scene={'kind':'video','verified':True,'source_count':1}
        plugin._video_transcript={'cues':[{'at_s':3.2,'text_candidate':'一段视频语音'}]}
        video=(await plugin.get_live_audio_content()).value
        assert video['video_speech_candidates'][0]['text_candidate']=='一段视频语音'
        guide=(await plugin.get_audio_tool_guide()).value
        assert guide['ok'] and len(guide['entrypoints'])>=50
        assert any(x['id']=='review_audio_source_label' for x in guide['entrypoints'])
        await plugin.shutdown()
    asyncio.run(run())


def test_ui_unifies_audio_without_manual_ai_state_switch():
    folder=Path(__file__).resolve().parents[1]
    ui=(folder/'ui/panel.tsx').read_text('utf-8')
    assert 'label:"音频"' in ui
    assert 'title="电脑音频感知"' in ui
    assert 'title="隔离 AI 自己的声音"' not in ui
    assert 'perform("set_ai_voice_playback"' not in ui
    assert 'audio.video_speech_candidates' in ui
    assert 'overscrollBehaviorY' in ui and 'maxHeight:"min(56dvh, 470px)"' in ui
    assert 'AI 主程序及子进程固定排除于其他应用' in ui


def test_distinct_song_boundaries_notify_separately_but_identical_one_is_deduped(plugin_fixture):
    plugin, *_=plugin_fixture
    messages=[]
    async def run():
        await plugin.startup()
        plugin._auto_consented=True;plugin._ai_delivery=True
        plugin.push_message=lambda **kw: messages.append(kw) or {'submitted':True}
        plugin._invalidate_pending_ai_music('track_ended',old_song_id='play:A',boundary_sequence=5)
        plugin._invalidate_pending_ai_music('track_ended',old_song_id='play:A',boundary_sequence=5)
        assert len(messages)==1
        plugin._invalidate_pending_ai_music('track_ended',old_song_id='play:B',boundary_sequence=6)
        assert len(messages)==2  # not the same track transition
        assert all('上一首歌曲' in m['parts'][0]['text'] for m in messages)
        await plugin.shutdown()
    asyncio.run(run())
