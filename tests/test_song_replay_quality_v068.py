"""v0.6.8: track isolation, replay reuse, quality selection and newest-song AI scope."""
from __future__ import annotations
import asyncio
import json
import numpy as np
from music_perception.song_memory import RecentSongMemory,trusted_song_identity,QUALITY_MODEL,audio_segment_fingerprint
from music_perception.auto_listener import FinishedClip
from test_plugin import plugin_fixture


def metadata(title,artist='Composer',duration=193,source='windows_smtc'):
    return {'available':True,'source':source,'title':title,
            'artist':artist,'duration_s':duration,'player':'Player X'}


def test_metadata_identity_rejects_title_only_without_duration():
    assert trusted_song_identity(metadata('Track',artist='',duration=0)) is None
    assert trusted_song_identity(metadata('Track',artist='',duration=180))
    assert trusted_song_identity(metadata('Track','Alice',duration=200)) != trusted_song_identity(metadata('Track','Bob',duration=200))
    assert trusted_song_identity(metadata('Track','Alice',duration=200)) != trusted_song_identity(metadata('Track','Alice',duration=202))
    assert trusted_song_identity(metadata('Track','Alice',duration=200)) == trusted_song_identity(metadata('  track  ','alice',duration=200))


def test_replay_two_distinct_slots_and_event_and_lyrics_per_song(tmp_path):
    store=RecentSongMemory(tmp_path/'songs.json')
    a=trusted_song_identity(metadata('Song A'))
    b=trusted_song_identity(metadata('Song B'))
    assert store.begin('first:A',title='A',identity=a)['reused'] is False
    assert store.observe('first:A',end_s=20,lyrics=[{'at_s':4.0,'text_preview':'first track only'}],lyrics_quality='base')
    assert store.cache_segment('first:A',start_s=0,duration_s=20,timeline={
        'beat_times_s':[.5,1.], 'lyrics':{'status':'asr_hypothesis','timed_cues':[{'at_s':4.0}]}})
    store.end('first:A')
    store.begin('second:B',title='B',identity=b)
    store.observe('second:B',lyrics=[{'at_s':5.,'text_preview':'second track only'}],lyrics_quality='small')
    store.end('second:B')
    result=store.begin('third:A',title='A',identity=a)
    assert result['reused'] and result['play_count']==2
    assert [e['title'] for e in store.recent()]==['B','A']
    assert store.cached_lyrics('third:A',quality='base')[0]['text_preview']=='first track only'
    assert store.cached_lyrics('third:A',quality='small')==[]
    assert store.cached_segment('third:A',start_s=.6,duration_s=20)
    assert not store.cached_segment('third:A',start_s=130,duration_s=20)
    assert not store.observe('first:A',lyrics=[{'at_s':33,'text_preview':'stale'}])
    # Replaying A doesn't consume a third distinct-song slot.
    assert len(RecentSongMemory(store.path).recent())==2
    store.begin('fourth:C',title='C',identity=trusted_song_identity(metadata('Song C')))
    assert [e['title'] for e in store.recent()]==['A','C']
    assert not any('second track only' in json.dumps(e) for e in store.recent())


def test_late_metadata_identification_only_when_empty(tmp_path):
    store=RecentSongMemory(tmp_path/'songs.json')
    identity=trusted_song_identity(metadata('Original'))
    store.begin('old',title='Original',identity=identity)
    store.end('old')
    store.begin('other',title='Other',identity=trusted_song_identity(metadata('Other')))
    store.end('other')
    # Old slot may be evicted when an unknown third starts; no unsafe reuse.
    store.begin('fresh:unknown')
    result=store.identify('fresh:unknown',identity,title='Original')
    assert not result.get('reused')
    assert store.current()['identity']==identity
    assert len(store.recent())==2


def test_legacy_quality_entry_cannot_activate_missing_weights(plugin_fixture):
    """v0.8+: legacy setters must respect download/SHA/inference gates."""
    plugin,_,_,_=plugin_fixture
    async def check():
        await plugin.startup()
        assert plugin._lyrics_profile is None
        assert (await plugin._settings())['lyrics_model']=='base'
        a=(await plugin.set_lyrics_quality('high')).value
        b=(await plugin.set_lyrics_quality('low')).value
        assert a['ok'] is False and a['next_action']=='stage_lyrics_model'
        assert b['ok'] is False and b['next_action']=='stage_lyrics_model'
        assert (await plugin.set_lyrics_quality('not-a-tier')).value['ok'] is False
        assert plugin._lyrics_profile is None
        assert not plugin._preferences_path().exists()
        await plugin.shutdown()
    asyncio.run(check())


def test_reused_segment_skips_expensive_music_analysis_and_lyrics_asr(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def check():
        await plugin.startup()
        plugin._auto_consented=True;plugin._ai_delivery=True
        plugin._singing_profile='low'
        plugin._listener_session=3
        plugin._track_identity=trusted_song_identity(metadata('Song A'))
        plugin._track_display_name='Song A'
        plugin._on_track_boundary('track_started',1)
        first=plugin._song_current_id
        samples=np.sin(np.arange(160000,dtype=np.float32)*2*np.pi*440/16000).astype(np.float32)*.05
        plugin._song_memory.cache_segment(first,start_s=0,duration_s=10,timeline={
            'schema':'music_events_v2','beat_times_s':[0.5,1.],
            'chord_progression':{'changes':[]},'lyrics':{'status':'asr_hypothesis','timed_cues':[]}},
            fingerprint=audio_segment_fingerprint(samples))
        plugin._song_memory.observe(first,lyrics_quality='sing-whisper-small',
            lyrics=[{'at_s':3.0,'text_preview':'old lyrics in Song A'}])
        plugin._on_track_boundary('track_ended',1)
        # This replay is confirmed by a fresh player metadata event.
        plugin._track_identity=trusted_song_identity(metadata('Song A'))
        plugin._track_display_name='Song A'
        plugin._on_track_boundary('track_started',2)
        assert plugin._song_reused and plugin._song_memory.current()['play_count']==2
        plugin._analyze_live_audio_isolated=lambda *args: (_ for _ in ()).throw(AssertionError('should reuse music scene'))
        plugin._enqueue_lyrics=lambda *args: (_ for _ in ()).throw(AssertionError('should reuse lyrics'))
        plugin._deliver_auditory_context=lambda *args:True
        plugin._live_session_config={'lyrics_model':'base','max_seconds':900,'segment_seconds':10}
        clip=FinishedClip(samples,'live_progress',5,
                          offset_s=0,sequence=2,part_index=1)
        await plugin._process_live_clip(clip,expected_session=3)
        latest=plugin._recent_auditory_context
        assert latest['music_capture']['song_id']==plugin._song_current_id
        assert latest['musical_event_timeline']['replay_cache']['reused']
        assert latest['musical_event_timeline']['lyrics']['timed_cues'][0]['text_preview']=='old lyrics in Song A'
        assert plugin._live_cache_reuses==1
        assert plugin._song_memory.current()['title']=='Song A'
        await plugin.shutdown()
    asyncio.run(check())


def test_ai_this_song_prefers_newest_and_never_old_stale_context(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def check():
        await plugin.startup()
        plugin._auto_consented=True;plugin._ai_delivery=True
        plugin._listener_session=4
        async def no_refresh():return None
        plugin._refresh_player_track=no_refresh
        plugin._track_identity=trusted_song_identity(metadata('Previous'))
        plugin._track_display_name='Previous'
        plugin._on_track_boundary('track_started',1)
        plugin._recent_auditory_context={'music_capture':{'song_id':plugin._song_current_id,
            'music_sequence':1},'musical_event_timeline':{'beat_times_s':[1]}}
        assert (await plugin.get_ai_auditory_context()).value['available']
        plugin._on_track_boundary('track_ended',1)
        plugin._track_identity=trusted_song_identity(metadata('Latest'))
        plugin._track_display_name='Latest'
        plugin._on_track_boundary('track_started',2)
        waiting=(await plugin.get_ai_auditory_context()).value
        assert not waiting['available'] and waiting['song_scope']=='current'
        previous=(await plugin.get_ai_auditory_context(song_scope='previous')).value
        assert previous['song']['title']=='Previous'
        plugin._recent_auditory_context={'music_capture':{'song_id':plugin._song_current_id,
            'music_sequence':2},'musical_event_timeline':{'beat_times_s':[2]}}
        current=(await plugin.get_ai_auditory_context()).value
        assert current['available'] and current['current_song']['title']=='Latest'
        assert current['context']['musical_event_timeline']['beat_times_s']==[2]
        await plugin.shutdown()
    asyncio.run(check())


def test_high_tier_uses_official_multilingual_small_not_english_only():
    from music_perception.lyrics_model_store import MODELS
    from music_perception.lyrics_backend import _make_model
    assert MODELS['small']['repo']=='Systran/faster-whisper-small'
    assert MODELS['small']['sha256']=='3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671'
    calls=[]
    def factory(*args,**kwargs):
        calls.append((args,kwargs))
        return object()
    _,device=_make_model('small','/tmp/test-small',factory,device_policy='cpu')
    assert device=='cpu' and calls[0][0]==('small',)
    assert calls[0][1]['compute_type']=='int8'


def test_changed_tier_invalidates_old_results_in_flight(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def check():
        await plugin.startup()
        plugin._auto_consented=True;plugin._ai_delivery=True
        plugin._listener_session=1
        plugin._on_track_boundary('track_started',1)
        before=plugin._lyrics_quality_generation
        attempt=(await plugin.set_lyrics_quality('high')).value
        assert not attempt['ok']
        assert plugin._lyrics_quality_generation==before  # rejected candidate cannot poison live ASR
        await plugin.shutdown()
    asyncio.run(check())


def test_empty_successful_lyrics_segment_is_not_retranscribed_on_replay(tmp_path):
    store=RecentSongMemory(tmp_path/'songs.json')
    identity=trusted_song_identity(metadata('Song with instrumental intro'))
    store.begin('play:1',identity=identity,title='Instrumental intro')
    assert store.mark_lyrics_scanned('play:1',start_s=0,end_s=20,quality='base')
    store.end('play:1')
    assert store.begin('play:2',identity=identity)['reused']
    assert store.lyrics_already_scanned('play:2',start_s=.5,end_s=19.8,quality='base')
    assert not store.lyrics_already_scanned('play:2',start_s=21,end_s=42,quality='base')
    assert not store.lyrics_already_scanned('play:2',start_s=0,end_s=20,quality='small')


def test_status_song_summary_does_not_embed_full_timeline(tmp_path):
    store=RecentSongMemory(tmp_path/'songs.json')
    store.begin('play:1',identity=trusted_song_identity(metadata('Long song')))
    assert store.cache_segment('play:1',start_s=0,duration_s=20,timeline={
        'chord_progression':{'changes':[{'at_s':2.1,'from':'C','to':'G'}]},
        'beat_times_s':[.5,1,1.5], 'some_long_string':'x'*8000})
    full=store.recent()
    small=store.recent_summary()
    assert full[0]['observed_segments']
    assert 'observed_segments' not in small[0]
    assert small[0]['cached_segment_count']==1
    assert 'x'*100 not in json.dumps(small)


def test_latest_song_query_rejects_scene_with_same_sequence_but_old_song_id(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def check():
        await plugin.startup()
        plugin._auto_consented=True; plugin._ai_delivery=True
        plugin._listener_session=1
        async def no_refresh(): return None
        plugin._refresh_player_track=no_refresh
        plugin._track_identity=trusted_song_identity(metadata('Track with distinct ID'))
        plugin._track_display_name='Track with distinct ID'
        plugin._on_track_boundary('track_started',1)
        latest_id=plugin._song_current_id
        plugin._recent_auditory_context={
            'music_capture':{'song_id':'stale-older-track','music_sequence':1},
            'musical_event_timeline':{'beat_times_s':[1]}}
        result=(await plugin.get_ai_auditory_context()).value
        assert result['available'] is False
        assert result['song_scope']=='current'
        assert '旧歌' in result['reason']
        plugin._recent_auditory_context['music_capture']['song_id']=latest_id
        assert (await plugin.get_ai_auditory_context()).value['available'] is True
        await plugin.shutdown()
    asyncio.run(check())


def test_quality_change_private_notice_is_not_false_song_change(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def check():
        await plugin.startup()
        plugin._auto_consented=True; plugin._ai_delivery=True
        receipts=[]
        plugin.push_message=lambda **kw: receipts.append(kw) or {'submitted':True}
        plugin._invalidate_pending_ai_music('lyrics_quality_changed')
        assert len(receipts)==1
        assert receipts[0]['visibility']==[]
        assert receipts[0]['ai_behavior']=='read'
        txt=receipts[0]['parts'][0]['text']
        assert '当前歌曲并未切换' in txt
        assert '上一首歌曲的歌词、节拍' not in txt
        await plugin.shutdown()
    asyncio.run(check())


def test_fingerprint_guard_prevents_reuse_when_same_song_played_from_different_position(tmp_path):
    store=RecentSongMemory(tmp_path/'songs.json')
    identity=trusted_song_identity(metadata('Track'))
    a=np.sin(np.arange(160000,dtype=np.float32)*2*np.pi*440/16000).astype(np.float32)*.05
    b=np.sin(np.arange(160000,dtype=np.float32)*2*np.pi*523/16000).astype(np.float32)*.05
    fa=audio_segment_fingerprint(a)
    fb=audio_segment_fingerprint(b)
    assert fa and fb and fa!=fb
    store.begin('first',identity=identity)
    store.cache_segment('first',start_s=0,duration_s=10,timeline={'beat_times_s':[1]},
                        fingerprint=fa,media_start_s=10)
    store.end('first')
    assert store.begin('replay',identity=identity)['reused']
    assert store.cached_segment('replay',0,10,fingerprint=fa,media_start_s=10)
    assert store.cached_segment('replay',0,10,fingerprint=fb,media_start_s=10) is None
    assert store.cached_segment('replay',0,10,fingerprint=fa,media_start_s=60) is None
    assert store.cached_segment('replay',0,10,fingerprint='pcm16k:unknown') is None
    assert audio_segment_fingerprint(np.zeros(160000,dtype=np.float32)) is None



def test_rejected_model_change_keeps_active_ai_history(plugin_fixture):
    """An unavailable ASR tier must not rewrite active audio/lyric context."""
    plugin,_,_,_=plugin_fixture
    async def check():
        await plugin.startup()
        plugin._auto_consented=True;plugin._ai_delivery=True
        plugin._song_boundary_enforced=False
        plugin._recent_auditory_context={'music_capture':{'music_sequence':1},
            'lyrics_understanding':{'timed_cues':[{'text_preview':'base result'}]},
            'musical_event_timeline':{'lyrics':{'timed_cues':[{'text_preview':'base result'}]}}}
        original=json.dumps(plugin._recent_auditory_context,sort_keys=True)
        result=(await plugin.set_lyrics_quality('high')).value
        assert not result['ok'] and result['next_action']=='stage_lyrics_model'
        assert json.dumps(plugin._recent_auditory_context,sort_keys=True)==original
        await plugin.shutdown()
    asyncio.run(check())

def test_natural_song_end_clears_stale_player_identity_and_local_lyrics(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def check():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._listener_session=1
        old=trusted_song_identity(metadata('Previous old player title'))
        plugin._track_identity=old
        plugin._track_metadata_key=('windows_smtc','X','previous','Artist','None')
        plugin._track_display_name='Previous old player title'
        plugin._player_lrc_lines=[{'at_s':0.0,'text':'stale song lyric'}]
        plugin._on_track_boundary('track_started',1)
        plugin._on_track_boundary('track_ended',1)
        assert plugin._track_identity is None
        assert plugin._track_display_name is None
        assert not plugin._player_lrc_lines
        assert plugin._player_metadata_unreliable_key == plugin._track_metadata_key
        plugin._on_track_boundary('track_started',2)
        assert plugin._song_reused is False
        assert plugin._song_memory.current().get('identity') is None
        await plugin.shutdown()
    asyncio.run(check())


def test_fresh_media_boundary_preserves_new_track_identity(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def check():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._listener_session=1
        plugin._track_display_name='Old'
        plugin._track_identity=trusted_song_identity(metadata('Old'))
        plugin._on_track_boundary('track_started',1)
        new_identity=trusted_song_identity(metadata('New'))
        # Media controller has already delivered verified New metadata.
        plugin._track_display_name='New'
        plugin._track_identity=new_identity
        plugin._on_track_boundary('track_ended_metadata',1)
        assert plugin._track_display_name=='New'
        assert plugin._track_identity==new_identity
        plugin._on_track_boundary('track_started',2)
        assert plugin._song_memory.current()['identity']==new_identity
        assert plugin._song_memory.current()['title']=='New'
        await plugin.shutdown()
    asyncio.run(check())
