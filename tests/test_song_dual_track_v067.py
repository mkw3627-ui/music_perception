"""No private audio, network, system devices, or large pretrained models needed."""
from __future__ import annotations

import asyncio
import json

import numpy as np
from music_perception.auto_listener import FinishedClip
from music_perception.lyrics_continuity import join_song_audio, merge_song_lyrics
from music_perception.music_events import compact_music_context, offset_event_timeline
from music_perception.music_harmony import beat_accent_candidates, chord_change_candidates, metrical_grouping_candidates
from music_perception.song_change import SongChangeDetector
from music_perception.song_memory import RecentSongMemory


def _frame(freqs, bpm):
    sr=16000
    t=np.arange(32000,dtype=np.float32)/sr
    wave=sum(.07*np.sin(2*np.pi*f*t) for f in freqs)
    wave+=.15*(np.mod(t,60./bpm)<.05)
    return wave.astype(np.float32)


def test_audio_only_transition_requires_sustained_combined_melody_rhythm_timbre_change():
    detector=SongChangeDetector()
    before=_frame([220,261.63,329.63],120)
    after=_frame([293.66,370,440],78)
    first=[detector.feed(before,is_music=True) for _ in range(10)]
    assert not any(first), 'same music must not be split'
    new=[detector.feed(after,is_music=True) for _ in range(8)]
    signals=[x for x in new if x]
    assert len(signals)==1 and signals[0]['status']=='candidate_not_confirmed_identity'
    assert signals[0]['pitch_novelty']>.28
    assert detector.feed(after,is_music=False) is None


def test_no_false_boundary_from_simple_volume_change():
    detector=SongChangeDetector()
    before=_frame([220,261.63,329.63],120)
    for _ in range(12):
        assert detector.feed(before,is_music=True) is None
    for _ in range(10):
        assert detector.feed(before*.5,is_music=True) is None


def test_lyrics_overlap_only_consecutive_same_song():
    def c(seq,index,offset,n=160000):
        return FinishedClip(np.ones(n,dtype=np.float32), 'live_progress',5,
                            offset_s=offset,sequence=seq,part_index=index)
    first,tail,key=join_song_audio(c(7,1,0),previous_tail=None,previous_sequence=None)
    assert first.offset_s==0 and len(tail)==64000
    second,tail,key=join_song_audio(c(7,2,10),previous_tail=tail,previous_sequence=key)
    assert second.offset_s==6.0 and len(second.samples)==224000
    next_song,tail,key=join_song_audio(c(8,1,0),previous_tail=tail,previous_sequence=key)
    assert next_song.offset_s==0.0 and len(next_song.samples)==160000
    gap,_,_=join_song_audio(c(8,3,20),previous_tail=tail,previous_sequence=key)
    assert gap.offset_s==20.0


def test_lyrics_merge_same_line_across_4second_overlap_but_keep_later_chorus():
    old=[{'at_s':17.7,'text_preview':'你好世界'}]
    incoming=[{'at_s':18.1,'text_preview':'你好世界！'},
              {'at_s':65.2,'text_preview':'你好世界'}]
    joined=merge_song_lyrics(old,incoming)
    assert len(joined)==2 and joined[0]['at_s']==17.7
    assert joined[1]['at_s']==65.2


def test_song_store_has_only_two_even_after_restart(tmp_path):
    path=tmp_path/'music'/'recent_songs.json'
    mem=RecentSongMemory(path)
    for i in range(5):
        mem.begin(f'uuid:{i}',title=f'第{i}首')
        mem.observe(f'uuid:{i}',end_s=20+i,
                    lyrics=[{'at_s':2.,'text_preview':'候选'}])
        mem.end(f'uuid:{i}')
    assert [x['id'] for x in mem.recent()]==['uuid:3','uuid:4']
    record=json.loads(path.read_text(encoding='utf8'))
    assert len(record['songs'])==2
    assert 'uuid:2' not in json.dumps(record,ensure_ascii=False)
    assert [x['id'] for x in RecentSongMemory(path).recent()]==['uuid:3','uuid:4']
    assert not mem.observe('uuid:1',lyrics=[{'text_preview':'stale'}])


def test_chord_change_and_stronger_beats_preserve_uncertainty():
    # 2 s with 100 frames, clear C-major followed by D-minor chroma.
    chroma=np.zeros((12,200),dtype=np.float32)
    chroma[[0,4,7],:100]=np.array([1.,.85,.7])[:,None]
    chroma[[2,5,9],100:]=np.array([1.,.85,.7])[:,None]
    report=chord_change_candidates(chroma,sample_rate=16000,hop_length=160,duration=2)
    assert report['segments'] and report['status']=='estimated'
    assert len(report['changes'])>=1
    assert all(c['status']=='candidate_not_verified' for c in report['changes'])
    beat_times=list(np.arange(24)*.5)
    strength=np.zeros(1300,dtype=float)
    for i,t in enumerate(beat_times):
        strength[int(round(t*100))] = 3 if i%4==0 else .2
    accents=beat_accent_candidates(beat_times,strength,sample_rate=16000,hop_length=160)
    assert accents.get('events') and accents['events'][0]['status']=='strong_beat_not_verified_downbeat'
    groups=metrical_grouping_candidates(beat_times,accents)
    assert groups['status'] in {'weak_meter_hypothesis','no_reliable_meter'}


def test_offsets_of_accent_metrical_and_chord_are_added_exactly_once():
    event={'type':'beat_accent_candidate','at_s':.5}
    structural={'type':'metrical_downbeat_hypothesis','at_s':1.0}
    source={'events':[event,structural], 'beat_times_s':[.5], 'downbeat_times_s':[],
            'beat_accents':{'events':[event]},'metrical_downbeat_hypotheses':{'events':[structural]},
            'chord_progression':{'changes':[{'at_s':1.5,'to':'C:maj'}]},'lyrics':{}}
    moved=offset_event_timeline(source,20)
    assert moved['events'][0]['at_s']==20.5
    assert moved['beat_accents']['events'][0]['at_s']==20.5
    assert moved['metrical_downbeat_hypotheses']['events'][0]['at_s']==21.0
    assert moved['chord_progression']['changes'][0]['at_s']==21.5
    compact=compact_music_context(moved)
    assert compact['chord_progression']['changes'][0]['to']=='C:maj'


def test_track_state_retains_previous_and_current_not_third(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def check():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._listener_session=2
        for seq in (1,2,3):
            plugin._on_track_boundary('track_started',seq)
            plugin._on_track_boundary('track_ended',seq)
        songs=plugin._song_memory.recent()
        assert len(songs)==2
        assert [s['id'].split(':')[-1] for s in songs]==['2','3']
        answer=(await plugin.recent_two_songs()).value
        assert answer['ok'] and len(answer['songs'])==2
        plugin._ai_delivery=False
        assert not (await plugin.recent_two_songs()).value['ok']
        await plugin.shutdown()
    asyncio.run(check())


def test_player_authorized_local_lrc_needs_title_and_player_clock(tmp_path):
    from music_perception.player_lyrics import find_authorized_player_lrc, preview_player_lyrics
    root=tmp_path/'music'
    root.mkdir()
    (root/'Demo Song.lrc').write_text('[00:12.00]候选一句\n[00:15.00]second line',encoding='utf-8')
    found=find_authorized_player_lrc('Demo Song','', [root])
    assert len(found)==2
    assert find_authorized_player_lrc('Other Song','', [root])==[]
    result=preview_player_lyrics(found,music_clip_offset_s=20,
                                 media_clip_start_s=11,duration_s=8)
    assert result['status']=='user_local_lrc_estimated_sync'
    assert result['timed_cues'][0]['at_s']==21.0
    assert result['timed_cues'][0]['player_at_s_estimated']==12
    assert result['timed_cues'][0]['provenance']=='user_supplied_local_lrc'


def test_audio_boundary_invalidates_old_lrc_without_mixing_tracks(plugin_fixture):
    plugin,_,_,_=plugin_fixture
    async def check():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._listener_session=1
        plugin._track_metadata_key=('source','player','old','artist','0')
        plugin._player_lrc_lines=[{'at_s':2,'text':'旧歌词'}]
        plugin._active_track_sequence=4
        plugin._song_boundary_enforced=True
        plugin._on_track_boundary('track_ended_audio',4)
        assert plugin._player_lrc_lines==[]
        assert plugin._player_metadata_unreliable_key==('source','player','old','artist','0')
        plugin._on_track_boundary('track_started',5)
        assert not plugin._track_lyrics_history
        await plugin.shutdown()
    asyncio.run(check())


def test_lyrics_are_queued_before_expensive_chord_rhythm_analysis(plugin_fixture,monkeypatch):
    plugin,_,_,_=plugin_fixture
    async def check():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._listener_session=1
        plugin._on_track_boundary('track_started',1)
        plugin._live_session_config={'segment_seconds':10,'max_seconds':900,
                                     'emotion_mode':'fast'}
        clip=FinishedClip(_frame([220,260,330],120).repeat(5)[:160000],
                          'live_progress',5,offset_s=0,sequence=1,part_index=1)
        seen=[]
        def fake_analyze(samples,cfg):
            seen.append(plugin._lyrics_queue.qsize())
            return {'duration_s':10,'rhythm':{'bpm_estimate':120,'beat_times_s':[1,2]},
                    'harmony':{'estimated_key':{'name':'C'}},'emotion':{'label':'unknown'},
                    'music_events':{'schema':'music_events_v2','events':[],
                                    'beat_times_s':[], 'downbeat_times_s':[],
                                    'lyrics':{'status':'not_available','timed_cues':[]}},
                    'spectrum':{},'segments':[]}
        plugin._analyze_live_audio_isolated=fake_analyze
        plugin._deliver_auditory_context=lambda context, clip, position:None
        await plugin._process_live_clip(clip,expected_session=1)
        assert seen==[1], 'lyric ASR must run concurrently with rhythm worker'
        assert plugin._live_reports_completed==1
        await plugin.shutdown()
    asyncio.run(check())


def test_lyrics_queue_overflow_preserves_overlapping_clip_offsets(plugin_fixture):
    """Queue eviction must not revert to unextended PCM with shifted timestamps."""
    plugin,_,_,_=plugin_fixture
    async def check():
        await plugin.startup()
        plugin._auto_consented=True
        plugin._ai_delivery=True
        plugin._listener_session=1
        plugin._on_track_boundary('track_started',9)
        c1=FinishedClip(np.ones(160000,dtype=np.float32),'live_progress',5,
                        offset_s=0.,sequence=9,part_index=1)
        c2=FinishedClip(np.ones(160000,dtype=np.float32),'live_progress',5,
                        offset_s=10.,sequence=9,part_index=2)
        context1={'music_capture':{'offset_s':0}}
        context2={'music_capture':{'offset_s':10}}
        assert plugin._enqueue_lyrics(c1,context1,1)
        q=plugin._lyrics_queue
        q.put_nowait((c1,{},1))  # queue maxsize=2: now full
        assert plugin._enqueue_lyrics(c2,context2,1)
        queued=list(q.queue)
        last_clip=queued[-1][0]
        assert last_clip.offset_s==6.
        assert len(last_clip.samples)==224000
        assert context2['music_capture']['asr_capture_offset_s']==6.
        await plugin.shutdown()
    asyncio.run(check())


def test_zero_lyrics_overlap_does_not_keep_whole_previous_clip():
    a=FinishedClip(np.ones(160000,dtype=np.float32),'live_progress',5,
                   offset_s=0.,sequence=1,part_index=1)
    _,tail,_=join_song_audio(a,previous_tail=None,previous_sequence=None,overlap_seconds=0)
    assert tail.size==0


def test_song_storage_permission_error_does_not_crash_audio_state(tmp_path, monkeypatch):
    import music_perception.song_memory as song_memory
    store = song_memory.RecentSongMemory(tmp_path / 'music' / 'recent_songs.json')
    def locked(*args, **kwargs):
        raise PermissionError(13, 'Windows file lock')
    monkeypatch.setattr(song_memory.os, 'replace', locked)
    for i in range(3):
        store.begin(f'song:{i}')
        store.observe(f'song:{i}', chords=[{'at_s':i,'to':'C:maj'}])
        store.end(f'song:{i}')
    assert [x['id'] for x in store.recent()] == ['song:1','song:2']
    assert 'PermissionError' in store.last_save_error
    monkeypatch.undo()
    store.observe('song:2', lyrics=[{'at_s':3,'text_preview':'short'}])
    assert store.last_save_error is None
    assert len(song_memory.RecentSongMemory(store.path).recent()) == 2
