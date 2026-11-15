"""Timestamped *observations* derived from system audio, not a verdict.

The light backend uses the existing librosa rhythm analysis and an
harmonic/percussive decomposition. Neither a spectral attack nor HPSS is a
verified drum instrument or a separated lead melody. Accurate downbeats need
an optional *trained* tracker, e.g. Beat This!, with a locally provisioned
checkpoint. No model weights are downloaded by this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

SR = 16000


def _seconds(value: Any, offset: float = 0.0) -> float:
    return round(max(0.0, float(value) + offset), 3)


def _times(raw: Any, duration: float) -> list[float]:
    """Stable, bounded timestamp extraction, never manufacture beat times."""
    import numpy as np
    xs = np.asarray(raw, dtype=float).ravel()
    return sorted(set(round(float(t), 4) for t in xs
                      if np.isfinite(t) and 0 <= float(t) <= duration))


def predict_beat_this(wave: Any, *, checkpoint: str | Path, sample_rate: int = SR,
                      predictor=None) -> tuple[list[float], list[float]]:
    """Predict beats/downbeats from a *user-installed* Beat This! checkpoint.

    The official Audio2Beats API takes (mono audio ndarray, sample rate) and
    returns (beat timestamps, downbeat timestamps), confirmed against the
    Postprocessor return value. Unlike the basic backend,
    this really contains learned downbeat candidates. Model isn't bundled.
    """
    import numpy as np
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError('Beat This! 权重未安装；不会自动从网络下载大型模型')
    if predictor is None:
        try:
            from beat_this.inference import Audio2Beats
        except ImportError as exc:
            raise RuntimeError('未安装可选节拍模型 beat-this；基础节拍仍然可用') from exc
        predictor = Audio2Beats(checkpoint_path=str(path), device='cpu', dbn=False)
    audio = np.asarray(wave, dtype=np.float32).reshape(-1)
    if len(audio) < sample_rate:
        raise ValueError('需要至少一秒音乐才能检测重拍')
    beats, downbeats = predictor(audio, sample_rate)
    duration = len(audio) / sample_rate
    return _times(beats, duration), _times(downbeats, duration)


def harmonic_percussive_observations(wave: Any, *, rate: int = SR,
                                     max_attacks: int = 2400) -> dict:
    """A frequency-domain proxy for tonal-vs-percussive activity.

    Cannot distinguish singer from guitar, bass from drums, or backing vocals
    from lead vocals; those require a pretrained source separation model.
    """
    import librosa
    import numpy as np
    x = np.asarray(wave, dtype=np.float32).reshape(-1)
    if len(x) < 1024:
        return {'method': 'hpss_spectral_proxy', 'status': 'insufficient_audio',
                'percussive_attack_times_s': [], 'tonal_share': None, 'percussive_share': None}
    hop = 256
    stft = librosa.stft(x, n_fft=1024, hop_length=hop)
    harmonic, percussive = librosa.decompose.hpss(stft)
    eh = float(np.sum(np.abs(harmonic)**2))
    ep = float(np.sum(np.abs(percussive)**2))
    denom = eh+ep+1e-12
    onset = librosa.onset.onset_strength(S=librosa.amplitude_to_db(np.abs(percussive)+1e-8),
                                         sr=rate, hop_length=hop)
    if not len(onset) or float(np.max(onset)) < 0.05:
        attacks = []
    else:
        frames = librosa.onset.onset_detect(onset_envelope=onset, sr=rate,
                                            hop_length=hop, backtrack=False)
        attacks = [round(float(frame)*hop/rate, 3) for frame in frames if frame*hop <= len(x)]
        # Onsets can be multiple per beat, and not all onsets are drums.
        attacks = attacks[:max_attacks]
    return {
        'method': 'hpss_spectral_proxy',
        'status': 'estimated',
        'tonal_share': round(eh/denom, 3),
        'percussive_share': round(ep/denom, 3),
        'percussive_attack_times_s': attacks,
        'not_true_stems': True,
        'cannot_isolate_voice': True,
        'warning': 'HPSS 仅粗略区分持续谐波和瞬态打击，不等于分离人声、鼓、贝斯和伴奏',
    }


def timed_music_events(report: dict, *, waveform: Any | None = None,
                       offset_s: float = 0.0, time_basis: str = 'music_only_elapsed',
                       beat_this_checkpoint: str | Path | None = None,
                       local_lyrics: list[dict] | None = None,
                       lyrics_aligned: bool = False) -> dict:
    """Convert acoustics into a compact, annotated event timeline.

    Offsets refer to *accepted audio duration* unless a caller provides a
    trustworthy media clock. Skipped non-music gaps make this clock diverge
    from real song progress; never label it a media/player timestamp.
    """
    duration = float(report.get('duration_s') or 0)
    rhythm = report.get('rhythm') or {}
    beats = _times(rhythm.get('beat_times_s') or [], duration)
    source = 'librosa_beat_track'
    downbeats = []
    downbeat_status = 'unavailable_without_trained_model'
    if beat_this_checkpoint and waveform is not None:
        try:
            beats, downbeats = predict_beat_this(waveform, checkpoint=beat_this_checkpoint)
            source = 'beat_this_pretrained'
            downbeat_status = 'estimated_by_trained_model'
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            downbeat_status = 'model_unavailable: '+type(exc).__name__
    events = []
    for t in beats[:5000]:
        events.append({'type':'beat', 'at_s':_seconds(t,offset_s),
                       'method':source,'status':'estimated', 'confidence': None,
                       'description':'节拍候选（不是保证由鼓乐器演奏）'})
    for t in downbeats[:2500]:
        events.append({'type':'downbeat', 'at_s':_seconds(t,offset_s),
                       'method':source,'status':'estimated','confidence':None,
                       'description':'可能的小节重拍（需训练模型，短片段可能漂移）'})
    # Additional low-cost evidence from existing librosa onset/chroma arrays.
    # Prominent beats are not reliably the first beat of a bar; chord changes
    # are major/minor template matches, not verified polyphonic transcription.
    accent_data = rhythm.get('accent_candidates') or {}
    for accent in accent_data.get('events', [])[:150]:
        row=dict(accent)
        row['at_s']=_seconds(row.get('at_s',0),offset_s)
        events.append(row)
    meter_data=rhythm.get('meter_candidates') or {}
    for group in meter_data.get('events',[]):
        row=dict(group);row['at_s']=_seconds(row.get('at_s',0),offset_s)
        events.append(row)
    chords = (report.get('harmony') or {}).get('chord_progression_estimate') or {}
    chord_changes=[]
    for change in chords.get('changes',[])[:120]:
        row=dict(change)
        row['at_s']=_seconds(row.get('at_s',0),offset_s)
        chord_changes.append(row)
        events.append({'type':'chord_change_candidate', **row})
    transient = None
    if waveform is not None:
        try:
            transient = harmonic_percussive_observations(waveform)
        except (ImportError, ValueError, RuntimeError) as exc:
            transient = {'status':'unavailable', 'reason':type(exc).__name__,
                         'not_true_stems':True, 'cannot_isolate_voice':True}
        for t in transient.get('percussive_attack_times_s', [])[:2400]:
            events.append({'type':'percussive_attack_candidate',
                           'at_s':_seconds(t,offset_s),
                           'method':'hpss_onset', 'status':'possible_not_confirmed_drum',
                           'description':'短促打击瞬态候选，不能确认是鼓'})
    else:
        transient = {'status':'audio_not_supplied', 'not_true_stems':True,
                     'cannot_isolate_voice':True}
    # Treat dominant mixed-spectrum pitch only as an observation, never as
    # a lead vocal/lead instrument transcribed note sequence.
    points = (report.get('melody') or {}).get('points_every_2s') or []
    for p, nxt in zip(points, points[1:]):
        a, b = p.get('midi'), nxt.get('midi')
        if a is None or b is None:
            continue
        delta = float(b)-float(a)
        motion = 'rising' if delta > 1.25 else 'falling' if delta < -1.25 else 'roughly_stable'
        events.append({'type':'dominant_pitch_motion',
                       'from_s':_seconds(p.get('t',0),offset_s),
                       'to_s':_seconds(nxt.get('t',0),offset_s),
                       'motion':motion, 'method':'mixed_audio_peak',
                       'status':'not_isolated_melody',
                       'description': '混合音频主导音高'+({'rising':'上行','falling':'下行','roughly_stable':'大致稳定'}[motion])})
    mood_sections = []
    for s in (report.get('segments') or [])[:120]:
        label = s.get('mood_cue') or (s.get('emotion_model') or {}).get('label') or '不确定'
        method = 'acoustic_heuristic_low_confidence'
        if s.get('emotion_model'):
            label = s['emotion_model'].get('label') or label
            method = 'pretrained_model_uncalibrated'
        mood_sections.append({'from_s':_seconds(s.get('start_s',0),offset_s),
                              'to_s':_seconds(s.get('end_s',duration),offset_s),
                              'candidate':label, 'method':method,
                              'certainty':'not_ground_truth',
                              'description':'这一时间段的音乐情绪线索仅为预测，需结合实际听感与歌词'})
    lyric_status = 'not_available'
    lyric_cues = []
    if local_lyrics:
        lyric_status = 'aligned_with_media_clock' if lyrics_aligned else 'exists_but_media_clock_unaligned'
        if lyrics_aligned:
            for line in local_lyrics:
                t = float(line.get('at_s',-1))
                if offset_s <= t < offset_s+duration:
                    # Only brief snippets; no copying whole copyrighted lyrics
                    # into unsolicited chat messages or API contexts.
                    lyric_cues.append({'at_s':round(t,3),'text_preview':str(line.get('text',''))[:32],
                                       'provenance':'user_supplied_lrc',
                                       'timestamp_confidence':'from_lrc_not_automatically_verified'})
    events.sort(key=lambda e:(e.get('at_s',e.get('from_s',0)), e['type']))
    return {
        'schema': 'music_events_v2',
        'timeline_basis': time_basis,
        'time_warning': ('若跳过非音乐间隔，这些时间戳相对于确认音乐的累积时间，'
                         '不是播放器时间；重拍需要经过训练的模型才能报告'),
        'duration_s': round(duration,3), 'offset_s':round(offset_s,3),
        'beat_source':source,'beat_times_s':[e['at_s'] for e in events if e['type']=='beat'],
        'downbeat_status':downbeat_status,
        'downbeat_times_s':[e['at_s'] for e in events if e['type']=='downbeat'],
        'beat_accents': {'status':accent_data.get('status','unavailable'),
                         'method':accent_data.get('method'),'events':[
                             e for e in events if e['type']=='beat_accent_candidate'][:96],
                         'not_verified_downbeat':True},
        'metrical_downbeat_hypotheses':{'status':meter_data.get('status','unavailable'),
              'meter_candidate':meter_data.get('meter_candidate'),
              'confidence_proxy':meter_data.get('confidence_proxy'),
              'events':[e for e in events if e['type']=='metrical_downbeat_hypothesis']},
        'chord_progression':{'status':chords.get('status','unavailable'),
                             'method':chords.get('method'),
                             'changes':chord_changes,
                             'segments':[dict(x,from_s=_seconds(x.get('from_s',0),offset_s),
                                              end_s=_seconds(x.get('end_s',0),offset_s))
                                         for x in chords.get('segments',[])][:60],
                             'warning':'和弦是粗略色度模板候选，不等于准确谱面'},
        'events':events,
        'sound_layers':transient,
        'sound_layer_warning':'HPSS 是谐波/打击代理，不是真正声源分离；主唱、旋律乐器和伴奏均无法可靠单独识别',
        'source_separation': {'status':'not_available', 'vocals':'unknown', 'accompaniment':'unknown',
                              'reason':'必须选择额外的经过训练的声源分离模型；基础模式不能声称人声与伴奏已分离'},
        'mood_sections':mood_sections,
        'lyrics': {'status':lyric_status, 'timed_cues':lyric_cues[:20],
                   'note':'没有可靠歌词时不能从频谱推测歌词；LRC 属于用户提供的文字'},
        'music_understanding_policy':'AI 结合事件与音频特征自行判断；不强制发言，不臆造歌词或真实听觉体验',
    }



def _spread(items: list, limit: int) -> list:
    """Sample beginning, middle AND end of a lengthy song's event track."""
    if len(items) <= limit:
        return items[:]
    if limit < 2:
        return [items[-1]] if limit else []
    return [items[round(i*(len(items)-1)/(limit-1))] for i in range(limit)]


def human_music_evidence(timeline: dict) -> list[str]:
    """Natural-language observation cues grounded in times and explicit uncertainty."""
    beat_times = timeline.get('beat_times_s') or []
    downbeats = timeline.get('downbeat_times_s') or []
    events = timeline.get('events') or []
    notes = []
    if beat_times:
        import statistics
        intervals = [b-a for a,b in zip(beat_times, beat_times[1:]) if .18 <= b-a <= 3.0]
        note = '检测到节拍候选 {} 次，例如 {} 秒'.format(len(beat_times),
                       '、'.join(str(v) for v in _spread(beat_times, min(5,len(beat_times)))))
        if len(intervals) >= 2:
            note += '；局部中位间隔约 {:.2f} 秒（约 {:.0f} BPM，存在半拍/倍拍歧义）'.format(
                     statistics.median(intervals), 60/statistics.median(intervals))
        notes.append(note)
    else:
        notes.append('没有足够的节拍证据，不能断言音乐速度或鼓点位置')
    meter=(timeline.get('metrical_downbeat_hypotheses') or {})
    if meter.get('events'):
        notes.append('拍号与小节重拍推测为 {}，示例 {} 秒；这是弱证据，不能当成真实重拍'.format(
            meter.get('meter_candidate'),','.join(str(x['at_s']) for x in _spread(meter['events'],4))))
    chords=(timeline.get('chord_progression') or {}).get('changes') or []
    if chords:
        notes.append('混合音频色度推测的和弦转换：'+ '、'.join(
            '{:.1f} 秒 {}→{}'.format(x['at_s'],x.get('from'),x.get('to'))
            for x in _spread(chords,min(5,len(chords))))+'（不能视为准确和弦谱）')
    accents=(timeline.get('beat_accents') or {}).get('events') or []
    if accents:
        notes.append('拍点力度较强的候选出现于 '+ '、'.join(str(v['at_s'])
                     for v in _spread(accents,min(6,len(accents))))+' 秒；未验证为小节重拍')
    if downbeats:
        example = '、'.join(map(str, _spread(downbeats, min(5, len(downbeats)))))
        notes.append(f'训练模型给出 {len(downbeats)} 个小节重拍候选，时间如 {example} 秒；短片段会漂移')
    else:
        notes.append('没有可验证的小节重拍时间（未安装或未输出训练重拍模型结果）')
    hits = [v['at_s'] for v in events if v.get('type') == 'percussive_attack_candidate']
    if hits:
        example = '、'.join(map(str, _spread(hits, min(6, len(hits)))))
        notes.append(f'短促打击瞬态候选出现在 {example} 秒附近；这可能是鼓，也可能是其他声音的起音')
    moves=[v for v in events if v.get('type')=='dominant_pitch_motion']
    if moves:
        changes={'rising':'上行','falling':'下行','roughly_stable':'大致平稳'}
        notes.append('观察到混合音频主导音高走势：'+ '；'.join(
            '{:.1f}–{:.1f} 秒{}'.format(v['from_s'],v['to_s'],changes.get(v.get('motion'),'变化'))
            for v in _spread(moves,min(3,len(moves))))+'。不能据此识别主唱或独立主旋律')
    moods=timeline.get('mood_sections') or []
    if moods:
        notes.append('逐段情绪线索：'+ '；'.join(
            '{:.1f}–{:.1f} 秒为「{}」候选'.format(v['from_s'],v['to_s'],v.get('candidate') or '不确定')
            for v in _spread(moods,min(3,len(moods))))+'；这是估计，并非听众的实际感受')
    lyrics=timeline.get('lyrics') or {}
    if lyrics.get('status') in ('not_available','exists_but_media_clock_unaligned'):
        notes.append('未获得可靠同步歌词；不能根据频谱推断唱词')
    elif lyrics.get('timed_cues'):
        notes.append('找到用户提供的同步歌词时间标记，但没有声学对齐校验，不应认定歌词内容识别准确')
    return notes


def compact_music_context(timeline: dict, *, max_beats: int = 36,
                          max_events: int = 32) -> dict:
    """Meaningful, time-addressed context instead of hundreds of grid numbers."""
    events = timeline.get('events') or []
    beats = timeline.get('beat_times_s') or []
    lyric_payload = dict(timeline.get('lyrics') or {})
    cues = lyric_payload.get('timed_cues') or []
    lyric_payload['timed_cues'] = _spread(cues, 12)
    lyric_payload['cues_sampled'] = len(cues) > 12
    return {
        'schema':timeline.get('schema'),
        'timeline_basis':timeline.get('timeline_basis'),
        'time_warning':timeline.get('time_warning'),
        'beat_source':timeline.get('beat_source'),
        'beat_times_s':_spread(beats,max_beats),
        'beat_times_sampled':len(beats)>max_beats,
        'downbeat_status':timeline.get('downbeat_status'),
        'downbeat_times_s':_spread((timeline.get('downbeat_times_s') or []),36),
        'beat_accents':{'status':(timeline.get('beat_accents') or {}).get('status'),
                        'not_verified_downbeat':True,
                        'events':_spread((timeline.get('beat_accents') or {}).get('events') or [],24)},
        'metrical_downbeat_hypotheses':{
             'status':(timeline.get('metrical_downbeat_hypotheses') or {}).get('status'),
             'meter_candidate':(timeline.get('metrical_downbeat_hypotheses') or {}).get('meter_candidate'),
             'events':_spread((timeline.get('metrical_downbeat_hypotheses') or {}).get('events') or [],16),
             'not_verified_downbeat':True},
        'chord_progression':{'method':(timeline.get('chord_progression') or {}).get('method'),
                             'status':(timeline.get('chord_progression') or {}).get('status'),
                             'changes':_spread((timeline.get('chord_progression') or {}).get('changes') or [],28),
                             'warning':'色度模板估计并非可信和弦谱'},
        'other_music_events':_spread([x for x in events if x.get('type')!='beat'],max_events),
        'sound_layers':{k:v for k,v in (timeline.get('sound_layers') or {}).items()
                        if k != 'percussive_attack_times_s'},
        'mood_sections':_spread((timeline.get('mood_sections') or []),10),
        'source_separation':timeline.get('source_separation'),
        'lyrics':lyric_payload,
        'listening_notes':human_music_evidence(timeline),
        'critical_limits': ['鼓点瞬态不等于明确的鼓乐器；没有分离人声',
                            '当前未安装训练重拍模型则重拍不可用',
                            '情绪推测不是可靠的快乐/悲伤结论'],
    }


def offset_event_timeline(timeline: dict, offset_s: float, *,
                          basis: str = 'confirmed_music_elapsed') -> dict:
    """Shift relative times to cumulative music-only time, without relabeling as media clock."""
    import copy
    out = copy.deepcopy(timeline)
    delta = float(offset_s)
    out['offset_s'] = round(delta,3)
    out['timeline_basis'] = basis
    for prop in ('beat_times_s','downbeat_times_s'):
        out[prop] = [_seconds(x, delta) for x in out.get(prop, [])]
    for e in out.get('events', []):
        for field in ('at_s','from_s','to_s'):
            if field in e and e[field] is not None:
                e[field] = _seconds(e[field],delta)
    shifted_ids={id(e) for e in out.get('events',[])}
    for a in (out.get('metrical_downbeat_hypotheses') or {}).get('events',[]):
        if id(a) not in shifted_ids and 'at_s' in a:a['at_s']=_seconds(a['at_s'],delta)
    for a in (out.get('beat_accents') or {}).get('events', []):
        if id(a) not in shifted_ids and 'at_s' in a: a['at_s'] = _seconds(a['at_s'],delta)
    for a in (out.get('chord_progression') or {}).get('changes', []):
        if 'at_s' in a: a['at_s'] = _seconds(a['at_s'],delta)
    for a in (out.get('chord_progression') or {}).get('segments', []):
        for field in ('from_s','end_s'):
            if field in a:a[field]=_seconds(a[field],delta)
    for e in out.get('mood_sections', []):
        for field in ('from_s','to_s'):
            if field in e: e[field] = _seconds(e[field],delta)
    for p in (out.get('lyrics') or {}).get('timed_cues', []):
        if 'at_s' in p: p['at_s'] = _seconds(p['at_s'],delta)
    return out


def enrich_with_user_lrc(timeline: dict, lines: list[dict]) -> dict:
    """An authorized *file* starts at media t=0, unlike loopback segments."""
    import copy
    out=copy.deepcopy(timeline)
    cues=[]
    for line in lines[:1000]:
        t=float(line.get('at_s', -1))
        if 0 <= t < float(out.get('duration_s', 0)):
            cues.append({'at_s':round(t,3), 'text_preview':str(line.get('text',''))[:32],
                         'provenance':'user_supplied_lrc', 'not_asr':True})
        if len(cues)>=1000: break
    out['lyrics']={'status':'user_supplied_lrc_aligned_to_file_start' if cues else 'not_available',
                   'timed_cues':cues,
                   'source':'adjacent_authorized_lrc',
                   'warning':'歌词时间戳来自用户本地 LRC，不是经过音频识别验证；只保留少量预览'}
    return out
