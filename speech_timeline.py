"""Conservative text grouping: music tracks beats, video tracks human pauses.

Source ASR timestamps remain unchanged. Musical beats are descriptive
alignment, never a forced quantization of spoken/sung syllables.
"""
from __future__ import annotations

import bisect
import math


def _finite(value, default=0.):
    try:
        f=float(value)
        return f if math.isfinite(f) else default
    except (TypeError,ValueError):
        return default


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith(('。', '！', '？', '!', '?', '；', ';', '…'))


def _word_join(previous: str, next_text: str) -> str:
    """Join ASR fragments without manufacturing a phrase break.

    CJK characters join naturally; English words retain a space. Timestamps
    remain on source cues; grouping is ONLY a UI/AI presentation hint.
    """
    previous, next_text = previous.rstrip(), next_text.lstrip()
    if not previous:
        return next_text
    if not next_text:
        return previous
    if next_text[0] in ',.!?;:，。！？；：、' or previous[-1] in '（("“‘':
        return previous + next_text
    if '\u4e00' <= previous[-1] <= '\u9fff' or '\u4e00' <= next_text[0] <= '\u9fff':
        return previous + next_text
    if '\u3040' <= previous[-1] <= '\u30ff' or '\u3040' <= next_text[0] <= '\u30ff':
        return previous + next_text
    return previous + ' ' + next_text


def _group_cues(cues, beats, *, bar_beats=4):
    beat_values=sorted(set(_finite(b) for b in beats if _finite(b,-1.)>=0))[:4096]
    rows=sorted((dict(c) for c in cues if isinstance(c,dict)),
                key=lambda c:_finite(c.get('at_s')))
    periods=[b-a for a,b in zip(beat_values,beat_values[1:]) if 0.25<=b-a<=2.5]
    beat_period=sorted(periods)[len(periods)//2] if periods else None
    # Line breaks are anchored to audible pauses and punctuation, NOT
    # beat subdivisions. A slow 3/4 ballad must not require a 2.4s pause
    # before it can break a lyric sentence.
    pause_cutoff=1.1
    group=-1
    prev_end=None
    prev_text=''
    prev_start=None
    for row in rows:
        at=max(0.,_finite(row.get('at_s')))
        end=max(at,_finite(row.get('end_s'),at))
        gap=at-prev_end if prev_end is not None else None
        text=str(row.get('text_preview') or row.get('text_candidate') or '').strip()
        explicit_end=bool(prev_text and _ends_sentence(prev_text))
        # Long melodic holds may separate word timestamps by a beat or two;
        # only pause, punctuation, and conservative maximum phrase duration
        # can create line breaks. ASR segment boundaries alone cannot.
        at_pause=bool(gap is not None and gap >= pause_cutoff)
        clause_pause=bool(gap is not None and gap>=.65 and
                          prev_text.endswith(('，',',','、')))
        long_phrase=(prev_start is not None and at-prev_start>13 and clause_pause)
        boundary=prev_end is None or explicit_end or at_pause or clause_pause or long_phrase
        if boundary:
            group+=1
            prev_start=at
        row['phrase_index']=group
        row['line_break_before']=bool(boundary)
        row['line_break_reason']=('first' if prev_end is None else
                   'punctuation' if explicit_end else 'audible_pause' if at_pause else
                   'clause_pause' if clause_pause else
                   'long_clause' if long_phrase else 'continue')
        row['line_pause_before_s']=round(max(0.,gap),2) if gap is not None else None
        if beat_values:
            idx=bisect.bisect_right(beat_values,at)-1
            if idx>=0:
                row['last_beat_s']=round(beat_values[idx],3)
                row['beat_index_estimated']=idx
                row['bar_index_estimated']=idx//max(1,int(bar_beats))
                if beat_period:
                    row['beat_phase_approx']=round(min(1.,max(0.,(at-beat_values[idx])/beat_period)),2)
        row['time_basis_note']='保留原ASR时间戳；语义/停顿断句，拍点只作参考'
        prev_end=max(end,prev_end or 0.)
        prev_text=text
    return rows,group+1,beat_period


def build_music_lines(cues, beats=(), *, limit=48):
    """Stable display rows by natural musical phrases, not by Whisper chunks.

    Calling this on the whole active song also bridges different ASR windows;
    never relies on per-chunk phrase_index resetting to zero.
    """
    rows,_,_=_group_cues(cues,beats)
    lines=[]
    for item in rows:
        phrase=item.get('phrase_index',0)
        text=str(item.get('text_preview') or item.get('text_candidate') or '').strip()
        if not text:
            continue
        if not lines or phrase!=lines[-1]['phrase_index']:
            lines.append({'phrase_index':phrase,
                'at_s':item.get('at_s',0), 'end_s':item.get('end_s',item.get('at_s',0)),
                'player_at_s_estimated':item.get('player_at_s_estimated',item.get('at_s',0)),
                'text_preview':text,'source_cues':1,
                'time_basis_note':'歌曲内自然乐句；时间为候选，不是逐字对齐'})
        else:
            line=lines[-1]
            line['text_preview']=_word_join(line['text_preview'],text)[:500]
            line['end_s']=max(_finite(line.get('end_s')), _finite(item.get('end_s')))
            line['source_cues']+=1
    return lines[-max(1,min(int(limit),96)):]


def music_phrases(cues, beats, *, bar_beats=4):
    annotated,groups,beat_period=_group_cues(cues,beats,bar_beats=bar_beats)
    return {'mode':'music_semantic_pause_phrase','beat_period_s_estimate':beat_period,
            'groups_count':groups,'cues':annotated,
            'display_lines':build_music_lines(annotated,beats)}


_TEMPLATE_MARKERS=(
    'subtitle','subtitles','字幕by','字幕 by','字幕由','字幕组','谢谢观看','感谢观看',
    '请不吝点赞','请订阅','点赞订阅','转载请注明','thanks for watching','amara.org',
)


def _norm_text(text: str) -> str:
    return ''.join(ch for ch in str(text or '').casefold() if ch.isalnum())


def annotate_video_evidence(cues, *, repeat_min=3):
    """Text-only hallucination screen for video ASR candidates.

    Marks, never rewrites, each cue with ``evidence_status``:
    - ``repeated_suspect``: the same normalized text recurs ``repeat_min``+
      times in the window, or matches a known ASR template phrase.
    - ``candidate``: otherwise.
    Returns (rows, suspect_count). Rows keep original text and timestamps.
    """
    rows=[dict(c) for c in cues if isinstance(c,dict)]
    counts={}
    for row in rows:
        key=_norm_text(row.get('text_candidate') or row.get('text_preview'))
        if key:
            counts[key]=counts.get(key,0)+1
    suspects=0
    for row in rows:
        raw=str(row.get('text_candidate') or row.get('text_preview') or '')
        key=_norm_text(raw)
        lowered=raw.casefold()
        reason=None
        if key and counts.get(key,0)>=max(2,int(repeat_min)):
            reason='repeated_text'
        elif any(marker in lowered for marker in _TEMPLATE_MARKERS):
            reason='template_phrase'
        row['evidence_status']='repeated_suspect' if reason else 'candidate'
        row['evidence_reason']=reason
        suspects+=1 if reason else 0
    return rows,suspects


def video_sentences(cues, *, pause_split_s=1.8):
    """Video speech may contain long gaps; do not use song beat semantics."""
    result=[]
    groups=0
    prev_end=None
    for raw in sorted((x for x in cues if isinstance(x,dict)),key=lambda x:_finite(x.get('at_s'))):
        row=dict(raw)
        at=max(0,_finite(row.get('at_s')))
        end=max(at,_finite(row.get('end_s'),at))
        gap=at-prev_end if prev_end is not None else None
        if prev_end is None or (gap is not None and gap>=pause_split_s):
            groups+=1
        row['speech_section']=groups
        row['pause_before_s']=round(max(0,gap),2) if gap is not None else None
        row['silence_boundary']=bool(gap is not None and gap>=pause_split_s)
        row['source']='video_speech_asr_candidate'
        row['time_basis_note']='来自同一视频会话的语音时间，不按音乐节拍拆句'
        result.append(row)
        prev_end=max(end, prev_end or 0)
    return {'mode':'video_speech_pauses','sections_count':groups,'cues':result[-64:]}
