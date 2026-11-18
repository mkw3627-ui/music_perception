"""Overlap and de-duplicate incremental singing transcription in one song.

The ASR receives up to 4 seconds of the previous *confirmed-music* clip so
word endings are not cut in half at a fixed ten/twenty-second boundary. Neither
PCM nor lyric context can cross a song ID, an interrupted interval or consent
revocation. All audio stays in volatile memory.
"""
from __future__ import annotations

import re
from dataclasses import replace
from difflib import SequenceMatcher


def join_song_audio(clip, *, previous_tail, previous_sequence,
                    overlap_seconds: float = 4.0,
                    previous_full_tail=None, previous_full_sequence=None):
    import numpy as np
    pcm=np.asarray(clip.samples,dtype=np.float32).reshape(-1)
    key=(clip.sequence,clip.part_index)
    tail=np.asarray(previous_tail,dtype=np.float32).reshape(-1) if previous_tail is not None else None
    use=(previous_sequence==(clip.sequence,clip.part_index-1) and
         clip.part_index>1 and tail is not None and len(tail)>0)
    prefix=tail if use else np.empty(0,dtype=np.float32)
    if prefix.size and prefix.size+pcm.size>90*16000:
        prefix=prefix[:max(0,90*16000-pcm.size)]
    merged=np.concatenate([prefix,pcm]) if prefix.size else pcm
    beginning=max(0.,float(clip.offset_s)-prefix.size/16000)
    high=getattr(clip,'full_band_samples',None)
    rate=int(getattr(clip,'full_band_rate',48000))
    if high is not None:
        high=np.asarray(high,dtype=np.float32)
        full_prefix=(np.asarray(previous_full_tail,dtype=np.float32)
                     if (use and previous_full_sequence==(clip.sequence,clip.part_index-1)
                         and previous_full_tail is not None) else None)
        if prefix.size:
            if full_prefix is None or full_prefix.ndim!=2 or full_prefix.shape[1]!=2:
                # Never use time-inconsistent high-band input for Whisper.
                high=None
            else:
                want=int(round(prefix.size*rate/16000))
                if abs(len(full_prefix)-want)>rate//4:
                    high=None
                else:
                    high=np.concatenate([full_prefix[-want:],high])
    enriched=replace(clip,samples=merged,offset_s=round(beginning,3),
                     full_band_samples=high)
    tail_count=int(min(max(overlap_seconds,0.),6)*16000)
    new_tail=pcm[-tail_count:].copy() if tail_count else np.empty(0,dtype=np.float32)
    # The previous tail includes ONLY currently confirmed music; callers own
    # the track-boundary event and clear this state before starting a new song.
    return enriched,new_tail,key


def merge_song_lyrics(previous: list, incoming: list, *, limit=48) -> list:
    """Merge overlapping hypotheses without treating wording as verified.

    A repeated refrain minutes later is preserved. Near-duplicate phrases
    within 2 seconds are coalesced; this reduces double-sent lines caused
    by intentional overlapping ASR windows.
    """
    def normalized(value):
        return re.sub(r'[\W_]+','',str(value or '').casefold(),flags=re.UNICODE)
    ordered=sorted([dict(x) for x in [*previous,*incoming]
                    if isinstance(x,dict) and isinstance(x.get('at_s'),(float,int))],
                   key=lambda item:float(item.get('at_s',0)))
    merged=[]
    for item in ordered:
        txt=normalized(item.get('text_preview'))
        if not txt:
            continue
        start=float(item['at_s'])
        best_idx=None
        for j in range(len(merged)-1,-1,-1):
            existing=merged[j]
            if start-float(existing['at_s'])>2.4:
                break
            old=normalized(existing.get('text_preview'))
            if old and (txt==old or (min(len(txt),len(old))>=4 and
                                       SequenceMatcher(None,txt,old).ratio()>=.80)):
                best_idx=j
                break
        if best_idx is None:
            merged.append(item)
        else:
            # Keep the more complete hypothesis at a shared timestamp.
            original=merged[best_idx]
            if len(txt)>len(normalized(original.get('text_preview'))):
                item['at_s']=original['at_s']
                merged[best_idx]=item
    return merged[-max(1,min(limit,48)):]
