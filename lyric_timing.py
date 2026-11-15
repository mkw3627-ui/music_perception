"""Align ASR time estimates with beats while preserving their real timestamps."""
from __future__ import annotations

import bisect
import copy
import math


def align_lyrics_to_music(lyrics: dict, *, offset_s: float, beats: list[float],
                          media_time_anchor_s: float | None = None) -> dict:
    """Return clips as absolute song-relative candidates, plus nearest beat.

    The offset is cumulative *accepted music* seconds unless an estimated player
    time anchor exists. Beats are additional evidence; DO NOT quantize lyrics
    onto beats, since syncopated singing is common.
    """
    out=copy.deepcopy(lyrics)
    bases=sorted(set(round(float(t),3) for t in beats if isinstance(t,(int,float)) and math.isfinite(t) and t>=0))
    anchor=float(offset_s)
    # The primary at_s remains in the exact same clock as beat_times_s.
    # When SMTC/player metadata exists, a second *estimated* clock is added.
    out['time_basis']='confirmed_music_elapsed_not_media_time'
    if media_time_anchor_s is not None:
        out['player_time_basis']='estimated_player_position_not_verified' 
    cues=[]
    for raw in out.get('timed_cues',[])[:20]:
        if not isinstance(raw,dict): continue
        item=copy.deepcopy(raw)
        for f in ('at_s','end_s'):
            if f in item:
                item[f]=round(max(0.0, float(item[f])+anchor),3)
        for w in item.get('words',[])[:16]:
            if 'at_s' in w: w['at_s']=round(float(w['at_s'])+anchor,3)
        if media_time_anchor_s is not None:
            item['player_at_s_estimated']=round(max(0,float(raw.get('at_s',0))+float(media_time_anchor_s)),3)
        t=float(item.get('at_s',anchor))
        if bases:
            idx=bisect.bisect_left(bases,t)
            near=min((bases[j] for j in (idx-1,idx) if 0<=j<len(bases)),
                     key=lambda v:abs(t-v))
            if abs(near-t) <= 0.85:
                item['nearest_beat_at_s']=round(near,3)
                item['offset_from_beat_s']=round(t-near,3)
        cues.append(item)
    out['timed_cues']=cues
    return out
