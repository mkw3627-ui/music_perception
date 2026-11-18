"""Lightweight beat-accent and chord-change *candidates* from existing DSP.

No additional checkpoints or packages. Chords inferred from mixed-audio chroma
are hypotheses, not verified transcription. Accents are NOT trained downbeats.
"""
from __future__ import annotations


def chord_change_candidates(chroma, *, sample_rate: int, hop_length: int,
                            duration: float, window_seconds: float = 0.75) -> dict:
    import numpy as np
    pcs = np.asarray(chroma, dtype=np.float32)
    if pcs.ndim != 2 or pcs.shape[0] != 12 or pcs.shape[1] < 2:
        return {'method':'chroma_template_major_minor','status':'insufficient_audio', 'segments':[], 'changes':[]}
    if sample_rate <= 0 or hop_length <= 0 or duration <= 0:
        raise ValueError('invalid chroma timebase')
    # Harmonic templates rotate through all 12 major/minor roots; each is a
    # rough tonal similarity rather than an automatic chord transcription.
    templates=[]
    for root in range(12):
        for quality, third in (('maj',4),('min',3)):
            weights=np.zeros(12,dtype=np.float32)
            weights[root]=1.0
            weights[(root+third)%12]=0.85
            weights[(root+7)%12]=0.72
            templates.append((root,quality,weights/(np.linalg.norm(weights)+1e-7)))
    template_matrix=np.stack([row[2] for row in templates])
    names='C C# D D# E F F# G G# A A# B'.split()
    length=max(1,int(round(window_seconds*sample_rate/hop_length)))
    labels=[]
    scores=[]
    for idx in range(0,pcs.shape[1],length):
        unit=pcs[:,idx:min(idx+length,pcs.shape[1])]
        v=np.mean(unit,axis=1)
        norm=float(np.linalg.norm(v))
        if norm < 1e-5:
            labels.append(None)
            scores.append(0.0)
            continue
        sims=template_matrix@(v/(norm+1e-7))
        best=int(np.argmax(sims))
        sorted_sims=np.sort(sims)
        margin=float(sorted_sims[-1]-sorted_sims[-2])
        # Template matches compete across nearby harmonies. Short/noisy
        # fragments must be allowed to remain unknown.
        confidence=round(min(1.,max(0.,(float(sims[best])-.60)*2+margin*2)),3)
        labels.append(names[templates[best][0]]+':'+templates[best][1]
                      if confidence>=0.28 else None)
        scores.append(confidence)
    # Median vote of 3 adjacent windows reduces one-frame chord flicker.
    smoothed=[]
    for i,label in enumerate(labels):
        recent=labels[max(0,i-1):min(len(labels),i+2)]
        majority=next((c for c in recent if c is not None and recent.count(c)>=2),None)
        smoothed.append(majority if majority is not None else label)
    segments=[]
    for i,label in enumerate(smoothed):
        start=round(i*length*hop_length/sample_rate,3)
        end=round(min(duration,(i+1)*length*hop_length/sample_rate),3)
        if end<=start:
            continue
        if segments and label==segments[-1]['label']:
            segments[-1]['end_s']=end
            segments[-1]['confidence']=round((segments[-1]['confidence']+scores[i])/2,3)
        else:
            segments.append({'from_s':start,'end_s':end,'label':label,
                             'confidence':scores[i],'status':'estimated_from_mixed_chroma'})
    # Unknowns can interrupt a progression but are NOT "chord changes".
    changes=[]
    previous=None
    for segment in segments:
        label=segment['label']
        if label is not None and previous is not None and label!=previous:
            changes.append({'at_s':segment['from_s'],'from':previous,'to':label,
                            'confidence':segment['confidence'],
                            'method':'chroma_template_major_minor',
                            'status':'candidate_not_verified'})
        previous=label
    return {'method':'chroma_template_major_minor','status':'estimated',
            'segments':segments[:160],'changes':changes[:120],
            'warning':'和弦使用混合音频的色度模板估计，转位、七和弦及强旋律可能被误判'}


def beat_accent_candidates(beats, onset_envelope, *, sample_rate: int,
                           hop_length: int, top_limit: int = 96) -> dict:
    import numpy as np
    onsets=np.asarray(onset_envelope,dtype=float).reshape(-1)
    if onsets.size<3 or len(beats)<3:
        return {'status':'insufficient_beats','events':[],'method':'onset_strength_near_beat'}
    picked=[]
    for t in beats:
        i=int(round(float(t)*sample_rate/hop_length))
        a=max(0,i-2)
        b=min(len(onsets),i+3)
        if b<=a:
            continue
        picked.append((float(t),float(np.max(onsets[a:b]))))
    if not picked:
        return {'status':'insufficient_beats','events':[]}
    levels=np.array([v for _,v in picked])
    median=float(np.median(levels))
    spread=float(np.std(levels))
    floor=median+0.55*spread
    events=[]
    for i,(at,level) in enumerate(picked):
        if level<=floor or level<=0.03:
            continue
        # This is a prominent beat, NOT necessarily the first beat in a bar.
        events.append({'type':'beat_accent_candidate','at_s':round(at,3),
                       'relative_strength':round(min(2.,level/max(median,1e-5)),2),
                       'status':'strong_beat_not_verified_downbeat',
                       'method':'onset_strength_near_beat',
                       'description':'较突出的拍点候选，不能确定是小节第一拍'})
    return {'status':'estimated','events':events[:top_limit],
            'method':'onset_strength_near_beat',
            'warning':'强拍不等于训练模型确认的小节重拍'}


def metrical_grouping_candidates(beats, accent_data: dict, *, max_events=28) -> dict:
    """Very cautious bar-phase guess from emphasis over consecutive beats.

    This is NOT Beat This! or a validated downbeat neural network. Only
    produce an estimate if a clear 3/4 or 4/4 accent pattern spans 3+ bars.
    """
    import numpy as np
    times=np.asarray(beats,dtype=float).reshape(-1)
    if len(times)<12:
        return {'status':'not_enough_beats','meter_candidate':None,'events':[]}
    accent_times={round(float(x.get('at_s',0)),2):float(x.get('relative_strength',1))
                  for x in accent_data.get('events',[])}
    # Strong-beat events are sparse; do not replace their missing strengths
    # by assumed zeros: the baseline intensity for other beats is 1.0.
    strengths=np.asarray([accent_times.get(round(t,2),1.0) for t in times])
    best=None
    for meter in (3,4):
        if len(times)<meter*3:
            continue
        for phase in range(meter):
            indexes=np.arange(len(times))
            a=strengths[(indexes-phase)%meter==0]
            b=strengths[(indexes-phase)%meter!=0]
            if len(a)<3 or not len(b):
                continue
            diff=float(np.mean(a)-np.mean(b))
            # Normalized contrast in [0,1], not a probability.
            score=diff/(float(np.mean(strengths))+1e-7)
            if best is None or score>best[0]:
                best=(score,meter,phase)
    if best is None or best[0]<.28:
        return {'status':'no_reliable_meter','meter_candidate':None,'events':[]}
    score,meter,phase=best
    timestamps=[round(float(times[i]),3) for i in range(phase,len(times),meter)][:max_events]
    return {'status':'weak_meter_hypothesis','meter_candidate':f'{meter}/4',
            'confidence_proxy':round(min(1.,score),3),
            'events':[{'type':'metrical_downbeat_hypothesis','at_s':t,
                       'status':'not_verified_downbeat','method':'pattern_of_beat_accents',
                       'meter_candidate':f'{meter}/4'} for t in timestamps],
            'warning':'推测强拍分组，不能视为歌曲拍号或准确小节重拍'}
