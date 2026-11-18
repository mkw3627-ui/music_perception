"""Conservative on-device song-transition evidence from 2-second music frames.

SMTC/player track identity remains authoritative. Audio-only novelty detects
obvious *possible* seamless transitions, not song identity. No fingerprint DB,
network calls, raw-audio caching or microphone capture.
"""
from __future__ import annotations

from collections import deque


class SongChangeDetector:
    def __init__(self, *, min_music_windows: int = 9, confirmation: int = 2,
                 cooldown_windows: int = 9):
        self.min_music_windows=min_music_windows
        self.confirmation=confirmation
        self.cooldown_windows=cooldown_windows
        self.reset()

    def reset(self):
        self._begin=[]
        self._recent=deque(maxlen=3)
        self._n=0
        self._last_reset_window=0
        self._candidate=0
        self._streak=0
        self.last_evidence=None

    @staticmethod
    def fingerprint(samples):
        import numpy as np
        x=np.asarray(samples,dtype=np.float32).ravel()
        if x.size<8000 or not np.isfinite(x).all():
            return None
        if float(np.sqrt(np.mean(x*x)))<.001:
            return None
        n=8192
        # Several Hann-windowed excerpts suppress reliance on one transient.
        slices=[]
        for loc in (.1,.5,.85):
            start=max(0,min(len(x)-n,int((len(x)-n)*loc)))
            mag=np.abs(np.fft.rfft(x[start:start+n]*np.hanning(n)))
            slices.append(mag)
        mag=np.mean(slices,axis=0)
        frequencies=np.fft.rfftfreq(n,1/16000)
        valid=(frequencies>=90)&(frequencies<=2400)
        pitches=np.zeros(12,dtype=np.float64)
        idx=np.rint(69+12*np.log2(frequencies[valid]/440.)).astype(int)%12
        np.add.at(pitches,idx,mag[valid])
        pitches/=np.linalg.norm(pitches)+1e-9
        bands=[]
        for low,high in zip((70,130,230,420,750,1300,2250,3900),
                            (130,230,420,750,1300,2250,3900,7500)):
            bands.append(float(np.sum(mag[(frequencies>=low)&(frequencies<high)])))
        bands=np.log1p(np.asarray(bands,dtype=float))
        bands-=bands.min()
        bands/=np.linalg.norm(bands)+1e-9
        # Tempo/accent proxy: local RMS activity at ~50 ms resolution. This
        # is rhythmic-change evidence, NOT a tempo or downbeat measurement.
        frame=max(400,int(16000*.05))
        usable=x[:len(x)//frame*frame]
        energy=np.sqrt(np.mean(usable.reshape(-1,frame)**2,axis=1))
        diff=np.maximum(0.,np.diff(energy,prepend=energy[0]))
        if diff.max()>0:
            diff/=diff.max()
        autoc=np.correlate(diff-diff.mean(),diff-diff.mean(),mode='full')[len(diff)-1:]
        lag=np.argmax(autoc[5:min(len(autoc),22)])+5 if len(autoc)>10 else 10
        # one lag (0.25-1.05s) is enough for novelty, not a BPM assertion.
        pulse=float(lag*.05)
        return pitches,bands,pulse

    def feed(self, samples, *, is_music: bool) -> dict | None:
        import numpy as np
        if not is_music:
            self._streak=0
            self._recent.clear()
            return None
        entry=self.fingerprint(samples)
        if entry is None:
            return None
        self._n+=1
        if len(self._begin)<4:
            self._begin.append(entry)
            return None
        self._recent.append(entry)
        if self._n<self.min_music_windows or self._n-self._last_reset_window<self.cooldown_windows:
            return None
        baseline=(np.mean([a[0] for a in self._begin],axis=0),
                  np.mean([a[1] for a in self._begin],axis=0),
                  float(np.median([a[2] for a in self._begin])))
        # Compare the last two sustained fingerprints with the opening
        # track reference. Changes within a verse can resemble transitions:
        # conservatively require simultaneous changes in 3 feature families.
        current=(np.mean([a[0] for a in self._recent],axis=0),
                 np.mean([a[1] for a in self._recent],axis=0),
                 float(np.median([a[2] for a in self._recent])))
        pitch=max(0.,1.-float(np.dot(baseline[0],current[0])/(np.linalg.norm(baseline[0])*np.linalg.norm(current[0])+1e-8)))
        timbre=max(0.,1.-float(np.dot(baseline[1],current[1])/(np.linalg.norm(baseline[1])*np.linalg.norm(current[1])+1e-8)))
        rhythm=min(1.,abs(current[2]-baseline[2])/max(current[2],baseline[2],.25))
        confidence=round(min(1.,pitch*.44+timbre*.28+rhythm*.28),3)
        # No novelty means no boundary. Meter/structure can naturally change
        # within one song; audio-only detection remains probabilistic.
        different=(pitch>.28 and timbre>.10 and rhythm>.24 and confidence>.25)
        self._streak=self._streak+1 if different else 0
        if self._streak<self.confirmation:
            return None
        self.last_evidence={'source':'audio_tonal_rhythm_timbre_change',
                            'status':'candidate_not_confirmed_identity',
                            'confidence':confidence,
                            'pitch_novelty':round(pitch,3),
                            'rhythm_novelty':round(rhythm,3),
                            'timbre_novelty':round(timbre,3)}
        evidence=dict(self.last_evidence)
        self.reset()
        self.last_evidence=evidence
        self._last_reset_window=self._n
        return evidence
