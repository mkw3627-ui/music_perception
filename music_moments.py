"""Turn consecutive music-clip analyses into *felt* change moments for 吐槽.

Input: the per-clip ``report`` produced by ``engine.analyze_waveform`` (segments with
raw ``loudness_rms`` / ``spectral_centroid_hz`` / ``bass_share`` / ``local_bpm``,
rhythm bpm, key mode, mood cue). Nothing here is shown to anyone verbatim:
``describe()`` maps a change to everyday words — "突然炸开了", "一下子安静下来" —
so the character can react like a person listening along, never by quoting BPM,
Hz, keys or scores.

Pure Python, no host imports: unit-tested offline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# A moment needs at least this much "how noticeable" (0..1) to be worth a remark.
MIN_STRENGTH = 0.45
# Ignore the first clip of a song: there is nothing to compare with yet, and
# the song start is already covered by the song-change path.
WARMUP_CLIPS = 1


@dataclass
class ClipFeel:
    """Coarse, comparable fingerprint of one analyzed clip (all relative, unitless)."""
    loud: float          # mean raw RMS of the clip
    loud_peak: float     # max segment RMS
    loud_jump: float     # largest RMS ratio between neighbouring 10 s segments (>=1)
    bright: float        # mean spectral centroid (Hz, only compared, never shown)
    bass: float          # mean bass share 0..1
    bpm: float | None
    major: bool | None   # key mode, only when the key estimate is reliable
    mood: str            # 可能偏欢快 / 可能偏忧伤 / 情绪不确定
    regular: float       # beat regularity 0..1


def clip_feel(report: dict) -> ClipFeel | None:
    segs = [s for s in (report.get('segments') or []) if isinstance(s, dict)]
    if not segs:
        return None
    rms = [float(s.get('loudness_rms') or 0.0) for s in segs]
    if max(rms) <= 0:
        return None
    jumps = [max(b, 1e-6) / max(a, 1e-6) for a, b in zip(rms, rms[1:])]
    jump = max([j for j in jumps] + [1 / j for j in jumps] + [1.0])
    key = ((report.get('harmony') or {}).get('estimated_key') or {})
    rhythm = report.get('rhythm') or {}
    bpm = rhythm.get('bpm_estimate')
    return ClipFeel(
        loud=sum(rms) / len(rms), loud_peak=max(rms), loud_jump=jump,
        bright=sum(float(s.get('spectral_centroid_hz') or 0) for s in segs) / len(segs),
        bass=sum(float(s.get('bass_share') or 0) for s in segs) / len(segs),
        bpm=float(bpm) if isinstance(bpm, (int, float)) and bpm > 0 else None,
        major=(key.get('mode') == 'major') if key.get('reliable') else None,
        mood=str((report.get('emotion') or {}).get('label') or ''),
        regular=float(rhythm.get('beat_regularity') or 0.0),
    )


@dataclass
class Moment:
    kind: str            # burst / calm / faster / slower / brighter / darker / heavier / lighter / mood_up / mood_down / style
    strength: float      # 0..1, how obvious
    feel: str            # everyday words for the change (no numbers)
    hints: list[str] = field(default_factory=list)


def _ratio(a: float, b: float) -> float:
    return math.log(max(b, 1e-9) / max(a, 1e-9))


def _tempo_ratio(a: float | None, b: float | None) -> float:
    """log tempo ratio, folding half/double-time ambiguity (a common estimator flip)."""
    if not a or not b:
        return 0.0
    r = math.log(b / a)
    for fold in (math.log(2), -math.log(2)):
        if abs(r - fold) < abs(r):
            r -= fold
    return r


def compare(prev: ClipFeel, cur: ClipFeel) -> list[Moment]:
    """All noticeable changes from the previous clip to this one, strongest first."""
    out: list[Moment] = []
    # 流行歌混音压得很狠：真实的「间奏收住 / 副歌炸开」整段响度也只差 15~25%，
    # 阈值按 2026-11 三首实际歌曲的 20 秒片段标定（见 tests/test_v0924_commentary.py）。
    loud = _ratio(prev.loud, cur.loud)
    # Within-clip jump catches a drop that lands mid-clip (e.g. chorus kicks in at 12 s).
    inner = math.log(cur.loud_jump)
    if loud > 0.13 or inner > 0.4:
        s = min(1.0, max(loud / 0.25, inner / 0.7))
        out.append(Moment('burst', s, '一下子炸开了，声音猛地变大、变满', ['副歌进来了', '高潮来了']))
    elif loud < -0.15:
        s = min(1.0, -loud / 0.3)
        out.append(Moment('calm', s, '突然安静下来了，只剩一点点声音', ['像是在酝酿下一段', '收住了']))
    tempo = _tempo_ratio(prev.bpm, cur.bpm)
    if tempo > 0.12:
        out.append(Moment('faster', min(1.0, tempo / 0.25), '节奏一下子快起来了，变得很赶、很想跟着动', []))
    elif tempo < -0.12:
        out.append(Moment('slower', min(1.0, -tempo / 0.25), '节奏慢下来了，变得拖拖的、很松弛', []))
    bright = _ratio(prev.bright, cur.bright)
    if bright > 0.3:
        out.append(Moment('brighter', min(1.0, bright / 0.6), '声音一下子变亮了，听起来更透、更清脆', []))
    elif bright < -0.3:
        out.append(Moment('darker', min(1.0, -bright / 0.6), '声音变闷变暗了，像是蒙了一层', []))
    bass = cur.bass - prev.bass
    if bass > 0.12:
        out.append(Moment('heavier', min(1.0, bass / 0.25), '低音一下子重起来了，咚咚的感觉很压耳朵', []))
    elif bass < -0.12:
        out.append(Moment('lighter', min(1.0, -bass / 0.25), '低音撤掉了，整首歌变得轻飘飘的', []))
    # 调式 / 情绪标签在相邻片段之间经常来回翻（实测同一首歌 20 秒一翻），单独不足以开口，
    # 只作为「风格变了」的弱佐证；真正的情绪转折要和别的变化一起出现。
    if '欢快' in cur.mood and '忧伤' in prev.mood:
        out.append(Moment('mood_up', 0.3, '感觉一下子明亮起来了，没那么难过了', []))
    elif '忧伤' in cur.mood and '欢快' in prev.mood:
        out.append(Moment('mood_down', 0.3, '情绪往下沉了，听着有点伤感', []))
    # Several dimensions moving at once reads as "this is a different kind of music now".
    if sum(1 for m in out if m.strength >= 0.5) >= 3 or (
            len(out) >= 3 and any(m.kind in ('mood_up', 'mood_down') for m in out)):
        s = min(1.0, sum(m.strength for m in out) / 3.0)
        out.insert(0, Moment('style', s, '整个风格都变了，像换了一种音乐', []))
    out.sort(key=lambda m: m.strength, reverse=True)
    return out


def describe_overall(feel: ClipFeel) -> str:
    """One everyday sentence about what this music feels like right now (no numbers)."""
    parts = []
    if feel.bpm:
        parts.append('节奏很快、很有劲' if feel.bpm >= 135 else ('节奏慢慢的' if feel.bpm < 85 else '节奏不快不慢'))
    if feel.bass >= 0.45:
        parts.append('低音很厚')
    if feel.bright >= 2600:
        parts.append('声音亮亮的')
    elif feel.bright and feel.bright < 1300:
        parts.append('声音偏柔、偏暗')
    if '欢快' in feel.mood:
        parts.append('听着挺开心')
    elif '忧伤' in feel.mood:
        parts.append('听着有点伤感')
    if feel.regular >= 0.8 and feel.bpm and feel.bpm >= 110:
        parts.append('很适合跟着点头')
    return '，'.join(parts)


class MusicMomentTracker:
    """Keeps the previous clip of the *same song* and yields the best new moment."""

    def __init__(self, min_strength: float = MIN_STRENGTH):
        self.min_strength = min_strength
        self._song = None
        self._prev: ClipFeel | None = None
        self._clips = 0
        self._said_kinds: list[str] = []

    def reset(self, song=None):
        self._song, self._prev, self._clips, self._said_kinds = song, None, 0, []

    def feed(self, song, report: dict) -> tuple[Moment | None, ClipFeel | None]:
        if song != self._song:
            self.reset(song)
        cur = clip_feel(report)
        if cur is None:
            return None, None
        prev, self._prev = self._prev, cur
        self._clips += 1
        if prev is None or self._clips <= WARMUP_CLIPS:
            return None, cur
        for m in compare(prev, cur):
            if m.strength < self.min_strength:
                break
            # Do not repeat the same kind of remark within one song.
            if m.kind in self._said_kinds:
                continue
            return m, cur
        return None, cur

    def mark_said(self, moment: Moment):
        self._said_kinds.append(moment.kind)
