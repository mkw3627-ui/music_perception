"""为她所听·本能: fixed auto-classify flow, eavesdrop dice, heard-song book.

Pure logic + small JSON persistence; no SDK imports so it is unit-testable.

* ``decide_label``: fixed instinct "heard a sound -> is it music or video?"
  from the process hint + lightweight YAMNet evidence of the selected app.
* ``EavesdropDice``: per-song probability that she secretly pays attention to
  what the owner is listening to (scaled by the shared 三档 tier).
* ``HeardSongBook``: quietly remembers frequently heard songs and their lyric
  lines; on re-hearing, cross-checks new ASR lines against remembered ones.
"""
from __future__ import annotations

import json
import os
import random
import re
import tempfile
import threading
import time
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

BROWSERS = frozenset({"chrome.exe", "msedge.exe", "firefox.exe", "opera.exe", "brave.exe",
                      "360se.exe", "360chrome.exe", "qqbrowser.exe", "sogouexplorer.exe",
                      "vivaldi.exe", "arc.exe"})
VIDEO_HINT_APPS = frozenset({"bilibili.exe", "哔哩哔哩.exe", "potplayer.exe", "potplayermini64.exe",
                             "mpc-hc64.exe", "mpc-be64.exe", "vlc.exe", "iqiyi.exe", "qqlive.exe",
                             "youku.exe", "mgtv.exe", "douyin.exe", "kugou_video.exe"})
MUSIC_HINT_APPS = frozenset({"spotify.exe", "cloudmusic.exe", "qqmusic.exe", "kugou.exe", "kwmusic.exe",
                             "foobar2000.exe", "musicbee.exe", "aimp.exe", "winamp.exe",
                             "applemusic.exe", "itunes.exe", "lx-music-desktop.exe"})

# A browser alone proves nothing, so it needs a title hint to count as video.
BROWSER_VIDEO_TITLE_HINTS = ("视频", "直播", "弹幕", "在线观看", "bilibili", "哔哩",
                             "youtube", "youku", "iqiyi", "爱奇艺", "优酷", "腾讯视频",
                             "mgtv", "芒果tv", "douyin", "抖音", "西瓜视频", "twitch")


def title_is_video(title: str) -> bool:
    low = str(title or "").casefold()
    return bool(low) and any(k in low for k in BROWSER_VIDEO_TITLE_HINTS)

TIER_FACTOR = {"hell": 0.6, "easy": 1.0, "very_easy": 1.5}
TIER_WAIT = {"hell": 1.0, "easy": 0.5, "very_easy": 0.25}
EAVESDROP_BASE_CHANCE = 0.30
FAMILIAR_PLAYS = 3
NUDGE_COOLDOWN_S = 1800.0


def current_tier() -> str:
    forced = str(os.environ.get("NEKO_AUTONOMY_TIER") or "").strip()
    if forced in TIER_FACTOR:
        return forced
    base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
    try:
        data = json.loads((base / "N.E.K.O" / "autonomy_tier_v1.json").read_text("utf-8-sig"))
        tier = str(data.get("tier") or "")
        return tier if tier in TIER_FACTOR else "easy"
    except (OSError, ValueError, AttributeError):
        return "easy"


# --------------------------------------------------------------------------
# Instinct: music or video?
# --------------------------------------------------------------------------
def decide_label(app: dict, evidence: dict | None) -> tuple[str, str]:
    """Return (label, basis) or ("", reason) when not decidable yet.

    Known players decide instantly; browsers/unknown apps wait for the
    selected-app classifier (music hint -> music, speech dominant -> video).
    """
    name = str((app or {}).get("name") or "").casefold()
    hint = str((app or {}).get("candidate_tag") or (app or {}).get("tag") or "")
    ev = evidence if isinstance(evidence, dict) and evidence.get("available") else None
    if name in MUSIC_HINT_APPS or (hint == "music" and name not in BROWSERS):
        return "music", "model_context_judgment"
    title = str((app or {}).get("title") or "")
    if name in VIDEO_HINT_APPS or (hint == "video" and
                                   (name not in BROWSERS or title_is_video(title))):
        if ev and ev.get("is_music_hint") and float(ev.get("music_score") or 0) >= 0.5:
            # A video container playing a song: still video container; the
            # scene arbiter routes live music inside video to the song path.
            return "video", "model_context_judgment"
        return "video", "model_context_judgment"
    if ev is None:
        return "", "waiting_for_classifier_evidence"
    music = float(ev.get("music_score") or 0)
    speech = float(ev.get("speech_score") or 0)
    if ev.get("is_music_hint"):
        return "music", "local_music_classifier"
    if speech >= 0.15 and speech >= music:
        return "video", "model_context_judgment"
    if music >= 0.12 and music > speech:
        return "music", "model_context_judgment"
    return "", "ambiguous_audio"


def pick_audible_source(apps: list[dict]) -> dict | None:
    """Loudest currently audible non-AI app."""
    rows = [a for a in apps or [] if a.get("audible") and not a.get("host_ai")]
    if not rows:
        return None
    return max(rows, key=lambda a: float(a.get("level_dbfs") or a.get("last_live_dbfs") or -120))


# --------------------------------------------------------------------------
# Eavesdrop dice
# --------------------------------------------------------------------------
class EavesdropDice:
    def __init__(self, *, rng: random.Random | None = None, clock=time.time):
        self._rng = rng or random.Random()
        self._clock = clock
        self.last_nudge_at = 0.0
        self.last_roll: dict[str, Any] = {}

    def chance(self, tier: str | None = None) -> float:
        t = tier or current_tier()
        return min(0.95, EAVESDROP_BASE_CHANCE * TIER_FACTOR.get(t, 1.0))

    def roll(self, *, familiar: bool = False, tier: str | None = None) -> bool:
        t = tier or current_tier()
        p = self.chance(t)
        if familiar:
            p = min(0.95, p + 0.2)
        hit = self._rng.random() < p
        self.last_roll = {"at": self._clock(), "chance": round(p, 3), "hit": hit, "tier": t}
        return hit

    def may_nudge(self, tier: str | None = None) -> bool:
        t = tier or current_tier()
        return self._clock() - self.last_nudge_at >= NUDGE_COOLDOWN_S * TIER_WAIT.get(t, 0.5)

    def mark_nudged(self) -> None:
        self.last_nudge_at = self._clock()


# --------------------------------------------------------------------------
# Heard-song book
# --------------------------------------------------------------------------
def _norm(text: Any) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").casefold(), flags=re.UNICODE)


def _similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if min(len(a), len(b)) >= 4 and (a in b or b in a):
        return 0.9
    return SequenceMatcher(None, a, b).ratio()


class HeardSongBook:
    """Small persistent memory; at most ``max_songs`` songs x ``max_lines`` lines."""

    def __init__(self, path: str | Path | None, *, max_songs: int = 120, max_lines: int = 80,
                 clock=time.time):
        self.path = Path(path) if path else None
        self.max_songs = max_songs
        self.max_lines = max_lines
        self._clock = clock
        self._lock = threading.RLock()
        self.songs: list[dict] = []
        self._load()

    # persistence -----------------------------------------------------------
    def _load(self) -> None:
        if not self.path:
            return
        try:
            data = json.loads(self.path.read_text("utf-8"))
            songs = data.get("songs") if isinstance(data, dict) else None
            if isinstance(songs, list):
                self.songs = [s for s in songs if isinstance(s, dict) and s.get("id")][: self.max_songs]
        except (OSError, ValueError):
            self.songs = []

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            payload = {"schema": "music_perception.heard_song_book.v1", "songs": self.songs}
            tmp = self.path.with_suffix(".tmp")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
                os.replace(tmp, self.path)
            except OSError:
                pass

    # matching --------------------------------------------------------------
    def _by_title(self, title: str) -> dict | None:
        key = _norm(title)
        if not key:
            return None
        return next((s for s in self.songs if _norm(s.get("title")) == key), None)

    def match_lines(self, lines: list[str], *, min_hits: int = 2) -> tuple[dict | None, float]:
        """Best remembered song for these lyric lines, with overlap ratio."""
        normed = [_norm(x) for x in lines if len(_norm(x)) >= 3]
        if not normed:
            return None, 0.0
        best, best_ratio = None, 0.0
        for song in self.songs:
            stored = [_norm(x.get("text")) for x in song.get("lines", [])]
            if not stored:
                continue
            hits = sum(1 for n in normed if any(_similar(n, s) >= 0.72 for s in stored))
            ratio = hits / len(normed)
            if hits >= min(min_hits, len(normed)) and ratio > best_ratio:
                best, best_ratio = song, ratio
        return best, round(best_ratio, 3)

    # recording -------------------------------------------------------------
    def begin_play(self, *, session_key: str, title: str | None, app: str | None) -> dict:
        """Start a listening session; returns info about prior familiarity."""
        with self._lock:
            song = self._by_title(title or "") if title else None
            info = {"session_key": session_key, "song_id": song["id"] if song else None,
                    "known_by_title": bool(song), "plays_before": int(song.get("plays", 0)) if song else 0}
            if song:
                song["plays"] = int(song.get("plays", 0)) + 1
                song["last_heard"] = self._clock()
                if app:
                    song["app"] = str(app)[:60]
                self._touch(song)
                self.save()
            return info

    def _touch(self, song: dict) -> None:
        try:
            self.songs.remove(song)
        except ValueError:
            pass
        self.songs.insert(0, song)
        del self.songs[self.max_songs:]

    def observe_lines(self, *, song_id: str | None, title: str | None, app: str | None,
                      lines: list[str], counted_play: bool) -> dict:
        """Merge recognised lines; cross-check against memory.

        Returns {"song_id", "plays", "crosscheck": {...}, "new_song": bool}.
        """
        clean = [" ".join(str(x).split())[:120] for x in lines if len(_norm(x)) >= 2]
        with self._lock:
            song = next((s for s in self.songs if s.get("id") == song_id), None) if song_id else None
            matched_ratio = 0.0
            matched_by_lyrics = False
            if song is None and title:
                song = self._by_title(title)
            if song is None and clean:
                cand, ratio = self.match_lines(clean)
                if cand is not None and ratio >= 0.5:
                    song, matched_ratio, matched_by_lyrics = cand, ratio, True
            new_song = song is None
            if song is None:
                if not clean:
                    return {"song_id": None, "plays": 0, "crosscheck": None, "new_song": False}
                song = {"id": uuid.uuid4().hex[:12], "title": (title or "")[:120] or None,
                        "app": (app or "")[:60] or None, "first_heard": self._clock(),
                        "last_heard": self._clock(), "plays": 1, "lines": []}
                counted_play = True
            elif not counted_play:
                song["plays"] = int(song.get("plays", 0)) + 1
                counted_play = True
            if title and not song.get("title"):
                song["title"] = title[:120]
            song["last_heard"] = self._clock()
            confirmed, added, differing = [], [], []
            stored = song.setdefault("lines", [])
            for text in clean:
                n = _norm(text)
                best, best_score = None, 0.0
                for item in stored:
                    sc = _similar(n, _norm(item.get("text")))
                    if sc > best_score:
                        best, best_score = item, sc
                if best is not None and best_score >= 0.72:
                    best["seen"] = int(best.get("seen", 1)) + 1
                    if best_score < 0.999:
                        differing.append({"remembered": best.get("text"), "heard_now": text,
                                          "similarity": round(best_score, 2)})
                        # Keep the longer (usually more complete) hypothesis
                        # once it has been heard at least as often.
                        if len(text) > len(str(best.get("text") or "")) and best.get("seen", 1) <= 2:
                            best["text"] = text
                    confirmed.append(best.get("text"))
                elif len(stored) < self.max_lines:
                    stored.append({"text": text, "seen": 1})
                    added.append(text)
            self._touch(song)
            self.save()
            plays = int(song.get("plays", 1))
            cross = None
            if not new_song:
                cross = {"matched_by": "lyrics" if matched_by_lyrics else ("title" if title else "session"),
                         "match_ratio": matched_ratio or (round(len(confirmed) / max(1, len(clean)), 3)),
                         "confirmed_lines": confirmed[:6], "new_lines": added[:6],
                         "differences": differing[:4],
                         "remembered_line_count": len(stored)}
            return {"song_id": song["id"], "plays": plays, "crosscheck": cross,
                    "new_song": new_song, "title": song.get("title"), "app": song.get("app")}

    def familiar(self, *, limit: int = 8) -> list[dict]:
        with self._lock:
            rows = sorted(self.songs, key=lambda s: (-int(s.get("plays", 0)), -float(s.get("last_heard", 0))))
            out = []
            for s in rows[:limit]:
                top = sorted(s.get("lines", []), key=lambda x: -int(x.get("seen", 1)))[:3]
                out.append({"title": s.get("title") or "（不知道歌名）", "plays": int(s.get("plays", 0)),
                            "app": s.get("app"), "last_heard": s.get("last_heard"),
                            "known_lines": [x.get("text") for x in top]})
            return out

    def get(self, song_id: str | None) -> dict | None:
        with self._lock:
            return next((dict(s) for s in self.songs if s.get("id") == song_id), None)
