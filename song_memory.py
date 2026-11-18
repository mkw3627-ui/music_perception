"""Two-track bounded musical memory; no PCM, model weights or microphone data.

A playback has a unique *id*. Trusted player metadata produces a separate
identity hash, so returning to a known recording can reuse confirmed, song-local
observations without treating two ordinary music windows as the same song.
No identity is derived from a title alone or an unverified tempo similarity.
"""
from __future__ import annotations
from collections import deque
import copy
import hashlib
import json
import os
from pathlib import Path
import threading


QUALITY_MODEL = {'low': 'tiny', 'medium': 'base', 'high': 'small',
                 'ultra': 'large-v3-turbo'}


def trusted_song_identity(metadata: dict | None) -> str | None:
    """Stable, conservative playback identity from a player (not from YAMNet).

    Avoid treating similarly named releases as identical when the player has
    no artist and no duration. Playlist index is NOT a song identifier.
    """
    if not isinstance(metadata, dict) or not metadata.get('available'):
        return None
    title = ' '.join(str(metadata.get('title') or '').casefold().split())
    artist = ' '.join(str(metadata.get('artist') or '').casefold().split())
    if len(title) < 2 or title in {'unknown', '未知歌曲', '未命名歌曲', 'music'}:
        return None
    try:
        duration = float(metadata.get('duration_s') or 0)
    except (ValueError, TypeError):
        duration = 0
    if not artist and not (30 <= duration <= 7200):
        return None
    player = str(metadata.get('source') or 'player').casefold().strip()
    # An artist alone is usually strong, but not infallible (remixes and covers).
    # Retain duration if present so radio edit/live release isn't mixed.
    safe_duration = round(duration) if 30 <= duration <= 7200 else None
    record = [player, title, artist, safe_duration]
    return 'media:' + hashlib.sha256(json.dumps(record, ensure_ascii=False).encode('utf-8')).hexdigest()[:32]


def audio_segment_fingerprint(samples) -> str | None:
    """Cheap nonreversible fingerprint of *this exact music interval*.

    Metadata proves track identity, but not position. Reuse cached lyrics only
    if the audible interval also matches; jumping into a chorus may otherwise
    return yesterday's intro. This is an intentionally conservative matcher:
    volume, device or seek changes can cause a harmless re-analysis.
    """
    import numpy as np
    arr=np.asarray(samples,dtype=np.float32).reshape(-1)
    if not 16000 <= arr.size <= 90*16000 or not np.isfinite(arr).all():
        return None
    # Sample ~4096 frames: bounded work even for a 90-second clip.
    indices=(np.arange(4096,dtype=np.int64)*arr.size//4096).astype(np.int64)
    sample=arr[indices].astype(np.float64)
    sample-=float(sample.mean())
    rms=float(np.sqrt(np.mean(sample*sample)))
    if rms < 1e-4:  # skip silence and nearly constant audio
        return None
    quantized=np.rint(np.clip(sample/rms,-8,8)*12).astype(np.int8)
    return 'pcm16k:' + hashlib.sha256(quantized.tobytes()+arr.size.to_bytes(8,'little')).hexdigest()[:24]


class RecentSongMemory:
    def __init__(self, path: str | Path | None = None, *, max_songs: int = 2):
        if max_songs != 2:
            raise ValueError('Only two-song retention is supported')
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._entries = deque(maxlen=2)
        self.last_save_error = None
        if self.path is not None:
            try:
                payload = json.loads(self.path.read_text(encoding='utf-8'))
                for entry in payload.get('songs', [])[-2:]:
                    if isinstance(entry, dict) and isinstance(entry.get('id'), str):
                        self._entries.append(entry)
            except (OSError, ValueError, TypeError, AttributeError):
                pass

    def _save(self):
        if self.path is None:
            return
        tmp = self.path.with_suffix('.tmp')
        data = {'schema': 'recent_songs_v2', 'retention': 2, 'songs': list(self._entries)}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open('w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            self.last_save_error = None
        except OSError as exc:
            # Antivirus/file readers must not crash capture, or destroy an old cache.
            self.last_save_error = f'{type(exc).__name__}: {str(exc)[:120]}'
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _new(song_id, title, origin, identity=None):
        return {'id': song_id, 'identity': identity,
                'title': str(title or '未命名歌曲')[:128],
                'origin': origin, 'status': 'listening', 'play_count': 1,
                'end_reason': None, 'end_position_s': None, 'duration_s': None,
                'last_position_s': None, 'segment_count': 0,
                'chord_changes': [], 'beat_accents': [], 'lyrics_cues': [],
                'lyrics_quality': None, 'observed_segments': [],
                'lyrics_scanned_ranges': {}}

    def _reuse(self, index: int, new_id: str, title: str | None):
        rows = list(self._entries)
        reused = rows.pop(index)
        reused['id'] = new_id
        reused['status'] = 'listening'
        reused['end_reason'] = None
        reused['end_position_s'] = None
        reused['duration_s'] = None
        reused['interrupted_early'] = False
        reused['play_count'] = int(reused.get('play_count', 1)) + 1
        # Replay fingerprint/cache may be reused for analysis. Previously
        # recorded ASR text MUST NOT be shown as lyrics for a new playhead.
        reused['replay_lyrics_pending'] = True
        if title:
            reused['title'] = str(title)[:128]
        rows.append(reused)
        self._entries = deque(rows, maxlen=2)
        self._save()
        return {'reused': True, 'play_count': reused['play_count'],
                'cached_segments': len(reused.get('observed_segments') or []),
                'cached_lyrics': len(reused.get('lyrics_cues') or [])}

    def begin(self, song_id: str, *, title: str | None = None,
              origin='music_gate', identity: str | None = None):
        with self._lock:
            if self._entries and self._entries[-1]['id'] == song_id:
                return {'reused': False, 'already_current': True}
            if identity:
                for i, entry in enumerate(self._entries):
                    if entry.get('identity') == identity:
                        return self._reuse(i, song_id, title)
            self._entries.append(self._new(song_id, title, origin, identity))
            self._save()
            return {'reused': False, 'play_count': 1, 'cached_segments': 0}

    def identify(self, song_id: str, identity: str, *, title: str | None = None):
        """Attach late media metadata and salvage a matching past recording.

        Only migrate the current, not-yet-populated track, otherwise a delayed
        media callback could overwrite lyrics already attributed to new audio.
        """
        with self._lock:
            current = next((e for e in self._entries if e['id'] == song_id), None)
            if current is None or not identity:
                return {'reused': False}
            if current.get('identity') == identity:
                return {'reused': int(current.get('play_count', 1)) > 1,
                        'play_count': int(current.get('play_count', 1))}
            # Once a song identity has been assigned, don't overwrite it with
            # stale metadata from another player.
            if current.get('identity'):
                return {'reused': False, 'conflict': True}
            if current.get('segment_count') or current.get('lyrics_cues'):
                current['identity'] = identity
                self._save()
                return {'reused': False, 'too_late_to_reuse': True}
            for i, cached in enumerate(self._entries):
                if cached['id'] != song_id and cached.get('identity') == identity:
                    self._entries = deque((e for e in self._entries if e['id'] != song_id), maxlen=2)
                    return self._reuse(next(j for j,e in enumerate(self._entries) if e.get('identity') == identity),song_id,title)
            current['identity'] = identity
            if title:
                current['title'] = str(title)[:128]
            self._save()
            return {'reused': False}

    def end(self, song_id: str, *, reason: str = 'music_ended',
            position_s: float | None = None, duration_s: float | None = None):
        with self._lock:
            for e in self._entries:
                if e['id'] == song_id:
                    # Do not claim the whole song was heard when the listener
                    # only observed a partial playback or a probable switch.
                    if reason in ('metadata_switch', 'audio_switch_candidate', 'manual_switch'):
                        e['status'] = 'switched'
                    else:
                        e['status'] = 'ended'
                    e['end_reason'] = reason
                    if position_s is not None:
                        try: e['end_position_s'] = round(max(0., float(position_s)), 2)
                        except (TypeError, ValueError): pass
                    if duration_s is not None:
                        try: e['duration_s'] = round(max(0., float(duration_s)), 2)
                        except (TypeError, ValueError): pass
                    position, duration = e.get('end_position_s'), e.get('duration_s')
                    e['interrupted_early'] = (e['status'] == 'switched' and
                        position is not None and duration is not None and duration > 0 and
                        position < max(0, duration - 7))
            self._save()

    def observe(self, song_id: str, *, title: str | None = None,
                end_s: float | None = None, chords=(), accents=(), lyrics=(),
                lyrics_quality: str | None = None):
        with self._lock:
            entry = next((e for e in self._entries if e['id'] == song_id), None)
            if entry is None:
                return False
            if title:
                entry['title'] = str(title)[:128]
            if end_s is not None:
                entry['segment_count'] = int(entry.get('segment_count', 0)) + 1
                entry['last_position_s'] = round(float(end_s), 2)
            for field, received, limit in (
                ('chord_changes', chords, 64), ('beat_accents', accents, 80),
                ('lyrics_cues', lyrics, 96)):
                if field == 'lyrics_cues' and not received:
                    continue
                if field == 'lyrics_cues' and received and entry.get('replay_lyrics_pending'):
                    # First actual new-playback recognition supersedes all
                    # historical lyrics; avoid an A/B/A alternating display.
                    entry['lyrics_cues'] = []
                    entry['replay_lyrics_pending'] = False
                if field == 'lyrics_cues' and lyrics_quality:
                    if entry.get('lyrics_quality') not in (None, lyrics_quality):
                        entry['lyrics_cues'] = []
                    entry['lyrics_quality'] = lyrics_quality
                combined = entry.get(field, []) + [copy.deepcopy(x) for x in list(received) if isinstance(x, dict)]
                if field == 'lyrics_cues':
                    for c in combined:
                        c['text_preview'] = str(c.get('text_preview', ''))[:140]
                        c.pop('words', None)
                unique = {}
                for x in combined:
                    stamp = (x.get('at_s'), x.get('type'), x.get('to'), x.get('text_preview'))
                    unique[stamp] = x
                entry[field] = list(unique.values())[-limit:]
            self._save()
            return True

    def cache_segment(self, song_id: str, *, start_s: float, duration_s: float,
                      timeline: dict, cues: dict | None = None,
                      fingerprint: str | None = None,
                      media_start_s: float | None = None):
        """Cache compact semantic events, never raw audio or numerical spectra."""
        with self._lock:
            entry = next((e for e in self._entries if e['id'] == song_id), None)
            if entry is None:
                return False
            try:
                trimmed = copy.deepcopy(timeline)
                # Cache a bounded amount of music events; no raw spectrogram.
                safe = json.dumps(trimmed, ensure_ascii=False, allow_nan=False)
                if len(safe.encode('utf-8')) > 16000:
                    return False
                position = round(max(0.0, float(start_s)), 2)
                duration = round(float(duration_s), 2)
            except (ValueError, TypeError):
                return False
            segments = entry.setdefault('observed_segments', [])
            segments[:] = [s for s in segments if abs(s.get('start_s', -999)-position) > 1.5]
            scene={'start_s':position,'duration_s':duration,'timeline':trimmed}
            if isinstance(fingerprint,str) and fingerprint.startswith('pcm16k:'):
                scene['fingerprint']=fingerprint
            if media_start_s is not None:
                try:
                    media=float(media_start_s)
                    if 0<=media<=86400 and __import__('math').isfinite(media):
                        scene['media_start_s']=round(media,2)
                except (TypeError,ValueError):pass
            segments.append(scene)
            segments[:] = segments[-20:]
            self._save()
            return True

    def cached_segment(self, song_id: str, start_s: float, duration_s: float,
                       *, require_replay=True,
                       fingerprint: str | None = None,
                       media_start_s: float | None = None) -> dict | None:
        with self._lock:
            entry = next((e for e in self._entries if e['id'] == song_id), None)
            if not entry or (require_replay and int(entry.get('play_count', 1)) < 2):
                return None
            segments = entry.get('observed_segments') or []
            for scene in reversed(segments):
                if (abs(float(scene.get('start_s', -999))-float(start_s)) <= 1.8
                        and abs(float(scene.get('duration_s', -999))-float(duration_s)) <= 2.5
                        and isinstance(scene.get('timeline'), dict)):
                    if fingerprint is not None and scene.get('fingerprint') != fingerprint:
                        continue
                    previous_media=scene.get('media_start_s')
                    if media_start_s is not None and previous_media is not None:
                        if abs(float(previous_media)-float(media_start_s)) > 2.5:
                            continue
                    return copy.deepcopy(scene)
            return None

    def cached_lyrics(self, song_id: str, *, quality: str | None = None,
                      start_s: float | None = None, end_s: float | None = None) -> list:
        with self._lock:
            entry = next((e for e in self._entries if e['id'] == song_id), None)
            if not entry or (quality is not None and entry.get('lyrics_quality') != quality):
                return []
            cues = entry.get('lyrics_cues') or []
            if start_s is not None and end_s is not None:
                cues = [c for c in cues if isinstance(c.get('at_s'), (float, int))
                        and float(start_s)-2 <= c['at_s'] <= float(end_s)+2]
            return copy.deepcopy(cues)

    def mark_lyrics_scanned(self, song_id: str, *, start_s: float, end_s: float,
                            quality: str):
        """Remember ASR success even when a vocal-free clip produced no lyrics.

        Failed downloads and aborted ASR attempts are NOT marked as scanned.
        """
        if (quality not in QUALITY_MODEL.values() and not (
                isinstance(quality, str) and '|' in quality and
                quality.split('|', 1)[0] in QUALITY_MODEL.values() and
                quality.split('|', 1)[1] in {'auto', 'zh', 'ja', 'en'})):
            return False
        with self._lock:
            entry = next((e for e in self._entries if e['id'] == song_id), None)
            if not entry:
                return False
            start = round(float(start_s), 2)
            end = round(float(end_s), 2)
            if not 0 <= start < end or end-start > 91:
                return False
            groups = entry.setdefault('lyrics_scanned_ranges', {})
            existing = groups.setdefault(quality, [])
            existing[:] = [x for x in existing if abs(float(x[0])-start)>1.5]
            existing.append([start,end])
            existing[:] = existing[-24:]
            self._save()
            return True

    def lyrics_already_scanned(self, song_id: str, *, start_s: float,
                               end_s: float, quality: str) -> bool:
        with self._lock:
            entry = next((e for e in self._entries if e['id'] == song_id), None)
            if not entry or int(entry.get('play_count',1))<2:
                return False
            segments = (entry.get('lyrics_scanned_ranges') or {}).get(quality, [])
            return any(abs(float(s)-float(start_s))<2.5 and
                       abs(float(e)-float(end_s))<2.5 for s,e in segments)

    @staticmethod
    def _summary(entry: dict) -> dict:
        return {key: copy.deepcopy(entry.get(key)) for key in
                ('id','title','origin','status','play_count','segment_count',
                 'last_position_s','lyrics_quality','end_reason','end_position_s',
                 'duration_s','interrupted_early')} | {
                    'cached_segment_count':len(entry.get('observed_segments') or []),
                    'lyrics_cue_count':len(entry.get('lyrics_cues') or []),
                    'chord_change_count':len(entry.get('chord_changes') or []),
                    'beat_accent_count':len(entry.get('beat_accents') or [])}

    def recent_summary(self):
        with self._lock:
            return [self._summary(entry) for entry in self._entries]

    def current_summary(self):
        with self._lock:
            return self._summary(self._entries[-1]) if self._entries else None

    def recent(self):
        with self._lock:
            return copy.deepcopy(list(self._entries))

    def lyric_display(self, song_id: str, *, limit: int = 96):
        """Panel-local lyric cues for ONE retained song, never cross-song text."""
        with self._lock:
            entry = next((e for e in self._entries if e.get('id') == song_id), None)
            if entry is None:
                return None
            cues = ([] if entry.get('replay_lyrics_pending') else
                    [copy.deepcopy(c) for c in entry.get('lyrics_cues', [])
                    if isinstance(c, dict) and str(c.get('text_preview') or '').strip()])
            cues.sort(key=lambda x: (float(x.get('player_at_s_estimated', x.get('at_s', 0)) or 0),
                                     float(x.get('at_s', 0) or 0)))
            return {'id': song_id, 'title': str(entry.get('title') or '未命名歌曲'),
                    'status': entry.get('status', 'unknown'),
                    'end_reason': entry.get('end_reason'),
                    'interrupted_early': bool(entry.get('interrupted_early', False)),
                    'lyrics_quality': entry.get('lyrics_quality'),
                    'cues': cues[-max(1, min(96, int(limit))):]}

    def clear_song_lyrics(self, song_id: str):
        with self._lock:
            entry = next((e for e in self._entries if e.get('id') == song_id), None)
            if entry is None:
                return False
            entry['lyrics_cues'] = []
            entry['lyrics_quality'] = None
            entry['lyrics_scanned_ranges'] = {}
            self._save()
            return True

    def current(self):
        with self._lock:
            return copy.deepcopy(self._entries[-1]) if self._entries else None

    def clear(self):
        with self._lock:
            self._entries.clear()
            if self.path:
                try: self.path.unlink(missing_ok=True)
                except OSError: pass
