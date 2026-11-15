"""Optional sync to user-owned, local LRC from a Windows media-session title.

Nothing is downloaded or scraped. Scanning is bounded to expressly permitted
music folders; timing is only an estimate because SMTC clocks can drift.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .lyrics_sync import parse_lrc


def _stem(value: str) -> str:
    value=unicodedata.normalize('NFKC',value or '').casefold()
    return re.sub(r'[^\w]+','',value)


def find_authorized_player_lrc(title: str, artist: str, roots, *, max_files=320):
    title_key=_stem(title)
    if len(title_key)<2:return []
    artist_key=_stem(artist)
    allowed={title_key,title_key+artist_key,artist_key+title_key}
    visited=0
    for root in roots[:12]:
        try:
            root=Path(root).expanduser().resolve(strict=True)
            if not root.is_dir():continue
            # Song folders under the root are common; avoid recursive scans.
            folders=[root]
            for sub in root.iterdir():
                if len(folders)>=24:break
                if sub.is_dir() and not sub.is_symlink():folders.append(sub)
            for folder in folders:
                for file in folder.iterdir():
                    visited+=1
                    if visited>max_files:return []
                    if file.suffix.casefold()!='.lrc' or file.is_symlink():continue
                    if _stem(file.stem) not in allowed:continue
                    path=file.resolve(strict=True)
                    if not path.is_relative_to(root) or path.stat().st_size>180000:continue
                    raw=path.read_bytes()
                    for enc in ('utf-8-sig','utf-16','gb18030'):
                        try:
                            lines=parse_lrc(raw.decode(enc))
                            return lines[:1000] if lines else []
                        except UnicodeError:pass
        except (OSError,ValueError,RuntimeError):continue
    return []


def preview_player_lyrics(lines, *, music_clip_offset_s: float,
                          media_clip_start_s: float, duration_s: float,
                          limit=8):
    """Preserve two clocks, don't pretend estimated SMTC is exact alignment."""
    start=float(media_clip_start_s)
    end=start+float(duration_s)
    cues=[]
    for line in lines:
        song_at=float(line.get('at_s',-1))
        if not start-.1<=song_at<end:continue
        # ASR, beat and melody events retain accepted-music clock.
        cue_at=max(0.,music_clip_offset_s+song_at-start)
        cues.append({'at_s':round(cue_at,3),
                     'player_at_s_estimated':round(song_at,3),
                     'text_preview':str(line.get('text',''))[:40],
                     'provenance':'user_supplied_local_lrc',
                     'status':'from_player_metadata_clock_not_acoustically_verified'})
        if len(cues)>=limit:break
    return {'status':'user_local_lrc_estimated_sync' if cues else 'lrc_outside_current_interval',
            'time_basis':'estimated_player_position_not_exact',
            'timed_cues':cues,
            'warning':'用户已有的 LRC，依据播放器时间估算对齐；不是歌声识别校验'}
