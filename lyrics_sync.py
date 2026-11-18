"""Safe parser for user-supplied synchronized LRC lyrics.

Never scrapes an unlicensed lyrics service or pretends timestamps have been
aligned to the user's Windows player. Song metadata and capture timing have
to be correctly anchored before using LRC against live loopback audio.
"""
from __future__ import annotations

import re
from pathlib import Path

TIMETAG = re.compile(r'\[(\d{1,3}):(\d{2})(?:[.,](\d{1,3}))?\]')
OFFSET = re.compile(r'^\[offset\s*:\s*([+-]?\d+)\s*\]$', re.I)


def parse_lrc(text: str, *, max_lines: int = 1000) -> list[dict]:
    if not isinstance(text, str) or len(text) > 180_000:
        raise ValueError('LRC 内容为空或超过大小上限')
    offset_ms = 0
    rows = text.lstrip('\ufeff').splitlines()
    for raw in rows:
        m = OFFSET.match(raw.strip())
        if m:
            offset_ms = max(-600000, min(600000, int(m.group(1))))
            break
    lines = []
    for raw in rows:
        tags = list(TIMETAG.finditer(raw))
        if not tags:
            continue
        content = TIMETAG.sub('',raw).strip()
        if not content or len(content) > 500:
            continue
        for tag in tags:
            mm = int(tag.group(1))
            sec = int(tag.group(2))
            if sec >= 60:
                continue
            frac = tag.group(3) or '0'
            # [mm:ss.xx] hundredths; [.x] tenths; [.xxx] milliseconds.
            ms = int(frac.ljust(3,'0'))
            at = mm*60 + sec + ms/1000 + offset_ms/1000
            if at >= 0:
                lines.append({'at_s':round(at,3),'text':content})
            if len(lines) >= max_lines:
                break
        if len(lines) >= max_lines:
            break
    return sorted(lines,key=lambda x:x['at_s'])


def load_adjacent_lrc(audio_path: str | Path, *, allowed_roots: list[Path]) -> list[dict]:
    """Only load alongside a user-authorized local music file, no network."""
    audio = Path(audio_path).expanduser().resolve(strict=True)
    path = audio.with_suffix('.lrc').resolve()
    roots = [root.expanduser().resolve() for root in allowed_roots]
    if not any(path.is_relative_to(root) for root in roots):
        raise PermissionError('歌词文件路径不属于用户授权的音乐目录')
    if not path.is_file():
        return []
    if path.stat().st_size > 180_000:
        raise ValueError('LRC 文件超过大小限制')
    raw = path.read_bytes()
    for encoding in ('utf-8-sig','utf-16','gb18030'):
        try:
            return parse_lrc(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    raise ValueError('LRC 编码无法识别')
