"""Infer where a video sits in a series from its title (heuristic, labelled as such).

Only the title text is inspected, and only the derived label leaves this module.
Raw titles must never be placed into AI-facing payloads.
"""
from __future__ import annotations

import re

_PART_PATTERNS = (
    re.compile(r'(?<![A-Za-z])P\s*(\d{1,3})(?![0-9])', re.IGNORECASE),
    re.compile(r'第\s*(\d{1,3})\s*[集话話部节節期章段]'),
    re.compile(r'[\(（\[【]\s*(\d{1,3})\s*[\)）\]】]\s*$'),
    re.compile(r'[-_#]\s*(\d{1,3})\s*$'),
)
_SERIES_STRIP = re.compile(
    r'\s*(?:P\s*\d{1,3}|第\s*\d{1,3}\s*[集话話部节節期章段]|[\(（\[【]\s*\d{1,3}\s*[\)）\]】]|[-_#]\s*\d{1,3})\s*$',
    re.IGNORECASE)


def _norm(text: str) -> str:
    return ' '.join(str(text or '').casefold().split())


def parse_part(title: str):
    """Return (series_key, part_number) or (series_key, None)."""
    raw = str(title or '').strip()
    if not raw:
        return '', None
    part = None
    for pattern in _PART_PATTERNS:
        match = pattern.search(raw)
        if match:
            part = int(match.group(1))
            break
    base = _SERIES_STRIP.sub('', raw) if part is not None else raw
    return _norm(base)[:120], part


def infer_video_position(previous_title, current_title):
    """Safe wrapper for AI payloads: never fails, never returns titles."""
    try:
        return infer_position(str(previous_title or ''), str(current_title or ''))
    except Exception:
        return {'position': 'unknown', 'inferred': True, 'confidence': 'none'}


def infer_position(previous_title: str, current_title: str, previous_series_total=None):
    """Label the current video relative to the previous one.

    Returns a dict containing only labels and a confidence flag, never titles.
    """
    cur_key, cur_part = parse_part(current_title)
    if not cur_key:
        return {'position': 'unknown', 'inferred': True, 'confidence': 'none'}
    if not previous_title:
        if cur_part is None:
            return {'position': 'standalone', 'inferred': True, 'confidence': 'low'}
        return {'position': 'first' if cur_part == 1 else 'middle',
                'inferred': True, 'confidence': 'medium'}
    prev_key, prev_part = parse_part(previous_title)
    if not prev_key or prev_key != cur_key:
        return {'position': 'standalone', 'inferred': True, 'confidence': 'low'}
    if cur_part is None or prev_part is None:
        return {'position': 'unknown', 'inferred': True, 'confidence': 'low'}
    if cur_part == prev_part + 1:
        return {'position': 'next', 'inferred': True, 'confidence': 'medium'}
    if cur_part == prev_part - 1:
        return {'position': 'previous', 'inferred': True, 'confidence': 'medium'}
    return {'position': 'same_series', 'inferred': True, 'confidence': 'low'}
