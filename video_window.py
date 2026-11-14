"""Decide how much of a video to process, without asking the user.

Rules:
- Playback position and duration are known: process from the current position
  to the end, but never more than ``max_window_s`` seconds from it.
- Position unknown but duration known: process the first ``max_window_s``.
- Neither known: process the first ``max_window_s`` seconds (fallback).

Returns a dict with ``start_s``, ``end_s``, ``basis`` and ``inferred``. It never
contains titles or any other source text.
"""
from __future__ import annotations

DEFAULT_WINDOW_S = 600.0
_MAX_DURATION_S = 86400.0


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number < 0:  # NaN or negative
        return None
    return number


def decide_video_window(position_s=None, duration_s=None, max_window_s=DEFAULT_WINDOW_S):
    window = float(max_window_s) if _number(max_window_s) else DEFAULT_WINDOW_S
    pos = _number(position_s)
    dur = _number(duration_s)
    if dur is not None and not (0 < dur < _MAX_DURATION_S):
        dur = None
    if pos is not None and dur is not None and pos >= dur:
        pos = None  # position past the end is not trustworthy; treat as unknown

    if pos is not None:
        end = pos + window
        if dur is not None:
            end = min(end, dur)
        return {'start_s': round(pos, 2), 'end_s': round(end, 2),
                'basis': 'playback_position', 'inferred': True}

    end = window if dur is None else min(window, dur)
    return {'start_s': 0.0, 'end_s': round(end, 2),
            'basis': 'start_fallback' if dur is None else 'duration_bounded_start',
            'inferred': True}
