"""Short lived Windows render-meter evidence (never PCM, never content proof).

Only data actually returned by live peak meter may enter the history. A very
short playback silence must not erase the PID + process-start identity, but a
recent meter is NOT evidence that sound is audible at the current instant.
"""
from __future__ import annotations

from typing import Callable, Any


def merge_recent_meter(current: list[dict], history: dict[str, dict], *,
                       now: float, is_same_process: Callable[[int, str], bool],
                       grace_s: float = 10.0, max_kept: int = 64,
                       selected_identity: str | None = None):
    """Return display rows + bounded history, with all recent-only rows muted.

    The explicitly selected capture source stays visible while silent (it is
    still the verified recognition target); other sources expire after grace_s.
    """
    from .audio_dispatch import source_id
    grace = max(0.5, min(12.0, float(grace_s)))
    fresh: dict[str, dict] = {}
    seen = set()
    result = []
    for row in current[:120]:
        if not isinstance(row, dict) or not row.get('audible'):
            continue
        try:
            identity=source_id(row)
        except (KeyError, TypeError, ValueError):
            continue
        live = dict(row, recently_audible=False, meter_state='live_peak',
                    last_live_at=now, last_live_dbfs=row.get('level_dbfs'))
        seen.add(identity)
        result.append(live)
        fresh[identity]={'app':dict(row), 'seen_at':now}
    for identity, rec in sorted(history.items(), key=lambda item: -float(item[1].get('seen_at',0)))[:max_kept]:
        if identity in seen or not isinstance(rec, dict):
            continue
        try:
            row=dict(rec['app'])
            age=now-float(rec['seen_at'])
            pinned = selected_identity is not None and identity == selected_identity
            if not pinned and not (0 <= age <= grace):
                continue
            if not is_same_process(int(row['pid']),str(row['created'])):
                continue
            if source_id(row)!=identity:
                continue
        except (KeyError, ValueError, TypeError, OSError):
            continue
        recent=dict(row, audible=False, recently_audible=True,
                    meter_state=('selected_source_silent_waiting_for_audio' if pinned
                                 else 'recent_peak_but_instant_currently_silent'),
                    last_live_at=float(rec['seen_at']),
                    last_live_dbfs=row.get('level_dbfs'),
                    level_dbfs=None,peak=0.0)
        result.append(recent)
        fresh[identity]=rec
    return result, fresh
