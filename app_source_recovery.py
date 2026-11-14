"""Fail-closed selection repair for Windows per-process audio capture.

A saved Windows PID is a session identifier, not a persistent application.
Only a unique verified executable can be rebound; recovery never switches to
whole-speaker capture or to the N.E.K.O host.
"""
from __future__ import annotations

from .app_channels import normalized_exe_path

LEGACY_PLAYER_NAMES = frozenset({
    'spotify.exe', 'cloudmusic.exe', 'qqmusic.exe', 'foobar2000.exe',
    'musicbee.exe', 'aimp.exe', 'winamp.exe', 'applemusic.exe',
    'potplayer.exe', 'potplayermini64.exe', 'mpc-hc64.exe', 'mpc-be64.exe',
})


def find_replacement(selected: dict | None, sources: list[dict], *,
                     require_audible: bool = True) -> tuple[dict | None, str]:
    """Return a unique identity match for a previously selected application."""
    if not isinstance(selected, dict):
        return None, 'selection_missing'
    old_path = normalized_exe_path(selected.get('path'))
    old_name = str(selected.get('name') or '').strip().casefold()
    if not old_path and old_name not in LEGACY_PLAYER_NAMES:
        return None, 'legacy_selection_needs_one_time_reselect'
    matches = []
    for row in sources:
        if (not isinstance(row, dict) or
                (require_audible and not row.get('audible')) or row.get('host_ai')):
            continue
        try:
            pid = int(row.get('pid', 0))
            created = str(row.get('created') or '')
        except (TypeError, ValueError):
            continue
        if pid <= 4 or not created.isdecimal():
            continue
        name = str(row.get('name') or '').strip().casefold()
        path = normalized_exe_path(row.get('path'))
        if old_path:
            if not path or path != old_path or name != old_name:
                continue
        elif name != old_name or not path:
            continue
        matches.append(row)
    if len(matches) > 1:
        # Multi-process players (NetEase CloudMusic, QQMusic helpers) expose
        # the same executable several times. The process-tree root is the
        # stable identity and its capture covers the whole family, so use it
        # when exactly one root exists.
        pids = {int(m['pid']) for m in matches}
        roots = [m for m in matches if int(m.get('parent_pid') or 0) not in pids]
        if len(roots) == 1:
            return roots[0], 'unique_executable_root'
        return None, 'ambiguous_same_executable'
    if not matches:
        return None, 'waiting_for_saved_app_audio'
    return matches[0], ('unique_audible_match' if require_audible
                        else 'unique_running_match')
