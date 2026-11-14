"""Download telemetry shared by isolated model installers (stdlib only).

Never fabricates completed bytes or an ETA.  The percent stays below 100
until a separate hash/integrity check succeeds.  Telemetry is a view of file
transfer, not permission to activate a model.
"""
from __future__ import annotations

import math
import time


class TransferMeter:
    def __init__(self, *, stall_after=25.0, clock=time.monotonic):
        self._clock = clock
        self.stall_after = float(stall_after)
        self.last_at = None
        self.last_bytes = None
        self.last_move_at = None
        self.last_move_wall = None
        self.started_at = self._clock()
        self.rate = None

    def update(self, received, total=None, *, restarted=False, estimated=True):
        now = self._clock()
        done = max(0, int(received or 0))
        maximum = int(total) if total is not None and int(total) > 0 else None
        if restarted or self.last_bytes is None or done < self.last_bytes:
            self.rate = None
            self.last_at = now
            self.last_bytes = done
            self.last_move_at = now
            self.last_move_wall = time.time()
        else:
            dt = now - self.last_at
            delta = done - self.last_bytes
            if delta > 0:
                if dt > 0.05:
                    sample = delta / dt
                    self.rate = sample if self.rate is None else .40 * sample + .60 * self.rate
                self.last_bytes = done
                self.last_at = now
                self.last_move_at = now
            self.last_move_wall = time.time()
        rate = self.rate if self.rate is not None and self.rate > 0 and math.isfinite(self.rate) else None
        eta = (max(0, maximum - done) / rate if rate and maximum is not None else None)
        pct = (min(99, max(0, round(100 * done / maximum, 1))) if maximum else None)
        return {
            'bytes_received': done,
            'total_bytes': maximum,
            'progress_pct': pct,
            'progress_estimated': bool(estimated),
            'speed_bytes_s': round(rate, 1) if rate else None,
            'eta_seconds': round(eta) if eta is not None and math.isfinite(eta) else None,
            'stalled': bool(self.last_move_at is not None and now - self.last_move_at >= self.stall_after),
            'last_progress_at': self.last_move_wall,
        }


def snapshot_bytes(folder):
    """Best effort HF local_dir observation, never reads model contents.

    Completed files appear in local_dir; in-flight Hugging Face downloads
    commonly appear under .cache/huggingface/download/*.incomplete.  We
    intentionally report this as an estimate since the Hub/Xet backend may
    write data in ways that do not expose live byte counts.
    """
    from pathlib import Path
    folder = Path(folder)
    completed = {}
    try:
        for entry in folder.iterdir():
            if entry.is_file() and not entry.is_symlink() and not entry.name.startswith('.'):
                if entry.suffix.lower() in {'.bin', '.safetensors', '.json', '.txt', '.model', '.tiktoken'}:
                    completed[entry.name] = max(0, entry.stat().st_size)
    except OSError:
        pass
    pending = 0
    cache = folder / '.cache' / 'huggingface' / 'download'
    try:
        for entry in cache.glob('*.incomplete'):
            if entry.is_file() and not entry.is_symlink():
                # Ignore an incomplete cache part after its destination file
                # already exists (cannot count the same data twice).
                if not any(entry.name.startswith(name+'.') for name in completed):
                    pending += max(0, entry.stat().st_size)
    except OSError:
        pass
    return sum(completed.values()) + pending
