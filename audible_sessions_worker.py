"""Read Windows render-session peaks in an isolated, killable process.

Runs the COM snapshot out of process so a hung audio driver cannot block the
plugin host. Emits only JSON rows of {pid, peak, shared_pid, system_audio};
never reads PCM and never opens a recording stream.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    sys.path.insert(0, str(base))
    try:
        from windows_audio_sessions import scan_audible_sessions
        rows = scan_audible_sessions(sample_interval=0.12)
        response = {'ok': True, 'sessions': [
            {'pid': int(r['pid']), 'peak': float(r.get('peak', 0.0)),
             'sample_count': int(r.get('sample_count', 0))}
            for r in rows[:256]]}
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
        response = {'ok': False, 'error': type(exc).__name__, 'sessions': []}
    sys.stdout.buffer.write(json.dumps(response, ensure_ascii=False).encode('utf-8') + b'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
