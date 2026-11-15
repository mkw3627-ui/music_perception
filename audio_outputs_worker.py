"""List render output endpoints for user-selected mixed-audio isolation."""
from __future__ import annotations
import json
from pathlib import Path
import sys


def main():
    if len(sys.argv) != 2:
        return 2
    vendor = Path(sys.argv[1]).resolve()
    if not vendor.is_dir():
        return 2
    sys.path.insert(0, str(vendor))
    try:
        import soundcard as sc
        default = sc.default_speaker()
        devices = [dict(id=str(item.id)[:512], name=str(item.name)[:160],
                        is_default=bool(default and str(default.id)==str(item.id)))
                   for item in list(sc.all_speakers())[:32]]
        response = {'ok': True, 'devices': devices}
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
        response = {'ok': False, 'error': type(exc).__name__, 'devices': []}
    sys.stdout.buffer.write(json.dumps(response, ensure_ascii=False).encode('utf-8')+b'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
