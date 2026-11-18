"""Explicitly approved GPU package installer (isolated standard-library process)."""
import sys
from pathlib import Path


def main():
    if len(sys.argv)!=4:
        return 2
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from stem_gpu_runtime import _status, install_gpu
    env,status,priority=sys.argv[1:]
    try:
        return 0 if install_gpu(Path(env),Path(status),priority) else 1
    except Exception as exc:
        _status(Path(status),state='failed',stage='bootstrap',progress_pct=0,
                message=f'{type(exc).__name__}: {str(exc)[:170]}')
        return 1
if __name__=='__main__':
    raise SystemExit(main())
