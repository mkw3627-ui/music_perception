"""Fail-open Windows SMTC metadata reader (isolated from frozen N.E.K.O)."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys


def read_system_media(*,python_exe,vendor,worker_script,timeout=4.0):
    if not (Path(vendor).is_dir() and Path(worker_script).is_file()):
        return {'available':False,'reason':'media_worker_missing'}
    if Path(python_exe).name.lower().startswith(('projectneko','neko_server')):
        return {'available':False,'reason':'invalid_python'}
    variables=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONEXECUTABLE','PYTHONSTARTUP','VIRTUAL_ENV','CONDA_PREFIX'):
        variables.pop(key,None)
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0) if sys.platform=='win32' else 0
    try:
        response=subprocess.run([str(python_exe),'-I','-S',str(worker_script),str(vendor)],
                                capture_output=True,timeout=timeout,env=variables,creationflags=flags,
                                check=False)
        if response.returncode or len(response.stdout)>16000:
            return {'available':False,'reason':'metadata_process_failure'}
        info=json.loads(response.stdout.decode('utf-8'))
        if isinstance(info,dict) and isinstance(info.get('available'),bool):
            return info
    except (OSError,UnicodeError,ValueError,subprocess.TimeoutExpired):
        pass
    return {'available':False,'reason':'metadata_unavailable'}
