"""Launch a bounded, read-only CUDA probe outside the N.E.K.O host process."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def probe_cuda_isolated(*, python_exe, vendor, script, timeout=12):
    python_exe=Path(python_exe)
    vendor=Path(vendor)
    script=Path(script)
    if python_exe.name.lower().startswith(('projectneko', 'neko_server')):
        return {'cuda_candidate':False,'message':'拒绝使用 N.E.K.O 主程序作为 Python。'}
    if not vendor.is_dir() or not script.is_file() or not python_exe.is_file():
        return {'cuda_candidate':False,'message':'独立歌词环境尚未就绪，请等待自动环境安装。'}
    env=os.environ.copy()
    for name in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','PYTHONEXECUTABLE','VIRTUAL_ENV','CONDA_PREFIX'):
        env.pop(name,None)
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0) if sys.platform=='win32' else 0
    try:
        proc=subprocess.run([str(python_exe),'-I','-S',str(script),str(vendor)],
                            capture_output=True,timeout=timeout,env=env,
                            creationflags=flags,check=False)
        if proc.returncode or len(proc.stdout)>8192:
            return {'cuda_candidate':False,'message':'显卡检测进程执行失败；CPU 识别仍可用。'}
        result=json.loads(proc.stdout.decode('utf-8'))
        if result.get('version')!=1 or type(result.get('cuda_candidate')) is not bool:
            raise ValueError('Invalid GPU probe output')
        return {key:result[key] for key in ('cuda_candidate','gpu_count','compute_types','message')
                if key in result}
    except (OSError, subprocess.TimeoutExpired, UnicodeError, ValueError):
        return {'cuda_candidate':False,'message':'显卡检测超时或环境不兼容；CPU 识别仍可用。'}
