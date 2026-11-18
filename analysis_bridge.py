"""Bounded, private, out-of-process music analysis for the packaged N.E.K.O host.

All PCM transfers remain in local stdin pipes. Diagnostic logs contain only
worker stage names, exit codes and sanitized exception summaries, never sound
samples, transcript text, or the worker's stdout (which contains music data).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


class AnalysisWorkerError(RuntimeError):
    pass


class AnalysisWorkerCancelled(AnalysisWorkerError):
    """Plugin stopped or changed sessions; this isn't an audio failure."""


def _append_diagnostic(path: str | Path | None, detail: str) -> None:
    """Best-effort local-only log. A locked log must not break audio capture."""
    if path is None:
        return
    try:
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        # Bound history and avoid log growth from repeated failed clips.
        if file.is_file() and file.stat().st_size > 128 * 1024:
            file.write_text('', encoding='utf-8')
        timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
        with file.open('a', encoding='utf-8') as out:
            out.write(f'[{timestamp}] {detail[:2200]}\n')
    except (OSError, ValueError, UnicodeError):
        pass


def _worker_failure(returncode: int, err: bytes, out_bytes: int) -> str:
    """Keep stage breadcrumbs separate from the worker's actual exception."""
    lines = err.decode('utf-8', errors='replace').splitlines()
    stages = [line.partition(':')[2].strip() for line in lines
              if line.startswith('ANALYSIS_STAGE:')]
    real_errors = [line for line in lines if not line.startswith('ANALYSIS_STAGE:')]
    stage = stages[-1] if stages else 'python_entry_not_reached'
    unsigned = returncode & 0xffffffff
    error_text = '\n'.join(real_errors).strip()[-850:]
    suffix = error_text if error_text else '子进程未输出异常详情（可能是原生模块退出、进程终止或解释器启动失败，不能仅凭退出码确定）'
    return (f'独立音乐分析进程退出 exit={returncode} (0x{unsigned:08X})，'
            f'最后阶段={stage}，stdout_bytes={out_bytes}；{suffix}')


def analyze_pcm_isolated(samples, *, python_exe: str, vendor: str | Path,
                         worker_script: str | Path, options: dict,
                         on_process: Callable[[subprocess.Popen | None], None] | None = None,
                         timeout: int = 120,
                         is_cancelled: Callable[[], bool] | None = None,
                         diagnostic_file: str | Path | None = None) -> dict:
    import numpy as np
    audio = np.asarray(samples, dtype='<f4').reshape(-1)
    if not 16000 <= audio.size <= 90 * 16000:
        raise AnalysisWorkerError('实时音乐片段长度应在 1～90 秒之间')
    if not np.isfinite(audio).all():
        raise AnalysisWorkerError('音频包含非有限浮点数，已停止分析')
    vendor = Path(vendor).resolve()
    worker_script = Path(worker_script).resolve()
    if not vendor.is_dir() or not worker_script.is_file():
        raise AnalysisWorkerError('独立 Python 运行环境或音频分析入口不存在')
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if sys.platform == 'win32' else 0
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONEXECUTABLE',
                'VIRTUAL_ENV', 'CONDA_PREFIX'):
        env.pop(key, None)
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'  # -I ignores it; worker uses explicit bytes
    if Path(python_exe).name.lower().startswith(('projectneko', 'neko_server')):
        raise AnalysisWorkerError('禁止用 N.E.K.O 宿主程序运行音乐分析')
    if is_cancelled is not None and is_cancelled():
        raise AnalysisWorkerCancelled('音乐监听已停止，取消未开始的分析')
    argv = [str(python_exe), '-I', '-S', str(worker_script), str(vendor),
            json.dumps(options, ensure_ascii=False, separators=(',', ':'))]
    child = None
    try:
        child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, env=env, creationflags=flags)
        if on_process:
            on_process(child)
        out, err = child.communicate(input=audio.tobytes(), timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if child is not None:
            child.kill()
            child.communicate()
        if is_cancelled is not None and is_cancelled():
            raise AnalysisWorkerCancelled('监听已停止，已终止分析进程') from exc
        detail = f'音频分析子进程超过 {timeout} 秒，已终止本次处理；stage=timeout'
        _append_diagnostic(diagnostic_file, detail)
        raise AnalysisWorkerError(detail) from exc
    except OSError as exc:
        if is_cancelled is not None and is_cancelled():
            raise AnalysisWorkerCancelled('监听已停止，取消分析') from exc
        detail = f'无法启动独立 Python 音乐分析进程：{type(exc).__name__}: {exc}'
        _append_diagnostic(diagnostic_file, detail)
        raise AnalysisWorkerError(detail) from exc
    finally:
        if on_process:
            on_process(None)
    if is_cancelled is not None and is_cancelled():
        raise AnalysisWorkerCancelled('监听或插件关闭时，取消未完成的音乐分析')
    if child.returncode:
        detail = _worker_failure(child.returncode, err, len(out))
        _append_diagnostic(diagnostic_file, detail)
        raise AnalysisWorkerError(detail)
    if len(out) > 16 * 1024 * 1024:
        detail = '音频分析结果超过大小限制'
        _append_diagnostic(diagnostic_file, detail)
        raise AnalysisWorkerError(detail)
    try:
        result = json.loads(out.decode('utf-8'))
        if (result.get('version') != 1 or not isinstance(result.get('spectrum'), dict)
                or ('report' not in result)):
            raise ValueError('missing protocol fields')
        return result
    except (UnicodeDecodeError, ValueError, AttributeError) as exc:
        detail = (f'独立音频分析返回的 JSON 无效: {type(exc).__name__}；'
                  f'bytes={len(out)}；不会保存或打印分析结果原文')
        _append_diagnostic(diagnostic_file, detail)
        raise AnalysisWorkerError(detail) from exc
