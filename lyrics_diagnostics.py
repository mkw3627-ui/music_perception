"""Privacy-preserving, standard-library diagnostic milestones for local ASR.

Worker snapshots never store PCM, lyrics, song names, arbitrary paths, error
messages or GPU identifiers. Reads and writes are best effort so a locked
Windows file never breaks audio capture or recognition.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

STAGES = frozenset({
    'worker_started', 'request_received', 'checking_model', 'model_ready',
    'loading_cuda', 'loading_cpu', 'gpu_fallback', 'model_reused',
    'decoding', 'preflight', 'completed', 'failed', 'idle', 'importing_dependencies',
})


def publish_stage(path, stage: str, *, job=0, kind='transcribe', model='base',
                  requested_device='cpu', actual_device=None, error_type=None):
    """Write small immutable metadata. Never let diagnostics interrupt the worker."""
    if not path or stage not in STAGES:
        return
    record = {
        'schema': 'lyrics_worker_diagnostics_v1',
        'stage': stage, 'pid': os.getpid(),
        'updated_at': round(time.time(), 2),
        'job': job if type(job) is int else 0,
        'kind': kind if kind in {'prepare', 'transcribe'} else 'transcribe',
        'model': model if model in {'tiny', 'base', 'small', 'large-v3-turbo',
                                    'sing-whisper-small', 'sing-whisper-large-v3',
                                    'sing-qwen-0.6b', 'sing-qwen-1.7b'} else 'unknown',
        'requested_device': requested_device if requested_device in {'cpu','cuda','auto'} else 'unknown',
        'actual_device': actual_device if actual_device in {'cpu','cuda'} else None,
        'error_type': (error_type if error_type in {
            'ImportError', 'ModuleNotFoundError', 'RuntimeError', 'OSError',
            'ValueError', 'MemoryError', 'LyricsModelDownloadError', 'TimeoutError',
            'TypeError', 'FileNotFoundError'} else None),
    }
    try:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A transient Windows file lock should only hide diagnostics, never
        # abort recognition. No raw sound or lyrics are ever written here.
        line = json.dumps(record, ensure_ascii=False)
        destination.write_text(line, encoding='utf-8')
        history = destination.with_name('lyrics_diagnostics.log')
        # Keep a small diagnostic timeline without transcript/audio content.
        if history.exists() and history.stat().st_size > 256_000:
            history.write_text('', encoding='utf-8')
        with history.open('a', encoding='utf-8') as f:
            f.write(line + '\n')
    except (OSError, ValueError):
        pass


def read_stage(path, *, max_age=86400):
    if not path:
        return None
    try:
        p = Path(path)
        if not p.is_file() or p.stat().st_size > 4096:
            return None
        record = json.loads(p.read_text('utf-8'))
        if record.get('schema') != 'lyrics_worker_diagnostics_v1' or record.get('stage') not in STAGES:
            return None
        age = max(0.0, time.time() - float(record['updated_at']))
        if age > max_age:
            return None
        record['stage_age_seconds'] = round(age, 1)
        return record
    except (OSError, ValueError, TypeError, UnicodeError, KeyError):
        return None


def diagnose_stall(*, stage, stage_age_seconds, queue_depth, completed_count, child_running):
    """Observed state, not an invented cause; explicitly flag uncertain cases."""
    if not child_running:
        return '未检测到正在运行的歌词子进程；可能尚未开始或进程已退出。'
    if stage in {'checking_model', 'loading_cuda', 'loading_cpu', 'preflight'}:
        if stage_age_seconds >= 45:
            return '模型准备超过45秒；可能在校验权重、初始化 CUDA 或加载大模型，需查看深度检测结果。'
        return '模型正在准备中，尚不能确定实际运行设备。'
    if stage == 'decoding' and stage_age_seconds >= 40:
        return '歌词推理超过40秒；需核对是否已回退 CPU，或 CUDA 原生运行库是否兼容。'
    if stage in {'request_received', 'checking_model'} and stage_age_seconds >= 60:
        return '歌词任务长时间无进展；可运行一键检测区分依赖、模型和进程问题。'
    if queue_depth >= 2 and completed_count == 0:
        return '歌词队列已满且尚无结果；请检查当前工作阶段，不能据此认定 GPU 算力不足。'
    return None
