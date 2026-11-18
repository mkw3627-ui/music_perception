"""Conservative, dependency-free VRAM planning for *local* N.E.K.O model workers.

Only user-verified models are eligible. No download, Windows global dependency,
LLM notifications or recording occurs in this module. Measurements use
NVIDIA's own `nvidia-smi`; a missing/failed probe is explicitly unknown.

A CPU decision means unload the CUDA worker and RELOAD weights on the CPU;
never attempt to swap CUDA allocations directly into system RAM.
"""
from __future__ import annotations

import shutil
import ctypes
import os
from pathlib import Path
import subprocess
import time

TIERS = ('low', 'medium', 'high', 'ultra')
DOMAINS = ('video', 'singing', 'stem')
DEFAULT_RESERVE_MIB = 2048


def free_system_ram_mib():
    """Best-effort actual available RAM, not total installed RAM.

    Windows GlobalMemoryStatusEx is read-only. If unavailable, *do not*
    guess that a GPU+RAM split is safe. Never count swap/pagefile as RAM.
    """
    try:
        if os.name == 'nt':
            from ctypes import wintypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_=[('dwLength',wintypes.DWORD),('dwMemoryLoad',wintypes.DWORD),
                    ('ullTotalPhys',ctypes.c_ulonglong),('ullAvailPhys',ctypes.c_ulonglong),
                    ('ullTotalPageFile',ctypes.c_ulonglong),('ullAvailPageFile',ctypes.c_ulonglong),
                    ('ullTotalVirtual',ctypes.c_ulonglong),('ullAvailVirtual',ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual',ctypes.c_ulonglong)]
            info=MEMORYSTATUSEX()
            info.dwLength=ctypes.sizeof(info)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(info)):
                return None
            return int(info.ullAvailPhys//1048576)
        # POSIX only for local tests/developer preview, not Windows behavior.
        if Path('/proc/meminfo').is_file():
            for line in Path('/proc/meminfo').read_text('ascii').splitlines():
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1])//1024
    except (OSError,ValueError,AttributeError,TypeError):
        pass
    return None


def gpu_memory(timeout=2.0, runner=None, executable=None):
    """Return GPU 0's free memory, never guess on unavailable systems.

    `nvidia-smi --query-gpu=...` reports global free VRAM, *not* VRAM held by
    just this plugin. This is sufficient for a conservative resource guard,
    but not for attributing GPU load to a particular app/game.
    """
    clock = time.time()
    name = executable if executable is not None else shutil.which('nvidia-smi')
    if not name:
        return {'available': False, 'free_mib': None, 'total_mib': None,
                'reason': 'nvidia-smi_not_found', 'updated_at': clock}
    try:
        proc = (runner or subprocess.run)(
            [name, '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits', '-i', '0'],
            capture_output=True, text=True, timeout=max(.5, min(4., float(timeout))),
            check=False)
        if proc.returncode != 0:
            raise ValueError('nvidia-smi_unavailable')
        first = proc.stdout.splitlines()[0].strip().split(',')
        free, total = (int(first[0].strip()), int(first[1].strip()))
        if free < 0 or total < 256 or free > total or total > 1048576:
            raise ValueError('invalid_GPU_memory')
        return {'available': True, 'free_mib': free, 'total_mib': total,
                'reason': 'nvidia_smi_gpu0', 'updated_at': clock}
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError, TypeError):
        return {'available': False, 'free_mib': None, 'total_mib': None,
                'reason': 'nvidia-smi_probe_failed', 'updated_at': clock}


def select_model(*, preferred_tier, preferred_device, installed, requirements,
                 free_mib=None, reserve_mib=DEFAULT_RESERVE_MIB,
                 released_vram_mib=0, allow_cpu=False, enabled=True,
                 allow_hybrid=False, hybrid_tiers=(), free_ram_mib=None):
    """Pick <= preferred tier, never a missing model or silent CPU fallback.

    `released_vram_mib` is an *estimate* of this plugin's resident checkpoints
    which will be terminated BEFORE loading the replacement. This is explicitly
    not a guaranteed available amount; real model preflight is still required.
    """
    if preferred_tier not in TIERS or preferred_device not in ('cuda', 'cpu'):
        raise ValueError('invalid preferred tier/device')
    if not isinstance(installed, dict) or not isinstance(requirements, dict):
        raise ValueError('invalid model inventory')
    reserve = max(512, min(8192, int(reserve_mib)))
    if not installed.get(preferred_tier, False):
        return {'state': 'model_missing', 'quality': None, 'device': None,
                'reason': 'selected_model_not_verified', 'requires_cpu_consent': False}
    if preferred_device == 'cpu':
        return {'state': 'ready', 'quality': preferred_tier, 'device': 'cpu',
                'reason': 'user_selected_ram_cpu', 'requires_cpu_consent': False}
    if not enabled or free_mib is None:
        return {'state': 'ready', 'quality': preferred_tier, 'device': 'cuda',
                'reason': 'gpu_unmeasured' if free_mib is None else 'automatic_disabled',
                'requires_cpu_consent': False}
    available = max(0, int(free_mib) + max(0, int(released_vram_mib)) - reserve)
    # Never upgrade above the user's preferred tier, never download implicitly.
    i = TIERS.index(preferred_tier)
    for tier in reversed(TIERS[:i+1]):
        if installed.get(tier, False) and int(requirements.get(tier, 10**12)) <= available:
            return {'state': 'ready', 'quality': tier, 'device': 'cuda',
                    'reason': ('gpu_budget_ok' if tier == preferred_tier else 'vram_pressure_downshift'),
                    'requires_cpu_consent': False}
    # Experimental Qwen/Accelerate weight offload: preserve CUDA for the
    # GPU-resident layers while keeping SOME weights in RAM. It is NOT
    # Windows WDDM shared-GPU allocation and is never available to ONNX/CT2.
    # Requires real RAM measurement and a GPU budget for activations/kernels.
    if (allow_hybrid and preferred_tier in hybrid_tiers and
            isinstance(free_ram_mib,int) and free_ram_mib >= 6144 and
            available >= 1536):
        gpu_budget=min(available, int(requirements[preferred_tier])-512)
        cpu_budget=min(32768,free_ram_mib-2048)
        return {'state':'ready','quality':preferred_tier,'device':'cuda',
                'placement':'gpu_ram_weights', 'gpu_budget_mib':gpu_budget,
                'cpu_budget_mib':cpu_budget,
                'reason':'experimental_qwen_gpu_ram_weight_offload',
                'requires_cpu_consent':False}
    if allow_cpu:
        return {'state': 'ready', 'quality': preferred_tier, 'device': 'cpu',
                'reason': 'vram_pressure_user_approved_cpu', 'requires_cpu_consent': False}
    return {'state': 'insufficient_vram', 'quality': None, 'device': None,
            'reason': 'no_verified_tier_fits_gpu', 'requires_cpu_consent': True}


class PressureTracker:
    """Two low-VRAM samples to avoid one-frame churn; stable-recovery delay.

    Pressure only triggers *decisions* at safe clip boundaries. It never kills
    an in-flight model inference or steals memory from other applications.
    """
    def __init__(self, reserve_mib=DEFAULT_RESERVE_MIB):
        self.reserve_mib = max(512, min(8192, int(reserve_mib)))
        self.low_hits = 0
        self.last_pressure = 0.
        self.latest = {'available': False, 'free_mib': None,
                       'total_mib': None, 'reason': 'not_sampled'}

    def update(self, sample, now=None):
        now = time.monotonic() if now is None else float(now)
        self.latest = dict(sample)
        free = sample.get('free_mib') if sample.get('available') else None
        if isinstance(free, int) and free < self.reserve_mib:
            self.low_hits += 1
            if self.low_hits >= 2:
                self.last_pressure = now
        else:
            self.low_hits = 0
        return self.pressured(now)

    def pressured(self, now=None):
        now = time.monotonic() if now is None else float(now)
        # Do not auto-upgrade during the first 60 seconds after low-VRAM alert.
        return self.low_hits >= 2 or (self.last_pressure > 0 and
                                     now - self.last_pressure < 60.)
