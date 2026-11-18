"""Dedicated Windows speaker-loopback subprocess; no N.E.K.O host SDK imports.

Launched only after user consent, by the already-provisioned local CPython.
Sends framed 16 kHz mono float32 in RAM over stdout to the plugin worker.
Only render-device loopback is enumerated; there is no microphone fallback.
"""
from __future__ import annotations

import json
import struct
import sys
import traceback
import types
from contextlib import redirect_stdout
from pathlib import Path

_HEADER = struct.Struct('!cI')
_OUT = sys.stdout.buffer
# 采集块：每 50ms 送一包起音强度（O），2 秒再送一包整段音频（D/H）。
# 起音包很小但够快，Live2D 的动作延迟从约 2 秒降到约 0.1 秒。
BLOCK_SECONDS = 0.05
ONSET_HOP_SECONDS = 0.025


class _OnsetTracker:
    """25ms 粒度的对数能量起音包络，和 auto_listener 里的算法保持一致。"""

    def __init__(self, rate: int):
        import numpy as np
        self._np = np
        self.hop = max(1, int(rate * ONSET_HOP_SECONDS))
        self.buf = np.zeros(0, dtype='float32')
        self.prev = None

    def push(self, block):
        np = self._np
        self.buf = np.concatenate((self.buf, np.asarray(block, dtype='float32')))
        n = self.buf.size // self.hop
        if not n:
            return None
        frames = self.buf[:n * self.hop].reshape(n, self.hop)
        self.buf = self.buf[n * self.hop:]
        rms = np.sqrt(np.mean(frames * frames, axis=1))
        log = np.log10(np.maximum(rms, 1e-5))
        if self.prev is None:
            delta = np.zeros(n, dtype='float32')
        else:
            delta = np.diff(np.concatenate((np.asarray([self.prev], dtype='float32'), log)))
        self.prev = float(log[-1])
        return np.clip(delta * 2.5, 0.0, 1.0).astype('<f4', copy=False)


def _send(tag: bytes, body: bytes):
    if len(body) > 1_000_000:
        raise ValueError('audio packet too large')
    _OUT.write(_HEADER.pack(tag, len(body)))
    _OUT.write(body)
    _OUT.flush()


def _json_packet(tag: bytes, value: dict):
    _send(tag, json.dumps(value, ensure_ascii=False).encode('utf-8'))


def main() -> int:
    # Diagnostic text from Windows subprocesses must use a stable encoding.
    # N.E.K.O collects the stderr pipe into loopback_diagnostics.log.
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
    except (AttributeError, OSError, ValueError):
        pass
    stage = '加载回环依赖'
    try:
        if sys.platform != 'win32':
            raise RuntimeError('当前隔离监听进程不在 Windows 系统运行')
        if len(sys.argv) not in (2,3,4):
            raise RuntimeError('隔离监听未收到插件专属音频依赖目录')
        vendor = Path(sys.argv[1]).resolve()
        capture_stereo = ('stereo' in sys.argv[2:])
        output_id = next((arg[10:] for arg in sys.argv[2:] if arg.startswith('device_id=')), '')
        if not vendor.is_dir():
            raise RuntimeError('音频依赖目录不存在')
        sys.path.insert(0, str(vendor))
        # Import auto_listener as a proper package module, without executing
        # music_perception/__init__.py (which requires N.E.K.O's host SDK).
        # The bare ``from auto_listener import ...`` previously crashed here:
        # auto_listener imports .song_change, which needs a package parent.
        # This mirrors analysis_worker.py and lyrics_worker.py exactly.
        source_dir = Path(__file__).resolve().parent
        package = types.ModuleType('music_perception')
        package.__path__ = [str(source_dir)]
        package.__package__ = 'music_perception'
        sys.modules['music_perception'] = package
        with redirect_stdout(sys.stderr):
            from music_perception.auto_listener import WINDOW_SECONDS, ListenerError, SoundCardLoopback
            # In v0.4.7 the SoundCard backend initializes its own COM. Avoid
            # any prior CoInitializeEx calls in this process (S_FALSE bug).
            last_error = None
            # Different speaker drivers advertise 44.1 kHz or 48 kHz; try both.
            # This retry never switches to an ordinary microphone.
            for rate in (48000, 44100):
                stage = f'打开扬声器回环（{rate}Hz）'
                try:
                    with SoundCardLoopback(capture_rate=rate,
                                           capture_stereo=capture_stereo,
                                           output_device_id=output_id) as source:
                        stage = '设备初始化完成'
                        _json_packet(b'R', {'device': str(getattr(source, 'device_name', '系统默认扬声器'))[:160],
                                             'sample_rate': 16000, 'capture_rate': rate})
                        stage = '正在读取扬声器回环帧'
                        import math

                        import numpy as np
                        from scipy.signal import resample_poly
                        blocks = max(1, int(round(WINDOW_SECONDS / BLOCK_SECONDS)))
                        onset = _OnsetTracker(rate)
                        div = math.gcd(rate, 16000)
                        while True:
                            parts = []
                            stereos = [] if capture_stereo else None
                            for _ in range(blocks):
                                blk = source.read_block(BLOCK_SECONDS)
                                parts.append(blk)
                                if capture_stereo:
                                    full = source.last_stereo
                                    if full is None:
                                        raise RuntimeError('full-band capture unavailable')
                                    stereos.append(np.asarray(full, dtype='float32'))
                                vals = onset.push(blk)
                                # 起音包先走：重活管线还在攒 2 秒，但动作这边已经拿到了
                                if vals is not None and vals.size:
                                    _send(b'O', vals.tobytes())
                            mono = np.concatenate(parts) if len(parts) > 1 else parts[0]
                            pcm = resample_poly(mono, 16000 // div, rate // div).astype('<f4', copy=False)
                            # No disk I/O and no cloud traffic. This pipe is local.
                            if capture_stereo:
                                _send(b'H', np.concatenate(stereos).astype('<f4', copy=False).tobytes())
                            _send(b'D', pcm.tobytes())
                except (ListenerError, RuntimeError, OSError, ValueError) as exc:
                    last_error = exc
                    if stage != f'打开扬声器回环（{rate}Hz）':
                        # Once READY is sent, switching formats would corrupt
                        # the stream; report the real read error and stop.
                        raise
                    print(f'回环尝试 {rate} Hz 失败：{type(exc).__name__}: {exc}', file=sys.stderr)
            raise RuntimeError(f'48k/44.1kHz 回环采集均失败：{last_error}')
    except BaseException as exc:
        print(f'[{stage}] {type(exc).__name__}: {exc}', file=sys.stderr)
        traceback.print_exc(file=sys.stderr, limit=8)
        try:
            _json_packet(b'E', {'stage': stage, 'error': f'{type(exc).__name__}: {exc}'})
        except (BrokenPipeError, OSError):
            pass
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
