"""Windows speaker-loopback reader isolated from the frozen N.E.K.O host.

Only the dedicated, locally managed CPython is allowed to launch the worker.
It captures Windows *render endpoint* loopback audio; microphone capture is not
an alternative. Audio frames stay in memory and pass through a local pipe.
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from collections import deque
from pathlib import Path
import struct
import subprocess
import sys
from typing import BinaryIO

from .auto_listener import ListenerError, SAMPLE_RATE, WINDOW_SECONDS

_HEADER = struct.Struct('!cI')
_MAX_PACKET = 1_000_000
FRAME_TIMEOUT_SECONDS = 15.0
# 25ms 粒度的起音包络：保留最近 2 秒（80 个点），和 auto_listener 的 hop 一致
ONSET_HOP_SECONDS = 0.025
ONSET_RING = 80


def _read_exact(pipe: BinaryIO, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        part = pipe.read(size - len(result))
        if not part:
            raise EOFError(f'子进程音频管道提前关闭（收到 {len(result)}/{size} 字节）')
        result.extend(part)
    return bytes(result)


def _read_packet(pipe: BinaryIO) -> tuple[bytes, bytes]:
    tag, length = _HEADER.unpack(_read_exact(pipe, _HEADER.size))
    if tag not in {b'R', b'D', b'E', b'H', b'O'} or length > _MAX_PACKET:
        raise ListenerError('隔离回环进程发送了无效数据帧，已拒绝读取')
    return tag, _read_exact(pipe, length)


class IsolatedWindowsLoopback:
    """Read signed-off Windows WASAPI loopback from a separate CPython process.

    The Popen invocation is never run with N.E.K.O's frozen sys.executable.
    No microphone is used and no raw audio is written to disk. stop() may be
    called from another thread, to unblock any pending audio pipe read.
    """

    def __init__(self, *, python: str, vendor: str | Path, worker_script: str | Path,
                 diagnostics: str | Path, start_timeout: float = 12,
                 capture_stereo: bool = False, output_device_id: str = '',
                 stream_onsets: bool = False):
        self.python = Path(python).resolve()
        self.vendor = Path(vendor).resolve()
        self.script = Path(worker_script).resolve()
        self.diagnostics = Path(diagnostics).resolve()
        self.start_timeout = float(start_timeout)
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.device = None
        self.capture_stereo = bool(capture_stereo)
        self.output_device_id = str(output_device_id or '')[:512]
        self.capture_rate = 48000
        self.last_stereo = None
        # 流式起音：工作进程每 50ms 送一小包 O，这里攒成一个 2 秒的滚动环，
        # 主循环每 100ms 取一次快照给 Live2D，动作延迟就只剩采集本身。
        self.stream_onsets = bool(stream_onsets)
        self._onset_lock = threading.Lock()
        self._onset_ring = deque(maxlen=ONSET_RING)
        self._frames = None
        self._pending_stereo = None
        self._reader = None
        self._stop_reader = threading.Event()

    def _absorb_onset(self, payload: bytes):
        """收一包 25ms 起音值，按到达时间倒推出每个点的时刻。"""
        import numpy as np
        try:
            arr = np.frombuffer(payload, dtype='<f4')
        except (TypeError, ValueError):
            return
        if not arr.size:
            return
        now = time.time()
        n = int(arr.size)
        with self._onset_lock:
            for i in range(n):
                self._onset_ring.append((now - (n - 1 - i) * ONSET_HOP_SECONDS,
                                         float(arr[i])))

    def take_onsets(self):
        """返回 (最近 2 秒的起音包络, 最新一点的时刻)；还没数据时返回 None。"""
        with self._onset_lock:
            if not self._onset_ring:
                return None
            items = list(self._onset_ring)
        at = items[-1][0]
        keep = ONSET_RING
        if len(items) > keep:
            items = items[-keep:]
        return [round(v, 3) for _, v in items], at

    def _next_data_packet(self, proc):
        """读包，顺便把中间的起音包吸收掉（它们不是音频帧）。"""
        while True:
            tag, payload = _read_packet(proc.stdout)
            if tag == b'O':
                self._absorb_onset(payload)
                continue
            return tag, payload

    def _push_frame(self, item):
        q = self._frames
        if q is None:
            return
        try:
            q.put_nowait(item)
        except queue.Full:
            # 主循环落后时丢最旧的音频：宁可丢一段，也不能把管道堵死，
            # 否则采集进程会被反压，起音数据也跟着停。
            try:
                q.get_nowait()
                q.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def _reader_loop(self):
        proc = self._proc
        try:
            while (proc is not None and proc.poll() is None
                   and not self._stop_reader.is_set()):
                tag, payload = self._next_data_packet(proc)
                if tag == b'H':
                    self._pending_stereo = payload
                elif tag == b'D':
                    self._push_frame((b'D', self._pending_stereo, payload))
                    self._pending_stereo = None
                elif tag == b'E':
                    self._push_frame((b'E', payload, None))
                    return
        except Exception as exc:                      # noqa: BLE001 - 交给 read() 复述
            self._push_frame((b'X', exc, None))

    def _pending_frames(self):
        """测试替身（内存管道）里没有后台线程时退回同步读。"""
        return self._frames is not None

    def _validate_paths(self):
        if sys.platform != 'win32':
            raise ListenerError('隔离式 WASAPI 监听仅适用于 Windows')
        # 名字放宽到 python.exe / python3.11.exe 这类（uv 装出来的解释器就叫
        # python3.11.exe，之前只认 python.exe，会把可用的解释器误判成没有）。
        name = self.python.name.lower()
        if (not re.fullmatch(r'python(?:\d+(?:\.\d+)?)?(?:\.exe)?', name)
                or not self.python.is_file()
                or 'projectneko' in str(self.python).lower()
                or 'neko_server' in str(self.python).lower()):
            raise ListenerError('没有可用的独立 python.exe；不能将 N.E.K.O 主程序当 Python 运行')
        if not self.vendor.is_dir():
            raise ListenerError('插件音频依赖目录未就绪；请先完成自动环境修复')
        if not self.script.is_file() or self.script.name not in {'loopback_worker.py', 'application_loopback_worker.py'}:
            raise ListenerError('隔离式回环程序文件缺失，请重新加载完整插件源码')

    def _launch_args(self):
        return ([str(self.python), '-I', '-S', str(self.script), str(self.vendor)]
                + (['stereo'] if self.capture_stereo else [])
                + (['device_id='+self.output_device_id] if self.output_device_id else []))

    def __enter__(self):
        self._validate_paths()
        self.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        if self.diagnostics.exists() and self.diagnostics.stat().st_size > 1_000_000:
            self.diagnostics.write_text('', encoding='utf-8')
        env = dict(os.environ)
        for key in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP', 'PYTHONUSERBASE', 'VIRTUAL_ENV'):
            env.pop(key, None)
        try:
            with self.diagnostics.open('ab') as log:
                proc = subprocess.Popen(
                    self._launch_args(),
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log,
                    cwd=str(self.script.parent), env=env, bufsize=0,
                )
            with self._lock:
                self._proc = proc
            # A helper thread enforces startup timeout without Windows select()
            # (select cannot poll anonymous pipes on Windows).
            result: list = []
            done = threading.Event()
            def handshake():
                try:
                    result.append(_read_packet(proc.stdout))
                except Exception as exc:
                    result.append(exc)
                finally:
                    done.set()
            thread = threading.Thread(target=handshake, name='neko-loopback-handshake', daemon=True)
            thread.start()
            if not done.wait(self.start_timeout):
                raise ListenerError(f'独立 WASAPI 音频进程启动超时（{self.start_timeout:g} 秒）；详见 {self.diagnostics}')
            if isinstance(result[0], Exception):
                raise ListenerError(f'独立 WASAPI 音频进程异常退出：{result[0]}；详见 {self.diagnostics}')
            tag, payload = result[0]
            if tag == b'E':
                raise ListenerError(self._worker_error(payload))
            if tag != b'R':
                raise ListenerError('独立音频进程启动协议异常；详见诊断日志')
            try:
                info = json.loads(payload)
                self.device = str(info.get('device', '系统默认扬声器'))[:200]
                self.capture_rate = int(info.get('capture_rate',48000))
            except (ValueError, TypeError, AttributeError):
                raise ListenerError('独立音频进程设备信息损坏')
            if self.stream_onsets:
                self._frames = queue.Queue(maxsize=8)
                self._stop_reader.clear()
                self._reader = threading.Thread(target=self._reader_loop,
                                                name='neko-loopback-reader', daemon=True)
                self._reader.start()
            return self
        except BaseException:
            self.cancel()
            raise

    def _worker_error(self, payload: bytes) -> str:
        try:
            detail = json.loads(payload)
            if isinstance(detail, dict):
                phase = str(detail.get('stage', '未知阶段'))[:60]
                cause = str(detail.get('error', '未提供详细原因'))[:350]
                return f'Windows 回环在「{phase}」阶段失败：{cause}；诊断日志：{self.diagnostics}'
        except (ValueError, UnicodeError):
            pass
        return f'隔离音频采集失败：{payload[:240]!r}；详见 {self.diagnostics}'

    def read(self):
        import numpy as np
        if self._frames is not None:
            return self._read_streamed(np)
        with self._lock:
            proc = self._proc
        if proc is None or proc.stdout is None:
            raise ListenerError('独立音频监听尚未启动或已停止')
        result: list = []
        done = threading.Event()
        def receive():
            try:
                first = self._next_data_packet(proc)
                if first[0] == b'H' and getattr(self,'capture_stereo',False):
                    # Both packets must arrive before the SAME bounded timeout.
                    result.append((first, self._next_data_packet(proc)))
                else:
                    result.append((first,))
            except Exception as exc:
                result.append(exc)
            finally:
                done.set()
        threading.Thread(target=receive, name='neko-loopback-packet', daemon=True).start()
        if not done.wait(FRAME_TIMEOUT_SECONDS):
            self.cancel()
            raise ListenerError(f'超过 {FRAME_TIMEOUT_SECONDS:g} 秒没有收到扬声器音频帧，已自动关闭卡住的采集进程；详见 {self.diagnostics}')
        if isinstance(result[0], Exception):
            raise ListenerError(f'回环工作进程退出或设备断开：{result[0]}；诊断日志：{self.diagnostics}')
        packets = result[0]
        tag, payload = packets[0]
        self.last_stereo = None
        if tag == b'H' and getattr(self,'capture_stereo',False):
            expected = self.capture_rate * WINDOW_SECONDS * 2 * 4
            if len(payload) != expected or len(packets) != 2:
                raise ListenerError('隔离分离音频帧大小不匹配')
            self.last_stereo = np.frombuffer(payload,dtype='<f4').reshape(-1,2).copy()
            tag,payload = packets[1]
        if tag == b'E':
            raise ListenerError(self._worker_error(payload))
        if tag != b'D' or len(payload) != SAMPLE_RATE * WINDOW_SECONDS * 4:
            raise ListenerError('隔离音频进程提供的帧长度不匹配（预期 2 秒 16kHz 单声道音频）')
        return np.frombuffer(payload, dtype='<f4').copy()

    def _read_streamed(self, np):
        """后台线程持续排空管道，这里只取已经攒好的 2 秒帧。"""
        try:
            item = self._frames.get(timeout=FRAME_TIMEOUT_SECONDS)
        except queue.Empty:
            self.cancel()
            raise ListenerError(f'超过 {FRAME_TIMEOUT_SECONDS:g} 秒没有收到扬声器音频帧，已自动关闭卡住的采集进程；详见 {self.diagnostics}')
        tag, stereo_payload, payload = item
        if tag == b'X':
            raise ListenerError(f'回环工作进程退出或设备断开：{stereo_payload}；诊断日志：{self.diagnostics}')
        if tag == b'E':
            raise ListenerError(self._worker_error(payload))
        self.last_stereo = None
        if stereo_payload is not None:
            expected = self.capture_rate * WINDOW_SECONDS * 2 * 4
            if len(stereo_payload) != expected:
                raise ListenerError('隔离分离音频帧大小不匹配')
            self.last_stereo = np.frombuffer(stereo_payload, dtype='<f4').reshape(-1, 2).copy()
        if len(payload) != SAMPLE_RATE * WINDOW_SECONDS * 4:
            raise ListenerError('隔离音频进程提供的帧长度不匹配（预期 2 秒 16kHz 单声道音频）')
        return np.frombuffer(payload, dtype='<f4').copy()

    def cancel(self):
        """Promptly release a blocking read when the user revokes consent."""
        self._stop_reader.set()
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        proc.kill()
                        proc.wait(timeout=2)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
            if proc.stdout is not None:
                proc.stdout.close()

    def __exit__(self, exc_type, exc, tb):
        self.cancel()
        return False


class IsolatedApplicationLoopback(IsolatedWindowsLoopback):
    """Only a verified Windows application process tree (never system fallback)."""

    def __init__(self, *, app_vendor, pid: int, created: str, **kwargs):
        super().__init__(**kwargs)
        self.app_vendor = Path(app_vendor).resolve()
        self.pid = int(pid)
        self.created = str(created)

    def _validate_paths(self):
        super()._validate_paths()
        from .app_channels import verify_selected_process
        if self.script.name != 'application_loopback_worker.py':
            raise ListenerError('应用采集必须使用专用 WASAPI 进程回环，禁止系统回环替代')
        if not self.app_vendor.is_dir():
            raise ListenerError('逐应用采集组件未安装。请在「音源中控」中确认安装')
        if not verify_selected_process(self.pid, self.created):
            raise ListenerError('已选择的应用关闭或进程身份改变；请重新扫描选择')

    def _launch_args(self):
        return ([str(self.python), '-I', '-S', str(self.script), str(self.vendor),
                 str(self.app_vendor), str(self.pid), self.created]
                + (['stereo'] if self.capture_stereo else []))
