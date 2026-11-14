"""Optional, bounded source-separation session in managed CPython.

No model downloads, no host Python imports and no on-disk recording. If the
worker dies or times out, the caller keeps using original music input.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading

_HEADER = struct.Struct('!I')
_MAX_META = 4096
_MAX_PCM = 32 * 16000 * 3 * 4


class StemWorkerError(RuntimeError):
    pass


def _read_exact(pipe, count):
    data = bytearray()
    while len(data) < count:
        chunk = pipe.read(count - len(data))
        if not chunk:
            raise StemWorkerError('声部分离子进程通信中断')
        data.extend(chunk)
    return bytes(data)


class PersistentStemSession:
    def __init__(self, *, python, vendor, script, weights, gpu_vendor=None, on_process=None, timeout=30):
        self.python, self.vendor = Path(python), Path(vendor)
        self.script, self.weights = Path(script), Path(weights)
        self.gpu_vendor = Path(gpu_vendor).resolve() if gpu_vendor is not None else None
        self.on_process = on_process
        self.timeout = min(120, max(5, float(timeout)))
        self._child = None
        self._call_lock = threading.Lock()
        self._closed = False
        self._id = 0
        self._replies = queue.Queue(maxsize=2)
        python_name = self.python.name.lower()
        # uv 装出来的解释器叫 python3.11.exe；旧白名单只认 python.exe/python3，
        # 会把已验证可用的独立 CPython 误判成缺失，导致分离直接回退原声。
        if (not self.vendor.is_dir() or not self.script.is_file()
                or not re.fullmatch(r'python(?:\d+(?:\.\d+)?)?(?:\.exe)?', python_name)
                or 'projectneko' in str(self.python).lower()
                or 'neko_server' in str(self.python).lower()):
            raise StemWorkerError('分离环境或独立 Python 缺失')

    def _start(self):
        env = os.environ.copy()
        for key in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV','PYTHONSTARTUP','CONDA_PREFIX'):
            env.pop(key,None)
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        flags = getattr(subprocess,'CREATE_NO_WINDOW',0) if sys.platform == 'win32' else 0
        child = subprocess.Popen(
            [str(self.python),'-I','-S',str(self.script),str(self.vendor),str(self.weights),
             str(self.gpu_vendor) if self.gpu_vendor is not None else '-'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env, creationflags=flags, bufsize=0)
        self._child = child
        if self.on_process:
            self.on_process(child)
        def reader():
            try:
                while True:
                    length_data = child.stdout.read(4)
                    if not length_data: break
                    if len(length_data)!=4: raise StemWorkerError('incomplete response header')
                    meta_size, = _HEADER.unpack(length_data)
                    if not 1<=meta_size<=_MAX_META: raise StemWorkerError('invalid response size')
                    response = json.loads(_read_exact(child.stdout,meta_size).decode('utf-8'))
                    if response.get('ok'):
                        count = (int(response.get('voice_frames',0)) + int(response.get('enhanced_frames',0)) + int(response.get('instrument_frames',0)))*4
                        if not 0<count<=_MAX_PCM: raise StemWorkerError('invalid response PCM size')
                        payload = _read_exact(child.stdout,count)
                    else:
                        payload = b''
                    self._replies.put((response,payload),timeout=2)
            except Exception as exc:
                try:self._replies.put_nowait(exc)
                except queue.Full:pass
            finally:
                try:self._replies.put_nowait(None)
                except queue.Full:pass
        threading.Thread(target=reader,name='neko-stem-reader',daemon=True).start()

    def separate(self, stereo, *, rate=48000, gpu=False, cpu_fallback=True, strength='balanced'):
        """Return both stems or fail within a deadline, including pipe writes.

        Windows unbuffered anonymous pipes can accept only a *part* of a large
        ``write`` call. The worker may also be stuck during native library
        import, before it starts reading. Use a bounded sender and a cumulative
        deadline so neither failure can hang the lyrics worker indefinitely.
        """
        import numpy as np
        data = np.asarray(stereo,dtype='<f4')
        if data.ndim != 2 or data.shape[1]!=2 or rate not in (16000,44100,48000) or not rate<=len(data)<=rate*32:
            raise StemWorkerError('分离要求1至32秒的双声道音乐片段')
        if not np.isfinite(data).all(): raise StemWorkerError('invalid PCM')
        with self._call_lock:
            if self._closed: raise StemWorkerError('分离子进程已关闭')
            if self._child is None: self._start()
            if self._child.poll() is not None:
                self.close()
                raise StemWorkerError('分离子进程已退出')
            self._id += 1
            request_id = self._id
            if strength not in ('off','gentle','balanced','strong'):
                raise StemWorkerError('无效人声增强配置')
            info = json.dumps({'cmd':'split','id':request_id,'rate':rate,
                               'frames':len(data),'cuda':bool(gpu),
                               'cpu_fallback_allowed':bool(cpu_fallback),
                               'strength':strength},separators=(',',':')).encode()
            payload = data.tobytes()
            sent = threading.Event()
            error = []
            def writer():
                try:
                    def write_all(blob):
                        # No assumptions about partial writes on unbuffered pipes.
                        for start in range(0,len(blob), 65536):
                            piece=memoryview(blob)[start:start+65536]
                            while piece:
                                n=self._child.stdin.write(piece)
                                if n is None or n<=0:
                                    raise StemWorkerError('声部分离写入管道中断')
                                piece=piece[n:]
                    write_all(_HEADER.pack(len(info)))
                    write_all(info)
                    write_all(payload)
                    self._child.stdin.flush()
                except (OSError, ValueError, AttributeError, StemWorkerError) as exc:
                    error.append(exc)
                finally:
                    sent.set()
            started=time.monotonic()
            threading.Thread(target=writer,name='neko-stem-writer',daemon=True).start()
            if not sent.wait(self.timeout):
                self.close()
                raise StemWorkerError('声部分离发送超时，已终止子进程并回退原始声音')
            if error:
                self.close()
                raise StemWorkerError('声部分离传输失败，已回退原声') from error[0]
            elapsed=time.monotonic()-started
            try:
                response = self._replies.get(timeout=max(0.01, self.timeout-elapsed))
            except queue.Empty as exc:
                self.close()
                raise StemWorkerError('声部分离推理超时，已终止子进程并回退原声') from exc
            if response is None or isinstance(response,Exception):
                self.close()
                raise StemWorkerError('声部分离没有返回有效结果')
            meta,output_pcm = response
            if meta.get('id')!=request_id or meta.get('ok') is not True:
                raise StemWorkerError(str(meta.get('error') or '分离输出无效')[:220])
            count = int(meta['voice_frames'])
            count_enhanced = int(meta.get('enhanced_frames',0))
            offset = 4*count
            voice = np.frombuffer(output_pcm[:offset],dtype='<f4').copy()
            enhanced = (np.frombuffer(output_pcm[offset:offset+4*count_enhanced],dtype='<f4').copy()
                        if count_enhanced else voice.copy())
            other = np.frombuffer(output_pcm[offset+4*count_enhanced:],dtype='<f4').copy()
            expected = int(round(len(data)*16000/rate))
            if (len(other)!=int(meta['instrument_frames']) or
                    abs(len(voice)-expected)>1 or len(other)!=len(voice) or
                    len(enhanced)!=len(voice) or
                    not np.isfinite(voice).all() or not np.isfinite(enhanced).all() or
                    not np.isfinite(other).all()):
                raise StemWorkerError('分离返回采样长度或采样数值异常')
            return {'vocals':voice, 'enhanced_vocals':enhanced,
                    'instrumental':other, 'model':meta.get('model'),
                    'device':meta.get('device','unknown'), 'warning':meta.get('warning',''),
                    'gpu_fallback_reason':meta.get('gpu_fallback_reason')}

    def close(self):
        self._closed=True
        child,self._child=self._child,None
        if child is not None:
            if child.poll() is None:
                try: child.terminate(); child.wait(timeout=1)
                except (subprocess.TimeoutExpired,OSError):
                    try: child.kill();child.wait(timeout=1)
                    except OSError:pass
            for pipe in (child.stdin,child.stdout):
                try:pipe.close()
                except (OSError,ValueError,AttributeError):pass
            if self.on_process:
                try:self.on_process(None)
                except Exception:pass

    @property
    def running(self):
        return not self._closed and self._child is not None and self._child.poll() is None
