"""Safe bridge between the plugin's music-only PCM and optional ASR worker."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable


class LyricsWorkerError(RuntimeError):
    pass


class LyricsModelDownloadError(LyricsWorkerError):
    """Catching this permits bounded retries without restarting the audio listener."""
    pass


def recognize_pcm_isolated(samples, *, python_exe, vendor, worker_script, model_dir,
                           model_name='base', on_process: Callable | None=None,
                           timeout=180) -> dict:
    import numpy as np
    samples = np.asarray(samples, dtype='<f4').reshape(-1)
    if not 16000 <= samples.size <= 90 * 16000 or not np.isfinite(samples).all():
        raise LyricsWorkerError('invalid PCM clip')
    if Path(python_exe).name.lower().startswith(('projectneko', 'neko_server')):
        raise LyricsWorkerError('Never run the host executable as Python')
    if not Path(vendor).is_dir() or not Path(worker_script).is_file():
        raise LyricsWorkerError('independent Python environment missing')
    env = os.environ.copy()
    for var in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','PYTHONEXECUTABLE','VIRTUAL_ENV','CONDA_PREFIX'):
        env.pop(var,None)
    flags = getattr(subprocess,'CREATE_NO_WINDOW',0) if sys.platform=='win32' else 0
    child = None
    try:
        child = subprocess.Popen([str(python_exe), '-I','-S',str(worker_script),
                                  str(vendor),str(model_dir),str(model_name),'mono16k-v1'],
                                 stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,env=env,creationflags=flags)
        if on_process: on_process(child)
        stdout,stderr = child.communicate(input=samples.tobytes(),timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if child is not None:
            child.kill()
            child.communicate()
        raise LyricsWorkerError('歌词模型推理或首次模型下载超时，已停止本次识别') from exc
    except OSError as exc:
        raise LyricsWorkerError('Cannot launch independent lyrics worker') from exc
    finally:
        if on_process: on_process(None)
    if len(stdout)>200000:
        raise LyricsWorkerError('lyrics payload too large')
    try:
        record = json.loads(stdout.decode('utf-8'))
    except (ValueError,UnicodeError) as exc:
        raise LyricsWorkerError('lyrics worker returned malformed UTF-8 JSON') from exc
    if child.returncode or record.get('error'):
        msg = str(record.get('error') or stderr.decode('utf-8',errors='replace'))[:300]
        raise LyricsWorkerError('歌词识别不可用：'+msg)
    if record.get('version')!=1 or not isinstance(record.get('lyrics'),dict):
        raise LyricsWorkerError('unrecognized lyrics protocol')
    return record['lyrics']


class PersistentLyricsSession:
    """One isolated model, one outstanding clip, bounded CPU use.

    A reusable ASR worker avoids loading Whisper weights from disk for every
    10/20-second music interval. It is started only after music is confirmed,
    never at N.E.K.O plugin startup, and stopped when listening is disabled.
    """
    MAX_REPLY_BYTES = 1_048_576

    def __init__(self, *, python_exe, vendor, worker_script, model_dir,
                 model_name='base', device='cpu', cpu_threads=2, timeout=180,
                 cold_start_timeout=None, allow_public_mirror=True,
                 on_process: Callable | None = None, diagnostics_file=None,
                 use_site_packages=False,placement='standard',
                 gpu_budget_mib=None,cpu_budget_mib=None):
        import queue
        import threading
        self.python_exe = Path(python_exe)
        self.vendor = Path(vendor)
        self.worker_script = Path(worker_script)
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.device = device
        self.cpu_threads = cpu_threads
        self.timeout = timeout
        self.cold_start_timeout = cold_start_timeout or timeout
        self.allow_public_mirror = bool(allow_public_mirror)
        self.use_site_packages = bool(use_site_packages)
        if placement not in {'standard','gpu_ram_weights'}:
            raise LyricsWorkerError('未知模型权重放置策略')
        if placement=='gpu_ram_weights' and (device!='cuda' or not model_name.startswith('sing-qwen-')):
            raise LyricsWorkerError('权重分层仅适用于 Qwen 歌声 CUDA')
        self.placement=placement
        self.gpu_budget_mib=int(gpu_budget_mib or 0)
        self.cpu_budget_mib=int(cpu_budget_mib or 0)
        if placement=='gpu_ram_weights' and not (1536<=self.gpu_budget_mib<=32768 and
                                                   4096<=self.cpu_budget_mib<=131072):
            raise LyricsWorkerError('分层显存或内存预算无效')
        self._model_ready = False
        self.diagnostics_file = Path(diagnostics_file) if diagnostics_file else None
        # Separate local-only process log for native CUDA/OOM/WinDLL crashes.
        # It contains no PCM by design, is never sent to the AI, and rotates
        # at every worker start to avoid accumulating environment diagnostics.
        self._stderr_file = (self.diagnostics_file.with_name(
            self.diagnostics_file.stem + '_stderr.log')
            if self.diagnostics_file is not None else None)
        self.on_process = on_process
        self._queue = queue.Queue(maxsize=3)
        self._call_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._child = None
        self._next_id = 0
        self._closed = False
        if (self.python_exe.name.lower().startswith(('projectneko', 'neko_server'))
                or not self.vendor.is_dir() or not self.worker_script.is_file()):
            raise LyricsWorkerError('独立歌词识别环境缺失或不允许使用宿主 EXE')
        if model_name not in ('tiny', 'base', 'small', 'large-v3-turbo',
                'sing-whisper-small','sing-qwen-0.6b','sing-whisper-large-v3','sing-qwen-1.7b') or device not in ('cpu', 'cuda', 'auto'):
            raise LyricsWorkerError('unsupported lightweight ASR policy')
        if type(cpu_threads) is not int or not 1 <= cpu_threads <= 4:
            raise LyricsWorkerError('invalid CPU thread limit')

    def _start(self):
        import threading
        env = os.environ.copy()
        for key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','PYTHONEXECUTABLE','VIRTUAL_ENV','CONDA_PREFIX'):
            env.pop(key, None)
        # Keep the ASR child from monopolizing host cores. GPU remains opt-in.
        env.update(OMP_NUM_THREADS=str(self.cpu_threads),
                   MKL_NUM_THREADS=str(self.cpu_threads),
                   OPENBLAS_NUM_THREADS='1')
        if self.diagnostics_file is not None:
            env['NEKO_LYRICS_WORKER_STATUS'] = str(self.diagnostics_file)
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if sys.platform == 'win32' else 0
        log_handle = None
        try:
            if self._stderr_file is not None:
                self._stderr_file.parent.mkdir(parents=True, exist_ok=True)
                log_handle = self._stderr_file.open('wb')
                if os.name != 'nt':
                    os.chmod(self._stderr_file, 0o600)
            child = subprocess.Popen(
                ([str(self.python_exe), '-I'] + ([] if self.use_site_packages else ['-S']) +
                 [str(self.worker_script), str(self.vendor), str(self.model_dir), self.model_name,
                 'mono16k-v1', 'stream-v2']),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                # A private disk file never fills a pipe and preserves native
                # DLL/GPU exceptions that cannot reach our JSON protocol.
                stderr=log_handle if log_handle is not None else subprocess.DEVNULL,
                env=env, creationflags=flags)
        except OSError as exc:
            raise LyricsWorkerError('无法启动独立歌词识别进程') from exc
        finally:
            if log_handle is not None:
                log_handle.close()  # the child inherited its own handle
        self._child = child
        try:
            if self.on_process:
                self.on_process(child)
        except BaseException:
            child.terminate()
            self._child = None
            raise

        def read_lines():
            # Some ASR backends write initialization messages to stdout.
            # Only JSON records are protocol frames; tolerate bounded noise,
            # never send it back to the UI or interpret it as song lyrics.
            noisy=0
            try:
                while True:
                    line=child.stdout.readline(self.MAX_REPLY_BYTES+1)
                    if not line:
                        try:self._queue.put(('exited',child.poll()),timeout=1)
                        except queue.Full:pass
                        break
                    if len(line)>self.MAX_REPLY_BYTES or not line.endswith(b'\n'):
                        try:self._queue.put(('reply_too_large',len(line)),timeout=1)
                        except queue.Full:pass
                        break
                    if not line.lstrip().startswith(b'{'):
                        noisy+=1
                        if noisy<=32:continue
                        try:self._queue.put(('stdout_contamination',noisy),timeout=1)
                        except queue.Full:pass
                        break
                    self._queue.put(line,timeout=2)
            except (OSError,ValueError) as exc:
                try:self._queue.put(('read_error',type(exc).__name__),timeout=1)
                except queue.Full:pass
            except queue.Full:
                pass
        # Each session owns exactly one reader: do not enqueue a stray None
        # which could hide the actual exit reason from the next request.
        threading.Thread(target=read_lines, name='neko-lyrics-json-reader',daemon=True).start()

    def _private_error_hint(self) -> str:
        """Return only a CATEGORY, never log content, paths or speech."""
        if self._stderr_file is None:
            return ''
        try:
            with self._stderr_file.open('rb') as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 16384))
                excerpt = f.read().decode('utf-8', errors='replace').lower()
        except (OSError, ValueError):
            return ''
        for markers, reason in (
            (('cuda out of memory', 'torch.cuda.outofmemoryerror', 'cudnn_status_alloc_failed'), 'CUDA 显存不足'),
            (('cudnn', 'cublas', 'cuda runtime'), 'CUDA/cuDNN 运行依赖或初始化异常'),
            (('dll load failed', 'winerror 126', 'winerror 127'), 'Windows 原生 DLL 加载失败'),
            (('no module named', 'modulenotfounderror'), '独立 Python 环境缺少模块'),
            (('memoryerror', 'cannot allocate memory'), '系统内存不足'),
        ):
            if any(marker in excerpt for marker in markers):
                return reason
        return '子进程异常；本地已保留诊断日志' if excerpt.strip() else ''

    def _exchange(self, *, payload: bytes, sample_count: int, kind: str,
                  language: str = 'auto') -> dict:
        import queue
        with self._call_lock:
            if self._closed:
                raise LyricsWorkerError('歌词识别进程已经关闭')
            if self._child is None:
                self._start()
            if self._child.poll() is not None:
                self.close()
                raise LyricsWorkerError('歌词识别进程意外退出')
            self._next_id += 1
            job = self._next_id
            header = json.dumps({'version': 2, 'id': job, 'kind': kind,
                                 'samples': sample_count, 'device': self.device,
                                 'cpu_threads': self.cpu_threads,
                                 'language': language,
                                 'allow_public_mirror': self.allow_public_mirror,
                                 'placement':self.placement,
                                 'gpu_budget_mib':self.gpu_budget_mib,
                                 'cpu_budget_mib':self.cpu_budget_mib},
                                separators=(',', ':')).encode('ascii') + b'\n'
            try:
                self._child.stdin.write(header)
                if payload:
                    self._child.stdin.write(payload)
                self._child.stdin.flush()
                wait = self.timeout if self._model_ready else self.cold_start_timeout
                line = self._queue.get(timeout=wait)
            except queue.Empty as exc:
                try:
                    from .lyrics_diagnostics import read_stage
                    latest = read_stage(self.diagnostics_file) if self.diagnostics_file else None
                except Exception:
                    latest = None
                at = ('；最后阶段：' + latest['stage']) if latest else ''
                self.close()
                raise LyricsWorkerError('歌词识别超时' + at + '；已停止占用资源的模型进程') from exc
            except (OSError, BrokenPipeError, ValueError) as exc:
                self.close()
                raise LyricsWorkerError('歌词识别子进程已关闭或通信失败') from exc
            if not isinstance(line,(bytes,bytearray)):
                issue=(line[0] if isinstance(line,tuple) and line else 'exited')
                exit_code=child_code=None
                if self._child is not None:
                    # stdout EOF can arrive before the process fully exits;
                    # wait briefly so the real return code is reported.
                    try:
                        exit_code=self._child.wait(timeout=2)
                    except Exception:
                        exit_code=self._child.poll()
                try:
                    from .lyrics_diagnostics import read_stage
                    recent=read_stage(self.diagnostics_file) if self.diagnostics_file else None
                    stage=str(recent.get('stage','unknown'))[:70] if recent else 'unknown'
                except Exception:
                    stage='unknown'
                diagnostic=self._private_error_hint()
                self.close()
                extra=('；疑似原因：'+diagnostic) if diagnostic else ''
                if issue=='reply_too_large':
                    raise LyricsWorkerError('歌词识别响应超过 1MiB 上限；请检查模型输出或协议日志'+extra)
                if issue=='stdout_contamination':
                    raise LyricsWorkerError('模型依赖持续向数据通道打印内容；已中断本次识别，等待下一段恢复'+extra)
                raise LyricsWorkerError('歌词识别子进程退出/通信中断；最后阶段：'+stage+
                                        '；进程返回码：'+str(exit_code)+extra)
            try:
                response = json.loads(line.decode('utf-8'))
            except (UnicodeError, ValueError) as exc:
                self.close()
                raise LyricsWorkerError('歌词识别返回了无效 UTF-8 数据') from exc
            if response.get('version') != 2 or response.get('id') != job:
                self.close()
                raise LyricsWorkerError('歌词识别响应与本次歌曲片段不匹配')
            if response.get('error'):
                if response.get('code') == 'model_download_failed':
                    raise LyricsModelDownloadError(str(response['error'])[:420])
                raise LyricsWorkerError('歌词识别失败：' + str(response['error'])[:240])
            # A quiet clip can return success before loading Whisper. Do
            # not treat a live process or a no_singing_detected reply as a
            # model-verified GPU/CPU inference result.
            if kind == 'prepare' and response.get('ready') is True and response.get('device') in ('cpu','cuda'):
                self._model_ready = True
            elif kind == 'transcribe' and isinstance(response.get('lyrics'),dict) and response['lyrics'].get('device') in ('cpu','cuda'):
                self._model_ready = True
            return response

    def prepare(self) -> dict:
        """Readiness + real inference preflight; no captured audio required."""
        result = self._exchange(payload=b'', sample_count=0, kind='prepare')
        if result.get('ready') is not True or result.get('device') not in {'cpu','cuda'}:
            raise LyricsWorkerError('新歌词模型预热结果无效')
        return result

    def transcribe(self, samples, *, language: str = 'auto') -> dict:
        import numpy as np
        if language not in ('auto', 'zh', 'ja', 'en'):
            raise LyricsWorkerError('不支持的歌词识别语言设置')
        audio = np.asarray(samples, dtype='<f4').reshape(-1)
        if not 16000 <= audio.size <= 90*16000 or not np.isfinite(audio).all():
            raise LyricsWorkerError('歌词识别音频格式无效')
        response = self._exchange(payload=audio.tobytes(),
                                  sample_count=int(audio.size), kind='transcribe',
                                  language=language)
        if not isinstance(response.get('lyrics'), dict):
            raise LyricsWorkerError('歌词识别未返回有效候选')
        return response['lyrics']

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            child = self._child
            self._child = None
            if child is not None:
                if child.poll() is None:
                    try: child.terminate()
                    except OSError: pass
                    try: child.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=1)
                for pipe in (child.stdin, child.stdout):
                    try: pipe.close()
                    except (OSError, ValueError): pass
                if self.on_process:
                    try: self.on_process(None)
                    except Exception: pass

    @property
    def running(self) -> bool:
        return not self._closed and self._child is not None and self._child.poll() is None
