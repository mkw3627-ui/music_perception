"""Explicitly consented, self-contained Windows per-process capture dependency.

Requires existing isolated CPython/audio vendor. Pinned Windows binary wheel
only; never edits host, system Python or a vendor with loaded native DLLs.
Download failed? Keep old working environment and never capture mixed audio.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import shutil
import subprocess
import threading
import time
import uuid

PACKAGE = 'proc-tap==1.1.1'
PINNED_VERSION = '1.1.1'
# PyPI 2026-07-22 Windows amd64 native wheels. All hashes are for the same
# release; uv/pip still selects a wheel compatible with the active CPython.
# An unverified cross-platform py3-none-any fallback cannot satisfy any hash.
WIN_NATIVE_WHEEL_SHA256 = {
    'cp311':'7ba5716a0f0be1ab999365c06e9adf4ce78d275a39498708689c965e0fda8396',
    'cp312':'84b69a124859dfdf0a6b00cdb194dfc3ee5ca962a39e873475204253ba10f420',
    'cp313':'330e03d1c9dc2e6e8e785521cba45404dc963a52cb8f197136011b8b54e16542',
}

PROBE = r'''
import sys
from pathlib import Path
import importlib.metadata as metadata
base, addon = map(str,sys.argv[1:3])
sys.path[:0] = [addon, base]
d = {x.metadata.get('Name','').lower().replace('_','-'):x.version for x in metadata.distributions(path=[addon])}
if d.get('proc-tap') != '1.1.1': raise RuntimeError('proc_tap_version_invalid')
from proctap import ProcessAudioCapture
assert callable(ProcessAudioCapture)
# Probe the documented public API, never an internal undocumented class.
# Require a Windows native .pyd instead of accepting a cross-platform pure-Python wheel.
assert any((Path(addon) / 'proctap').rglob('*.pyd')), 'windows_native_extension_missing'
print('PER_PROCESS_WASAPI_IMPORT_OK')
'''


class ProcessCaptureInstaller:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self._lock = threading.RLock()
        self._thread = None
        self._state = {'state':'not_installed','running':False,'ready':False,'message':''}

    def _pointer_path(self):
        return self.root / 'process_capture_active.json'

    def _active_dir(self):
        try:
            item = json.loads(self._pointer_path().read_text(encoding='utf-8'))
            directory = Path(item.get('dir','')).resolve()
            if (item.get('version') == PINNED_VERSION and
                    directory.is_relative_to(self.root) and
                    directory.is_dir() and (directory/'.neko-process-capture-ready').is_file()
                    and any((directory/'proctap').rglob('*.pyd'))):
                return directory
        except (OSError,ValueError,TypeError):pass
        return None

    def installed(self):
        return self._active_dir() is not None

    def directory(self):
        return self._active_dir()

    def status(self):
        with self._lock:
            info = dict(self._state)
        info['ready'] = self.installed()
        return info

    def start(self, *, confirmed: bool, env, source='mirror'):
        if confirmed is not True:
            return {'ok':False, 'error':'用户未确认，不安装逐应用音频采集组件'}
        if os.name != 'nt' or sys.platform!='win32':
            return {'ok':False,'error':'此采集组件仅用于 Windows 10 Build 20348+ / Windows 11'}
        try:
            if sys.getwindowsversion().build < 20348:
                return {'ok':False,'error':'当前 Windows 构建版本不支持逐进程回环（最低 Build 20348）'}
        except AttributeError:
            return {'ok':False,'error':'无法确认 Windows 构建版本；已拒绝安装'}
        if env is None or not env.snapshot().get('ready'):
            return {'ok':False,'error':'请先准备插件的独立 Python 音频环境'}
        if self.installed():
            return {'ok':True,'ready':True,'note':'逐应用采集组件已安装并持久复用，不重复下载'}
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {'ok':True,'running':True,'note':'安装正在进行，不会重复启动'}
            py, vendor = env.worker_runtime()
            self._state = {'state':'installing','running':True,'ready':False,
                           'message':'正在下载 Windows 逐进程采集组件（固定版本，二进制轮子）'}
            self._thread = threading.Thread(target=self._install,
                args=(str(py),str(vendor),source),daemon=True, name='neko-process-capture-install')
            self._thread.start()
        return {'ok':True,'running':True,'note':'已开始在插件独立目录安装；完成前不能开启逐应用采集'}

    def _install(self, python: str, base_vendor: str, source: str):
        stage = self.root / ('.app-audio-stage-'+uuid.uuid4().hex[:12])
        committed = False
        try:
            self.root.mkdir(parents=True,exist_ok=True)
            source_urls = (['https://pypi.tuna.tsinghua.edu.cn/simple','https://pypi.org/simple']
                if source == 'mirror' else
                ['https://pypi.org/simple','https://pypi.tuna.tsinghua.edu.cn/simple'])
            uv = shutil.which('uv')
            failures = []
            requirements = self.root / 'proc_tap_locked_requirements.txt'
            requirements.write_text(
                PACKAGE+' '+ ' '.join('--hash=sha256:'+sha for sha in WIN_NATIVE_WHEEL_SHA256.values())+'\n',
                encoding='ascii')
            for index_url in source_urls:
                if stage.exists():
                    shutil.rmtree(stage,ignore_errors=True)
                stage.mkdir(parents=True)
                if uv:
                    argv=[uv,'pip','install','--python',python,'--target',str(stage),
                          '--only-binary',':all:','--no-deps','--link-mode','copy',
                          '--require-hashes','-r',str(requirements),
                          '--index-url',index_url]
                else:
                    argv=[python,'-m','pip','--isolated','install','--target',str(stage),
                          '--only-binary=:all:','--no-deps','--require-hashes',
                          '--disable-pip-version-check', '--no-input',
                          '-r',str(requirements),'--index-url',index_url]
                try:
                    done=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,
                                        timeout=240,check=False)
                    if done.returncode:
                        failures.append(f'{index_url}:installer_exit_{done.returncode}')
                        continue
                    run=subprocess.run([python,'-I','-S','-c',PROBE,base_vendor,str(stage)],
                                       stdin=subprocess.DEVNULL,capture_output=True,timeout=25,check=False)
                    if run.returncode or b'PER_PROCESS_WASAPI_IMPORT_OK' not in run.stdout:
                        failures.append(f'{index_url}:native_import_probe_failed')
                        continue
                    # Never rename or overwrite a directory potentially loaded
                    # by a running native capture. Commit immutable generation.
                    (stage/'.neko-process-capture-ready').write_text(PINNED_VERSION,encoding='ascii')
                    dest=self.root/('proc-tap-1.1.1-'+uuid.uuid4().hex[:10])
                    try:
                        stage.rename(dest)
                    except PermissionError:
                        # Windows may lock newly-imported DLL files for a short
                        # time. Reuse the verified immutable staging directory
                        # rather than deleting a good download or touching an
                        # active native DLL directory (earlier WinError 5).
                        dest = stage
                    pointer=self._pointer_path()
                    temp=pointer.with_suffix('.tmp')
                    temp.write_text(json.dumps({'dir':str(dest),'version':PINNED_VERSION}),encoding='utf-8')
                    os.replace(temp,pointer)
                    committed = True
                    with self._lock:
                        self._state={'state':'ready','running':False,'ready':True,
                                     'message':'独立采集组件安装成功；请选择应用并开启监听'}
                    return
                except (OSError,subprocess.TimeoutExpired) as exc:
                    failures.append(index_url+':'+type(exc).__name__)
            raise RuntimeError(' | '.join(failures) or 'no_available_package_index')
        except Exception as exc:
            with self._lock:
                self._state={'state':'failed','running':False,'ready':self.installed(),
                             'message':f'{type(exc).__name__}: {str(exc)[:320]}'}
        finally:
            if not committed and stage.exists(): shutil.rmtree(stage,ignore_errors=True)
            try: (self.root / 'proc_tap_locked_requirements.txt').unlink(missing_ok=True)
            except OSError: pass
