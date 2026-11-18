"""Isolated singing-ASR environment lifecycle for the N.E.K.O plugin.

The *only* disposable location is <plugin-data>/environment/singing_asr_venv.
Do not install into system Python or the host's shared CUDA/Python environment.
All commands use trusted, fixed package specifications without a shell.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SINGING_WHISPER_PACKAGES = ('faster-whisper>=1.1,<2','huggingface-hub>=0.31,<2','numpy>=1.24,<3')
SINGING_QWEN_PACKAGES = ('qwen-asr==0.0.6','huggingface-hub>=0.31,<2','numpy>=1.24,<3',
                         'accelerate>=1.0,<2')
PRIVATE_ENV_NAME = 'singing_asr_venv'
PRIVATE_LOG_NAME = 'singing_asr_install.log'
PYPI_URL = 'https://pypi.org/simple'
MIRROR_URL = 'https://pypi.tuna.tsinghua.edu.cn/simple'
# torch 2.11 的 CUDA/CPU 轮子约 1~3 GB。download.pytorch.org 国内常只有几百 KB/s；
# 南大镜像是官方索引的完整同步（同名同哈希），2026-11 实测 17 MB/s。
TORCH_VERSION = 'torch==2.11.0'
TORCH_INDEXES = {
    'cu128': ('https://mirror.nju.edu.cn/pytorch/whl/cu128', 'https://download.pytorch.org/whl/cu128'),
    'cpu': ('https://mirror.nju.edu.cn/pytorch/whl/cpu', 'https://download.pytorch.org/whl/cpu'),
}

class SingingInstallError(RuntimeError):
    pass

class SingingInstallCancelled(SingingInstallError):
    pass


def paths(env_root):
    venv=Path(env_root)/PRIVATE_ENV_NAME
    exe=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    vendor=venv/('Lib/site-packages' if os.name=='nt' else
                 f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')
    return venv,exe,vendor


def private_environment_status(env_root):
    venv, exe, vendor = paths(env_root)
    # Read-only and stat-only: the Hosted UI snapshot never imports torch/pip.
    return {'present':venv.is_dir() and not venv.is_symlink(),
            'python_present':exe.is_file(), 'packages_present':vendor.is_dir(),
            'install_log':str(Path(env_root)/PRIVATE_LOG_NAME),
            'models_preserved_on_uninstall':True}


def _diagnostic_log_tail(log_path, maximum=1200):
    if not log_path or not Path(log_path).is_file():
        return ''
    try:
        with Path(log_path).open('rb') as f:
            f.seek(0,2)
            f.seek(max(0,f.tell()-4096))
            raw=f.read().decode('utf-8',errors='replace').strip()
        # Pip can include index URLs and credentials inherited from the user's
        # machine. Only return a short *redacted* diagnostic to the UI.
        raw=re.sub(r'https?://[^\s<>\]\)]+', '[下载地址已隐藏]', raw)
        raw=re.sub(r'(?i)(token|password|secret|authorization)\s*[=:]\s*\S+', r'\1=[已隐藏]',raw)
        raw=re.sub(r'(?i)([A-Za-z]:\\Users\\|/home/)[^\\/\s]+', '[用户目录]', raw)
        return raw[-maximum:]
    except OSError:
        return ''


def _error_hint(output):
    text=output.lower()
    if 'typeerror:' in text and 'singing_prepare_worker' in text:
        return '专业歌声下载器存在代码异常（TypeError）；这是插件问题，不是你的 CUDA 或网络错误。'
    if 'modulenotfounderror:' in text or 'importerror:' in text:
        return '独立进程导入依赖失败；请查看具体模块名并检查此独立环境。'
    if 'singingmodelerror:' in text:
        return '模型下载或 SHA-256 完整性校验失败；可以重试，不会覆盖原有已验证权重。'
    if 'no module named pip' in text:
        return '独立 Python 缺少 pip，需要重新创建此独立环境。'
    if 'no matching distribution' in text or 'could not find a version' in text:
        return '依赖无兼容发行包；请检查独立 Python 版本、Windows 架构与下载源。'
    if 'certificate verify failed' in text or 'ssl' in text:
        return '下载源的 TLS/证书校验失败。'
    if 'readtimeout' in text or 'timed out' in text or 'proxyerror' in text or 'connection' in text:
        return '下载网络超时、代理或镜像连接异常；可切换来源后重试。'
    if 'no space left' in text or 'disk full' in text:
        return '磁盘可用空间不足。'
    if 'access is denied' in text or 'permission denied' in text or 'winerror 32' in text:
        return 'Windows 文件占用或目录权限不足，请先停止正在运行的专业歌声进程。'
    return '请打开插件私有环境安装日志查看 pip/venv 的实际错误。'


def _launch(argv, *, timeout, on_process=None, log_path=None, cancel_event=None):
    """Bounded subprocess, cancellation, and persistent failure evidence.

    A separate log is essential: discarding stderr created the original
    `SingingInstallError: return code 1` with no actionable cause.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise SingingInstallCancelled('已取消专业歌声安装；模型文件不会被删除')
    env=os.environ.copy()
    for variable in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','CONDA_PREFIX',
                     'PYTHONEXECUTABLE','PIP_INDEX_URL','PIP_EXTRA_INDEX_URL',
                     'PIP_TRUSTED_HOST','PIP_FIND_LINKS','PIP_REQUIRE_VIRTUALENV'):
        env.pop(variable,None)
    env['PIP_CONFIG_FILE']=os.devnull
    env['PIP_NO_INPUT']='1'
    env['PYTHONIOENCODING']='utf-8'
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0) if sys.platform=='win32' else 0
    log_file=Path(log_path) if log_path else None
    if log_file:
        log_file.parent.mkdir(parents=True,exist_ok=True)
    # Keep a temporary error sink if no persistent log was supplied.
    with (log_file.open('ab') if log_file else tempfile.TemporaryFile('w+b')) as out:
        out.write(('\n== 子进程命令：'+str(Path(str(argv[0])).name)+' / '+
                   ('pip 安装' if 'pip' in argv else '准备/校验')+' ==\n').encode('utf-8'))
        out.flush()
        try:
            p=subprocess.Popen([str(x) for x in argv],stdin=subprocess.DEVNULL,
                stdout=out,stderr=subprocess.STDOUT,env=env,creationflags=flags)
        except OSError as exc:
            raise SingingInstallError('无法启动独立环境命令：'+type(exc).__name__+'。请检查 Python 与文件权限') from exc
        try:
            if on_process:
                on_process(p)
            end=time.monotonic()+timeout
            while p.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    p.terminate()
                    try:
                        p.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait(timeout=3)
                    raise SingingInstallCancelled('已取消专业歌声安装；模型文件不会被删除')
                if time.monotonic()>=end:
                    p.terminate()
                    try:
                        p.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait(timeout=3)
                    raise SingingInstallError('独立环境安装超时；可查看插件私有安装日志')
                try:
                    p.wait(timeout=min(.5,max(.1,end-time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            if p.returncode:
                if cancel_event is not None and cancel_event.is_set():
                    raise SingingInstallCancelled('已取消专业歌声安装；不会继续修改私有环境')
                out.flush()
                if log_file:
                    tail=_diagnostic_log_tail(log_file,maximum=500)
                else:
                    out.seek(max(0,out.tell()-2048))
                    tail=out.read().decode('utf-8',errors='replace')[-500:]
                operation=('专业歌声模型下载/校验子进程' if any(
                    str(item).endswith('singing_prepare_worker.py') for item in argv)
                    else '独立 Python 环境或依赖安装命令')
                raise SingingInstallError(operation+'失败（返回码 '+str(p.returncode)+
                    '）：'+_error_hint(tail)+'；完整错误已写入 singing_asr_install.log。')
            return 0
        finally:
            if on_process:
                on_process(None)


def _install_torch(exe,flavor,source,*,log,cancel_event,on_process,force=False):
    """按来源偏好在南大镜像与 PyTorch 官方之间依次尝试安装固定版本 torch。"""
    indexes=TORCH_INDEXES[flavor]
    if source=='official':
        indexes=indexes[::-1]
    errors=[]
    for index in indexes:
        try:
            _launch([exe,'-I','-m','pip','install','--disable-pip-version-check',
                     '--timeout','30','--retries','3',
                     *(['--force-reinstall'] if force else []),
                     '--index-url',index,TORCH_VERSION],timeout=2400,log_path=log,
                     cancel_event=cancel_event,on_process=on_process)
            return
        except SingingInstallCancelled:
            raise
        except SingingInstallError as exc:
            errors.append(str(exc))
    raise SingingInstallError('PyTorch 两个下载源都失败；'+errors[-1])


def ensure_environment(python_exe,env_root,quality,*,device='cpu',source='official',
                       on_stage=None,on_process=None,cancel_event=None):
    from .singing_catalog import PROFILES
    if quality not in PROFILES or device not in {'cpu','cuda'} or source not in {'official','mirror'}:
        raise ValueError('Unsupported singing tier/device/source')
    venv,exe,vendor=paths(env_root)
    venv.parent.mkdir(parents=True,exist_ok=True)
    log=Path(env_root)/PRIVATE_LOG_NAME
    if not exe.is_file():
        if on_stage:
            on_stage('creating_private_venv')
        _launch([python_exe,'-I','-m','venv',str(venv)],timeout=180,
                log_path=log,cancel_event=cancel_event,on_process=on_process)
    required='qwen_asr' if PROFILES[quality]['engine']=='qwen_asr' else 'faster_whisper'
    if on_stage:
        on_stage('checking_dependencies')
    probe=[str(exe),'-I','-c',f'import {required}, huggingface_hub, numpy'+
           (', accelerate' if required=='qwen_asr' else '')+'; print("ok")']
    env=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','CONDA_PREFIX'):
        env.pop(k,None)
    try:
        result=subprocess.run(probe,env=env,capture_output=True,timeout=90)
    except (OSError,subprocess.TimeoutExpired) as exc:
        raise SingingInstallError('无法执行独立环境依赖检查：'+type(exc).__name__) from exc
    if result.returncode!=0:
        if on_stage:
            on_stage('installing_private_dependencies')
        if required=='qwen_asr' and device=='cuda' and os.name=='nt':
            if on_stage:
                on_stage('installing_private_torch_cuda128')
            _install_torch(exe,'cu128',source,log=log,cancel_event=cancel_event,on_process=on_process)
        if required=='qwen_asr' and device=='cpu' and os.name=='nt':
            # PyPI's default modern PyTorch wheel can carry CUDA runtimes even
            # for CPU-only use; install an explicit CPU build in this venv.
            if on_stage:
                on_stage('installing_private_torch_cpu')
            _install_torch(exe,'cpu',source,log=log,cancel_event=cancel_event,on_process=on_process)
        packages=(SINGING_QWEN_PACKAGES if required=='qwen_asr' else SINGING_WHISPER_PACKAGES)
        # Give the user's selected source a real meaning for Python packages.
        indexes=(MIRROR_URL,PYPI_URL) if source=='mirror' else (PYPI_URL,MIRROR_URL)
        attempts=[]
        for index in indexes:
            if cancel_event is not None and cancel_event.is_set():
                raise SingingInstallCancelled('用户已取消安装')
            try:
                _launch([exe,'-I','-m','pip','install','--disable-pip-version-check',
                    '--only-binary=:all:','--timeout','25','--retries','2',
                    '--index-url',index,*packages],timeout=1800,
                    log_path=log,cancel_event=cancel_event,on_process=on_process)
                break
            except SingingInstallCancelled:
                raise
            except SingingInstallError as exc:
                attempts.append(str(exc))
        else:
            raise SingingInstallError('已尝试官方与备用镜像；'+attempts[-1])
        check=subprocess.run(probe,env=env,capture_output=True,timeout=90)
        if check.returncode!=0:
            with log.open('ab') as f:
                f.write(b'\nIMPORT PROBE FAILED\n'+check.stderr[-2000:]+b'\n')
            raise SingingInstallError('独立环境依赖安装后仍无法导入：'+_error_hint(
                check.stderr.decode('utf-8',errors='replace'))+'；请检查私有安装日志')
    if required=='qwen_asr' and device=='cuda':
        if on_stage:
            on_stage('checking_private_cuda')
        cuda_probe=[str(exe),'-I','-c','import torch; assert torch.cuda.is_available(), "CUDA unavailable"']
        initial=subprocess.run(cuda_probe,env=env,capture_output=True,timeout=90)
        if initial.returncode!=0 and os.name=='nt':
            if on_stage:
                on_stage('repairing_private_torch_cuda128')
            _install_torch(exe,'cu128',source,log=log,cancel_event=cancel_event,
                           on_process=on_process,force=True)
        gpu_check=subprocess.run(cuda_probe,env=env,capture_output=True,timeout=90)
        if gpu_check.returncode!=0:
            with log.open('ab') as f:
                f.write(b'\nCUDA PROBE FAILED\n'+gpu_check.stderr[-2000:]+b'\n')
            raise SingingInstallError('专业歌声 CUDA 在独立 Python 环境未通过试运行；请检查驱动和安装日志')
    if not vendor.is_dir():
        candidates=list((venv/'lib').glob('python*/site-packages')) if (venv/'lib').is_dir() else []
        if candidates:
            vendor=candidates[0]
    if not vendor.is_dir():
        raise SingingInstallError('无法定位独立环境 site-packages')
    return exe,vendor


def uninstall_environment(env_root, *, retries=4, wait=.3):
    """Delete *only* the private singing venv; never delete weights or CUDA.

    Callers must stop our worker processes and wait until installation ends.
    """
    root=Path(env_root)
    venv,_,_=paths(root)
    if venv.name!=PRIVATE_ENV_NAME or root.is_symlink() or venv.is_symlink() or \
            venv.parent.resolve()!=root.resolve():
        raise SingingInstallError('独立环境目录不安全，拒绝卸载')
    if not venv.exists():
        return {'ok':True,'state':'absent','note':'独立环境原本不存在，无需删除'}
    last=None
    for attempt in range(retries):
        try:
            shutil.rmtree(venv)
            if venv.exists():
                raise OSError('卸载后独立环境仍存在')
            return {'ok':True,'state':'removed',
                    'note':'仅专业歌声 Python 环境已卸载；模型权重和私有 CUDA 均保留'}
        except OSError as exc:
            last=exc
            if attempt+1<retries:
                time.sleep(wait*(attempt+1))
    return {'ok':False,'state':'failed',
            'error':'独立环境卸载未完成（Windows 文件可能仍被占用）：'+type(last).__name__+
                     '。请停止插件并重试，未删除其他模型。'}
