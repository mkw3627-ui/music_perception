"""v0.9.5: CUDA DLLs belong to plugin data and are never mistaken for GPU proof."""
import hashlib
import json
from pathlib import Path
import os
import sys
import zipfile

import pytest
from music_perception import stem_cuda_bundle as cb
from music_perception import stem_gpu_runtime as gpu
from _version import ui_build, panel_ui_build, plugin_version  # noqa: E402


def build_fake_bundle(tmp_path):
    folder = cb._folder(tmp_path)
    for spec in cb.CUDA_WHEELS:
        directory = folder/'nvidia'/spec['family']/'bin'
        directory.mkdir(parents=True, exist_ok=True)
        for dll in spec['dlls']:
            (directory/dll).write_bytes(b'\x01'*2048)
    (folder/cb.CUDA_MARKER).write_text(json.dumps({
        'version':cb.CUDA_BUNDLE_VERSION,
        'wheels':{cb.wheel_name(w):w['sha256'] for w in cb.CUDA_WHEELS}}))
    return folder


def test_bundle_does_not_appear_ready_with_ort_only(tmp_path):
    assert cb.cuda_runtime_path(tmp_path) is None
    assert gpu.inventory(tmp_path)['installed'] is False
    assert 'private CUDA' in gpu.inventory(tmp_path)['note'] or 'CUDA/cuDNN' in gpu.inventory(tmp_path)['note']


def test_bundle_requires_pinned_sha_marker_not_just_dlls(tmp_path):
    folder=build_fake_bundle(tmp_path)
    assert cb.cuda_runtime_path(tmp_path) == folder
    record=json.loads((folder/cb.CUDA_MARKER).read_text())
    record['wheels'][cb.wheel_name(cb.CUDA_WHEELS[0])] = 'tampered'
    (folder/cb.CUDA_MARKER).write_text(json.dumps(record))
    assert cb.cuda_runtime_path(tmp_path) is None


def test_bundle_missing_dll_not_ready(tmp_path):
    folder=build_fake_bundle(tmp_path)
    (folder/'nvidia'/'cudnn'/'bin'/'cudnn64_9.dll').unlink()
    assert cb.cuda_runtime_path(tmp_path) is None
    assert any('cudnn64_9.dll' in x for x in cb.missing_private_cuda_files(folder))


def test_cuda_wheels_have_pinned_sha_and_fixed_win64_tag():
    assert len(cb.CUDA_WHEELS) >= 6
    assert len({spec['package'] for spec in cb.CUDA_WHEELS})==len(cb.CUDA_WHEELS)
    for spec in cb.CUDA_WHEELS:
        assert len(spec['sha256'])==64 and int(spec['sha256'],16)>0
        assert cb.wheel_name(spec).endswith('py3-none-win_amd64.whl')
    assert cb.CUDA_ESTIMATED_DOWNLOAD_BYTES > 1_500_000_000


def test_mirror_requires_https_and_pinned_package_name():
    spec=cb.CUDA_WHEELS[0]
    assert cb._sources(spec['package'],spec['version'],'official')[0][0]=='PyPI 官方'
    assert cb._sources(spec['package'],spec['version'],'mirror')[0][0]=='清华镜像'
    with pytest.raises(cb.PrivateCudaError):cb._sources('arbitrary','1','evil')


def test_cuda_wheel_rejects_zip_slip(tmp_path):
    archive=tmp_path/'malicious.whl'
    with zipfile.ZipFile(archive,'w') as zipped:
        zipped.writestr('../inject.dll','malware')
    dest=tmp_path/'staging';dest.mkdir()
    with pytest.raises(cb.PrivateCudaError,match='Unsafe'):
        cb._safe_extract_wheel(archive,dest,cb.CUDA_WHEELS[0])
    assert not (tmp_path/'inject.dll').exists()


def test_cuda_wheel_rejects_host_python_package_injection(tmp_path):
    archive=tmp_path/'wrong.whl'
    with zipfile.ZipFile(archive,'w') as zipped:
        zipped.writestr('sitecustomize.py','bad')
    dest=tmp_path/'stage';dest.mkdir()
    with pytest.raises(cb.PrivateCudaError,match='unexpected filesystem'):
        cb._safe_extract_wheel(archive,dest,cb.CUDA_WHEELS[0])


def test_cuda_wheel_can_extract_nvidia_bin_and_distinfo(tmp_path):
    archive=tmp_path/'ok.whl'
    spec=cb.CUDA_WHEELS[0]
    with zipfile.ZipFile(archive,'w') as zipped:
        zipped.writestr('nvidia/cuda_runtime/bin/cudart64_12.dll','testdata')
        zipped.writestr(f"{spec['package'].replace('-','_')}-{spec['version']}.dist-info/METADATA",'Name: test')
    dest=tmp_path/'stage';dest.mkdir()
    cb._safe_extract_wheel(archive,dest,spec)
    assert (dest/'nvidia/cuda_runtime/bin/cudart64_12.dll').is_file()


def test_windows_loader_path_changes_are_child_only(monkeypatch,tmp_path):
    folder=build_fake_bundle(tmp_path)
    handles=[]
    monkeypatch.setattr(cb.sys,'platform','win32')
    monkeypatch.setattr(cb.os,'add_dll_directory',lambda folder:handles.append(folder) or object(),raising=False)
    old=os.environ.get('PATH','')
    try:
        paths=cb.configure_private_cuda(folder)
        assert len(paths)==len(cb.CUDA_WHEELS)
        assert len(handles)==len(cb.CUDA_WHEELS)
        assert str(folder) in os.environ['PATH']
        assert cb._DLL_HANDLES
    finally:
        os.environ['PATH']=old
        cb._DLL_HANDLES.clear()


def test_loader_refuses_missing_cudnn_even_when_selected_gpu(monkeypatch,tmp_path):
    folder=build_fake_bundle(tmp_path)
    (folder/'nvidia/cudnn/bin/cudnn64_9.dll').unlink()
    monkeypatch.setattr(cb.sys,'platform','win32')
    with pytest.raises(cb.PrivateCudaError,match='不完整'):
        cb.configure_private_cuda(folder)


def test_installer_refuses_linux_and_never_modifies_vendor(monkeypatch,tmp_path):
    vendor=tmp_path/'environment'/'vendor'
    vendor.mkdir(parents=True)
    sentinel=vendor/'numpy_is_not_touched'
    sentinel.write_text('original')
    with pytest.raises(cb.PrivateCudaError,match='Windows'):
        cb.install_cuda_runtime(vendor.parent,vendor.parent/'status.json')
    assert sentinel.read_text()=='original'


def test_only_real_probe_may_say_gpu_verified():
    root=Path(__file__).parents[1]
    source=(root/'stem_gpu_probe_worker.py').read_text('utf8')
    assert 'evidence[\'cuda_nodes\'] > 0' in source
    assert "'real_inference_not_run'" in source
    assert "'private_cuda_not_installed'" in source
    assert "configure_private_cuda(private_cuda)" in source


def test_private_path_in_worker_before_native_ort_import():
    root=Path(__file__).parents[1]
    text=(root/'stem_worker.py').read_text('utf8')
    assert text.index('configure_private_cuda(cuda_private)') < text.index('model=TwoStemMDX(weight,request_gpu=wants_gpu)')
    assert 'gpu_runtime_ready' in text


def test_package_versions_and_ui_confirm_download_size():
    root=Path(__file__).parents[1]
    assert f'version = "{plugin_version()}"' in (root/'plugin.toml').read_text('utf8')
    ui=(root/'ui/panel.tsx').read_text('utf8')
    assert ui_build() in ui
    assert '1.8 GB' in ui and '确认安装' in ui
    assert '6 GB' in ui


def _small_verified_wheels(tmp_path, monkeypatch):
    """Make a miniature pinned wheel index to test full offline atomic install."""
    specs=[]
    wheel_files={}
    for spec in cb.CUDA_WHEELS:
        record=dict(spec)
        record['approximate_bytes']=500
        wheel=tmp_path / cb.wheel_name(record)
        with zipfile.ZipFile(wheel,'w') as z:
            for dll in record['dlls']:
                z.writestr('nvidia/'+record['family']+'/bin/'+dll,b'X'*2048)
            z.writestr(record['package'].replace('-','_')+'-'+record['version']+'.dist-info/METADATA','Name: '+record['package'])
        record['sha256']=hashlib.sha256(wheel.read_bytes()).hexdigest()
        wheel_files[record['package']]=wheel
        specs.append(record)
    monkeypatch.setattr(cb,'CUDA_WHEELS',tuple(specs))
    monkeypatch.setattr(cb,'CUDA_ESTIMATED_DOWNLOAD_BYTES',3000)
    monkeypatch.setattr(cb,'_check_target',lambda:None)
    monkeypatch.setattr(cb,'_fetch_url',lambda _index,spec:'https://example.test/'+spec['package'])
    return wheel_files


def test_cuda_full_offline_installer_atomic_and_idempotent(monkeypatch,tmp_path):
    wheels=_small_verified_wheels(tmp_path,monkeypatch)
    from music_perception import lyrics_model_store
    calls=[]
    def download(url,dest,**kwargs):
        name=url.split('/')[-1]
        calls.append(name)
        dest.write_bytes(wheels[name].read_bytes())
        if kwargs.get('transfer_progress'):
            kwargs['transfer_progress'](dest.stat().st_size,dest.stat().st_size,False)
    monkeypatch.setattr(lyrics_model_store,'_stream_to_file',download)
    monkeypatch.setattr(cb.shutil,'disk_usage',lambda _p:type('usage',(),{'free':10**10})())
    vendor=tmp_path/'env'/'vendor';vendor.mkdir(parents=True)
    (vendor/'sentinel').write_text('keep')
    status=tmp_path/'env'/'status.json'
    assert cb.install_cuda_runtime(vendor.parent,status,'official')
    assert len(calls)==len(cb.CUDA_WHEELS)
    assert cb.cuda_runtime_path(vendor.parent)
    assert (vendor/'sentinel').read_text()=='keep'
    assert cb.install_cuda_runtime(vendor.parent,status,'mirror')
    assert len(calls)==len(cb.CUDA_WHEELS)  # no new downloads


def test_install_download_failure_has_no_activated_cuda(monkeypatch,tmp_path):
    _small_verified_wheels(tmp_path,monkeypatch)
    monkeypatch.setattr(cb.shutil,'disk_usage',lambda _p:type('usage',(),{'free':10**10})())
    monkeypatch.setattr(cb,'_fetch_url',lambda *_:(_ for _ in ()).throw(OSError('offline')))
    environment=tmp_path/'env'
    environment.mkdir()
    (environment/'safe-audio').write_text('original')
    with pytest.raises(cb.PrivateCudaError,match='下载失败'):
        cb.install_cuda_runtime(environment,environment/'gpu-status.json')
    assert cb.cuda_runtime_path(environment) is None
    assert (environment/'safe-audio').read_text()=='original'


def test_status_lock_does_not_kill_installer(monkeypatch,tmp_path):
    import errno
    from music_perception import stem_gpu_runtime as gpu
    original=gpu.os.replace
    attempts=[]
    def lock_once(a,b):
        attempts.append(1)
        if len(attempts)<3:
            raise PermissionError(errno.EACCES,'temporarily locked')
        return original(a,b)
    monkeypatch.setattr(gpu.os,'replace',lock_once)
    monkeypatch.setattr(gpu.time,'sleep',lambda _:None)
    dest=tmp_path/'status.json'
    assert gpu._status(dest,state='running',progress_pct=51)
    assert json.loads(dest.read_text())['progress_pct']==51
    assert len(attempts)==3
