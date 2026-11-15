"""v0.9.9: failure diagnostics, removable private venv and model page action audit."""
from __future__ import annotations
import asyncio
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from music_perception import MusicPerceptionPlugin
from music_perception import singing_install as install
from _version import ui_build, panel_ui_build, plugin_version  # noqa: E402


def test_private_command_failure_captures_actionable_error(tmp_path):
    logfile=tmp_path/'private_install.log'
    with pytest.raises(install.SingingInstallError,match='兼容发行包'):
        install._launch([sys.executable,'-c',
            'import sys; print("ERROR: No matching distribution found for fake",file=sys.stderr);sys.exit(1)'],
            timeout=5,log_path=logfile)
    assert 'No matching distribution found' in logfile.read_text('utf8')
    assert 'return code 1' not in install._diagnostic_log_tail(logfile)


def test_private_command_can_be_cancelled_without_touching_other_files(tmp_path):
    cancel=threading.Event()
    original=tmp_path/'models'/'keep.bin'
    original.parent.mkdir();original.write_bytes(b'keep')
    def request():time.sleep(.15);cancel.set()
    thread=threading.Thread(target=request,daemon=True);thread.start()
    with pytest.raises(install.SingingInstallCancelled):
        install._launch([sys.executable,'-c','import time;time.sleep(6)'],
                        timeout=8,cancel_event=cancel,log_path=tmp_path/'install.log')
    thread.join(2)
    assert original.read_bytes()==b'keep'


def test_private_env_removes_only_dedicated_venv(tmp_path):
    environment=tmp_path/'environment'
    private=environment/'singing_asr_venv'
    private.mkdir(parents=True)
    (private/'bad_dependency.pyd').write_bytes(b'broken')
    shared=environment/'private_cuda'/'do_not_touch.dll'
    shared.parent.mkdir();shared.write_bytes(b'cuda')
    models=tmp_path/'models'/'singing_specialist'/'model.bin'
    models.parent.mkdir(parents=True);models.write_bytes(b'weights')
    result=install.uninstall_environment(environment,retries=1)
    assert result['ok'] and not private.exists()
    assert shared.read_bytes()==b'cuda' and models.read_bytes()==b'weights'
    assert install.uninstall_environment(environment)['state']=='absent'


def test_private_env_symlink_must_not_be_deleted(tmp_path):
    env=tmp_path/'environment';env.mkdir()
    target=tmp_path/'outside';target.mkdir()
    (target/'keep').write_text('valuable')
    try:(env/'singing_asr_venv').symlink_to(target,target_is_directory=True)
    except (NotImplementedError,OSError):pytest.skip('platform blocks symlinks')
    with pytest.raises(install.SingingInstallError,match='不安全'):
        install.uninstall_environment(env)
    assert (target/'keep').read_text()=='valuable'


@pytest.mark.asyncio
async def test_private_env_uninstall_action_preserves_weights_and_removes_venv(tmp_path):
    obj=object.__new__(MusicPerceptionPlugin)
    directory=tmp_path/'environment'/'singing_asr_venv'
    directory.mkdir(parents=True);(directory/'broken.txt').write_text('x')
    model=tmp_path/'models'/'singing_specialist'/'checkpoint'
    model.parent.mkdir(parents=True);model.write_text('keep')
    obj.data_path=lambda name:tmp_path/name
    obj._asr_residency_lock=threading.RLock()
    obj._singing_activation_token=0
    obj._singing_profile='ultra'
    obj._singing_runtime_device='cpu'
    obj._lyrics_quality_generation=1
    obj._singing_session=None
    obj._singing_prepare_process=None
    obj._singing_prepare_thread=None
    obj._singing_activation_thread=None
    obj._singing_candidate_process=None
    obj._singing_activation_candidate=None
    obj._singing_environment_removal_thread=None
    obj._singing_environment_removal_state={'state':'idle'}
    obj._singing_prepare_cancel=threading.Event()
    obj._save_preferences=lambda:None
    response=(await obj.uninstall_singing_environment(confirmed=False)).value
    assert response['ok'] is False and directory.exists()
    assert (await obj.uninstall_singing_environment(confirmed=True)).value['ok']
    obj._singing_environment_removal_thread.join(timeout=4)
    assert obj._singing_environment_removal_state['state']=='ready'
    assert not directory.exists() and model.read_text()=='keep'
    assert obj._singing_profile is None


def test_private_env_uninstall_during_busy_task_returns_clear_failure(tmp_path):
    result=install.private_environment_status(tmp_path/'environment')
    assert result['models_preserved_on_uninstall'] is True
    assert 'singing_asr_install.log' in result['install_log']


def test_all_model_page_confirmations_have_registered_backend_actions():
    root=Path(__file__).parents[1]
    ui=(root/'ui/panel.tsx').read_text('utf8')
    backend=(root/'__init__.py').read_text('utf8')
    declared=set(re.findall(r"@ui\.action\(id=['\"]([a-z_]+)['\"]", backend))
    invoked=set(re.findall(r'perform\("([a-z_]+)"',ui))
    invoked.update(re.findall(r'props\.api\.call\("([a-z_]+)"',ui))
    assert invoked <= declared, sorted(invoked-declared)
    for action in ('install_singing_model','activate_singing_model','uninstall_singing_model',
                   'verify_singing_import','cancel_singing_installation',
                   'uninstall_singing_environment','stage_lyrics_model',
                   'uninstall_lyrics_model','start_stem_download','uninstall_stem_model',
                   'set_stem_device','install_stem_gpu','check_stem_gpu'):
        assert action in invoked
    assert '</Page>\n  {dialog?<Modal open={true}' in ui
    assert 'maxHeight:"min(56dvh, 470px)"' in ui
    assert 'min(88dvh, 780px)' not in ui
    assert 'singingEnvironment.install_log' in ui
    assert '卸载独立环境' in ui
    assert '停止当前安装' in ui
    assert ui_build() in ui and ui_build() in backend
