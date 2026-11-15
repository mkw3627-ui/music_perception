"""Regression: observed Windows SingingInstallError and fixed-inset modal anchoring.

No model network access / no weights. Exercise the REAL worker subprocess
with a tiny fake singing_catalog, rather than monkeypatching its source.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import sys

from music_perception.singing_install import SingingInstallError, _launch
import pytest

ROOT = Path(__file__).parents[1]


def install_stub(tmp_path: Path, source: str) -> Path:
    work = tmp_path / '空 格╱loopback'
    work.mkdir(parents=True)
    shutil.copy2(ROOT / 'singing_prepare_worker.py', work / 'singing_prepare_worker.py')
    (work / 'singing_catalog.py').write_text(source, encoding='utf-8')
    return work / 'singing_prepare_worker.py'


def test_worker_from_real_script_creates_progress_and_finishes(tmp_path):
    script = install_stub(tmp_path, '''def acquire_model(root, quality, source, on_stage, on_progress):
    on_stage('downloading:official')
    on_progress('downloading', {'bytes_received': 250, 'total_bytes': 1000,
        'progress_pct': 25, 'speed_bytes_s': 1024, 'source_name': 'official'})
    on_stage('sha256_verification')
    return {'state': 'verified'}
''')
    downloads = tmp_path / 'private models' / '中日英'
    downloads.mkdir(parents=True)
    process = subprocess.run([sys.executable, '-I', str(script),
                              str(tmp_path / 'vendor'), str(downloads), 'ultra', 'official'],
                             capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    state_path = downloads / 'singing_download_ultra.json'
    assert state_path.exists(), f'Wrong status path: {list(downloads.iterdir())}'
    state = json.loads(state_path.read_text('utf-8'))
    assert state['state'] == 'ready' and state['stage'] == 'verified_and_ready'
    assert state['progress_pct'] == 100 and state['total_bytes'] == 1000


def test_worker_failure_writes_actionable_log_and_state(tmp_path):
    script = install_stub(tmp_path, '''def acquire_model(root, quality, source, on_stage, on_progress):
    raise RuntimeError('injected process failure')
''')
    dest = tmp_path / 'model files'
    dest.mkdir()
    log = tmp_path / 'install.log'
    with pytest.raises(SingingInstallError, match='模型下载/校验子进程'):
        _launch([sys.executable, '-I', str(script), 'vendor', str(dest), 'low', 'official'],
                timeout=15, log_path=log)
    # The child process writes with the platform's console encoding (GBK on
    # Chinese Windows), so decode defensively as the production tail reader does.
    output = log.read_bytes().decode('utf-8', errors='replace')
    assert 'Traceback (most recent call last)' in output
    assert 'injected process failure' in output
    state = json.loads((dest / 'singing_download_low.json').read_text('utf-8'))
    assert state['state'] == 'failed' and state['stage'] == 'failed'
    assert 'RuntimeError' in state['error']


def test_host_modal_is_sibling_to_scrolling_page_and_uses_native_layer():
    # The v0.9.10 bug came from putting position:fixed under the Page shell:
    # the containing block/iframe scrolling could pull it out of view.
    ui = (ROOT / 'ui/panel.tsx').read_text('utf-8')
    assert 'Modal,' in ui.split('from "@neko/plugin-ui"',1)[0]
    assert 'return <>' in ui and '</Page>\n  {dialog?<Modal open={true}' in ui
    assert '<Modal open={true} title={dialogTitle} size="md"' in ui
    assert 'closeOnBackdrop={false}' in ui
    assert 'onClose={()=>setDialog(null)}' in ui
    assert 'footer={' in ui and 'onClick={onConfirm}' in ui
    assert 'maxHeight:"min(56dvh, 470px)"' in ui
    # No second, hand-built overlay that tries to beat host stacking contexts.
    assert '<div role="dialog" aria-modal="true"' not in ui
    assert 'position:"fixed",inset:0,zIndex:2147483000' not in ui
    assert 'margin:"auto"' not in ui.split('  {dialog?<Modal',1)[-1]
    # All actions still require explicit user interaction.
    assert 'const onConfirm=()=>{' in ui
    assert '{quality:act.quality,source,device:singingDevice,confirmed:true}' in ui


def test_model_install_failure_is_not_triplicated():
    ui = (ROOT / 'ui/panel.tsx').read_text('utf-8')
    assert '安装失败：{short(singing.preparation.error' not in ui
    assert 'title={"专业歌声 · "+tierLabels[singingQuality]} task={singing.preparation}' not in ui
    assert '完整原因见上方下载任务' in ui
