"""Regression for v0.9.10 user screenshot: popup escaped viewport on scroll.

Tests inspect the compiled panel's host integration contract. Visual/scroll
behavior has separate Chromium fixture; N.E.K.O Windows host still needs QA.
"""
from pathlib import Path

from _version import ui_build  # noqa: E402

ROOT=Path(__file__).resolve().parents[1]

def test_native_host_modal_is_outside_page_scroll_context():
    ui=(ROOT/'ui/panel.tsx').read_text('utf8')
    assert 'Modal,' in ui.split('from "@neko/plugin-ui"')[0]
    assert '</Page>\n  {dialog?<Modal open={true}' in ui
    assert 'title={dialogTitle}' in ui and 'size="md"' in ui
    assert 'closeOnBackdrop={false}' in ui and 'onClose={()=>setDialog(null)}' in ui
    assert '<div role="dialog" aria-modal="true"' not in ui
    assert 'position:"fixed",inset:0,zIndex:' not in ui

def test_confirmation_actions_and_close_handling_unchanged():
    ui=(ROOT/'ui/panel.tsx').read_text('utf8')
    assert 'footer={' in ui and 'onClick={onConfirm}' in ui
    assert 'const act=dialog' in ui and 'setDialog(null)' in ui
    assert 'install_singing_model' in ui and 'uninstall_singing_environment' in ui
    assert '{quality:act.quality,source,device:singingDevice,confirmed:true}' in ui
    assert '{quality:act.quality,confirmed:true}' in ui
    assert 'maxHeight:"min(56dvh, 470px)"' in ui

def test_backend_ui_version_consistent():
    ui=(ROOT/'ui/panel.tsx').read_text('utf8')
    backend=(ROOT/'__init__.py').read_text('utf8')
    marker=ui_build()
    assert marker in ui and marker in backend
