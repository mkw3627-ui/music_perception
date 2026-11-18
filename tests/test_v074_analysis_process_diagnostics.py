"""Regression for Windows no-stderr exit=1 after a plugin restart.

Synthetic process fixtures, no real N.E.K.O installation, internet or Windows
sound device required. These tests do not capture or retain audio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from music_perception.analysis_bridge import (
    AnalysisWorkerCancelled,
    AnalysisWorkerError,
    _worker_failure,
    analyze_pcm_isolated,
)

ROOT = Path(__file__).resolve().parents[1]


def _args(tmp_path):
    return {'samples': np.zeros(16000,dtype=np.float32),
            'python_exe': sys.executable, 'vendor': tmp_path,
            'worker_script': ROOT / 'analysis_worker.py', 'options':{},
            'diagnostic_file': tmp_path / 'analysis_diagnostics.log'}


def test_unknown_exit_has_stage_and_private_diagnostic_file(monkeypatch,tmp_path):
    from music_perception import analysis_bridge
    class Child:
        returncode=1
        def communicate(self, input, timeout):
            assert len(input)==16000*4
            return b'PRIVATE-RESULT-MUST-NOT-BE-LOGGED', b'ANALYSIS_STAGE:import_scipy\n'
    monkeypatch.setattr(analysis_bridge.subprocess,'Popen',lambda *a,**k:Child())
    with pytest.raises(AnalysisWorkerError,match='import_scipy'):
        analyze_pcm_isolated(**_args(tmp_path))
    content=(tmp_path/'analysis_diagnostics.log').read_text('utf-8')
    assert 'exit=1' in content
    assert 'import_scipy' in content
    assert 'PRIVATE-RESULT' not in content


def test_exit_one_without_stderr_is_not_mistaken_for_missing_package():
    result=_worker_failure(1,b'',0)
    assert 'python_entry_not_reached' in result
    assert '不能仅凭退出码确定' in result
    assert 'ModuleNotFoundError' not in result


def test_native_windows_exit_code_is_preserved():
    # Windows access violation: preserve Windows unsigned HRESULT form.
    assert '0xC0000005' in _worker_failure(3221225477,b'ANALYSIS_STAGE:import_librosa\n',0)


def test_shutdown_termination_is_cancellation_not_real_error(monkeypatch,tmp_path):
    from music_perception import analysis_bridge
    state={'cancelled':False}
    class Child:
        returncode=1
        def communicate(self, input, timeout):
            state['cancelled']=True
            return b'', b''
    monkeypatch.setattr(analysis_bridge.subprocess,'Popen',lambda *a,**k:Child())
    with pytest.raises(AnalysisWorkerCancelled,match='取消'):
        analyze_pcm_isolated(**_args(tmp_path),is_cancelled=lambda:state['cancelled'])
    assert not (tmp_path/'analysis_diagnostics.log').exists()


def test_earlier_cancellation_does_not_start_child(monkeypatch,tmp_path):
    from music_perception import analysis_bridge
    def forbidden(*a,**k):
        raise AssertionError('cancellation should happen before spawn')
    monkeypatch.setattr(analysis_bridge.subprocess,'Popen',forbidden)
    with pytest.raises(AnalysisWorkerCancelled):
        analyze_pcm_isolated(**_args(tmp_path),is_cancelled=lambda:True)


def test_worker_records_pipeline_stages_in_code():
    src=(ROOT/'analysis_worker.py').read_text('utf-8')
    for stage in ('worker_entered','vendor_ready','import_numpy','import_scipy',
                  'import_librosa','read_pcm','extract_spectrum','extract_events',
                  'write_result'):
        assert f"_stage('{stage}')" in src


def test_stop_cancellation_has_distinct_handling():
    src=(ROOT/'__init__.py').read_text('utf-8')
    assert 'except AnalysisWorkerCancelled:' in src
    assert "diagnostic_file=self.data_path('analysis_diagnostics.log')" in src
