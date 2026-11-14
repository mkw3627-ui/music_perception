"""Regression for N.E.K.O's frozen host lacking scipy.signal.

These tests do not change host SciPy or install packages; the subprocess must
import the already-installed scientific stack from its own vendor directory.
"""
from __future__ import annotations

import asyncio
import importlib.util
import math
import os
from pathlib import Path
import sys
import sysconfig

import numpy as np
import pytest

from music_perception.analysis_bridge import analyze_pcm_isolated, AnalysisWorkerError
from music_perception.runtime_environment import PROBE_SCRIPT
from test_plugin import plugin_fixture

ROOT = Path(__file__).resolve().parents[1]


def _vendored_site_packages():
    """Developer env is only a test fixture; runtime uses MusicEnvironment."""
    root = Path(sysconfig.get_path('purelib'))
    if (root / 'scipy' / 'signal').is_dir() and (root / 'librosa').is_dir():
        return root
    pytest.skip('test environment has no scientific site-packages')


def _test_wave(length=10):
    t = np.arange(16000 * length, dtype=np.float32) / 16000
    mono = .2 * np.sin(2*np.pi*220*t) + .04 * np.sin(2*np.pi*440*t)
    for s in range(length):
        mono[s*16000:s*16000+400] += np.linspace(.28, 0, 400)
    return mono.astype(np.float32)


def _options():
    return {'max_seconds': 90, 'segment_seconds': 10, 'emotion_mode':'fast',
            'emotion_models_dir':'', 'beat_this_checkpoint':''}


def _invoke(samples):
    return analyze_pcm_isolated(samples, python_exe=sys.executable,
        vendor=_vendored_site_packages(), worker_script=ROOT / 'analysis_worker.py',
        options=_options(), timeout=100)


def test_regression_host_scipy_signal_missing_but_external_worker_succeeds(monkeypatch):
    # This exact failure occurred after 10 seconds of real WASAPI capture:
    # ModuleNotFoundError: No module named 'scipy.signal'. The isolated child
    # must not inherit the host's sys.modules or import hooks.
    monkeypatch.setitem(sys.modules, 'scipy.signal', None)
    result = _invoke(_test_wave())
    assert result['analysis_error'] is None
    assert result['report']['duration_s'] == 10.0
    assert result['report']['music_events']['beat_times_s']
    assert len(result['spectrum']['levels_time_by_frequency_0_100']) == 24
    assert result['report']['music_events']['downbeat_times_s'] == []  # no model


def test_subprocess_analysis_does_not_persist_audio(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = {p.name for p in tmp_path.iterdir()}
    result = _invoke(_test_wave())
    after = {p.name for p in tmp_path.iterdir()}
    assert result['report']['processed_entire_song'] is True  # flag rewritten for live by caller
    assert before == after


def test_worker_rejects_wrong_size_and_partial_pcm_without_spawning(monkeypatch):
    def must_not_spawn(*_args,**_kwargs):
        raise AssertionError('should be rejected before subprocess')
    from music_perception import analysis_bridge
    monkeypatch.setattr(analysis_bridge.subprocess, 'Popen', must_not_spawn)
    with pytest.raises(AnalysisWorkerError,match='片段长度'):
        _invoke(np.zeros(1,dtype=np.float32))
    with pytest.raises(AnalysisWorkerError,match='非有限'):
        _invoke(np.full(16000,np.nan,dtype=np.float32))


def test_worker_checks_scipy_submodules_not_only_package_name():
    assert 'import numpy, scipy, scipy.signal, scipy.ndimage, scipy.fft' in PROBE_SCRIPT
    assert 'resample_poly' in PROBE_SCRIPT
    assert 'median_filter' in PROBE_SCRIPT


def test_refuses_frozen_host_as_python_interpreter():
    with pytest.raises(AnalysisWorkerError,match='禁止用 N.E.K.O'):
        analyze_pcm_isolated(_test_wave(1),
            python_exe='D:/Programs/projectneko_server.exe',
            vendor=_vendored_site_packages(),worker_script=ROOT/'analysis_worker.py',
            options=_options())


def test_minimal_fallback_events_never_invent_downbeats(plugin_fixture):
    plugin, *_ = plugin_fixture
    spectrum={'energy_envelope_0_100':[10,10,80,20,10],
              'time_centers_s':[.2,.6,1,1.4,1.8]}
    events=plugin._minimal_spectral_events(spectrum,3)
    assert events['downbeat_times_s'] == []
    assert events['beat_times_s'] == []
    assert events['lyrics']['status'] == 'not_available'
    assert events['events'][0]['at_s'] == 4
    assert events['events'][0]['status'] == 'not_verified_drum_or_beat'


def test_plugin_processes_real_music_in_private_interpreter(plugin_fixture, monkeypatch):
    plugin, *_ = plugin_fixture
    class IsolatedReady:
        def worker_runtime(self):
            return sys.executable, _vendored_site_packages()
        def stop(self):
            pass
    async def run():
        await plugin.startup()
        plugin._environment=IsolatedReady()
        plugin._auto_consented=True
        plugin._ai_delivery=False
        plugin._live_session_config=_options()
        from music_perception.auto_listener import FinishedClip
        clip=FinishedClip(_test_wave(), 'live_progress', 5,sequence=1,part_index=1)
        await plugin._process_live_clip(clip)
        assert plugin._live_reports_completed == 1
        report=(await plugin.get_last_report()).value['report']
        assert report['music_events']['beat_times_s']
        assert report['processed_entire_song'] is False
        assert report['source']=='speaker_loopback_after_yamnet_music_gate'
        assert not list(plugin.data_path().glob('*.wav'))
        await plugin.shutdown()
    asyncio.run(run())


def test_auxiliary_failure_still_returns_primary_spectrum():
    bad = _options()
    bad['segment_seconds'] = 99  # analyser rejects, spectrum should still be valid
    result = analyze_pcm_isolated(_test_wave(2), python_exe=sys.executable,
        vendor=_vendored_site_packages(), worker_script=ROOT / 'analysis_worker.py',
        options=bad, timeout=100)
    assert result['report'] is None
    assert 'segment_seconds' in result['analysis_error']
    assert result['spectrum']['duration_s'] == 2.0


def test_stop_terminates_active_private_analysis_process(plugin_fixture):
    plugin, *_ = plugin_fixture
    class FakeChild:
        def __init__(self): self.terminated=False
        def poll(self): return None
        def terminate(self): self.terminated=True
    async def scenario():
        await plugin.startup()
        child=FakeChild()
        with plugin._analysis_children_lock:
            plugin._analysis_children.add(child)
        plugin._stop_background_processor()
        assert child.terminated
        await plugin.shutdown()
    asyncio.run(scenario())


def test_visual_builder_import_probe_checks_scipy_signal():
    from music_perception.install_prepare import _NATIVE_PROBE
    assert 'scipy.signal' in _NATIVE_PROBE
    assert 'scipy.ndimage' in _NATIVE_PROBE
