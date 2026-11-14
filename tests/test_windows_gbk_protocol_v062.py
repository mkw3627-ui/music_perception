"""v0.6.4 regression: N.E.K.O Windows GBK must not corrupt music JSON.

The -I child interpreter ignores PYTHONIOENCODING, so both stdout and
stderr are explicitly verified under a GBK-only text stream.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import numpy as np
import pytest

from music_perception import analysis_worker

ROOT = Path(__file__).resolve().parents[1]


def _gbk_stream():
    buf = io.BytesIO()
    return buf, io.TextIOWrapper(buf, encoding='gbk', errors='strict')


def test_music_json_emitted_as_utf8_bytes_not_gbk(monkeypatch):
    captured, gbk_only = _gbk_stream()
    with pytest.raises(UnicodeEncodeError):
        gbk_only.write('F♯4')
    with monkeypatch.context() as m:
        m.setattr(analysis_worker.sys, 'stdout', gbk_only)
        analysis_worker._emit_json({'version': 1, 'spectrum': {}, 'report': {
            'melody': [{'note': 'F♯4'}, {'note': 'B♭3'}],
            'lyrics': '风吹过山谷 🎵'}, 'analysis_error': None})
    raw = captured.getvalue()
    assert raw.endswith(b'\n')
    assert b'F\xe2\x99\xaf4' in raw
    assert json.loads(raw.decode('utf-8'))['report']['melody'][0]['note'] == 'F♯4'


def test_worker_error_and_warning_are_utf8_even_under_gbk(monkeypatch):
    captured, gbk_only = _gbk_stream()
    with monkeypatch.context() as m:
        m.setattr(analysis_worker.sys, 'stderr', gbk_only)
        analysis_worker._utf8_line(analysis_worker.sys.stderr, 'ANALYSIS_AUX_WARNING: F♯4 有效')
        with pytest.raises(SystemExit) as e:
            analysis_worker._fail('model: 和弦 A♯/B♭', exit_code=2)
        assert e.value.code == 2
    body = captured.getvalue().decode('utf-8')
    assert 'ANALYSIS_AUX_WARNING: F♯4 有效' in body
    assert 'ANALYSIS_WORKER_ERROR: model: 和弦 A♯/B♭' in body


def test_full_pcm_worker_under_forced_gbk_console(tmp_path):
    """Use the real analyzer behind -I -S, but force text stdout to GBK."""
    vendor = Path(sysconfig.get_path('purelib'))
    if not (vendor / 'scipy' / 'signal').is_dir() or not (vendor / 'librosa').is_dir():
        pytest.skip('scientific test libraries unavailable')
    launcher = tmp_path / 'gbk_launcher.py'
    launcher.write_text('''import runpy
import sys
# Mimic the Windows default console encoding and failure on non-GBK codepoints.
sys.stdout.reconfigure(encoding="gbk", errors="strict")
sys.stderr.reconfigure(encoding="gbk", errors="strict")
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
''', encoding='utf-8')
    sr = 16000
    t = np.arange(4*sr, dtype=np.float32) / sr
    samples = (.18 * np.sin(2 * np.pi * 369.9944 * t)).astype('<f4')
    params = {'segment_seconds': 10, 'max_seconds': 90, 'emotion_mode': 'fast',
              'emotion_models_dir': '', 'beat_this_checkpoint': ''}
    # Important: -I ignores this environment value; binary writes still work.
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    p = subprocess.run([sys.executable, '-I', '-S', str(launcher),
                        str(ROOT/'analysis_worker.py'), str(vendor),
                        json.dumps(params, ensure_ascii=False)],
                       input=samples.tobytes(), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env=env, timeout=85)
    assert p.returncode == 0, p.stderr.decode('utf-8', errors='replace')[-600:]
    obj = json.loads(p.stdout.decode('utf-8'))
    assert obj['report'] is not None
    notes = [x['note'] for x in obj['report']['melody']['points_every_2s'] if x['note']]
    assert 'F♯4' in notes, notes
    assert 'F♯4'.encode('utf-8') in p.stdout
    assert len(obj['spectrum']['levels_time_by_frequency_0_100']) == 24


def test_worker_protocol_does_not_depend_on_environment_variable(monkeypatch):
    # The parent starts the managed interpreter with -I; env is not enough.
    from music_perception import analysis_bridge
    source = Path(analysis_bridge.__file__).read_text(encoding='utf-8')
    assert "'-I', '-S'" in source
    assert 'sys.stdout.buffer.write' in Path(analysis_worker.__file__).read_text(encoding='utf-8')
