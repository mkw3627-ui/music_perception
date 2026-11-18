"""Independent PCM analyzer for Windows packaged N.E.K.O.

Run using the plugin's verified, independent CPython: python -I -S analysis_worker.py
<VENDOR> <OPTIONS_JSON>.  Standard input contains float32 mono 16kHz PCM.
Standard output contains a single JSON result. No PCM files are written.

Importing music_perception normally would execute its N.E.K.O SDK bootstrap; use a
package stub so only the pure analyzer modules load in this child process.
"""
from __future__ import annotations

import importlib
import json
import pathlib
import sys
import types

MAX_SECONDS = 90  # A live segment, not an arbitrary recording or whole song.
RATE = 16000


def _utf8_line(stream, message: str) -> None:
    """Write protocol text as UTF-8 *bytes*, never through Windows' GBK console.

    The managed interpreter is launched with -I, which ignores the parent
    process' PYTHONIOENCODING / PYTHONUTF8 environment variables.  The pipe
    protocol must therefore own its encoding explicitly on every platform.
    """
    stream.buffer.write((message + '\n').encode('utf-8', errors='replace'))
    stream.buffer.flush()


def _stage(name: str) -> None:
    # Crash breadcrumbs go through stderr; never include samples/lyrics.
    _utf8_line(sys.stderr, 'ANALYSIS_STAGE:' + name)


def _fail(message: str, *, exit_code: int = 2):
    _utf8_line(sys.stderr, 'ANALYSIS_WORKER_ERROR: ' + message[:1600])
    raise SystemExit(exit_code)


def _emit_json(result: dict) -> None:
    """One newline-terminated UTF-8 JSON document on binary stdout."""
    wire = json.dumps(result, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')
    sys.stdout.buffer.write(wire + b'\n')
    sys.stdout.buffer.flush()


def main() -> None:
    _stage('worker_entered')
    if len(sys.argv) != 3:
        _fail('expected <plugin-private-vendor-path> <options-json>')
    vendor = pathlib.Path(sys.argv[1]).resolve()
    if not vendor.is_dir():
        _fail('isolated vendor directory does not exist')
    _stage('vendor_ready')
    sys.path.insert(0, str(vendor))
    source_dir = pathlib.Path(__file__).resolve().parent
    package = types.ModuleType('music_perception')
    package.__path__ = [str(source_dir)]
    package.__package__ = 'music_perception'
    sys.modules['music_perception'] = package

    try:
        options = json.loads(sys.argv[2])
        if not isinstance(options, dict):
            raise ValueError('options must be a JSON object')
        # Use exactly the audio environment prepared for this plugin, rather
        # than the frozen host's shadow/partial scipy distribution.
        _stage('import_numpy')
        import numpy as np
        _stage('import_scipy')
        # deliberate: catches partial scipy installations
        importlib.import_module('scipy.signal')
        importlib.import_module('scipy.ndimage')
        _stage('import_librosa')
        importlib.import_module('librosa')
        _stage('import_audio_engine')
        from music_perception.engine import analyze_waveform
        from music_perception.spectral_context import spectrum_for_ai
    except Exception as exc:
        _fail(f'isolated import: {type(exc).__name__}: {exc}')

    _stage('read_pcm')
    pcm = sys.stdin.buffer.read((MAX_SECONDS * RATE * 4) + 1)
    if not pcm or len(pcm) > MAX_SECONDS * RATE * 4 or len(pcm) % 4:
        _fail('invalid or oversized PCM payload')
    samples = np.frombuffer(pcm, dtype='<f4').copy()
    if not np.isfinite(samples).all():
        _fail('PCM has nonfinite values')

    _stage('extract_spectrum')
    try:
        primary = spectrum_for_ai(samples)
    except Exception as exc:
        _fail(f'spectrogram: {type(exc).__name__}: {exc}')

    analysis_error = None
    report = None
    _stage('extract_events')
    try:
        # A malformed optional model must not destroy the main spectrum.
        report = analyze_waveform(
            samples, display_name='电脑系统正在播放的音乐（实时分段）',
            segment_seconds=int(options.get('segment_seconds', 10)),
            max_seconds=int(options.get('max_seconds', 900)),
            emotion_mode=str(options.get('emotion_mode', 'fast')),
            emotion_models_dir=str(options.get('emotion_models_dir', '')),
            beat_this_checkpoint=str(options.get('beat_this_checkpoint', '')),
        )
    except Exception as exc:
        analysis_error = f'{type(exc).__name__}: {str(exc)[:330]}'
        _utf8_line(sys.stderr, 'ANALYSIS_AUX_WARNING: ' + analysis_error)

    _stage('write_result')
    try:
        _emit_json({'version': 1, 'spectrum': primary, 'report': report,
                    'analysis_error': analysis_error})
    except Exception as exc:
        _fail(f'JSON output: {type(exc).__name__}: {exc}')


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        # Show location, not private audio samples or raw Python data.
        _fail(f'unexpected {type(exc).__name__}: {exc}')
