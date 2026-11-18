"""Local lyrics-only ASR in independent CPython (no host Python imports).

Legacy mode (v1): stdin contains mono 16kHz f32 PCM, stdout one UTF-8 JSON.
Persistent mode (v2): input header JSON line + exactly `samples` float32
frames, output one bounded JSON line per request. A single loaded Whisper
model is reused, with at most one outstanding request. No audio files saved.
"""
from __future__ import annotations
import json
import contextlib
import os
import pathlib
import sys
import types

RATE = 16000
MAX_SECONDS = 90
MAX_HEADER = 2048


def _reply(record, exit_code=0):
    data = json.dumps(record, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')
    sys.stdout.buffer.write(data + b'\n')
    sys.stdout.buffer.flush()
    return exit_code


def _read_exact(reader, size: int) -> bytes | None:
    """Read raw bytes from a pipe; avoid partial-read corruption."""
    chunks = bytearray()
    while len(chunks) < size:
        part = reader.read(size - len(chunks))
        if not part:
            return None
        chunks.extend(part)
    return bytes(chunks)


def _persistent_main(vendor, model_dir, model_name):
    from music_perception.lyrics_diagnostics import publish_stage
    status_file = os.environ.get('NEKO_LYRICS_WORKER_STATUS')
    publish_stage(status_file, 'worker_started', model=model_name)
    publish_stage(status_file, 'importing_dependencies', model=model_name)
    try:
        # Third-party model initialization may print banners to stdout.
        # Keep stdout exclusively for machine-readable JSON replies.
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            if model_name.startswith('sing-'):
                from music_perception.singing_backend import (transcribe_singing_candidate, prepare_singing_model)
            else:
                from music_perception.lyrics_backend import (transcribe_singing_candidate, prepare_singing_model)
            from music_perception.lyrics_model_store import LyricsModelDownloadError
    except Exception as exc:
        publish_stage(status_file, 'failed', model=model_name, error_type=type(exc).__name__)
        raise
    # Backend drift guard. The whisper backend (lyrics_backend) and the
    # specialist singing backends do NOT accept the same keywords: only the
    # singing backends understand the GPU/RAM layered-weight arguments. Sending
    # `placement` unconditionally raised
    #   TypeError: transcribe_singing_candidate() got an unexpected keyword
    #   argument 'placement'
    # on every non-sing model (base / small / large-v3-turbo), which is exactly
    # the whole normal lyrics + video-speech path. Forward only what the
    # imported callable really accepts.
    import inspect

    def _accepted(fn):
        try:
            return frozenset(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            return frozenset()

    transcribe_params = _accepted(transcribe_singing_candidate)
    prepare_params = _accepted(prepare_singing_model)

    def _supported(supported, values):
        return {k: v for k, v in values.items() if k in supported}

    model_cache = {}
    inp = sys.stdin.buffer
    while True:
        header = inp.readline(MAX_HEADER + 1)
        if not header:
            return 0
        try:
            if len(header) > MAX_HEADER or not header.endswith(b'\n'):
                return _reply({'version': 2, 'error': 'header too large'}, 2)
            instruction = json.loads(header.decode('utf-8'))
            if not isinstance(instruction, dict) or instruction.get('version') != 2:
                return _reply({'version': 2, 'error': 'unexpected protocol'}, 2)
            request_id = instruction.get('id')
            samples = instruction.get('samples')
            operation = instruction.get('kind','transcribe')
            if (type(request_id) is not int or not 0 <= request_id <= 2**31
                    or operation not in ('transcribe','prepare')
                    or type(samples) is not int
                    or (samples != 0 if operation == 'prepare' else
                        not RATE <= samples <= RATE * MAX_SECONDS)):
                return _reply({'version': 2, 'error': 'invalid sample count'}, 2)
            device = instruction.get('device', 'cpu')
            threads = instruction.get('cpu_threads', 2)
            language_hint = instruction.get('language', 'auto')
            allow_mirror = instruction.get('allow_public_mirror', True)
            placement=instruction.get('placement','standard')
            gpu_budget=int(instruction.get('gpu_budget_mib') or 0)
            cpu_budget=int(instruction.get('cpu_budget_mib') or 0)
            if (placement not in ('standard','gpu_ram_weights') or
                (placement=='gpu_ram_weights' and (not model_name.startswith('sing-qwen-') or
                  device!='cuda' or not 1536<=gpu_budget<=32768 or not 4096<=cpu_budget<=131072))):
                return _reply({'version':2,'id':request_id,'error':'无效的GPU/RAM分层请求'})
            if type(allow_mirror) is not bool:
                return _reply({'version': 2, 'error': 'invalid mirror setting'}, 2)
            if (device not in ('cpu', 'cuda', 'auto') or type(threads) is not int
                    or not 1 <= threads <= 4 or language_hint not in ('auto', 'zh', 'ja', 'en')):
                return _reply({'version': 2, 'error': 'invalid processing policy'}, 2)
            actual_device = None
            def mark(stage):
                publish_stage(status_file, stage, job=request_id, kind=operation,
                              model=model_name, requested_device=device,
                              actual_device=actual_device)
            mark('request_received')
            if operation == 'prepare':
                with contextlib.redirect_stdout(sys.stderr):
                    details = prepare_singing_model(**_supported(
                    prepare_params, dict(
                    model_dir=model_dir, model_name=model_name,
                    model_cache=model_cache, device_policy=device,
                    cpu_threads=threads, allow_public_mirror=allow_mirror,
                    placement=placement, gpu_budget_mib=gpu_budget,
                    cpu_budget_mib=cpu_budget, stage_hook=mark)))
                actual_device = details.get('device')
                mark('completed')
                _reply({'version':2, 'id':request_id, 'ready':True, **details})
                continue
            body = _read_exact(inp, samples * 4)
            if body is None:
                return 2
            pcm = np.frombuffer(body, dtype='<f4')
            with contextlib.redirect_stdout(sys.stderr):
                lyrics = transcribe_singing_candidate(pcm, **_supported(
                    transcribe_params, dict(
                    model_dir=model_dir, model_name=model_name,
                    model_cache=model_cache, device_policy=device, cpu_threads=threads,
                    allow_public_mirror=allow_mirror, language_hint=language_hint,
                    placement=placement, gpu_budget_mib=gpu_budget,
                    cpu_budget_mib=cpu_budget, stage_hook=mark)))
            actual_device = lyrics.get('device')
            mark('completed')
            _reply({'version': 2, 'id': request_id, 'lyrics': lyrics})
        except LyricsModelDownloadError as exc:
            publish_stage(status_file, 'failed', job=locals().get('request_id'),
                          kind=locals().get('operation'), model=model_name,
                          requested_device=locals().get('device'),
                          error_type='LyricsModelDownloadError')
            _reply({'version': 2, 'id': request_id if 'request_id' in locals() else None,
                    'code': 'model_download_failed', 'error': str(exc)[:420]})
        except Exception as exc:
            publish_stage(status_file, 'failed', job=locals().get('request_id'),
                          kind=locals().get('operation'), model=model_name,
                          requested_device=locals().get('device'),
                          error_type=type(exc).__name__)
            # Errors are returned as UTF-8 JSON without a corrupt GBK traceback.
            # Do not include PCM, lyrics, credentials or environment paths.
            _reply({'version': 2, 'id': request_id if 'request_id' in locals() else None,
                    'error': type(exc).__name__ + ': ' + str(exc)[:180]})


def main():
    if len(sys.argv) not in (5, 6):
        return _reply({'version': 1, 'error': 'bad arguments'}, 2)
    audio_vendor = pathlib.Path(sys.argv[1]).resolve()
    model_dir = pathlib.Path(sys.argv[2]).resolve()
    model_name = sys.argv[3]
    if (sys.argv[4] != 'mono16k-v1' or not audio_vendor.is_dir()
            or model_name not in {'tiny', 'base', 'small', 'large-v3-turbo',
                                'sing-whisper-small','sing-qwen-0.6b',
                                'sing-whisper-large-v3','sing-qwen-1.7b'}):
        return _reply({'version': 1, 'error': 'invalid private environment'}, 2)
    sys.path.insert(0, str(audio_vendor))
    directory = pathlib.Path(__file__).resolve().parent
    package = types.ModuleType('music_perception')
    package.__path__ = [str(directory)]
    package.__package__ = 'music_perception'
    sys.modules['music_perception'] = package
    if len(sys.argv) == 6:
        if sys.argv[5] != 'stream-v2':
            return _reply({'version': 1, 'error': 'unexpected protocol'}, 2)
        try:
            return _persistent_main(audio_vendor, model_dir, model_name)
        except Exception as exc:
            return _reply({'version': 2, 'error': type(exc).__name__ + ': ' + str(exc)[:160]}, 2)
    body = sys.stdin.buffer.read(MAX_SECONDS * RATE * 4 + 1)
    if not (RATE * 4 <= len(body) <= MAX_SECONDS * RATE * 4) or len(body) % 4:
        return _reply({'version': 1, 'error': 'audio length invalid'}, 2)
    try:
        import numpy as np
        from music_perception.lyrics_backend import transcribe_singing_candidate
        pcm = np.frombuffer(body, dtype='<f4')
        transcription = transcribe_singing_candidate(pcm, model_dir=model_dir,
                                                      model_name=model_name)
        return _reply({'version': 1, 'lyrics': transcription})
    except Exception as exc:
        return _reply({'version': 1, 'error': type(exc).__name__ + ': ' + str(exc)[:240]}, 2)


if __name__ == '__main__':
    raise SystemExit(main())
