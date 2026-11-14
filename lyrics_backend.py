"""Small multilingual ASR *hypotheses* for a consented speaker-loopback music clip.

Run ONLY in the independent Python worker. Whisper base covers zh/ja/en,
but is a speech model, not a validated sung-lyrics model. Recognition can be
wrong for accompaniment-heavy music, harmonies, long held notes and code-switching.
It never determines the song's identity and never emits a 'verified lyric'.
"""
from __future__ import annotations

from pathlib import Path

from .lyrics_model_store import find_local_model, LyricsModelDownloadError

SUPPORTED_LANGUAGES = {'zh', 'ja', 'en'}
LANGUAGE_MODES = {'auto', 'zh', 'ja', 'en'}


def _make_model(model_name: str, root: Path, model_factory=None,
                *, device_policy: str = 'auto', cpu_threads: int = 2,
                allow_public_mirror: bool = True, diagnostics: dict | None = None, stage_hook=None):
    """Create one low-contention Whisper model, preferring predictable CPU usage.

    In CPU mode no GPU libraries are loaded. Explicit 'cuda' and legacy
    'auto' are opt-ins: an unavailable CUDA/cuDNN backend transparently
    falls back to CPU INT8, with an explanatory status for the user.
    """
    if model_name not in {'tiny', 'base', 'small', 'large-v3-turbo'}:
        raise ValueError('Only audited multilingual tiny/base/small models are supported')
    if device_policy not in {'cpu', 'cuda', 'auto'}:
        raise ValueError('Unknown lyrics device policy')
    if not isinstance(cpu_threads,int) or not 1 <= cpu_threads <= 4:
        raise ValueError('Lyrics CPU threads out of range')
    if stage_hook: stage_hook('checking_model')
    if model_factory is None:
        # v0.8.4: NEVER perform network I/O in a live lyrics worker. In the
        # previous build a clip could sit at `checking_model` for minutes
        # because ensure_local_model() tried several public endpoints. The
        # explicit model-preparation subprocess is the sole network owner.
        verified_dir = find_local_model(model_name, root)
        if verified_dir is None:
            raise LyricsModelDownloadError(
                '当前歌词档位没有通过 SHA-256 的本地模型；请先在「模型」中下载校验，'
                '或使用已经验证的 Base/Small。识别子进程不会自动联网。')
        model_path = str(verified_dir)
        from faster_whisper import WhisperModel
        model_factory = WhisperModel
    else:
        # Fake factories used by unit tests must not touch the network.
        model_path = model_name
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cpu_options = dict(device='cpu', compute_type='int8', cpu_threads=cpu_threads,
                       num_workers=1, download_root=str(root))
    if stage_hook: stage_hook('model_ready')
    if device_policy in {'cuda', 'auto'}:
        try:
            if stage_hook: stage_hook('loading_cuda')
            return model_factory(model_path, device='cuda', compute_type='float16',
                                 num_workers=1, download_root=str(root),
                                 local_files_only=True), 'cuda'
        except (RuntimeError, OSError, ValueError, ImportError) as exc:
            if stage_hook: stage_hook('gpu_fallback')
            if diagnostics is not None:
                diagnostics['gpu_warning'] = (
                    'CUDA 歌词加速初始化失败（' + type(exc).__name__ + '）；已回退 CPU INT8。'
                    '请检查 NVIDIA 显卡驱动、CUDA 12 和 cuDNN 9。')
    if stage_hook: stage_hook('loading_cpu')
    return model_factory(model_path, local_files_only=True, **cpu_options), 'cpu'



def prepare_singing_model(*, model_dir, model_name, model_cache,
                          device_policy='cpu', cpu_threads=2,
                          allow_public_mirror=True, stage_hook=None,
                          placement='standard', gpu_budget_mib=0,
                          cpu_budget_mib=0):
    """Load and exercise a standby model without touching captured music.

    Startup and option changes run this in the isolated ASR subprocess; a
    switch is committed by the host only after a successful readiness reply.
    """
    if placement != 'standard':
        # Whisper weights cannot be split across GPU/CPU. Fail with a readable
        # reason instead of silently ignoring the owner's memory budget.
        raise ValueError('权重分层仅适用于 Qwen 歌声 CUDA')
    import numpy as np
    key = (model_name, str(Path(model_dir)), device_policy, cpu_threads,
           allow_public_mirror)
    if model_cache.get('key') == key and model_cache.get('instance') is not None:
        model = model_cache['instance']
        device = model_cache['device']
        details = model_cache.get('diagnostics') or {}
        if stage_hook: stage_hook('model_reused')
    else:
        details = {}
        model, device = _make_model(model_name, Path(model_dir),
                                    device_policy=device_policy,
                                    cpu_threads=cpu_threads,
                                    allow_public_mirror=allow_public_mirror,
                                    diagnostics=details, stage_hook=stage_hook)
    # CUDA may load successfully but crash at first inference because cuDNN
    # is unavailable. A tiny synthetic silence check catches this before
    # retiring the old lyrics worker; no system audio enters this preflight.
    if stage_hook: stage_hook('preflight')
    try:
        result, _ = model.transcribe(np.zeros(16000, dtype=np.float32),
                                     task='transcribe', language='en',
                                     beam_size=1, vad_filter=False,
                                     condition_on_previous_text=False)
        list(result)
    except Exception as exc:
        raise RuntimeError('歌词模型预热推理失败：'+type(exc).__name__) from exc
    model_cache.clear()
    model_cache.update(key=key, instance=model, device=device,
                       diagnostics=dict(details))
    return {'ready':True, 'device':device,
            'requested_device':device_policy,
            'device_note':details.get('gpu_warning'),
            'model':model_name}

def transcribe_singing_candidate(pcm, *, model_dir: str | Path, model_name: str = 'base',
                                 model_factory=None, model_cache: dict | None = None,
                                 device_policy: str = 'auto', cpu_threads: int = 2,
                                 allow_public_mirror: bool = True,
                                 language_hint: str = 'auto', stage_hook=None,
                                 placement: str = 'standard',
                                 gpu_budget_mib: int = 0,
                                 cpu_budget_mib: int = 0) -> dict:
    """Return time-stamped, unverified words; optionally reuse the loaded model.

    Never cache PCM, words, or decoding state across songs. Only weights stay
    resident in the dedicated lyrics child process while music is active.
    """
    if placement != 'standard':
        # Same contract as the singing backend: layering is Qwen/CUDA only.
        raise ValueError('权重分层仅适用于 Qwen 歌声 CUDA')
    import numpy as np
    if language_hint not in LANGUAGE_MODES:
        raise ValueError('Unsupported lyrics language preference')
    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if not 16000 <= len(audio) <= 90 * 16000 or not np.isfinite(audio).all():
        raise ValueError('Expected finite mono 16-kHz PCM, 1 to 90 seconds')
    if float(np.sqrt(np.mean(audio * audio))) < 0.0005:
        return {'status':'no_singing_detected', 'timed_cues':[], 'model':'not_used'}
    key = (model_name, str(Path(model_dir)), device_policy, cpu_threads, allow_public_mirror)
    gpu_diagnostics = {}
    if model_cache is not None and model_cache.get('key') == key and model_cache.get('instance') is not None:
        model, device = model_cache['instance'], model_cache['device']
        gpu_diagnostics.update(model_cache.get('diagnostics') or {})
        if stage_hook: stage_hook('model_reused')
    else:
        model, device = _make_model(model_name, Path(model_dir), model_factory,
                                    device_policy=device_policy, cpu_threads=cpu_threads,
                                    allow_public_mirror=allow_public_mirror,
                                    diagnostics=gpu_diagnostics, stage_hook=stage_hook)
        if model_cache is not None:
            model_cache.clear()
            model_cache.update(key=key, instance=model, device=device,
                               diagnostics=dict(gpu_diagnostics))

    def decode(instance):
        pieces, info = instance.transcribe(
            # Short 6-second sung passages are easy to misclassify. Users can
            # explicitly lock Chinese/Japanese/English, without forcing English
            # translation or changing the ASR model files.
            audio, language=None if language_hint == 'auto' else language_hint,
            task='transcribe', multilingual=language_hint == 'auto',
            beam_size=1 if model_name == 'large-v3-turbo' else 2, best_of=1,
            temperature=0.0, condition_on_previous_text=False,
            # First-pass subtitles favor rapid *line* text delivery. Whisper's
            # word-level alignment is an extra inference pass and can dominate
            # CPU latency; phrase/segment start-end timestamps remain available.
            # Never make users wait for per-word timing before showing lines.
            vad_filter=False, word_timestamps=False,
            no_speech_threshold=0.65, log_prob_threshold=-1.1,
        )
        # Inference only starts when the iterator is consumed.
        return list(pieces), info

    if stage_hook: stage_hook('decoding')
    try:
        segments, info = decode(model)
    except (RuntimeError, OSError, ValueError):
        if device != 'cuda':
            raise
        factory = model_factory
        if factory is None:
            from faster_whisper import WhisperModel
            factory = WhisperModel
        local_path = find_local_model(model_name, Path(model_dir)) if model_factory is None else None
        model = factory(str(local_path) if local_path else model_name, device='cpu', compute_type='int8',
                        cpu_threads=cpu_threads, num_workers=1,
                        download_root=str(model_dir), local_files_only=True)
        device = 'cpu'
        if stage_hook: stage_hook('gpu_fallback')
        gpu_diagnostics['gpu_warning'] = (
            'CUDA 首次推理失败；已回退 CPU INT8。请核对 NVIDIA 显卡驱动和 CUDA/cuDNN。')
        # Persist CPU fallback for subsequent clips; do not retry broken GPU.
        if model_cache is not None:
            model_cache.update(key=key, instance=model, device=device,
                               diagnostics=dict(gpu_diagnostics))
        segments, info = decode(model)
    language = str(getattr(info, 'language', '') or '').lower()
    candidates = []
    for segment in segments:
        if len(candidates) >= 20:
            break
        text = str(getattr(segment, 'text', '') or '').strip()
        if not text:
            continue
        start = max(0.0, float(getattr(segment, 'start', 0.0)))
        end = min(len(audio)/16000, float(getattr(segment, 'end', start)))
        if end <= start:
            continue
        avg_logprob = getattr(segment, 'avg_logprob', None)
        speech_prob = getattr(segment, 'no_speech_prob', None)
        if avg_logprob is not None and float(avg_logprob) < -1.1:
            continue
        if speech_prob is not None and float(speech_prob) > 0.8:
            continue
        # Repetitive/hallucinated text is common on instrumental music too.
        compression_ratio=getattr(segment,'compression_ratio',None)
        if compression_ratio is not None and float(compression_ratio)>2.6:
            continue
        if len(text) > 120:
            text = text[:120]
        # Do not force lyrics onto beats: actual singing often starts between
        # beats. Word timestamps remain model estimates, not alignment proof.
        words = []
        for token in (getattr(segment, 'words', None) or [])[:16]:
            word = str(getattr(token,'word','') or '').strip()
            ts = float(getattr(token,'start',start))
            if word and start <= ts <= end:
                words.append({'at_s':round(ts,3), 'word_preview':word[:18]})
        candidates.append({
            'at_s':round(start,3), 'end_s':round(end,3), 'text_preview':text,
            'words':words[:12], 'provenance':'local_whisper_multilingual_asr',
            'language_candidate': language_hint if language_hint != 'auto' else (language or 'unknown'),
            'language_mode': language_hint,
            'certainty':'unverified_singing_transcription',
        })
    return {
        'status':'asr_hypothesis' if candidates else 'no_reliable_lyrics',
        'model':'whisper-'+model_name, 'engine':'faster_whisper',
        'device': device,
        'requested_device': device_policy,
        'gpu_fallback': device_policy in {'cuda', 'auto'} and device != 'cuda',
        'device_note': gpu_diagnostics.get('gpu_warning'),
        'language_candidate':(language_hint if language_hint != 'auto' else
                              (language if language in SUPPORTED_LANGUAGES else (language or 'unknown'))),
        'language_mode':language_hint,
        'language_note':'片段语言检测并非逐句验证；中日英混唱可能误判',
        'timed_cues':candidates,
        'warning':'自动语音识别对歌唱不可靠，未经原唱或合法 LRC 校验，不应当作准确歌词',
    }
