"""Offline, full-track music feature extraction for N.E.K.O.

The fast mode uses librosa + numpy and makes *tentative* mood observations.
The optional Essentia backend uses separately installed licensed classifiers.
No audio or data is transmitted from this module.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16000
HOP = 512
N_FFT = 2048
MAX_AUDIO_BYTES = 256 * 1024 * 1024
EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif"})
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


class AudioAnalysisError(Exception):
    """An expected, user-facing audio analysis error."""


def _round(value: Any, ndigits: int = 3) -> float:
    """Always return a JSON-compatible finite float."""
    try:
        num = float(value)
        return round(num, ndigits) if math.isfinite(num) else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _decode(path: Path, max_seconds: int):
    """Bound decoded samples, preferring sf for WAV/FLAC and ffmpeg for other formats."""
    import numpy as np

    if not path.is_file():
        raise AudioAnalysisError("找不到音乐文件")
    if path.suffix.lower() not in EXTENSIONS:
        raise AudioAnalysisError("不支持此文件扩展名；请使用 WAV/MP3/FLAC/OGG/M4A/AAC/OPUS")
    if path.stat().st_size > MAX_AUDIO_BYTES:
        raise AudioAnalysisError("文件过大（上限 256 MiB）")

    # Use an inexpensive native decoder for common lossless formats.
    if path.suffix.lower() in {".wav", ".flac", ".ogg", ".aiff", ".aif"}:
        try:
            import soundfile as sf
            with sf.SoundFile(str(path)) as audio:
                if len(audio) / audio.samplerate > max_seconds + 0.05:
                    raise AudioAnalysisError(f"歌曲太长（当前上限 {max_seconds} 秒）")
                if audio.samplerate < 4000 or audio.samplerate > 384000:
                    raise AudioAnalysisError("无效或不支持的采样率")
                # Stream small chunks so a multi-channel hi-res source does not
                # allocate gigabytes of decoded PCM before downmix/resampling.
                import soxr
                sr = int(audio.samplerate)
                if not 1 <= audio.channels <= 16:
                    raise AudioAnalysisError("音轨声道数不在支持范围内")
                resampler = soxr.ResampleStream(sr, SAMPLE_RATE, 1, dtype="float32", quality="HQ") if sr != SAMPLE_RATE else None
                pieces = []
                for block in audio.blocks(blocksize=32768, dtype="float32", always_2d=True):
                    mono_block = block.mean(axis=1, dtype=np.float32)
                    if resampler is not None:
                        mono_block = resampler.resample_chunk(mono_block, last=False)
                    if mono_block.size:
                        pieces.append(mono_block)
                if resampler is not None:
                    tail = resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
                    if tail.size:
                        pieces.append(tail)
            return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        except AudioAnalysisError:
            raise
        except Exception:  # Some OGG variants are not supported by libsndfile.
            pass

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            # Optional local, package-scoped binary; no system PATH or global
            # installation required for Windows MP3/M4A/AAC decoding.
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, OSError, RuntimeError):
            raise AudioAnalysisError("缺少音频解码器；请等待插件自动修复环境（包含 FFmpeg）") from None
    # Explicit duration guard: -t ensures malformed headers never lead to unbounded decoding.
    args = [ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-i", str(path),
            "-map", "0:a:0", "-t", str(max_seconds + 1), "-vn", "-sn", "-dn",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    try:
        result = subprocess.run(args, capture_output=True, timeout=180, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AudioAnalysisError("解码超时，请转换为 WAV 后重试") from exc
    if result.returncode or not result.stdout:
        raise AudioAnalysisError("解码失败：请检查文件是否包含可播放的音轨")
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if samples.size / SAMPLE_RATE > max_seconds + 0.03:
        raise AudioAnalysisError(f"歌曲太长（当前上限 {max_seconds} 秒）")
    return samples


def _key_estimate(chroma_mean):
    """Krumhansl-Schmuckler key estimation: only a rough tonal hypothesis."""
    import numpy as np
    x = np.asarray(chroma_mean, dtype=np.float64)
    if float(x.sum()) < 0.001 or np.std(x) < 0.001:
        return {"name": "不确定", "mode": "unknown", "confidence": "low", "reliable": False}
    scores = []
    for tonic in range(12):
        for mode, template in [("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)]:
            corr = np.corrcoef(x, np.roll(template, tonic))[0, 1]
            scores.append((_round(corr, 5), tonic, mode))
    scores.sort(reverse=True)
    winner, second = scores[0], scores[1]
    reliable = bool(winner[0] >= 0.68 and winner[0] - second[0] >= 0.10)
    return {
        "name": f"{NOTE_NAMES[winner[1]]} {'大调' if winner[2] == 'major' else '小调'}" + ("（推测）" if not reliable else ""),
        "mode": winner[2], "tonic": NOTE_NAMES[winner[1]],
        "correlation": _round(winner[0]), "margin": _round(winner[0] - second[0]),
        "confidence": "medium" if reliable else "low", "reliable": reliable,
    }


def _quick_mood(*, key: dict, tempo: float, brightness: float, energy: float, consistency: float) -> dict:
    """Exploratory *unvalidated* cues, NOT a trained music emotion prediction."""
    # Tempo/brightness/volume do not reliably imply emotional valence.
    # This conservative combination only offers cues, and always flags low confidence.
    score = 0.0
    if key.get("reliable"):
        score += 0.36 if key["mode"] == "major" else -0.36
    score += 0.15 * max(-1.0, min(1.0, (tempo - 100.0) / 90.0))
    score += 0.17 * max(-1.0, min(1.0, (brightness - 1500.0) / 1800.0))
    score += 0.20 * max(-1.0, min(1.0, (energy - 0.42) * 1.8))
    score += 0.12 * max(-1.0, min(1.0, (consistency - 0.50) * 2.0))
    if score >= 0.23:
        label = "可能偏欢快"
    elif score <= -0.23:
        label = "可能偏忧伤"
    else:
        label = "情绪不确定"
    return {
        "label": label,
        "method": "unvalidated_acoustic_heuristic",
        "confidence": "low",
        "score_not_probability": _round(score),
        "notice": "这是节奏/调性/音色启发式推测，不是可靠的快乐或悲伤分类；不包含歌词语义。",
    }


def _essentia_emotion(wave, models_dir: str, *, duration: float) -> dict:
    """Optional pretrained happy and sad models, NOT downloaded/bundled here.

    Model files belong to their publishers; review license terms separately.
    One VGGish embedding extraction is reused by both classification heads.
    """
    import numpy as np
    d = Path(models_dir).expanduser().resolve() if models_dir else None
    if not d or not d.is_dir():
        raise AudioAnalysisError("Essentia 模式需要配置合法的 emotion_models_dir 路径")
    files = [d / "audioset-vggish-3.pb", d / "mood_happy-audioset-vggish-1.pb", d / "mood_sad-audioset-vggish-1.pb"]
    if any(not p.is_file() for p in files):
        raise AudioAnalysisError("缺少 Essentia 权重文件，请参照 README 安装三个 .pb 模型（权重不随插件分发）")
    try:
        from essentia.standard import TensorflowPredict2D, TensorflowPredictVGGish
    except ImportError as exc:
        raise AudioAnalysisError("未安装支持 TensorFlow 的 Essentia；可使用 fast 模式") from exc
    try:
        embeddings = TensorflowPredictVGGish(graphFilename=str(files[0]), output="model/vggish/embeddings")(
            np.asarray(wave, dtype=np.float32))
        responses = {}
        per_window = {}
        for mood, f, positive_class in [("happy", files[1], 0), ("sad", files[2], 1)]:
            # Official metadata: happy=[happy, non_happy]; sad=[non_sad, sad].
            scores = np.asarray(TensorflowPredict2D(graphFilename=str(f), output="model/Softmax")(embeddings), dtype=np.float64)
            if scores.size == 0 or scores.ndim != 2 or scores.shape[1] < 2:
                raise AudioAnalysisError("Essentia 模型返回数据格式不符合预期")
            per_window[mood] = np.clip(scores[:, positive_class], 0.0, 1.0)
            responses[mood] = _round(np.mean(per_window[mood]))
        h, s = responses["happy"], responses["sad"]
        if h >= 0.6 and h > s + 0.15:
            label = "偏欢快"
        elif s >= 0.6 and s > h + 0.15:
            label = "偏忧伤"
        elif h >= 0.6 and s >= 0.6:
            label = "混合情绪"
        else:
            label = "情绪不确定"
        count = min(len(per_window["happy"]), len(per_window["sad"]))
        return {
            "label": label, "method": "essentia_vggish_pretrained", "confidence": "model_estimate",
            "happy_score": h, "sad_score": s,
            "notice": "模型分数为算法输出，不等于人类感受或经校准的真实概率；不分析歌词语义。",
            "frame_scores": [
                {"t": _round(i * duration / max(count, 1), 2), "happy": _round(per_window["happy"][i]), "sad": _round(per_window["sad"][i])}
                for i in range(count)
            ],
        }
    except AudioAnalysisError:
        raise
    except Exception as exc:
        raise AudioAnalysisError(f"Essentia 推理失败：{type(exc).__name__}") from exc


def _spectrogram_thumbnail(spectrum, *, duration: float, sr: int, n_fft: int, max_cols: int = 160) -> dict:
    """Small 32-row, logarithmic-frequency heatmap; covers the *whole* track."""
    import numpy as np
    n_cols = min(max_cols, max(32, int(math.ceil(duration / 2))))
    mag = np.asarray(spectrum, dtype=np.float32)
    edges = np.geomspace(65.0, min(7200.0, sr * 0.48), 33)
    freq_hz = np.fft.rfftfreq(n_fft, 1.0 / sr)
    freq_indices = [np.flatnonzero((freq_hz >= edges[i]) & (freq_hz < edges[i + 1])) for i in range(32)]
    times = np.linspace(0, mag.shape[1], n_cols + 1, dtype=np.int64)
    raw = np.zeros((32, n_cols), dtype=np.float32)
    for col in range(n_cols):
        s = mag[:, times[col]:max(times[col] + 1, times[col + 1])]
        if s.size == 0:
            continue
        for row, bins in enumerate(freq_indices):
            if bins.size:
                raw[row, col] = float(np.sqrt(np.mean(s[bins] ** 2)))
    peak = max(float(np.max(raw)), 1e-8)
    db = 20 * np.log10(np.maximum(raw, 1e-9) / peak)
    intensity = np.rint(np.clip((db + 65.0) / 65.0, 0, 1) * 100).astype(np.uint8)
    return {
        "format": "percent_0_100", "frequency_scale": "log", "frequency_min_hz": 65,
        "frequency_max_hz": _round(edges[-1], 1), "duration_s": _round(duration, 2),
        "rows_low_to_high": intensity.tolist(),
    }


def analyze_audio(path: str | Path, *, segment_seconds: int = 10, max_seconds: int = 900,
                  emotion_mode: str = "fast", emotion_models_dir: str = "",
                  beat_this_checkpoint: str = "") -> dict:
    """Decode a complete authorized file and analyze its waveform."""
    start_time = time.perf_counter()
    song_path = Path(path).expanduser().resolve(strict=True)
    wave = _decode(song_path, max_seconds=max_seconds)
    return analyze_waveform(wave, display_name=song_path.name,
                            segment_seconds=segment_seconds, max_seconds=max_seconds,
                            emotion_mode=emotion_mode, emotion_models_dir=emotion_models_dir,
                            beat_this_checkpoint=beat_this_checkpoint,
                            _start_time=start_time)


def analyze_waveform(wave, *, display_name: str = "当前播放的音乐片段",
                     segment_seconds: int = 10, max_seconds: int = 900,
                     emotion_mode: str = "fast", emotion_models_dir: str = "",
                     beat_this_checkpoint: str = "",
                     _start_time: float | None = None) -> dict:
    """Analyze 16 kHz mono PCM entirely in memory (no raw audio file needed).

    Live captures represent the actually recorded span; they cannot prove
    complete song boundaries. Caller must record provenance.
    """
    import librosa
    import numpy as np
    from scipy.ndimage import median_filter

    start_time = _start_time if _start_time is not None else time.perf_counter()
    display_name = str(display_name).replace("/", "_").replace(chr(92), "_")[:100]
    if not 4 <= segment_seconds <= 60:
        raise AudioAnalysisError("segment_seconds 必须在 4 到 60 之间")
    if not 30 <= max_seconds <= 3600:
        raise AudioAnalysisError("max_seconds 必须在 30 到 3600 之间")
    if emotion_mode not in ("fast", "essentia"):
        raise AudioAnalysisError("emotion_mode 只能为 fast 或 essentia")
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or wave.size > max_seconds * SAMPLE_RATE + 2:
        raise AudioAnalysisError("实时音频必须为16kHz单声道，且不得超过配置的时长上限")
    duration = wave.size / SAMPLE_RATE
    if duration < 1.0:
        raise AudioAnalysisError("歌曲至少需要 1 秒有效音频")
    if not bool(np.isfinite(wave).all()):
        raise AudioAnalysisError("歌曲包含无效音频采样值")
    rms = float(np.sqrt(np.mean(wave ** 2)))
    if rms < 5e-5:
        raise AudioAnalysisError("文件几乎是静音，无法分析音乐")

    spec = np.abs(librosa.stft(wave, n_fft=N_FFT, hop_length=HOP)).astype(np.float32)
    n_frames = spec.shape[1]
    freq = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    power = np.square(spec, dtype=np.float32)
    # log-magnitude spectral flux from the *entire* track (avoids decoding only a 30 s clip)
    log_spec = librosa.amplitude_to_db(spec, ref=np.max)
    onset = librosa.onset.onset_strength(S=log_spec, sr=SAMPLE_RATE, hop_length=HOP)
    try:
        tempo_result, beat_times = librosa.beat.beat_track(
            onset_envelope=onset, sr=SAMPLE_RATE, hop_length=HOP, units="time", trim=False)
        tempo = float(np.asarray(tempo_result).ravel()[0])
        beats = [float(t) for t in np.asarray(beat_times).ravel() if t <= duration]
    except Exception:  # Irregular soundtracks may have no detectable beat.
        tempo, beats = 0.0, []
    if len(beats) < 3:
        tempo = 0.0
    beats_interval = np.diff(beats) if len(beats) >= 3 else np.asarray([])
    beat_regular = (
        float(max(0, 1 - (np.std(beats_interval) / max(np.mean(beats_interval), 1e-5))))
        if len(beats_interval) >= 3 else 0.0
    )
    centroid = librosa.feature.spectral_centroid(S=spec, sr=SAMPLE_RATE)[0]
    brightness = float(np.median(centroid))
    # chroma on FFT power -> a 12-pitch-class profile, not a polyphonic note transcript
    chroma = librosa.feature.chroma_stft(S=power, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, tuning=0.0)
    weights = np.maximum(spec.sum(axis=0), 1e-7)
    chroma_mean = np.average(chroma, axis=1, weights=weights)
    key = _key_estimate(chroma_mean)
    from .music_harmony import chord_change_candidates, beat_accent_candidates, metrical_grouping_candidates
    chord_evidence = chord_change_candidates(chroma, sample_rate=SAMPLE_RATE,
                                             hop_length=HOP, duration=duration)
    accent_evidence = beat_accent_candidates(beats, onset, sample_rate=SAMPLE_RATE,
                                             hop_length=HOP)
    meter_evidence=metrical_grouping_candidates(beats,accent_evidence)
    # Chroma no longer needs the full power matrix: save memory on long tracks.
    del power
    main_classes = np.argsort(chroma_mean)[::-1][:4]

    # Dominant spectral peak as a *rough* melodic contour. Percussion/bass can contaminate it.
    pitch_values, pitch_magnitudes = librosa.piptrack(
        S=spec, sr=SAMPLE_RATE, hop_length=HOP, fmin=125, fmax=1800, threshold=0.24)
    strongest_indices = np.argmax(pitch_magnitudes, axis=0)
    frame_index = np.arange(n_frames)
    dominant_hz = pitch_values[strongest_indices, frame_index]
    strengths = pitch_magnitudes[strongest_indices, frame_index]
    dominant_hz = np.where(strengths > 0, dominant_hz, 0.0)
    # Mild smoothing for display only; ignore low-energy frames.
    dominant_hz = median_filter(dominant_hz, size=5, mode="nearest")
    energy_frames = librosa.feature.rms(S=spec, frame_length=N_FFT)[0]
    dominant_hz[energy_frames < max(0.005, float(np.quantile(energy_frames, 0.2)) * 0.2)] = 0
    midi = np.zeros_like(dominant_hz, dtype=np.float32)
    pitched = dominant_hz > 0
    midi[pitched] = librosa.hz_to_midi(dominant_hz[pitched])

    def bound_frames(start: float, stop: float) -> tuple[int, int]:
        f0 = max(0, min(n_frames - 1, int(start * SAMPLE_RATE / HOP)))
        f1 = max(f0 + 1, min(n_frames, int(math.ceil(stop * SAMPLE_RATE / HOP))))
        return f0, f1

    window_energy = float(max(np.quantile(energy_frames, 0.95), 1e-6))
    melody_points = []
    for second in range(0, math.ceil(duration), 2):
        a, b = bound_frames(second, min(duration, second + 2))
        if int(np.count_nonzero(pitched[a:b])) >= max(2, int((b - a) * 0.25)):
            note_midi = float(np.median(midi[a:b][pitched[a:b]]))
            note_name = librosa.midi_to_note(round(note_midi))
            melody_points.append({"t": _round(second, 1), "midi": _round(note_midi, 1), "note": str(note_name)})
        else:
            melody_points.append({"t": _round(second, 1), "midi": None, "note": None})
    segments = []
    for i in range(math.ceil(duration / segment_seconds)):
        seg_start = i * segment_seconds
        seg_end = min(duration, (i + 1) * segment_seconds)
        a, b = bound_frames(seg_start, seg_end)
        raw_energy = float(np.mean(energy_frames[a:b]))
        local_beats = [t for t in beats if seg_start <= t < seg_end]
        local_intervals = np.diff(local_beats)
        local_bpm = float(60.0 / np.median(local_intervals)) if len(local_intervals) > 2 else 0.0
        pc = np.mean(chroma[:, a:b], axis=1)
        dominant_notes = [NOTE_NAMES[int(n)] for n in np.argsort(pc)[::-1][:3]]
        seg_pitch = midi[a:b][pitched[a:b]]
        median_pitch = float(np.median(seg_pitch)) if seg_pitch.size else None
        low = np.any((freq >= 40) & (freq < 220))
        bass_energy = float(np.square(spec[(freq >= 40) & (freq < 220), a:b], dtype=np.float32).sum()) if low else 0.0
        total_energy = float(np.square(spec[:, a:b], dtype=np.float32).sum())
        energy_norm = max(0.0, min(1.0, raw_energy / window_energy))
        local_brightness = float(np.median(centroid[a:b]))
        mood_cue = _quick_mood(key=key, tempo=local_bpm or tempo,
                               brightness=local_brightness, energy=energy_norm,
                               consistency=beat_regular)
        segments.append({
            "start_s": _round(seg_start, 2), "end_s": _round(seg_end, 2),
            "energy": _round(energy_norm, 2), "loudness_rms": _round(raw_energy, 4),
            "spectral_centroid_hz": _round(local_brightness, 0),
            "bass_share": _round(bass_energy / max(total_energy, 1e-8), 3),
            "beats": len(local_beats), "local_bpm": _round(local_bpm, 1),
            "top_pitch_classes": dominant_notes,
            "dominant_pitch_midi_estimate": _round(median_pitch, 1) if median_pitch is not None else None,
            "mood_cue": mood_cue["label"],
        })
    thumbnail = _spectrogram_thumbnail(spec, duration=duration, sr=SAMPLE_RATE, n_fft=N_FFT)
    mood = _quick_mood(key=key, tempo=tempo, brightness=brightness,
                       energy=float(np.mean(energy_frames) / window_energy), consistency=beat_regular)
    if emotion_mode == "essentia":
        prediction = _essentia_emotion(wave, emotion_models_dir, duration=duration)
        # Map classifier frames to coarse segments; only this mode uses learned class scores.
        for seg in segments:
            inside = [x for x in prediction["frame_scores"] if seg["start_s"] <= x["t"] < seg["end_s"]]
            if inside:
                happy = float(np.mean([x["happy"] for x in inside]))
                sad = float(np.mean([x["sad"] for x in inside]))
                seg["emotion_model"] = {"happy_score": _round(happy), "sad_score": _round(sad),
                                        "label": "偏欢快" if happy > sad + 0.15 and happy > .6 else
                                        "偏忧伤" if sad > happy + 0.15 and sad > .6 else "不确定/混合"}
        mood = {k: v for k, v in prediction.items() if k != "frame_scores"}
        mood["frame_count"] = len(prediction["frame_scores"])

    report = {
        "schema_version": 1,
        "file_name": display_name,  # No local absolute path in AI data.
        "duration_s": _round(duration, 2), "sample_rate_hz": SAMPLE_RATE,
        "processed_entire_song": True,
        "rhythm": {"bpm_estimate": _round(tempo, 1), "beat_times_s": [_round(t, 2) for t in beats],
                   "beat_count": len(beats), "beat_regularity": _round(beat_regular),
                   "accent_candidates": accent_evidence, "meter_candidates": meter_evidence},
        "harmony": {"estimated_key": key, "prominent_pitch_classes": [NOTE_NAMES[int(n)] for n in main_classes],
                    "chromagram_mean": [_round(n, 3) for n in chroma_mean.tolist()],
                    "chord_progression_estimate": chord_evidence},
        "melody": {"method": "dominant_spectral_peak_not_isolated_melody",
                   "voiced_frame_fraction": _round(float(np.mean(pitched))), "points_every_2s": melody_points,
                   "warning": "复音音乐的主导频率可能来自低音、鼓或伴奏，不等于精确主旋律或人声音高。"},
        "emotion": mood,
        "segments": segments,
        "spectrum": thumbnail,
        "summary": {"bpm": _round(tempo, 1), "key_guess": key["name"], "mood_guess": mood["label"],
                    "duration": _round(duration, 2), "segment_count": len(segments)},
        "limitations": ["仅分析本地音频；没有分析歌词意义", "高低音与速度不等于积极或消极情绪",
                        "仅基于整个音频的音高类分布与主导谱峰，不能保证正确分离主旋律",
                        "BPM、调性与每段估值会受复节奏、混音、鼓声等因素影响"],
        "elapsed_seconds": _round(time.perf_counter() - start_time, 2),
    }
    # v0.6: produce musical *events* while original waveform is in memory.
    # The basic detector never invents downbeats or separately identified
    # vocals; a trained beat/stem model must be opted into separately.
    try:
        from .music_events import timed_music_events
    except ImportError:  # Standalone engine.py tests outside the SDK package
        from music_perception.music_events import timed_music_events
    report["music_events"] = timed_music_events(report, waveform=wave,
                                                    time_basis="file_relative_seconds",
                                                    beat_this_checkpoint=beat_this_checkpoint or None)
    return report


def _even_sample(items: list, limit: int) -> list:
    """Context-limit a timeline without accidentally losing the second half."""
    import numpy as np
    if len(items) <= limit:
        return items
    return [items[int(i)] for i in np.linspace(0, len(items) - 1, limit, dtype=int)]


def brief_for_ai(report: dict, *, max_segments: int = 120) -> dict:
    """Bounded LLM response. Timeline samples span start, middle, and end."""
    rhythm = dict(report.get("rhythm", {}))
    all_beats = rhythm.get("beat_times_s", [])
    rhythm["beat_times_s"] = _even_sample(all_beats, 60)
    rhythm["beat_times_sampled"] = len(all_beats) > 60
    melody = dict(report.get("melody", {}))
    all_points = melody.get("points_every_2s", [])
    melody["points_every_2s"] = _even_sample(all_points, 90)
    melody["points_sampled"] = len(all_points) > 90
    all_segs = report.get("segments", [])
    try:
        from .music_events import compact_music_context
    except ImportError:
        from music_perception.music_events import compact_music_context
    return {
        "file_name": report.get("file_name"), "duration_s": report.get("duration_s"),
        "processed_entire_song": report.get("processed_entire_song"),
        "rhythm": rhythm, "harmony": report.get("harmony"), "melody": melody,
        "emotion": report.get("emotion"),
        "music_events": compact_music_context(report["music_events"]) if report.get("music_events") else None,
        "segments": _even_sample(all_segs, max_segments),
        "segments_sampled": len(all_segs) > max_segments,
        "limitations": report.get("limitations"),
        "elapsed_seconds": report.get("elapsed_seconds"),
    }
