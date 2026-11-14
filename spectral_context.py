"""AI-facing auditory *measurements*, not a classifier's interpretation.

The host cannot currently inject plugin audio parts into a dialog LLM, so
provide an honest, time-indexed logarithmic spectrum representation. No PCM is
sent or persisted. Mood, melody and rhythm estimates belong in separate
auxiliary metadata; the spectrum is not equivalent to hearing native audio.
"""
from __future__ import annotations

from typing import Any

SAMPLE_RATE = 16000


def spectrum_for_ai(samples: Any, *, rate: int = SAMPLE_RATE,
                    freq_bands: int = 48, time_bins: int = 24) -> dict:
    """Return a bounded log-frequency/time grid, with global (not per-frame)
    normalization so variations over time remain observable. Audio only in RAM.

    Each column averages several Hann-windowed FFTs in its time slice. Values
    are relative decibel levels (0..100), not probabilities or emotion labels.
    """
    import numpy as np

    if rate != SAMPLE_RATE:
        raise ValueError('AI 频谱目前只接受 16kHz 单声道音乐 PCM')
    if not (8 <= freq_bands <= 96 and 4 <= time_bins <= 48):
        raise ValueError('频率/时间分辨率超过允许范围')
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size < 2048 or not np.isfinite(x).all():
        raise ValueError('音乐段过短或包含无效采样点')
    size = 2048
    window = np.hanning(size).astype(np.float32)
    frequencies = np.fft.rfftfreq(size, d=1.0/rate)
    edges = np.geomspace(55., 7900., freq_bands + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    cols = []
    energy = []
    for i in range(time_bins):
        start = int(i * x.size / time_bins)
        stop = max(start + 1, int((i + 1) * x.size / time_bins))
        part = x[start:stop]
        # Capture attack and sustained characteristics rather than one center
        # FFT snapshot of the whole bin. Never use an unbounded STFT matrix.
        positions = np.linspace(start, max(start, stop-size), 4).astype(int)
        strengths = np.zeros(freq_bands, dtype=np.float64)
        for pos in positions:
            sample = x[pos:pos+size]
            if sample.size < size:
                sample = np.pad(sample, (0, size - sample.size))
            magnitude = np.abs(np.fft.rfft(sample * window))
            for j in range(freq_bands):
                band = magnitude[(frequencies >= edges[j]) & (frequencies < edges[j+1])]
                strengths[j] += float(np.sqrt(np.mean(band*band))) if band.size else 0.0
        cols.append(strengths / len(positions))
        energy.append(float(np.sqrt(np.mean(part*part))))
    signal = np.asarray(cols)
    actual_peak = float(signal.max())
    # Silence has no spectral content; never normalize silence to maximum.
    if actual_peak <= 1e-10:
        levels = np.zeros((time_bins, freq_bands), dtype=np.uint8)
    else:
        relative_db = 20.0*np.log10(np.maximum(signal, 1e-12)/actual_peak)
        levels = np.rint(100.0*np.clip((relative_db+72.0)/72.0, 0, 1)).astype(np.uint8)
    energy_arr = np.asarray(energy)
    e_max = max(float(energy_arr.max()), 1e-12)
    envelope = np.rint(100.0*np.clip(energy_arr/e_max, 0, 1)).astype(np.uint8)
    return {
        'representation': 'log_frequency_time_spectrogram_v1',
        'primary_evidence': True,
        'not_native_audio': True,
        'source': 'wasapi_default_render_endpoint_mix',
        'sample_rate_hz': rate,
        'duration_s': round(float(x.size/rate), 3),
        'frequency_centers_hz_low_to_high': [round(float(f), 1) for f in centers],
        'time_centers_s': [round((i+.5)*x.size/rate/time_bins, 3) for i in range(time_bins)],
        'levels_time_by_frequency_0_100': levels.tolist(),
        'energy_envelope_0_100': envelope.tolist(),
        'level_scale': 'relative_log_magnitude_db_-72_to_0_across_entire_segment',
        'limitations': '频谱数值不等于原始听觉输入；无法直接恢复歌词、音色或精确主旋律；全段的相对强弱可比较，不能用于跨歌曲绝对音量比较',
    }


def auxiliary_cues(report: dict) -> dict:
    """Small secondary metadata. Make no categorical emotion assertions."""
    rhythm = report.get('rhythm') or {}
    harmony = report.get('harmony') or {}
    melody = report.get('melody') or {}
    parts = report.get('segments') or []
    emotion = report.get('emotion') or {}
    return {
        'rhythm_estimate': {'bpm': rhythm.get('bpm_estimate'),
                            'regularity': rhythm.get('beat_regularity'),
                            'beat_times_s_sample': (rhythm.get('beat_times_s') or [])[:18]},
        'harmonic_hints': {'estimated_key': (harmony.get('estimated_key') or {}).get('name'),
                           'pitch_classes': harmony.get('prominent_pitch_classes', [])},
        'melodic_peak_hints': (melody.get('points_every_2s') or [])[:16],
        'timbre_proxy': {'centroid_hz_per_section': [p.get('spectral_centroid_hz') for p in parts[:8]],
                         'bass_share_per_section': [p.get('bass_share') for p in parts[:8]]},
        'emotion_hint': {'hypothesis': emotion.get('label'),
                         'method': emotion.get('method'),
                         'warning': '只是辅助推测，速度/大小调/亮度不能判定音乐快乐或悲伤'},
        'certainty': 'auxiliary_estimates_not_ground_truth',
    }


def ai_auditory_context(primary: dict, auxiliary: dict, capture: dict,
                        *, music_events: dict | None = None) -> dict:
    """Time-stamped semantic events FIRST, raw spectrum still queryable.

    The dialog host receives text, not PCM/audio tokens. A text event timeline
    is a useful observation but must not be claimed as native listening.
    """
    output = {
        'schema': 'wei_ta_suo_ting_music_perception_v2',
    }
    if music_events is not None:
        from .music_events import compact_music_context
        output['musical_event_timeline'] = compact_music_context(music_events)
    output['auditory_observation'] = primary
    output['auxiliary_feature_estimates'] = auxiliary
    output['music_capture'] = {key: capture.get(key) for key in
                               ('music_sequence', 'song_id', 'song_title', 'replay_of_cached_song',
                                'part_index', 'offset_s', 'end_s', 'end_reason',
                                'estimated_player_clip_start_s', 'player_clock_accuracy')}
    output['time_reference'] = ('confirmed_music_elapsed_time_not_original_media_timestamp'
                                if capture.get('streaming', False) else 'relative_to_audio_start')
    output['interpretation_policy'] = (
        '按时间线理解：节拍时间、鼓点候选、谐波/打击声层、音高走势和分段情绪；'
        '所有来源和不确定性必须保留。模型没听到原始音频，也没有可靠歌词时不得编造。'
        '不需要主动或逐项汇报，不强制聊天回复。'
    )
    return output


def condensed_history_item(context: dict) -> dict:
    """Tiny acoustic history item; numbers are observations, mood stays hints."""
    spec = context['auditory_observation']
    cells = spec['levels_time_by_frequency_0_100']
    # 24x48 => 6 temporal slices x 12 logarithmic frequency regions.
    matrix = []
    for r in range(0, len(cells), 4):
        block = cells[r:r+4]
        if not block: break
        matrix.append([round(sum(row[c] for row in block for c in range(f, min(f+4, len(row))))/
                             sum(min(4, len(row)-f) for row in block))
                       for f in range(0, len(block[0]), 4)])
    return {
        'music_capture': context['music_capture'],
        'time_frequency_thumbnail_6x12_0_100': matrix,
        'rhythm_bpm_hint': context['auxiliary_feature_estimates']['rhythm_estimate']['bpm'],
        'not_definitive': True,
    }
