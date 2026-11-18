"""Bounded CPU-only adaptive vocal cleanup after model source separation.

It is NOT a second trained model and deliberately does not try to reconstruct
information already removed by the separator. Spectral subtraction is soft,
frequency/time smoothed, guarded against silent inputs and output clipping.
Preserves original timestamps and length for the Whisper recognizer.
"""
from __future__ import annotations


def enhance_vocals(vocals, backing, *, rate=16000, strength='balanced'):
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from scipy.signal import stft, istft

    if strength not in {'off', 'gentle', 'balanced', 'strong'}:
        raise ValueError('无效的人声增强强度')
    voice = np.asarray(vocals, dtype=np.float32).reshape(-1)
    instrumental = np.asarray(backing, dtype=np.float32).reshape(-1)
    if (rate != 16000 or len(voice) != len(instrumental) or len(voice) > 32*rate
            or not np.isfinite(voice).all() or not np.isfinite(instrumental).all()):
        raise ValueError('音频增强要求最长32秒、16kHz、同长度的有效双音轨')
    if not len(voice) or strength == 'off' or len(voice) < 1024:
        return voice.copy()
    if float(np.max(np.abs(voice))) < 0.00005:
        return voice.copy()
    settings = {
        'gentle': (0.22, 0.48, 0.25),
        'balanced': (0.34, 0.64, 0.42),
        'strong': (0.48, 0.77, 0.58),
    }
    floor, attenuation, blend = settings[strength]
    nperseg, hop = 1024, 256
    _, _, vfft = stft(voice, fs=rate, nperseg=nperseg, noverlap=nperseg-hop,
                       boundary='zeros', padded=True)
    _, _, bfft = stft(instrumental, fs=rate, nperseg=nperseg, noverlap=nperseg-hop,
                       boundary='zeros', padded=True)
    magnitude = np.abs(vfft)
    bleed = np.abs(bfft)
    # Gentle ratio mask: no hard-gate consonant clipping or spectral holes.
    # Leakage should only be reduced where estimated accompaniment dominates.
    mask = magnitude / (magnitude + attenuation*bleed + 1e-7)
    mask = gaussian_filter(mask.astype(np.float32), sigma=(0.8, 1.3), mode='nearest')
    mask = np.clip(mask, floor, 1.0)
    # Preserve very low and high energy bands naturally, avoid rumble.
    hz = np.arange(mask.shape[0], dtype=np.float32) * (rate / nperseg)
    low_guard = np.clip((hz-85)/70,0,1)
    upper_guard = np.clip((7300-hz)/750,0,1)
    gate = (low_guard*upper_guard)[:,None]
    mask = (1-gate)+gate*mask
    _, clean = istft(vfft*mask, fs=rate, nperseg=nperseg, noverlap=nperseg-hop,
                     input_onesided=True, boundary=True)
    if len(clean) < len(voice):
        clean = np.pad(clean,(0,len(voice)-len(clean)))
    clean = clean[:len(voice)].astype(np.float32)
    # Wet-dry makes conservative changes and lowers the chance of damaging vocals.
    output = voice * (1.0 - blend) + clean * blend
    # Avoid aggressive loudness normalization which can magnify residual artefacts.
    original_rms = float(np.sqrt(np.mean(voice.astype(np.float64)**2)))
    output_rms = float(np.sqrt(np.mean(output.astype(np.float64)**2)))
    if output_rms > 1e-7:
        gain = min(1.18, max(0.85, original_rms / output_rms))
        output *= gain
    return np.clip(output, -1.0, 1.0).astype(np.float32)
