"""Regression checks using synthetic sounds, no copyrighted music or network."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from music_perception import engine  # now imports shared, package-relative time-event helpers


def make_song(path: Path, seconds: float = 12.0, samplerate: int = 16000):
    """Click-like 120-BPM transient mixed with changing sustained tones."""
    n = int(seconds * samplerate)
    t = np.arange(n, dtype=np.float32) / samplerate
    # First half has C4, second half uses E4; visible note/segment changes.
    wave = np.where(t < seconds / 2, np.sin(2 * np.pi * 261.63 * t),
                    np.sin(2 * np.pi * 329.63 * t)).astype(np.float32) * 0.13
    for beat in np.arange(0.5, seconds, 0.5):
        k = int(beat * samplerate)
        end = min(k + int(0.03 * samplerate), n)
        sl = np.arange(end - k, dtype=np.float32)
        wave[k:end] += .72 * np.exp(-sl / 35) * np.sin(2 * np.pi * 990 * sl / samplerate)
    sf.write(path, np.clip(wave, -1, 1), samplerate)
    return path


def test_full_song_covers_beginning_and_end(tmp_path):
    p = make_song(tmp_path / "tone.wav", seconds=12)
    r = engine.analyze_audio(p, segment_seconds=4, max_seconds=300)
    assert r["processed_entire_song"] is True
    assert r["duration_s"] == pytest.approx(12, abs=0.03)
    assert len(r["segments"]) == 3
    assert r["segments"][0]["start_s"] == 0
    assert r["segments"][-1]["end_s"] == pytest.approx(12, abs=0.05)
    assert len(r["spectrum"]["rows_low_to_high"]) == 32
    assert len(r["spectrum"]["rows_low_to_high"][0]) >= 32
    assert r["emotion"]["confidence"] == "low"
    assert r["emotion"]["method"] == "unvalidated_acoustic_heuristic"
    assert r["melody"]["method"] == "dominant_spectral_peak_not_isolated_melody"
    # JSON output is safe for the N.E.K.O tool and no absolute user paths are forwarded.
    assert json.loads(json.dumps(r, allow_nan=False))["schema_version"] == 1
    assert str(tmp_path) not in json.dumps(r, ensure_ascii=False)


def test_short_silence_and_extension_errors(tmp_path):
    p = tmp_path / "empty.wav"
    sf.write(p, np.zeros(16000 * 2, np.float32), 16000)
    with pytest.raises(engine.AudioAnalysisError, match="静音"):
        engine.analyze_audio(p)
    bad_extension = make_song(tmp_path / "valid.wav").rename(tmp_path / "valid.exe")
    with pytest.raises(engine.AudioAnalysisError, match="扩展名"):
        engine.analyze_audio(bad_extension)


def test_duration_is_not_silently_truncated(tmp_path):
    # 12 sec file and max 10 sec would be rejected at the decoder, but
    # analyze_audio's max_seconds itself must be in [30,3600].
    p = make_song(tmp_path / "duration.wav", seconds=3)
    with pytest.raises(engine.AudioAnalysisError, match="max_seconds"):
        engine.analyze_audio(p, max_seconds=10)
    with pytest.raises(engine.AudioAnalysisError, match="segment_seconds"):
        engine.analyze_audio(p, segment_seconds=2)


def test_sad_classifier_uses_correct_label_index(monkeypatch, tmp_path):
    """Official Sad metadata says index 0=non_sad and index 1=sad."""
    for name in ("audioset-vggish-3.pb", "mood_happy-audioset-vggish-1.pb", "mood_sad-audioset-vggish-1.pb"):
        (tmp_path / name).write_bytes(b"dummy")

    class FakeEmbedding:
        def __init__(self, **kw): pass
        def __call__(self, wave): return np.zeros((3, 128), np.float32)

    class FakeHead:
        def __init__(self, **kw): self.file = kw["graphFilename"]
        def __call__(self, embeddings):
            if "mood_happy" in self.file:
                return np.array([[0.1, 0.9], [0.12, 0.88], [0.1, .9]])
            return np.array([[0.16, 0.84], [0.17, 0.83], [0.15, 0.85]])

    fake = types.ModuleType("essentia")
    fake_std = types.ModuleType("essentia.standard")
    fake_std.TensorflowPredict2D = FakeHead
    fake_std.TensorflowPredictVGGish = FakeEmbedding
    monkeypatch.setitem(sys.modules, "essentia", fake)
    monkeypatch.setitem(sys.modules, "essentia.standard", fake_std)
    emo = engine._essentia_emotion(np.zeros(32000, np.float32), str(tmp_path), duration=2.0)
    assert emo["label"] == "偏忧伤"
    assert emo["sad_score"] > .8
    assert emo["happy_score"] < .2
    assert len(emo["frame_scores"]) == 3


def test_brief_samples_entire_timeline():
    points = [{"t": i} for i in range(600)]
    r = {"file_name": "demo.wav", "rhythm": {"beat_times_s": list(range(600))},
         "melody": {"points_every_2s": points}, "segments": points,
         "emotion": {"label": "不确定"}}
    brief = engine.brief_for_ai(r)
    assert brief["segments"][-1]["t"] == 599
    assert brief["melody"]["points_every_2s"][-1]["t"] == 599
    assert brief["rhythm"]["beat_times_s"][-1] == 599


def test_chunked_stereo_48k_audio(tmp_path):
    """ResampleStream must preserve full song length and not drop the tail."""
    from music_perception.engine import _decode, SAMPLE_RATE
    p = tmp_path / "stereo-48k.wav"
    sr, seconds = 48000, 4
    t = np.arange(sr * seconds, dtype=np.float64) / sr
    stereo = np.stack([0.2*np.sin(t*2*np.pi*261.6), 0.2*np.sin(t*2*np.pi*440)], axis=1)
    sf.write(p, stereo.astype(np.float32), sr)
    mono = _decode(p, max_seconds=30)
    assert mono.dtype == np.float32
    assert abs(len(mono)/SAMPLE_RATE - seconds) < .002
    assert .04 < np.sqrt(np.mean(mono**2)) < .16


def test_ffmpeg_mp3_full_file(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg not on PATH")
    wav = make_song(tmp_path / "from.wav", seconds=5)
    mp3 = tmp_path / "to.mp3"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
                    "-codec:a", "libmp3lame", "-qscale:a", "6", str(mp3)], check=True)
    r = engine.analyze_audio(mp3, segment_seconds=4, max_seconds=90)
    assert r["processed_entire_song"]
    assert r["duration_s"] == pytest.approx(5.0, abs=0.15)
    assert r["segments"][-1]["end_s"] == pytest.approx(r["duration_s"])
