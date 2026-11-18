"""Consent-gated, speaker-loopback-only music detection for 为她所听.

Uses local YAMNet TFLite to reject non-music before buffering a music clip.
No microphone fallback, raw audio disk writes, or remote inference. A dedicated
thread owns the SoundCard recorder (required for CoreAudio/WASAPI lifecycles).
"""
from __future__ import annotations

from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import sys
from typing import Callable
from .song_change import SongChangeDetector

SAMPLE_RATE = 16000
CAPTURE_RATE = 48000
WINDOW_SECONDS = 2
ONSET_HOP_SECONDS = 0.025
# How many consecutive non-music windows may pass before the current song is
# really considered over. A music video flickers between 'music' and 'video'
# every couple of seconds; a real pause in the song is much longer than this.
# 15 windows = 30 s.
NON_MUSIC_RESET_WINDOWS = 15
# Google lite-model/yamnet/classification/tflite/1 requires exactly 15,600
# mono float32 16 kHz samples per inference. It is NOT a variable-length model.
YAMNET_FRAME_SAMPLES = 15600
YAMNET_HOP_SAMPLES = 7680


class ListenerError(RuntimeError):
    pass


@dataclass
class Decision:
    music: bool
    music_score: float
    speech_score: float
    model: str = "YAMNet LiteRT"
    reason: str = "classified"
    top_class_index: int | None = None
    top_class_score: float = 0.0


@dataclass
class FinishedClip:
    samples: object  # numpy float32 [samples]
    end_reason: str
    music_windows: int
    # Position within the current continuously detected music session. Live
    # reports are incremental, not proof that an entire song was captured.
    offset_s: float = 0.0
    sequence: int = 0
    part_index: int = 1
    capture_end_monotonic_s: float | None = None
    # Optional high-bandwidth stereo for *authorized* neural separation only.
    # Ordinary YAMNet/music/Whisper paths remain 16k mono by default.
    full_band_samples: object | None = None
    full_band_rate: int = CAPTURE_RATE


class YamnetMusicGate:
    """YAMNet has 521 *independent* scores (not probabilities summing to 1).

    Source label map: TensorFlow models/research/audioset/yamnet.
    music=132; genres and song/background music=211..262. Only those labels
    count as music. A single instrument sound alone is not enough.
    """

    def __init__(self, model_file: str | Path, *, min_score: float = 0.24):
        path = Path(model_file).expanduser().resolve()
        if not path.is_file():
            raise ListenerError("缺少 YAMNet 音乐判别模型；请先在插件面板执行自动准备/安装")
        if not 0.05 <= min_score <= 0.95:
            raise ListenerError("音乐判断阈值需要在 0.05～0.95 内")
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:
            raise ListenerError("缺少 ai-edge-litert；请先通过插件面板自动准备依赖") from exc
        self._threshold = min_score
        self._interpreter = Interpreter(model_path=str(path), num_threads=2)
        self._input = self._interpreter.get_input_details()[0]
        shape = tuple(int(n) for n in self._input["shape"])
        if shape not in {(YAMNET_FRAME_SAMPLES,), (1, YAMNET_FRAME_SAMPLES)}:
            raise ListenerError(f"当前模型输入形状 {shape} 不受支持，请使用官方 0.975 秒分类版 YAMNet Lite")
        # Backends may omit dtype in test stubs; normal LiteRT always provides it.
        import numpy as np
        if self._input.get("dtype", np.float32) != np.float32:
            raise ListenerError("YAMNet 需要 float32 输入；模型格式不兼容")
        self._output_index = None
        self._interpreter.allocate_tensors()
        for output in self._interpreter.get_output_details():
            shape = tuple(int(n) for n in output["shape"])
            if shape and shape[-1] == 521:
                self._output_index = output["index"]
                break
        if self._output_index is None:
            raise ListenerError("YAMNet 模型输出无 521 维音频类别，文件可能不兼容")

    def classify(self, pcm) -> Decision:
        """Classify consecutive 0.975 second windows of the provided 2s buffer.

        A fixed input length is required by the *lite-model* (as distinct from
        tensorflow_hub's SavedModel which accepts arbitrary-length waveforms).
        Only music-class evidence permits a downstream music clip to begin.
        """
        import numpy as np
        wave = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if len(wave) < YAMNET_FRAME_SAMPLES:
            return Decision(False, 0.0, 0.0, reason="insufficient_samples")
        if not np.all(np.isfinite(wave)):
            return Decision(False, 0.0, 0.0, reason="invalid_pcm")
        rms = float(np.sqrt(np.mean(wave ** 2)))
        # Do not confuse quiet Windows outputs with a YAMNet rejection:
        # 0.001 (-60dBFS) previously skipped all classifier inference, yet
        # the dashboard displayed a misleading 'music score = 0'. Genuine
        # silence still fails closed; threshold-based music-only filtering
        # remains in effect after this small level guard.
        if rms < 0.0001:
            return Decision(False, 0.0, 0.0, reason="input_too_quiet")
        shape = tuple(int(n) for n in self._input["shape"])
        if shape not in {(YAMNET_FRAME_SAMPLES,), (1, YAMNET_FRAME_SAMPLES)}:
            raise ListenerError(f"YAMNet 的输入形状 {shape} 不兼容，应该是 (15600,)；请选择官方 lite-model")
        rows = []
        for start in range(0, len(wave) - YAMNET_FRAME_SAMPLES + 1, YAMNET_HOP_SAMPLES):
            sub = wave[start:start + YAMNET_FRAME_SAMPLES]
            self._interpreter.set_tensor(self._input["index"], sub.reshape(shape))
            self._interpreter.invoke()
            row = np.asarray(self._interpreter.get_tensor(self._output_index), dtype=np.float32).reshape(-1)
            if row.size != 521 or not np.isfinite(row).all():
                return Decision(False, 0.0, 0.0, reason="invalid_model_output")
            rows.append(row)
        if not rows:
            return Decision(False, 0.0, 0.0)
        # Mean scores suppress occasional spikes from notification sounds.
        mean = np.stack(rows).mean(axis=0)
        music_score = float(max(mean[132], mean[211:263].max()))
        # Speech/Conversation/Narration/Speech-synth: don't mistakenly feed
        # normal speaking voice into song analysis (vocals may still register).
        speech_score = float(max(mean[0], mean[2], mean[3], mean[5]))
        music = music_score >= self._threshold and music_score >= speech_score * 0.75
        top_index = int(np.argmax(mean))
        return Decision(bool(music), round(music_score, 4), round(speech_score, 4),
                        reason="confirmed_music" if music else "classified_as_other_audio",
                        top_class_index=top_index,
                        top_class_score=round(float(mean[top_index]), 4))


class MusicClipCollector:
    """Emit ONLY music, including while a long song is *still playing*.

    The old implementation waited for three non-music windows (or the 15-minute
    cap) before emitting anything. For continuously playing music that meant
    the downstream AI never received a report. Now music is released in short,
    sequential, non-overlapping, confirmed-music segments. Silence, speech,
    rejected candidates, and trailing non-music windows are never emitted.

    All PCM lives only in memory until analyzed. Short classification dropouts
    can be tolerated without breaking the session, but the rejected audio
    itself is NEVER added to the music buffer. A session duration cap rotates
    the analysis timeline; it does not permanently stop continuous listening.
    """

    def __init__(self, *, min_music_windows: int = 2, end_nonmusic_windows: int = 3,
                 max_seconds: int = 900, min_clip_seconds: int = 8,
                 progress_seconds: int = 20, first_progress_seconds: int = 10):
        if min_music_windows < 1 or end_nonmusic_windows < 1:
            raise ValueError('invalid hysteresis')
        if (min_clip_seconds < 1 or progress_seconds < min_clip_seconds
                or first_progress_seconds < min_clip_seconds or max_seconds < min_clip_seconds):
            raise ValueError('invalid duration limits')
        self.min_music_windows = min_music_windows
        self.end_nonmusic_windows = end_nonmusic_windows
        self.max_samples = int(max_seconds * SAMPLE_RATE)
        self.min_samples = int(min_clip_seconds * SAMPLE_RATE)
        self.progress_samples = int(progress_seconds * SAMPLE_RATE)
        self.first_progress_samples = int(first_progress_seconds * SAMPLE_RATE)
        self._next_progress_samples = self.first_progress_samples
        self._candidate = deque(maxlen=min_music_windows)
        self._candidate_full = deque(maxlen=min_music_windows)
        self._music = []
        self._music_full = []
        self._full_band_rate = CAPTURE_RATE
        self._buffer_samples = 0
        self._total_samples = 0
        self._emitted_samples = 0
        self._nonmusic_count = 0
        self._music_windows = 0
        self._sequence = 0
        self._part_index = 1
        self.recording = False

    @property
    def buffered_seconds(self) -> float:
        """Unprocessed *confirmed* music, not unfiltered microphone audio."""
        return round(self._buffer_samples / SAMPLE_RATE, 2)

    @property
    def progress_percent(self) -> int:
        return min(100, round(self._buffer_samples / self._next_progress_samples * 100))

    @property
    def progress_target_seconds(self) -> float:
        return round(self._next_progress_samples / SAMPLE_RATE, 1)

    def _emit(self, reason: str, *, final: bool):
        import numpy as np
        count = self._buffer_samples
        start = self._emitted_samples / SAMPLE_RATE
        index = self._part_index
        sequence = self._sequence
        windows = self._music_windows
        data = np.concatenate(self._music).astype(np.float32, copy=False) if self._music else np.empty(0, dtype=np.float32)
        full_band = (np.concatenate(self._music_full).astype(np.float32, copy=False)
                     if len(self._music_full) == len(self._music) and self._music_full else None)
        self._music.clear()
        self._music_full.clear()
        self._buffer_samples = 0
        self._music_windows = 0
        self._emitted_samples += count
        self._part_index += 1
        self._next_progress_samples = self.first_progress_samples if final else self.progress_samples
        if final:
            self._candidate.clear()
            self._total_samples = 0
            self._emitted_samples = 0
            self._nonmusic_count = 0
            self.recording = False
        if count < self.min_samples:
            return None
        return FinishedClip(data, reason, windows, offset_s=start,
                            sequence=sequence, part_index=index,
                            full_band_samples=full_band,
                            full_band_rate=self._full_band_rate)

    def _end(self, reason: str):
        return self._emit(reason, final=True)

    def feed(self, pcm, is_music: bool, *, full_band=None,
             full_band_rate: int = CAPTURE_RATE) -> FinishedClip | None:
        import numpy as np
        if full_band is not None and full_band_rate not in (44100, 48000):
            raise ValueError('unsupported stereo capture rate')
        if full_band is not None and self.recording and full_band_rate != self._full_band_rate:
            # Never merge samples recorded at two different rates in one stem.
            self._music_full.clear()
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return None
        if not self.recording:
            if not is_music:
                self._candidate.clear()
                self._candidate_full.clear()
                return None
            self._candidate.append(samples.copy())
            self._full_band_rate = full_band_rate
            self._candidate_full.append(np.asarray(full_band, dtype=np.float32).copy()
                                        if full_band is not None else None)
            if len(self._candidate) < self.min_music_windows:
                return None
            self._sequence += 1
            self._part_index = 1
            self._next_progress_samples = self.first_progress_samples
            self._emitted_samples = 0
            self._music = list(self._candidate)
            self._music_full = (list(self._candidate_full)
                                if len(self._candidate_full) == len(self._candidate)
                                and all(x is not None for x in self._candidate_full) else [])
            self._buffer_samples = sum(part.size for part in self._music)
            self._total_samples = self._buffer_samples
            self._music_windows = len(self._candidate)
            self._candidate.clear()
            self._candidate_full.clear()
            self.recording = True
            self._nonmusic_count = 0
        elif is_music:
            # Append ONLY a YAMNet-confirmed music frame, not a rejected gap.
            self._music.append(samples.copy())
            if self._music_full:
                if full_band is None: self._music_full.clear()
                else: self._music_full.append(np.asarray(full_band,dtype=np.float32).copy())
            self._buffer_samples += samples.size
            self._total_samples += samples.size
            self._music_windows += 1
            self._nonmusic_count = 0
        else:
            self._nonmusic_count += 1
            # Do not even buffer rejected sound in memory. Hysteresis is
            # tracked by a small integer only, never by non-music PCM.
            if self._nonmusic_count >= self.end_nonmusic_windows:
                return self._end('non_music_boundary')
            return None
        # The cap rotates the music timeline instead of disabling live detection
        # for arbitrarily long playlists. Memory stays bounded by progress size.
        if self._total_samples >= self.max_samples:
            return self._end('duration_limit')
        if self._buffer_samples >= self._next_progress_samples:
            # Critically: don't stop the listener, wait for silence, or repeat
            # previously analyzed music. Emit this chunk and keep capturing.
            return self._emit('live_progress', final=False)
        return None

    def discard(self) -> None:
        self._candidate.clear()
        self._candidate_full.clear()
        self._music.clear()
        self._music_full.clear()
        self._buffer_samples = 0
        self._total_samples = 0
        self._emitted_samples = 0
        self._nonmusic_count = 0
        self._music_windows = 0
        self._part_index = 1
        self._next_progress_samples = self.first_progress_samples
        self.recording = False


class SoundCardLoopback:
    """Capture OS speaker output, never any hardware microphone.

    Windows: SoundCard initializes COM itself when `soundcard` is imported.
    Do NOT call CoInitializeEx before that import. In SoundCard 0.4.6 its
    `_COMLibrary.check_error` rejects S_FALSE (1, already initialized) and
    formats it as the bogus "0x100000001" error; `_COMLibrary.__del__` then
    fails because `com_loaded` was not set. The isolated Windows worker owns
    the audio thread/process and lets SoundCard initialize COM exactly once.
    Linux: monitor source usually has the speaker's ID plus '.monitor'.
    macOS: no built-in OS loopback source is available to this implementation.
    """

    def __init__(self, *, capture_rate: int = CAPTURE_RATE, capture_stereo: bool = False, output_device_id: str = ""):
        if capture_rate not in (44100, 48000):
            raise ValueError('仅支持 44100 或 48000 Hz 回环采集')
        self.capture_rate = capture_rate
        self.capture_stereo = capture_stereo
        self.output_device_id = str(output_device_id or "")[:512]
        self.last_stereo = None
        self.device_name = None
        self.stage = '初始化 SoundCard/WASAPI'

    def __enter__(self):
        if sys.platform == "darwin":
            raise ListenerError("当前 macOS 版本未实现可验证的扬声器回环采集；不允许自动改为真实麦克风")
        self._stack = ExitStack()
        try:
            # SoundCard owns COM initialization when its WASAPI module is
            # imported. Never pre-initialize COM here: if COM is already
            # initialized, CoInitializeEx returns S_FALSE (1), which
            # SoundCard 0.4.6 incorrectly treats as a fatal error.
            # The Windows listener runs in a fresh, isolated CPython process.
            try:
                import soundcard as sc
            except ImportError as exc:
                raise ListenerError("缺少 SoundCard 回环采集依赖，请先在 N.E.K.O 插件面板准备依赖") from exc
            self.stage = '枚举默认扬声器'
            if self.output_device_id:
                # Exact ID matching only; a renamed/disconnected endpoint
                # must NOT silently fall back to a microphone or other output.
                candidates = list(sc.all_speakers())
                speaker = next((s for s in candidates if str(getattr(s,'id','')) == self.output_device_id), None)
                if speaker is None:
                    raise ListenerError("指定的音频输出设备已不存在；请重新选择监听设备")
            else:
                speaker = sc.default_speaker()
            if speaker is None:
                raise ListenerError("系统找不到默认扬声器")
            self.stage = '枚举回环设备'
            monitors = sc.all_microphones(include_loopback=True)
            monitor = pick_speaker_monitor(speaker, monitors, sys.platform)
            if monitor is None:
                raise ListenerError("没有找到与默认扬声器匹配的系统回环设备；不会改用麦克风")
            try:
                self.device_name = str(getattr(speaker, 'name', '系统默认扬声器'))[:160]
            except (OSError, RuntimeError, AttributeError):
                # Device names are diagnostic-only and must not block capture.
                self.device_name = '系统默认扬声器'
            self.stage = '打开回环录音器'
            # The SoundCard recorder belongs to this isolated worker; COM
            # lifetime is handled by SoundCard itself at process exit.
            self._recorder = self._stack.enter_context(monitor.recorder(
                samplerate=self.capture_rate,
                channels=min(2, max(1, int(speaker.channels))),
                blocksize=4096,
            ))
            self.stage = '采集就绪'
            return self
        except BaseException as exc:
            self._stack.close()
            if isinstance(exc, ListenerError):
                raise
            raise ListenerError(f'{self.stage}失败：{type(exc).__name__}: {exc}') from exc

    def _record(self, seconds):
        import numpy as np
        frame = self._recorder.record(numframes=int(self.capture_rate * float(seconds)))
        if frame is None:
            raise ListenerError("回环设备未返回音频")
        stereo = np.asarray(frame, dtype=np.float32)
        if stereo.ndim != 2 or not 1 <= stereo.shape[1] <= 8:
            raise ListenerError("回环设备返回了无法识别的声道数据")
        if not len(stereo):
            raise ListenerError("回环音频缓冲区为空")
        return stereo

    def _stereo_of(self, stereo):
        import numpy as np
        if not self.capture_stereo:
            self.last_stereo = None
            return
        self.last_stereo = (stereo.copy() if stereo.shape[1] == 2
                            else np.repeat(stereo[:, :1], 2, axis=1).copy())

    def read(self):
        import numpy as np
        from scipy.signal import resample_poly
        import math
        stereo = self._record(WINDOW_SECONDS)
        self._stereo_of(stereo)
        mono = stereo.mean(axis=1)
        div = math.gcd(self.capture_rate, SAMPLE_RATE)
        return np.asarray(resample_poly(mono, SAMPLE_RATE // div, self.capture_rate // div), dtype=np.float32)

    def read_block(self, seconds):
        """读一小段（几十毫秒）原始采样率的单声道音频。

        Live2D 要跟着鼓点动，就不能等满 2 秒才拿到音频。改成小块读、
        每块算一次起音强度，动作延迟从 2 秒降到 0.1 秒左右。
        重活管线（识别/歌词）仍然用 read() 的 2 秒窗，互不影响。
        """
        import numpy as np
        stereo = self._record(seconds)
        self._stereo_of(stereo)
        return np.asarray(stereo.mean(axis=1), dtype=np.float32)

    def __exit__(self, exc_type, exc, tb):
        return self._stack.__exit__(exc_type, exc, tb)


def pick_speaker_monitor(speaker, candidates, platform: str):
    """Strictly match a real OS loopback source; reject ordinary microphones."""
    speaker_id = str(speaker.id)
    if platform == "win32":
        for item in candidates:
            if str(item.id) == speaker_id and getattr(item, "isloopback", False) is True:
                return item
    elif platform.startswith("linux"):
        for item in candidates:
            monitor_id = str(item.id)
            if monitor_id == speaker_id + ".monitor":
                return item
    return None


class RapidLyricsSampler:
    """Produce song-local early 6s / subsequent 10s lyric clips.

    This sampler never holds raw audio beyond its bounded window, never
    delivers rejected YAMNet frames, and cannot cross a detected song switch.
    Music rhythm/feature analysis still uses its original 10/20s intervals.
    """
    def __init__(self, first_seconds=6, later_seconds=10):
        if not 2 <= first_seconds <= later_seconds <= 30:
            raise ValueError('Invalid lyrics preview interval')
        self.first_samples = first_seconds * SAMPLE_RATE
        self.later_samples = later_seconds * SAMPLE_RATE
        self.reset()

    def reset(self):
        self._sequence = None
        self._frames = []
        self._full_frames = []
        self._full_ok = True
        self._pending = 0
        self._emitted = 0
        self._part = 0

    def feed(self, sequence, frames, *, full_band_frames=None, full_band_rate=CAPTURE_RATE):
        import numpy as np
        if sequence is None or not frames:
            return None
        if self._sequence != sequence:
            self.reset()
            self._sequence = sequence
        for i, frame in enumerate(frames):
            samples = np.asarray(frame, dtype=np.float32).reshape(-1)
            full = (full_band_frames[i] if full_band_frames and i < len(full_band_frames)
                    else None)
            if full is None: self._full_ok = False
            elif self._full_ok:
                full = np.asarray(full,dtype=np.float32)
                if full.ndim != 2 or full.shape[1] != 2: self._full_ok = False
                else: self._full_frames.append(full.copy())
            if samples.size:
                self._frames.append(samples.copy())
                self._pending += samples.size
        threshold = self.first_samples if self._part == 0 else self.later_samples
        if self._pending < threshold:
            return None
        values = np.concatenate(self._frames).astype(np.float32, copy=False)
        high = (np.concatenate(self._full_frames) if self._full_ok and self._full_frames
                and len(self._full_frames)==len(self._frames) else None)
        clip = FinishedClip(values, 'early_lyrics_preview', len(self._frames),
                            offset_s=self._emitted / SAMPLE_RATE,
                            sequence=sequence, part_index=self._part+1,
                            full_band_samples=high, full_band_rate=full_band_rate)
        self._part += 1
        self._emitted += self._pending
        self._frames = []
        self._full_frames = []
        self._full_ok = True
        self._pending = 0
        return clip


class LiveListener:
    """Dedicated thread + callbacks into the N.E.K.O async event loop."""

    def __init__(self, *, model_file: Path, max_seconds: int, threshold: float,
                 on_clip: Callable[[FinishedClip], None], source_factory=SoundCardLoopback,
                 gate_factory=YamnetMusicGate, progress_seconds: int = 20,
                 on_track_boundary=None, on_lyrics_preview=None,
                 scene_decider=None, on_video_preview=None, on_scene_change=None,
                 classifier_allowed=None):
        self._model_file = model_file
        self._max_seconds = max_seconds
        self._threshold = threshold
        self._progress_seconds = progress_seconds
        self._on_clip = on_clip
        self._on_lyrics_preview = on_lyrics_preview
        self._on_track_boundary = on_track_boundary
        self._scene_decider = scene_decider
        self._on_video_preview = on_video_preview
        self._on_scene_change = on_scene_change
        self._pending_track_change = threading.Event()
        self._onset_stop = threading.Event()
        self._audio_boundary_detector = SongChangeDetector()
        self._source_factory = source_factory
        self._gate_factory = gate_factory
        # When provided, NO classifier weights are loaded until AI/window
        # reviewed the unique audible app. Legacy direct callers use None.
        self._classifier_allowed = classifier_allowed
        self._stop = threading.Event()
        self._thread = None
        self._source = None  # active source; cancel can be invoked after consent is revoked
        self._state_lock = threading.Lock()
        self._state = {"status": "stopped", "last_decision": "尚未检测", "last_music_score": 0.0,
                       "last_gate_reason": "not_started", "last_top_class_index": None,
                       "last_top_class_score": 0.0,
                       "last_speech_score": 0.0, "music_clips_completed": 0, "error": None,
                       "capturing_music": False, "last_update_unix": None,
                       "playback_paused": False,
                       "frames_received": 0, "audio_seconds_received": 0.0,
                       "input_level_dbfs": -120.0, "pending_music_seconds": 0.0,
                       "segment_progress_pct": 0, "progress_target_seconds": 10,
                       "spectrum_bands_0_100": [0] * 16,
                       "audio_track_change_candidates":0,
                       "last_audio_boundary_evidence":None,
                       "scene_kind":"unknown", "scene_reason":"not_started",
                       "scene_verified":False, "scene_source_count":0,
                       "scene_blocked_windows":0,"video_previews_sent":0}

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def status(self):
        with self._state_lock:
            return dict(self._state)

    def _state_set(self, **kwargs):
        with self._state_lock:
            self._state.update(kwargs)
            self._state["last_update_unix"] = round(time.time(), 1)

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._state_set(status="starting", error=None)
        self._thread = threading.Thread(target=self._run, name="neko-wei-ta-suo-ting-loopback", daemon=True)
        self._thread.start()

    def mark_track_changed(self):
        """Called when a supported music player announces a different song.

        The audio capture thread owns the collector; never mutate it from an
        async SDK call thread. It will atomically drop the mixed boundary clip.
        """
        self._pending_track_change.set()

    def stop(self, *, timeout: float = 7):
        self._stop.set()
        # In the Windows isolated backend, terminate only our own child process
        # so a blocked audio pipe read unblocks before join().
        active = self._source
        if active is not None and hasattr(active, 'cancel'):
            active.cancel()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)
        if not self.running:
            self._state_set(status="stopped", capturing_music=False, playback_paused=False)
        return not self.running

    def _run(self):
        """Restart a failed capture backend, never a different audio source.

        A Windows WASAPI child can terminate when a player switches audio
        endpoints. Previously that left the UI authorized but with no live
        audio/ASR until the user disabled and re-enabled listening. Keep the
        same user-selected process/output and bound the number of retries.
        """
        retries = 0
        while not self._stop.is_set():
            self._run_once()
            if self._stop.is_set() or self.status().get('status') != 'error':
                break
            if retries >= 3:
                error = str(self.status().get('error') or '未知采集错误')
                self._state_set(status='error', capturing_music=False,
                    error='自动恢复失败（已重试3次）；请检查播放设备或重新选择应用。'+error[:300])
                break
            retries += 1
            self._state_set(status='recovering', capture_retries=retries,
                            capturing_music=False)
            # A failed per-app capture is retried only with the *same* verified
            # PID. Never switch to the system-wide speaker/microphone.
            if self._stop.wait(min(2.0 * retries, 6.0)):
                break

    def _start_onset_publisher(self, source):
        """把流式起音包络持续写进 state.json。

        以前 onset_env 是每 2 秒（读完一整窗）才算一次，动作天生慢两秒。
        现在采集进程每 50ms 就送一小包，这里每 90ms 把最近 2 秒的包络快照
        发出去，Live2D 那边 100ms 轮询一次，延迟只剩采集本身。
        """
        self._onset_stop.clear()
        if not hasattr(source, 'take_onsets'):
            return   # 直连 SoundCard 或测试替身：仍走每窗口一次的老路径

        def run():
            while not self._stop.is_set() and not self._onset_stop.is_set():
                snap = None
                try:
                    snap = source.take_onsets()
                except Exception:
                    snap = None
                if snap:
                    env, at = snap
                    if env:
                        self._state_set(
                            onset_env=env,
                            onset_hop=ONSET_HOP_SECONDS,
                            onset_at=round(at, 3),
                            onset_peak=round(max(env), 3),
                        )
                self._onset_stop.wait(0.09)

        threading.Thread(target=run, name='neko-onset-publisher', daemon=True).start()

    def _run_once(self):
        collector = MusicClipCollector(max_seconds=self._max_seconds, progress_seconds=self._progress_seconds)
        lyrics_sampler = RapidLyricsSampler() if self._on_lyrics_preview else None
        video_sampler = RapidLyricsSampler() if self._on_video_preview else None
        video_sequence = 0
        previous_scene = 'unknown'
        previous_video_source_key = ''
        previous_source_key = ''
        # 播放暂停识别：只认全零帧，避免把弱奏当成暂停。
        self._digital_silence_windows = 0
        self._playback_paused = False
        non_music_windows = 0
        full_candidates = deque(maxlen=collector.min_music_windows)
        try:
            # Open the lightweight capture transport immediately; do NOT
            # allocate the classification model until actual audible PCM arrives.
            gate = None
            source_object = self._source_factory()
            self._source = source_object
            with source_object as source:
                self._state_set(status="listening", device=str(getattr(source, 'device', '') or '')[:160])
                self._start_onset_publisher(source_object)
                while not self._stop.is_set():
                    wave = source.read()
                    if self._stop.is_set():
                        break
                    if self._pending_track_change.is_set():
                        self._pending_track_change.clear()
                        previous_sequence = collector._sequence
                        collector.discard()
                        full_candidates.clear()
                        if lyrics_sampler: lyrics_sampler.reset()
                        self._audio_boundary_detector.reset()
                        if self._on_track_boundary and previous_sequence:
                            self._on_track_boundary('track_ended', previous_sequence)
                        # This just-received window can straddle the old and
                        # new song. Drop it instead of feeding mixed lyrics to
                        # the first transcription of the next song.
                        continue
                    # An OS output can be silent for hours. Staying idle must
                    # not load even the lightweight YAMNet checkpoint, let alone
                    # Whisper, Qwen or the stem separator.
                    import numpy as np
                    pcm = np.asarray(wave, dtype=np.float32)
                    rms_for_gate = float(np.sqrt(np.mean(pcm * pcm))) if pcm.size else 0.0
                    # Live2D 律动用：25 ms 粒度的起音包络。
                    # 主循环每 2 秒才走一次，只靠能量没法跟上鼓点；把这一窗
                    # 细粒度的起音强度发出去，角色才能真正踩着拍子动，
                    # 而不是跑引擎里写死的 100 BPM。计算量很小（一次 reshape）。
                    try:
                        _hop = max(1, int(SAMPLE_RATE * 0.025))
                        _mono = pcm.reshape(-1)
                        _usable = (_mono.size // _hop) * _hop
                        if _usable >= _hop * 8:
                            _frames = _mono[:_usable].reshape(-1, _hop)
                            _env = np.sqrt(np.mean(_frames * _frames, axis=1))
                            _log = np.log10(np.maximum(_env, 1e-5))
                            _delta = np.diff(_log, prepend=_log[0])
                            _onset = np.clip(_delta * 2.5, 0.0, 1.0)
                            self._state_set(
                                onset_env=[round(float(v), 3) for v in _onset],
                                onset_hop=round(_hop / float(SAMPLE_RATE), 4),
                                onset_at=round(time.time(), 3),
                                onset_peak=round(float(_onset.max()) if _onset.size else 0.0, 3),
                            )
                    except Exception:
                        pass
                    # 播放器暂停时，逐进程采集仍然会送全零帧。此时停止识别、
                    # 保留当前歌曲，不把它计入“非音乐窗口”，恢复后继续同一首。
                    # 全零（而不是“安静”）才判暂停，正常歌曲的弱奏不会误判。
                    if rms_for_gate <= 1e-6:
                        self._digital_silence_windows = int(
                            getattr(self, '_digital_silence_windows', 0)) + 1
                    else:
                        self._digital_silence_windows = 0
                    playback_paused = (self._digital_silence_windows >= 3
                                       and previous_scene in {'music', 'video'})
                    if playback_paused:
                        if not getattr(self, '_playback_paused', False):
                            self._playback_paused = True
                            self._state_set(playback_paused=True,
                                            last_decision='播放器已暂停，等待继续播放',
                                            capturing_music=False,
                                            pending_music_seconds=0.0)
                        frames = self.status().get('frames_received', 0)
                        self._state_set(
                            frames_received=frames + 1,
                            audio_seconds_received=round((frames + 1) * WINDOW_SECONDS, 1),
                            input_level_dbfs=-120.0)
                        continue
                    if getattr(self, '_playback_paused', False):
                        self._playback_paused = False
                        self._state_set(playback_paused=False,
                                        last_decision='播放器继续播放')
                    try:
                        reviewed = (self._classifier_allowed() if self._classifier_allowed
                                    else True)
                    except Exception:
                        reviewed = False  # Never start inference on failed verification.
                    if not reviewed:
                        gate = None  # Release the classifier when a reviewed label is revoked.
                        result = Decision(False, 0.0, 0.0, reason='awaiting_ai_or_window_review')
                    elif gate is None and rms_for_gate < 0.0001:
                        result = Decision(False, 0.0, 0.0, reason='input_too_quiet')
                    else:
                        if gate is None:
                            gate = self._gate_factory(self._model_file, min_score=self._threshold)
                        result = gate.classify(wave)
                    # Verified playback type and explicit TTS status are checked
                    # BEFORE buffering any PCM, so a mixed window never reaches
                    # either the song-ASR or the private AI channel.
                    try:
                        scene = (self._scene_decider(result) if self._scene_decider
                                 else {'kind':'music',
                                       'reason':'legacy_yamnet_only','verified':False,
                                       'source_count':0})
                    except Exception:
                        # Any classifier or metadata failure closes the gate.
                        scene = {'kind':'unknown','reason':'source_check_failed',
                                 'verified':False,'source_count':0}
                    scene_kind = str(scene.get('kind','unknown'))
                    if scene_kind not in {'music','video','mixed','unknown','ai_output'}:
                        scene_kind = 'unknown'
                    video_key = (str(scene.get('source_key') or '')
                                 if scene_kind == 'video' else '')
                    scene_source_key = str(scene.get('source_key') or '')
                    non_music_windows = (0 if scene_kind == 'music'
                                         else non_music_windows + 1)
                    source_changed = bool(previous_source_key) and (
                        scene_source_key != previous_source_key)
                    hard_block = scene_kind in {'mixed', 'ai_output'}
                    # A music video's acoustic scene flips between 'music',
                    # 'video' and the occasional 'unknown' while its speech
                    # score hovers around the 0.10 threshold. Treating every
                    # 2 s flicker as a track boundary discarded the music
                    # collector before it could ever fill (9 windows), so no
                    # clip was emitted and NO lyric was ever produced — the
                    # song id even advanced every few seconds. Only a real
                    # source change, a hard block, or a genuinely long silence
                    # may end the song.
                    reset_collector = (source_changed or hard_block or
                                       non_music_windows > NON_MUSIC_RESET_WINDOWS)
                    if (scene_kind != previous_scene or
                            (scene_kind == 'video' and video_key != previous_video_source_key)):
                        if reset_collector:
                            prior = collector._sequence if collector.recording else None
                            collector.discard()
                            full_candidates.clear()
                            if lyrics_sampler: lyrics_sampler.reset()
                            if video_sampler: video_sampler.reset()
                            self._audio_boundary_detector.reset()
                            if prior and self._on_track_boundary:
                                self._on_track_boundary('track_ended',prior)
                        if scene_kind == 'video':
                            video_sequence += 1
                        if self._on_scene_change:
                            try:self._on_scene_change(scene)
                            except Exception:pass
                        previous_scene = scene_kind
                        previous_video_source_key = video_key
                        if scene_source_key:
                            previous_source_key = scene_source_key
                    elif self._on_scene_change:
                        # Refresh scene state on every window, not only on a
                        # class change: otherwise the panel and the AI keep a
                        # stale 'not_started' reason while the label arrives.
                        try:self._on_scene_change(scene)
                        except Exception:pass
                    self._state_set(
                        scene_kind=scene_kind,
                        scene_reason=str(scene.get('reason','unknown'))[:150],
                        scene_verified=bool(scene.get('verified',False)),
                        scene_source_count=int(scene.get('source_count',0)),
                    )
                    if scene_kind != 'music':
                        # Music and video need separate pipelines. Video-only
                        # audio is sent to its own ASR queue, not music DSP.
                        if (scene_kind == 'video' and video_sampler and
                                result.reason not in {'input_too_quiet','invalid_pcm'}):
                            preview = video_sampler.feed(video_sequence,[wave])
                            if preview is not None and not self._stop.is_set():
                                preview.end_reason='video_speech_preview'
                                preview.capture_end_monotonic_s=time.monotonic()
                                try:
                                    self._on_video_preview(preview)
                                    self._state_set(video_previews_sent=self.status().get('video_previews_sent',0)+1)
                                except Exception:pass
                        if scene_kind in {'mixed','ai_output'}:
                            self._state_set(scene_blocked_windows=self.status().get('scene_blocked_windows',0)+1)
                        # Preserve the capture heartbeat while excluding PCM.
                        import numpy as np
                        rms=float(np.sqrt(np.mean(np.asarray(wave,dtype=np.float32)**2)))
                        frames=self.status().get('frames_received',0)
                        self._state_set(frames_received=frames+1,
                                        audio_seconds_received=round((frames+1)*WINDOW_SECONDS,1),
                                        input_level_dbfs=round(20*np.log10(max(rms,1e-10)),1),
                                        last_decision='视频' if scene_kind=='video' else
                                                     '混合/AI语音：识别已暂停' if scene_kind in {'mixed','ai_output'} else
                                                     '音源待确认',
                                        last_music_score=result.music_score,
                                        last_speech_score=result.speech_score,
                                        last_gate_reason=scene.get('reason','unverified'),
                                        capturing_music=False,
                                        pending_music_seconds=0.0)
                        continue
                    # A player with a media session gives stronger song identity
                    # evidence; this sound-only heuristic is intentionally more
                    # conservative. Two adjacent confirmed windows must show
                    # pronounced combined rhythm, pitch and timbre novelty.
                    evidence = self._audio_boundary_detector.feed(wave,is_music=result.music)
                    if evidence is not None and collector.recording:
                        prior=collector._sequence
                        collector.discard()
                        full_candidates.clear()
                        if lyrics_sampler: lyrics_sampler.reset()
                        last_count=self.status().get('audio_track_change_candidates',0)
                        self._state_set(audio_track_change_candidates=last_count+1,
                                        last_audio_boundary_evidence=evidence)
                        if self._on_track_boundary and prior:
                            self._on_track_boundary('track_ended_audio',prior)
                        # This 2s window may span two songs, so drop it entirely.
                        continue
                    previous_sequence = collector._sequence
                    source_full = getattr(source,'last_stereo',None)
                    full_rate = int(getattr(source,'capture_rate',CAPTURE_RATE))
                    if not collector.recording:
                        if result.music: full_candidates.append(source_full)
                        else: full_candidates.clear()
                    clip = collector.feed(wave, result.music, full_band=source_full,
                                          full_band_rate=full_rate)
                    if lyrics_sampler:
                        if (clip is not None and clip.end_reason in
                                {'non_music_boundary', 'duration_limit'}):
                            lyrics_sampler.reset()
                            full_candidates.clear()
                        elif result.music and collector.recording:
                            # At first confirmation the collector contains both
                            # music-confirmed YAMNet windows; include both once.
                            frames = (collector._music if collector._sequence != previous_sequence
                                      else [wave])
                            full_frames = (list(full_candidates)
                                           if collector._sequence != previous_sequence
                                           else [source_full])
                            preview = lyrics_sampler.feed(collector._sequence, frames,
                                                          full_band_frames=full_frames,
                                                          full_band_rate=full_rate)
                            if preview is not None and not self._stop.is_set():
                                preview.capture_end_monotonic_s = time.monotonic()
                                try:
                                    self._on_lyrics_preview(preview)
                                except Exception:
                                    # Preview is optional: never break the WASAPI
                                    # capture loop because ASR is overloaded.
                                    pass
                    if (self._on_track_boundary and collector._sequence != previous_sequence
                            and collector.recording):
                        self._on_track_boundary('track_started', collector._sequence)
                    if (self._on_track_boundary and clip is not None
                            and clip.end_reason == 'non_music_boundary'):
                        # Invalidate prior song before delivering its short tail.
                        self._on_track_boundary('track_ended', clip.sequence)
                    import numpy as np
                    rms = float(np.sqrt(np.mean(np.asarray(wave, dtype=np.float32) ** 2)))
                    # Tiny 16-band telemetry preview, not a raw audio feed.
                    # It proves the OS loopback is receiving real sound, even
                    # when the music gate rejects a speech/noise window.
                    bands = []
                    if rms >= 0.0003:
                        subsampled = np.asarray(wave, dtype=np.float32)[::4][:8192]
                        fft = np.abs(np.fft.rfft(subsampled * np.hanning(len(subsampled))))
                        freqs = np.fft.rfftfreq(len(subsampled), d=4 / SAMPLE_RATE)
                        edges = np.geomspace(60, 7900, 17)
                        strengths = []
                        for low, high in zip(edges[:-1], edges[1:]):
                            segment = fft[(freqs >= low) & (freqs < high)]
                            strengths.append(float(np.sqrt(np.mean(segment**2))) if segment.size else 0.0)
                        peak = max(strengths) or 1.0
                        bands = [round(min(100, 100 * value / peak)) for value in strengths]
                    else:
                        bands = [0] * 16
                    previous_frames = self.status()["frames_received"]
                    self._state_set(
                        last_decision="音乐" if result.music else "非音乐 / 不确定",
                        last_music_score=result.music_score,
                        last_speech_score=result.speech_score,
                        last_gate_reason=getattr(result, 'reason', 'unknown'),
                        last_top_class_index=getattr(result, 'top_class_index', None),
                        last_top_class_score=getattr(result, 'top_class_score', 0.0),
                        frames_received=previous_frames + 1,
                        audio_seconds_received=round((previous_frames + 1) * WINDOW_SECONDS, 1),
                        input_level_dbfs=round(max(-120.0, 20 * __import__('math').log10(max(rms, 1e-6))), 1),
                        pending_music_seconds=collector.buffered_seconds,
                        segment_progress_pct=collector.progress_percent,
                        progress_target_seconds=collector.progress_target_seconds,
                        spectrum_bands_0_100=bands,
                        capturing_music=collector.recording,
                    )
                    if clip is not None and not self._stop.is_set():
                        clip.capture_end_monotonic_s = time.monotonic()
                        self._on_clip(clip)
                        with self._state_lock:
                            count = self._state["music_clips_completed"] + 1
                        self._state_set(music_clips_completed=count)
        except Exception as exc:
            self._state_set(status="error", error=f"监听不可用：{type(exc).__name__}: {str(exc)[:480]}")
        finally:
            self._onset_stop.set()
            # A capture crash must close the current song's identity before a
            # restarted collector reuses sequence 1. Otherwise the new track
            # may be mistaken for the old and its lyrics rejected as stale.
            ended_sequence = collector._sequence if collector.recording else None
            if ended_sequence and self._on_track_boundary:
                try:
                    self._on_track_boundary('track_ended', ended_sequence)
                except Exception:
                    pass
            self._source = None
            if lyrics_sampler: lyrics_sampler.reset()
            collector.discard()  # no incomplete audio on disk or sent to the AI
            if self._stop.is_set():
                self._state_set(status="stopped", capturing_music=False, playback_paused=False)
            else:
                # An unexpected exit can never be represented as listening.
                self._state_set(capturing_music=False)
