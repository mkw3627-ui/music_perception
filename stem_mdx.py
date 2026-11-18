"""Opt-in, fully local UVR MDX-Net 2-stem separation (no torch).

Model: UVR-MDX-NET-Inst_HQ_3.onnx (instrumental-primary).
The model sees genuine 44.1 kHz stereo when available; it predicts the
instrumental, and vocals are the mixture minus that predicted instrumental.

This file is only imported inside a locally managed isolated worker. Inference
has NO network access, never writes captured PCM to disk, and cannot affect
speaker capture if the model is unavailable. Neither stem is mathematically
perfect: vocals can bleed into accompaniment and vice versa.
"""
from __future__ import annotations

from pathlib import Path

from .stem_catalog import STEM_PROFILES, verify_local_model

MODEL_FILENAME = 'UVR-MDX-NET-Inst_HQ_3.onnx'
MODEL_SHA256 = '317554b07fe1ea5279a77f2b1520a41ea4b93432560c4ffd08792c30fddf9adc'
MODEL_SOURCE = ('https://huggingface.co/Politrees/UVR_resources/resolve/'
                '495f5b4625eb092fe1fbe336f0b67c847566e52e/'
                'MDXNet_models/' + MODEL_FILENAME + '?download=true')
MODEL_ALT_SOURCE = ('https://huggingface.co/seanghay/uvr_models/resolve/main/'
                    + MODEL_FILENAME + '?download=true')
RATE = 44100
FFT = 6144
HOP = 1024
DIM_F = 3072
DIM_T = 256
CHUNK = HOP * (DIM_T - 1)
TRIM = FFT // 2
STRIDE = CHUNK - 2 * TRIM
# Inst HQ 3 model settings (UVR MDX-Net config); small gain compensation is
# applied to the predicted instrumental before calculating the vocal residual.
COMPENSATION = 1.022


class StemError(RuntimeError):
    pass

class StemGpuUnavailable(StemError):
    """GPU was explicitly requested, but no working CUDA execution provider exists."""
    pass


def verify_model(path: str | Path) -> bool:
    """Only allow published weights, verified by full SHA-256."""
    record=next((k for k,v in STEM_PROFILES.items() if v['file']==Path(path).name),None)
    return bool(record and verify_local_model(path,record))


def stft_channels(audio, *, fft=FFT, dim_f=DIM_F):
    """Torch.stft-equivalent centered reflect-pad, periodic Hann, NCHW."""
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32)
    if audio.shape != (CHUNK, 2):
        raise StemError('MDX-Net input must have 261120 stereo frames')
    if fft%2 or dim_f>fft//2+1 or fft<2048:
        raise StemError("不支持的 MDX 频谱配置")
    trim=fft//2
    window = np.hanning(fft + 1)[:-1].astype(np.float32)
    parts = []
    for channel in range(2):
        padded = np.pad(audio[:, channel], (trim, trim), mode='reflect')
        slices = np.lib.stride_tricks.sliding_window_view(padded, fft)[::HOP]
        if slices.shape[0] != DIM_T:
            raise StemError('MDX frame count mismatch')
        bins = np.fft.rfft(slices * window, n=fft, axis=-1).T[:dim_f]
        parts.extend((bins.real.astype(np.float32), bins.imag.astype(np.float32)))
    return np.stack(parts, axis=0)[None]


def istft_channels(model_output, *, fft=FFT, dim_f=DIM_F):
    """Inverse MDX-Net output with normalized Hann overlap-add."""
    import numpy as np
    pred = np.asarray(model_output, dtype=np.float32)
    if pred.shape != (1, 4, dim_f, DIM_T) or not np.isfinite(pred).all():
        raise StemError(f'MDX output shape/data invalid: {pred.shape}')
    trim=fft//2
    window = np.hanning(fft + 1)[:-1].astype(np.float32)
    length = fft + HOP * (DIM_T - 1)
    scale = np.zeros(length, dtype=np.float64)
    for i in range(DIM_T):
        scale[i*HOP:i*HOP+fft] += window.astype(np.float64)**2
    result = np.empty((CHUNK, 2), dtype=np.float32)
    for channel in range(2):
        complex_spec = (pred[0,2*channel].astype(np.float64)
                        + 1j * pred[0,2*channel+1].astype(np.float64))
        full = np.zeros((fft//2+1, DIM_T), dtype=np.complex128)
        full[:dim_f] = complex_spec
        waves = np.fft.irfft(full, n=fft, axis=0).T
        overlap = np.zeros(length, dtype=np.float64)
        for i in range(DIM_T):
            overlap[i*HOP:i*HOP+fft] += waves[i] * window
        overlap /= np.maximum(scale, 1e-8)
        result[:,channel] = overlap[trim:trim+CHUNK].astype(np.float32)
    return result


class TwoStemMDX:
    """Lazy-load one ONNX session and reuse it; CPU is the safe default."""
    def __init__(self, model_file, *, request_gpu: bool = False, session_factory=None, profile_dir=None):
        self.model_file = Path(model_file)
        self.model_record = next((x for x in STEM_PROFILES.values()
                                  if x['file'] == self.model_file.name), STEM_PROFILES['ultra'])
        if self.model_record is None or not verify_model(self.model_file):
            raise StemError('分离模型未安装或损坏，请在插件设置页确认自动下载并校验')
        if session_factory is None:
            import onnxruntime as ort
            if request_gpu and hasattr(ort, 'preload_dlls'):
                # ORT GPU >= 1.21 can locate compatible CUDA/cuDNN DLLs without
                # changing system PATH or the host N.E.K.O process.
                try:
                    import contextlib
                    import sys
                    # Native ORT may print warnings. On the binary stem-worker
                    # protocol stdout is RESERVED for framed PCM responses.
                    with contextlib.redirect_stdout(sys.stderr):
                        try:
                            ort.preload_dlls(directory='')
                        except TypeError:
                            ort.preload_dlls()
                except (OSError, RuntimeError) as exc:
                    self.gpu_preload_warning=f'{type(exc).__name__}: {str(exc)[:160]}'
            providers = ort.get_available_providers()
            if request_gpu and 'CUDAExecutionProvider' not in providers:
                raise StemGpuUnavailable(
                    f'CUDAExecutionProvider 不可用；ONNX Runtime {ort.__version__} '
                    f'仅检测到 {providers}，请安装并检测分离专用 GPU 运行包')
            selected = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                        if request_gpu else ['CPUExecutionProvider'])
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Only the explicit GPU self-check profiles operators. Never write
            # any capture content or a profiling trace during normal listening.
            if profile_dir is not None:
                options.enable_profiling = True
                options.profile_file_prefix = str(Path(profile_dir) / 'stem_cuda_probe')
            try:
                session = ort.InferenceSession(str(self.model_file), sess_options=options, providers=selected)
            except Exception as exc:
                if request_gpu:
                    raise StemGpuUnavailable('CUDA 模型会话创建失败：'+f'{type(exc).__name__}: {str(exc)[:240]}') from exc
                raise
        else:
            session = session_factory(str(self.model_file))
        if request_gpu and hasattr(session, 'get_providers') and 'CUDAExecutionProvider' not in session.get_providers():
            raise StemGpuUnavailable('GPU 已选择，但 ONNX Runtime 创建模型会话时回退 CPU（可能缺 CUDA/cuDNN DLL）')
        if request_gpu and hasattr(session, 'disable_fallback'):
            session.disable_fallback()  # do not silently switch GPU execution to CPU
        self.request_gpu=bool(request_gpu)
        self.session = session
        self.fft=int(self.model_record['fft'])
        self.dim_f=int(self.model_record['dim_f'])
        self.compensation=float(self.model_record['compensation'])
        self.trim=self.fft//2
        self.stride=CHUNK-2*self.trim
        try:
            inp = session.get_inputs()[0]
            self.input_name = inp.name
            if list(inp.shape[-3:]) != [4, self.dim_f, DIM_T]:
                raise StemError(f'MDX model dimensions incompatible: {inp.shape}')
        except (AttributeError, IndexError, TypeError) as exc:
            raise StemError('The ONNX file does not contain expected MDX input') from exc
        active = session.get_providers() if hasattr(session, 'get_providers') else ['CPUExecutionProvider']
        self.device = 'cuda' if 'CUDAExecutionProvider' in active else 'cpu'

    def separate(self, pcm, sample_rate: int, *, strength='balanced'):
        from math import gcd

        import numpy as np
        from scipy.signal import resample_poly
        if type(sample_rate) is not int or sample_rate not in (16000, 44100, 48000):
            raise StemError('Unsupported stem input sample rate')
        data = np.asarray(pcm, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] < sample_rate or data.shape[0] > sample_rate*32:
            raise StemError('Stem data must be bounded 1–32s stereo')
        if not np.isfinite(data).all():
            raise StemError('Invalid stem PCM')
        if sample_rate != RATE:
            div = gcd(sample_rate, RATE)
            mix = resample_poly(data, RATE//div, sample_rate//div, axis=0).astype(np.float32)
        else:
            mix = data.copy()
        sample_count = len(mix)
        predicted = np.empty_like(mix)
        # Never silently discard the last partial chunk. Padding and de-padding
        # preserve a strict sample-accurate timeline across the two stems.
        for start in range(0, sample_count, self.stride):
            segment = mix[start:start+self.stride]
            padded = np.zeros((CHUNK, 2), dtype=np.float32)
            # A short guard on either side controls STFT edge artefacts.
            left = max(0, start-self.trim)
            right = min(sample_count, start+self.stride+self.trim)
            framed = mix[left:right]
            insert_at = self.trim - (start-left)
            padded[insert_at:insert_at+len(framed)] = framed
            input_spec = stft_channels(padded,fft=self.fft,dim_f=self.dim_f)
            try:
                instrument_spec = self.session.run(None, {self.input_name: input_spec})[0]
            except Exception as exc:
                if self.request_gpu:
                    raise StemGpuUnavailable('CUDA 模型实际推理失败：'+f'{type(exc).__name__}: {str(exc)[:220]}') from exc
                raise
            if self.request_gpu and hasattr(self.session,'get_providers') and 'CUDAExecutionProvider' not in self.session.get_providers():
                raise StemGpuUnavailable('CUDA Provider 在实际推理期间被卸载，禁止误报 GPU 已运行')
            instrument_audio = istft_channels(instrument_spec,fft=self.fft,dim_f=self.dim_f)
            count = len(segment)
            predicted[start:start+count] = instrument_audio[self.trim:self.trim+count]
        if self.model_record['output'] == 'vocals':
            predicted *= self.compensation
            vocals = np.clip(predicted, -1.5, 1.5)
            instrumental = np.clip(mix - vocals, -1.5, 1.5)
        else:
            predicted *= self.compensation
            instrumental = np.clip(predicted, -1.5, 1.5)
            vocals = np.clip(mix - instrumental, -1.5, 1.5)
        def mono16(stereo):
            mixed = stereo.mean(axis=1).astype(np.float32)
            return resample_poly(mixed, 160, 441, axis=0).astype(np.float32)
        raw_voice = mono16(vocals)
        raw_instrumental = mono16(instrumental)
        from .stem_postprocess import enhance_vocals
        enhanced = enhance_vocals(raw_voice, raw_instrumental, rate=16000, strength=strength)
        return {
            'vocals': raw_voice,
            'enhanced_vocals': enhanced,
            'instrumental': raw_instrumental,
            'device': self.device, 'model': self.model_file.name,
            'source_rate': sample_rate, 'separated_rate': RATE,
            'warning': '分离与波形增强均为估计，请试听对比；过度增强会损伤人声',
        }
