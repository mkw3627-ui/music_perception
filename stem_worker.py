"""Isolated ONNX MDX stem worker. PCM exists only in local anonymous pipes."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import traceback
import types

LEN = struct.Struct('!I')
MAX_HEADER = 4096
MAX_INPUT = 32 * 48000 * 2 * 4


def exact(stream, size):
    blocks = bytearray()
    while len(blocks) < size:
        chunk = stream.read(size-len(blocks))
        if not chunk:
            return None
        blocks.extend(chunk)
    return bytes(blocks)


def send(info, pcm=b''):
    header = json.dumps(info, ensure_ascii=False, separators=(',',':')).encode('utf-8')
    sys.stdout.buffer.write(LEN.pack(len(header)))
    sys.stdout.buffer.write(header)
    if pcm:
        sys.stdout.buffer.write(pcm)
    sys.stdout.buffer.flush()


def main():
    try:
        if len(sys.argv) not in (3,4):
            raise RuntimeError('invalid local stem worker arguments')
        vendor, weight = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
        if not vendor.is_dir():
            raise RuntimeError('managed environment missing')
        sys.path.insert(0, str(vendor))
        # A verified plugin-private GPU wheel shadows the CPU ORT only in this
        # isolated music separator, not in lyrics/YAMNet or N.E.K.O.
        if len(sys.argv)==4 and sys.argv[3]!='-':
            from pathlib import Path as _P
            import json as _json
            overlay=_P(sys.argv[3]).resolve()
            sys.path.insert(0,str(Path(__file__).resolve().parent))
            from stem_gpu_runtime import GPU_WHEEL, GPU_WHEEL_SHA256, MARKER
            marker=_json.loads((overlay/MARKER).read_text(encoding='utf-8'))
            if marker.get('wheel')!=GPU_WHEEL or marker.get('sha256')!=GPU_WHEEL_SHA256:
                raise RuntimeError('invalid GPU overlay marker')
            sys.path.insert(0,str(overlay))
        package = types.ModuleType('music_perception')
        package.__path__ = [str(Path(__file__).resolve().parent)]
        package.__package__ = 'music_perception'
        sys.modules['music_perception'] = package
        import numpy as np
        from music_perception.stem_cuda_bundle import cuda_runtime_path, configure_private_cuda
        from music_perception.stem_mdx import TwoStemMDX, StemError, StemGpuUnavailable
        from music_perception.stem_gpu_runtime import overlay_path
        # A shared Windows CUDA toolkit is NOT an accepted acceleration source.
        # Both binary layers have to be installed under this plugin's data path.
        env_root = (overlay.parent.parent if len(sys.argv)==4 and sys.argv[3]!='-'
                    else vendor.parent)
        onnx_private = overlay_path(env_root)
        cuda_private = cuda_runtime_path(env_root)
        gpu_runtime_ready = (onnx_private is not None and cuda_private is not None)
        cuda_prepare_error = None
        if gpu_runtime_ready:
            try:
                configure_private_cuda(cuda_private)
            except (OSError, RuntimeError) as exc:
                cuda_prepare_error = f'{type(exc).__name__}: {str(exc)[:200]}'
                gpu_runtime_ready = False
        # No SDK or N.E.K.O main process imported here.
        model = None
        loaded_for_gpu = None
        failed_gpu_reason = None
        while True:
            hdr = exact(sys.stdin.buffer, 4)
            if hdr is None: break
            size, = LEN.unpack(hdr)
            if not 1 <= size <= MAX_HEADER: raise RuntimeError('oversized stem header')
            request = json.loads(exact(sys.stdin.buffer, size).decode('utf-8'))
            job = int(request.get('id', 0))
            frames = int(request.get('frames', 0))
            rate = int(request.get('rate', 0))
            if request.get('cmd') != 'split' or rate not in {16000,44100,48000} or frames < rate or frames > rate*32:
                send({'id':job, 'ok':False, 'error':'invalid music clip metadata'})
                continue
            pcm = exact(sys.stdin.buffer, frames * 2 * 4)
            if pcm is None: break
            try:
                wants_gpu=bool(request.get('cuda',False))
                cpu_allowed=bool(request.get('cpu_fallback_allowed',True))
                # When switching CPU/GPU, do not reuse a session created for the
                # other device. GPU failures are cached for this worker only,
                # preventing an expensive CUDA initialization on every clip.
                if model is None or loaded_for_gpu != wants_gpu:
                    model=None
                    loaded_for_gpu=wants_gpu
                    if wants_gpu and not gpu_runtime_ready:
                        if not cpu_allowed:
                            raise StemGpuUnavailable('GPU 不可用。CPU 内存回退尚未获得用户授权')
                        failed_gpu_reason = (cuda_prepare_error or
                            '插件专用 CUDA/cuDNN/ONNX GPU 环境未完整安装；已使用 CPU 兜底')
                        model=TwoStemMDX(weight,request_gpu=False)
                    elif wants_gpu and failed_gpu_reason:
                        if not cpu_allowed:raise StemGpuUnavailable('CUDA 故障；需先允许 CPU 内存运行')
                        model=TwoStemMDX(weight,request_gpu=False)
                    else:
                        try:
                            model=TwoStemMDX(weight,request_gpu=wants_gpu)
                        except StemGpuUnavailable as exc:
                            if not cpu_allowed:raise
                            failed_gpu_reason=str(exc)[:300]
                            model=TwoStemMDX(weight,request_gpu=False)
                data = np.frombuffer(pcm, dtype='<f4').reshape((frames,2))
                try:
                    result = model.separate(data, rate,strength=str(request.get('strength','balanced')))
                except StemGpuUnavailable as exc:
                    if not wants_gpu or not cpu_allowed:raise
                    failed_gpu_reason=str(exc)[:300]
                    model=TwoStemMDX(weight,request_gpu=False)
                    result=model.separate(data,rate,strength=str(request.get('strength','balanced')))
                voice = result['vocals'].astype('<f4', copy=False)
                enhanced = result['enhanced_vocals'].astype('<f4', copy=False)
                other = result['instrumental'].astype('<f4', copy=False)
                send({'id':job, 'ok':True, 'voice_frames':len(voice),
                      'enhanced_frames':len(enhanced), 'instrument_frames':len(other),
                      'device':result['device'], 'model':result['model'],
                      'gpu_fallback_reason':failed_gpu_reason if wants_gpu and result['device']!='cuda' else None,
                      'warning':result['warning']},
                     voice.tobytes()+enhanced.tobytes()+other.tobytes())
            except Exception as exc:
                # Never log samples / lyrics. A failing model must not take
                # down the rest of the plugin or overwrite Whisper models.
                print('stem_failure:',type(exc).__name__,str(exc)[:200],file=sys.stderr)
                send({'id':job, 'ok':False, 'error':f'{type(exc).__name__}: {str(exc)[:180]}'})
    except Exception as exc:
        print('stem_worker_failure:',type(exc).__name__,str(exc)[:200],file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
