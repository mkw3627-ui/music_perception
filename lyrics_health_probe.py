"""On-demand, isolated local ASR self-check. Never records or transmits sound.

A GPU model preflight uses one second of generated silence, ONLY when the
plugin is idle. It never downloads weights or repairs system drivers.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def _save(path, payload):
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    except (OSError, ValueError):
        pass


def run_probe(vendor, model_dir, model_name, requested_device, report_file,
              *, deep=False, module_loader=importlib.import_module):
    """Return low-sensitivity system diagnostics. No file paths in report."""
    started = time.monotonic()
    out = {'schema':'lyrics_full_diagnostics_v1', 'state':'running',
           'stage':'checking_python', 'requested_device':requested_device,
           'model':model_name, 'deep_requested':bool(deep), 'steps':[],
           'python_version':platform.python_version(),
           'python_64bit':sys.maxsize > 2**32}
    def step(stage, status, note=None):
        row={'stage':stage,'status':status}
        if note:
            row['note']=str(note)[:170]
        out['steps'].append(row)
        out['stage']=stage
        out['elapsed_seconds']=round(time.monotonic()-started,2)
        _save(report_file,out)
    missing_dependencies = []
    ctranslate2_import_ok = False
    if requested_device == 'cuda':
        try:
            gpu_query=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.free',
                         '--format=csv,noheader,nounits'],capture_output=True,
                         text=True,timeout=4,check=False)
            if gpu_query.returncode==0:
                first=(gpu_query.stdout or '').splitlines()
                fields=first[0].split(',') if first else []
                if len(fields)>=3:
                    out['gpu_name']=fields[0].strip()[:100]
                    out['nvidia_driver']=fields[1].strip()[:40]
                    try: out['gpu_free_mib']=int(fields[2].strip())
                    except ValueError: pass
                    step('nvidia_driver','ok','驱动与显存信息已读取（不代表 CUDA 推理成功）')
            else:
                step('nvidia_driver','unknown','nvidia-smi 不可用；不代表一定没有 NVIDIA 显卡')
        except (OSError,subprocess.TimeoutExpired):
            step('nvidia_driver','unknown','无法读取 nvidia-smi，仍会尝试 CTranslate2 检测')
    if not Path(vendor).is_dir():
        step('python_environment','failed','插件隔离依赖目录不可用')
        out['state']='failed';_save(report_file,out);return out
    # Metadata queries do not import the entire DSP stack or start CUDA.
    names = {'numpy':'numpy', 'faster-whisper':'faster-whisper',
             'ctranslate2':'ctranslate2', 'tokenizers':'tokenizers',
             'soundfile':'soundfile', 'scipy':'scipy',
             'librosa':'librosa', 'SoundCard':'SoundCard',
             'ai-edge-litert':'ai-edge-litert', 'imageio-ffmpeg':'imageio-ffmpeg'}
    for label, package in names.items():
        try:
            v=importlib.metadata.version(package)
            step('dependency_'+label,'ok',v)
        except importlib.metadata.PackageNotFoundError:
            missing_dependencies.append(label)
            step('dependency_'+label,'missing')
    try:
        ct2=module_loader('ctranslate2')
        ctranslate2_import_ok = True
        step('ctranslate2_import','ok')
        if requested_device == 'cuda':
            count=int(ct2.get_cuda_device_count())
            types=sorted(str(x) for x in ct2.get_supported_compute_types('cuda')) if count else []
            out['cuda_device_count']=count
            out['cuda_compute_types']=types[:12]
            out['cuda_candidate']=count>0 and 'float16' in types
            step('cuda_probe','ok' if out['cuda_candidate'] else 'unavailable',
                 '检测 CUDA 设备与 FP16，不能代替实际模型推理')
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
        out['cuda_candidate']=False
        step('ctranslate2_or_cuda','failed',type(exc).__name__)
    # Check cached weights quickly. SHA check of >1.6GB weights belongs to
    # existing model verifier and must not run during every UI refresh.
    model_root=Path(model_dir)
    model_path=model_root/model_name
    cached=False
    # This is a quick file-presence check; the heavy SHA verifier is used by
    # the downloader/real worker. Include old HF snapshots in the search so
    # a valid model downloaded by an earlier release isn't falsely 'missing'.
    out['model_cache_origin'] = 'none'
    out['model_missing_items'] = []
    try:
        from music_perception.lyrics_model_store import MODELS, candidate_model_directories
        spec=MODELS.get(model_name,{})
        expected=spec.get('min_bytes',2**62)
        names=['model.bin','config.json','tokenizer.json']
        if model_name=='large-v3-turbo':
            names.extend(['vocabulary.json','preprocessor_config.json'])
        possible = candidate_model_directories(model_name,model_root)
        best_missing = names
        for candidate,origin in possible:
            missing=[]
            for name in names:
                fp=candidate/name
                if not fp.is_file() or (name=='model.bin' and fp.stat().st_size<expected):
                    missing.append(name)
            if len(missing)<len(best_missing):
                best_missing=missing
            if not missing:
                model_path=candidate
                cached=True
                out['model_cache_origin']=origin
                break
        if not cached:
            out['model_missing_items']=best_missing
    except (OSError,ValueError):
        cached=False
        out['model_missing_items']=['unknown_files']
    out['weight_files_present']=cached
    step('model_weight_files','ok' if cached else 'missing',
         '仅检测文件和尺寸；完整 SHA-256 与实际推理由歌词工作进程验证')
    inference_passed = False
    inference_failed = False
    if deep:
        if not cached:
            step('synthetic_inference','skipped','本地无完整模型文件，不会自动下载')
        elif requested_device=='cuda' and not out.get('cuda_candidate'):
            step('synthetic_inference','skipped','CUDA 不可用；请先检查驱动与 CUDA/cuDNN')
        else:
            try:
                import numpy as np
                from faster_whisper import WhisperModel
                # No Hub queries, no live audio and no forced CPU fallback:
                # a failing GPU test must truthfully show the CUDA error.
                model=WhisperModel(str(model_path), device=requested_device,
                                   compute_type='float16' if requested_device=='cuda' else 'int8',
                                   cpu_threads=2, num_workers=1, local_files_only=True)
                step('model_loaded','ok')
                transcript,_=model.transcribe(np.zeros(16000,dtype='float32'),
                    task='transcribe',language='zh',beam_size=1,
                    vad_filter=False,condition_on_previous_text=False)
                list(transcript)
                out['actual_test_device']=requested_device
                inference_passed = True
                step('synthetic_inference','ok','只用一秒生成的静音进行推理')
            except Exception as exc:
                inference_failed = True
                step('synthetic_inference','failed',type(exc).__name__)
    else:
        step('synthetic_inference','skipped','已有工作进程正在运行，为避免重复占用显存跳过')
    # Completing the check is not the same as a healthy ASR environment.
    # An active lyrics worker deliberately skips a second heavy model load.
    if (missing_dependencies or not cached or not ctranslate2_import_ok
            or (requested_device == 'cuda' and not out.get('cuda_candidate'))):
        out['health'] = 'needs_attention'
    elif inference_failed:
        out['health'] = 'inference_failed'
    elif inference_passed:
        out['health'] = 'verified_'+requested_device
    else:
        out['health'] = 'not_fully_tested'
    out['missing_dependencies'] = missing_dependencies
    out['state']='complete'
    out['stage']='complete'
    out['elapsed_seconds']=round(time.monotonic()-started,2)
    _save(report_file,out)
    return out


def main():
    if len(sys.argv)!=7:
        return 2
    vendor, model_dir, model_name, device, report_file, deep=sys.argv[1:]
    if model_name not in {'tiny','base','small','large-v3-turbo'} or device not in {'cpu','cuda'} or deep not in {'0','1'}:
        return 2
    sys.path.insert(0,str(Path(vendor).resolve()))
    # The script runs under -I -S without invoking the plugin's heavy startup.
    import types
    pkg=types.ModuleType('music_perception')
    pkg.__path__=[str(Path(__file__).resolve().parent)]
    pkg.__package__='music_perception'
    sys.modules['music_perception']=pkg
    result=run_probe(vendor,model_dir,model_name,device,report_file,deep=deep=='1')
    sys.stdout.buffer.write(json.dumps({'state':result['state'],
                            'stage':result['stage'],'elapsed_seconds':result.get('elapsed_seconds')}).encode('utf-8')+b'\n')
    sys.stdout.buffer.flush()
    return 0 if result['state']=='complete' else 1

if __name__=='__main__':
    raise SystemExit(main())
