"""Song-vocal ASR backend. Loaded exclusively in the singing private venv.

Video speech continues through lyrics_backend.py / the original Whisper worker.
No model downloads are permitted inside this worker.
"""
from __future__ import annotations
from pathlib import Path
import re

from .singing_catalog import MODEL_BY_ID, verified_folder

LANGUAGE_NAMES={'zh':'Chinese','ja':'Japanese','en':'English'}


def _open_model(model_name, model_dir, device_policy, cpu_threads, stage_hook=None,
                placement='standard',gpu_budget_mib=0,cpu_budget_mib=0):
    import numpy as np
    info=MODEL_BY_ID[model_name]
    root=Path(model_dir)
    local=verified_folder(root,info['quality'],deep=True)
    if local is None:
        raise RuntimeError('专业歌声模型未通过本地SHA-256校验；请先在模型页点击安装')
    if stage_hook:stage_hook('verified_singing_model')
    if info['engine']=='faster_whisper':
        # In Windows, CTranslate2 CUDA DLLs must come from v0.9.5's
        # SHA-verified plugin-private runtime, never an arbitrary global PATH.
        if device_policy in {'cuda','auto'}:
            import os
            if os.name=='nt':
                from .stem_cuda_bundle import cuda_runtime_path, configure_private_cuda
                # model_dir is <plugin-data>/models/singing_specialist
                private_environment=Path(model_dir).parent.parent/'environment'
                private_cuda=cuda_runtime_path(private_environment)
                if private_cuda is None and device_policy=='cuda':
                    raise RuntimeError('专业歌声 Whisper CUDA 需要先在「设置→GPU 环境」安装插件私有 CUDA/cuDNN；不会使用系统全局 DLL')
                if private_cuda is not None:
                    configure_private_cuda(private_cuda)
        from faster_whisper import WhisperModel
        if device_policy in {'cuda','auto'}:
            try:
                if stage_hook:stage_hook('loading_cuda')
                obj=WhisperModel(str(local),device='cuda',compute_type='float16',
                                 num_workers=1,local_files_only=True)
                return obj,'cuda'
            except (RuntimeError,OSError,ImportError,ValueError):
                if device_policy=='cuda':
                    raise RuntimeError('专业歌词模型 CUDA 初始化失败；未悄悄宣称 GPU 已运行')
        if stage_hook:stage_hook('loading_cpu')
        return WhisperModel(str(local),device='cpu',compute_type='int8',
                            cpu_threads=cpu_threads,num_workers=1,local_files_only=True),'cpu'
    from qwen_asr import Qwen3ASRModel
    import torch
    if device_policy=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('当前专业歌词独立环境的 PyTorch CUDA 不可用；检查显卡驱动及 CUDA 版本')
    use_gpu=device_policy!='cpu' and torch.cuda.is_available()
    device='cuda' if use_gpu else 'cpu'
    if stage_hook:stage_hook('loading_'+device)
    # Official transformer backend API; model path is local, not an HF repo ID.
    kwargs={'dtype':torch.float16 if use_gpu else torch.float32,
            'device_map':'cuda:0' if use_gpu else 'cpu',
            'max_inference_batch_size':1,'max_new_tokens':384,'local_files_only':True}
    if placement=='gpu_ram_weights':
        if not use_gpu or not 1536<=gpu_budget_mib<=32768 or not 4096<=cpu_budget_mib<=131072:
            raise RuntimeError('无法安全规划 GPU + 系统RAM权重分层')
        # Qwen3-ASR forwards these arguments to Transformers.from_pretrained.
        # Accelerate may execute CPU-assigned layers on CPU, but GPU-resident
        # layers stay CUDA. This does NOT force Windows shared GPU memory.
        import accelerate  # noqa: F401; fail clearly if private env lacks it
        kwargs['device_map']='auto'
        kwargs['max_memory']={0:f'{gpu_budget_mib}MiB',
                              'cpu':f'{cpu_budget_mib}MiB'}
        if stage_hook:stage_hook('loading_experimental_gpu_ram_layers')
    obj=Qwen3ASRModel.from_pretrained(str(local),**kwargs)
    if placement=='gpu_ram_weights':
        underlying=getattr(obj,'model',None)
        allocation=getattr(underlying,'hf_device_map',None)
        if not isinstance(allocation,dict) or not any(
             isinstance(target,int) or str(target).startswith('cuda') for target in allocation.values()):
            raise RuntimeError('分层模型未确认有层驻留CUDA；拒绝将纯CPU误报GPU')
    return obj,device


def prepare_singing_model(*,model_dir,model_name,model_cache,
                          device_policy='cpu',cpu_threads=2,
                          allow_public_mirror=True,stage_hook=None,
                          placement='standard',gpu_budget_mib=0,cpu_budget_mib=0):
    """Actually load and perform a short silence inference before activating."""
    import numpy as np
    key=((model_name,str(model_dir),device_policy,cpu_threads) if placement=='standard' else
         (model_name,str(model_dir),device_policy,cpu_threads,placement,gpu_budget_mib,cpu_budget_mib))
    if model_cache.get('key')==key and model_cache.get('instance') is not None:
        obj,device=model_cache['instance'],model_cache['device']
    else:
        obj,device=_open_model(model_name,model_dir,device_policy,cpu_threads,stage_hook,
                               placement,gpu_budget_mib,cpu_budget_mib)
    if stage_hook:stage_hook('singing_preflight')
    audio=np.zeros(16000,dtype=np.float32)
    info=MODEL_BY_ID[model_name]
    if info['engine']=='faster_whisper':
        chunks,_=obj.transcribe(audio,language='en',task='transcribe',beam_size=1,
                                vad_filter=False,condition_on_previous_text=False)
        list(chunks)
    else:
        # Device availability and first-pass kernels are checked here; model
        # download never occurs inside the inference child.
        obj.transcribe(audio=(audio,16000),language='English')
    model_cache.clear()
    model_cache.update(key=key,instance=obj,device=device)
    return {'ready':True,'device':device,'requested_device':device_policy,
            'model':model_name,'source_kind':'separated_singing_vocals',
            'placement':placement}


def _timed_qwen_lines(text, seconds, language):
    """Coarse *estimated* phrase positions (NOT forced alignment)."""
    text=' '.join(str(text).split()).strip()
    if not text:return []
    # Bound private AI payloads even if the model hallucinates a long response.
    text=text[:1200]
    phrases=[s.strip() for s in re.split(r'(?<=[。！？!?；;\n])\s*',text) if s.strip()]
    if not phrases: phrases=[text]
    # Never create fake lines at arbitrary character counts. Qwen's plain
    # transcript lacks measured phoneme boundaries: keep an unpunctuated
    # passage as one provisional cue instead of inventing per-65-char times.
    phrases=phrases[:18]
    lengths=[max(1,len(p)) for p in phrases]
    total=sum(lengths)
    pos=0.0
    out=[]
    for p,n in zip(phrases,lengths):
        start=pos; pos += seconds*n/total
        out.append({'at_s':round(start,3),'end_s':round(min(seconds,pos),3),
                    'text_preview':p[:160], 'words':[],
                    'language_candidate':language,'provenance':'local_qwen3_singing_asr',
                    'time_alignment':'estimated_phrase_span_not_word_aligned',
                    'certainty':'unverified_singing_transcription'})
    return out


def transcribe_singing_candidate(pcm,*,model_dir,model_name,model_cache=None,
                                 device_policy='cpu',cpu_threads=2,
                                 allow_public_mirror=True,language_hint='auto',stage_hook=None,
                                 placement='standard',gpu_budget_mib=0,cpu_budget_mib=0):
    import numpy as np
    if model_name not in MODEL_BY_ID or language_hint not in {'auto','zh','ja','en'}:
        raise ValueError('Unsupported specialist model / language')
    audio=np.asarray(pcm,dtype=np.float32).reshape(-1)
    if not 16000 <= len(audio) <= 16000*90 or not np.isfinite(audio).all():
        raise ValueError('Expected finite 16k PCM')
    if float(np.sqrt(np.mean(audio*audio)))<0.0005:
        return {'status':'no_singing_detected','timed_cues':[],
                'model':model_name,'source_kind':'separated_singing_vocals',
            'placement':placement}
    cache=model_cache if model_cache is not None else {}
    key=((model_name,str(model_dir),device_policy,cpu_threads) if placement=='standard' else
         (model_name,str(model_dir),device_policy,cpu_threads,placement,gpu_budget_mib,cpu_budget_mib))
    if cache.get('key')==key and cache.get('instance') is not None:
        obj,device=cache['instance'],cache['device']
    else:
        obj,device=_open_model(model_name,model_dir,device_policy,cpu_threads,stage_hook,
                               placement,gpu_budget_mib,cpu_budget_mib)
        cache.clear();cache.update(key=key,instance=obj,device=device)
    if stage_hook:stage_hook('singing_decoding')
    spec=MODEL_BY_ID[model_name]
    if spec['engine']=='qwen_asr':
        lang=LANGUAGE_NAMES.get(language_hint)
        output=obj.transcribe(audio=(audio,16000),language=lang)
        if not output:
            text=''; detected=language_hint
        else:
            text=str(getattr(output[0],'text','') or '')
            detected=str(getattr(output[0],'language','unknown'))
        cues=_timed_qwen_lines(text,len(audio)/16000,detected)
        engine='qwen_asr'
    else:
        chunks,info=obj.transcribe(audio,task='transcribe',
                language=None if language_hint=='auto' else language_hint,
                beam_size=4,temperature=0.0,condition_on_previous_text=False,
                vad_filter=False,word_timestamps=False)
        cues=[]
        detected=getattr(info,'language',language_hint)
        for piece in list(chunks)[:100]:
            text=str(getattr(piece,'text','') or '').strip()
            start=float(getattr(piece,'start',0));end=float(getattr(piece,'end',start))
            if text and end>start and len(text)<180:
                cues.append({'at_s':round(start,3),'end_s':round(end,3),
                             'text_preview':text,'words':[],
                             'language_candidate':detected,'provenance':'local_whisper_singing_asr',
                             'time_alignment':'model_segment_estimate',
                             'certainty':'unverified_singing_transcription'})
        engine='faster_whisper'
    return {'status':'asr_hypothesis' if cues else 'no_reliable_lyrics',
            'timed_cues':cues,'model':model_name,'device':device,
            'requested_device':device_policy,'gpu_fallback':device_policy=='auto' and device=='cpu',
            'engine':engine,'source_kind':'separated_singing_vocals',
            'placement':placement,
            'language_candidate':detected,'language_mode':language_hint,
            'warning':'歌声识别结果仅为候选。极高档≤5%错误率是待实测目标；没有真实校对数据不得宣称达标。'}
