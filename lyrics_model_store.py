"""Verified, resumable Whisper CT2 model acquisition for the isolated ASR worker.

Only *public model weights* are downloaded. Captured audio never enters these
requests. Downloads are not attempted if a verified local model exists. A
complete cache remains usable with no internet connection on later sessions.

No N.E.K.O SDK import: this runs in the independently managed CPython worker.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import urllib.error
import http.client
import re
from urllib.parse import quote, urlparse
from pathlib import Path
from typing import Callable


class LyricsModelDownloadError(RuntimeError):
    """No verified model is available; may be retried after connectivity returns."""


# Official Systran CTranslate2 multilingual models (not English-only). Model
# binary hashes from the model.bin entries on huggingface.co, checked 2026-10.
MODELS = {
    'tiny': {
        'repo': 'Systran/faster-whisper-tiny',
        'revision': 'd90ca5fe260221311c53c58e660288d3deb8d356',
        'sha256': 'dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1',
        'min_bytes': 70_000_000,
    },
    'base': {
        'repo': 'Systran/faster-whisper-base',
        'revision': 'ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66',
        'sha256': 'd01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9',
        'min_bytes': 130_000_000,
    },
    'small': {
        'repo': 'Systran/faster-whisper-small',
        'revision': '03feb38459a0c9abc5336827440d32db28b7c373',
        'sha256': '3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671',
        'min_bytes': 480_000_000,
    },
    # High-capability, speed-optimized multilingual large-v3 conversion. It is
    # NOT a specialized sung-lyrics model and is optional (>1.6 GB download).
    'large-v3-turbo': {
        'repo': 'deepdml/faster-whisper-large-v3-turbo-ct2',
        'revision': '44cbbd1adefe7387c83df88963a6d9ac4c9adea5',
        'sha256': 'e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da',
        'min_bytes': 1_580_000_000,
    },
}
ALLOWED_FILES = ('config.json', 'model.bin', 'tokenizer.json', 'vocabulary.json', 'vocabulary.txt', 'preprocessor_config.json')
# Download-source allowlist: main Hub + two independently hosted public mirrors.
# Backup hosts are not trusted to choose model bytes: the pinned official
# model.bin SHA-256 in MODELS remains mandatory for EVERY source.
PUBLIC_SOURCES = ('https://huggingface.co', 'https://hf-mirror.com', 'https://hf-mirror.net')
# 魔搭（ModelScope）托管了同一批权重，model.bin 的 SHA-256 与上面固定值一致；
# 国内 2026-11 实测 10 MB/s 以上，hf-mirror 常只有几百 KB/s，所以镜像优先时排第一。
MODELSCOPE = 'https://www.modelscope.cn'
MODELSCOPE_REPOS = {
    'tiny': 'Systran/faster-whisper-tiny',
    'base': 'Systran/faster-whisper-base',
    'small': 'Systran/faster-whisper-small',
    'large-v3-turbo': 'pengzhendong/faster-whisper-large-v3-turbo',
}
SOURCE_TITLES = {
    'https://huggingface.co': 'Hugging Face 官方',
    'https://hf-mirror.com': 'HF-Mirror 公益镜像',
    'https://hf-mirror.net': 'HF-Mirror 备用节点',
    MODELSCOPE: '魔搭 ModelScope',
}
# Only these files may be downloaded from the model repository.
REQUIRED_FILES = {
    'tiny': ('config.json','tokenizer.json','vocabulary.txt'),
    'base': ('config.json','tokenizer.json','vocabulary.txt'),
    'small': ('config.json','tokenizer.json','vocabulary.txt'),
    'large-v3-turbo': ('config.json','tokenizer.json','vocabulary.json','preprocessor_config.json'),
}


def validate_sidecar_file(path: Path, filename: str) -> tuple[bool, str]:
    """Validate the *actual* CTranslate2 sidecar format, not just JSON syntax.

    CTranslate2 writes vocabulary.json with ``json.dump(tokens)`` where tokens
    is a **list of strings**. Older versions of this plugin incorrectly
    demanded a JSON object for every .json file. That rejected a legitimate
    turbo vocabulary from every mirror, even when its bytes were correct.

    A mirror's access-denied HTML or API error JSON must not be promoted as a
    usable model file. This also validates already downloaded local files, so
    a successful old download is reused without new network requests.
    """
    path = Path(path)
    if filename not in ALLOWED_FILES or filename == 'model.bin':
        return False, '不支持的模型配套文件'
    try:
        if not path.is_file() or path.stat().st_size < 20:
            return False, f'{filename} 缺失或未下载完整'
        if filename == 'vocabulary.txt':
            # The Systran tiers have a plain-text vocabulary, not JSON.
            with path.open('rb') as stream:
                start = stream.read(1024).lstrip().lower()
            if (b'\x00' in start or start.startswith((b'<html', b'<!doctype',
                  b'<?xml', b'{"error"', b'{"message"'))):
                return False, f'{filename} 收到了非词表内容（可能是镜像错误页）'
            return True, 'valid'
        # Sidecar JSON is at most a few MiB; the model.bin is never loaded here.
        with path.open('r', encoding='utf-8-sig') as stream:
            document = json.load(stream)
        if filename == 'vocabulary.json':
            # Real Whisper CT2 vocabularies contain tens of thousands of
            # token strings; an error message such as {"error": "..."} is
            # not a vocabulary even if it happens to be valid JSON.
            if not (isinstance(document, list) and len(document) >= 1000 and
                    all(isinstance(token, str) for token in document)):
                return False, f'{filename} 必须是 CTranslate2 的 JSON 词元数组'
        elif not isinstance(document, dict) or not document:
            return False, f'{filename} 必须是非空 JSON 对象'
        return True, 'valid'
    except (OSError, UnicodeError, ValueError, TypeError):
        return False, f'{filename} 不是可解析的 JSON（可能是镜像错误页或下载截断）'



# Only base/small were measured exactly on real verified Windows files.
# The other tiers use estimates, so their UI explicitly marks the percentage
# as an estimate; 100% is reserved for a completed SHA-256 check.
EXPECTED_DOWNLOAD_BYTES = {
    'tiny': 75_000_000,
    'base': 145_217_532,
    'small': 483_546_902,
    'large-v3-turbo': 1_620_000_000,
}


def observed_download_bytes(model_name: str, root: Path) -> int:
    """Observe real bytes written by Hugging Face without parsing secrets/logs.

    Depending on huggingface_hub version, a partial weight may live in the
    model directory's hidden cache, or a finished model.bin may be present.
    Hidden caches outside the plugin are deliberately not scanned. If nothing
    observable exists, the frontend must use an indeterminate progress bar.
    """
    if model_name not in MODELS:
        raise ValueError('Unknown model')
    target = Path(root) / model_name
    complete = target / 'model.bin'
    try:
        completed_size = complete.stat().st_size if complete.is_file() else 0
    except OSError:
        completed_size = 0
    sizes = [completed_size]
    for direct_name in ('model.bin.partial','model.bin.download'):
        direct = target / direct_name
        try:
            if direct.is_file(): sizes.append(direct.stat().st_size)
        except OSError:
            pass
    cache = target / '.cache' / 'huggingface' / 'download' 
    try:
        if cache.is_dir():
            for file in cache.iterdir():
                if (file.is_file() and file.name.startswith('model.bin')
                        and (file.name.endswith('.incomplete') or '.partial' in file.name)):
                    sizes.append(file.stat().st_size)
    except OSError:
        pass
    return max(sizes)


def _sidecar_present_cheap(path: Path, filename: str) -> bool:
    """Presence hint only; a full parse + SHA still gates activation.

    A plausible size is not evidence of valid JSON. Never expose this check
    to the readiness/verification path.
    """
    try:
        size = Path(path).stat().st_size
        return filename in ALLOWED_FILES and 20 <= size <= 12_000_000
    except OSError:
        return False


def inspect_model_files(model_name: str, root: Path, *, deep: bool = True) -> dict:
    """Per-tier inventory; Hosted UI may request stat-only mode.

    deep=False checks only file metadata. Parsing 2.7 MB tokenizer files plus
    1 MB vocabulary every 3 seconds once blocked the Host's UI heartbeat.
    This is NEVER a SHA verdict and cannot authorize model activation.
    """
    if model_name not in MODELS:
        raise ValueError('Unknown model')
    root = Path(root)
    spec = MODELS[model_name]
    result = {'model': model_name, 'state': 'missing',
              'sha256_verified': False, 'received_bytes': observed_download_bytes(model_name,root),
              'expected_bytes_estimate': EXPECTED_DOWNLOAD_BYTES[model_name],
              'progress_estimated': True}
    for folder, origin in candidate_model_directories(model_name,root):
        try:
            if not folder.is_dir():
                continue
            weight = folder / 'model.bin'
            required = list(REQUIRED_FILES[model_name])
            if weight.is_file():
                size = weight.stat().st_size
                result['received_bytes'] = max(result['received_bytes'],size)
                complete = size >= spec['min_bytes'] and all(
                    (validate_sidecar_file(folder / name, name)[0] if deep else
                     _sidecar_present_cheap(folder / name, name))
                    for name in required)
                if complete:
                    result.update(state='present_unverified',origin=origin)
                    return result
                result.update(state='incomplete',origin=origin)
            elif any((folder / name).is_file() for name in required):
                result.update(state='incomplete',origin=origin)
        except OSError:
            result.update(state='unreadable',origin=origin)
    if result['received_bytes'] and result['state']=='missing':
        result['state']='incomplete'
    return result



def sidecar_stat_signature(directory: Path, model_name: str) -> list[dict] | None:
    """Cheap, complete sidecar identity for a *previous* SHA verification.

    UI refreshes cannot rehash gigabyte weights or parse megabyte tokenizers.
    A saved SHA verdict is reusable only while the exact sidecars and weight
    retain their verified stat identity. This signature is a cache invalidator,
    never an independent proof of correct model bytes.
    """
    if model_name not in REQUIRED_FILES:
        return None
    try:
        result = []
        for name in REQUIRED_FILES[model_name]:
            info = (Path(directory) / name).stat()
            if not 20 <= info.st_size <= 12_000_000:
                return None
            result.append({'file':name, 'size':info.st_size, 'mtime_ns':info.st_mtime_ns})
        return result
    except OSError:
        return None


def check_local_model(directory: Path, model_name: str, progress_callback=None) -> tuple[bool, str]:
    """Require valid tokenizer/config AND authentic, complete published weight.

    Large weight SHA-256 is read in bounded chunks, not loaded into RAM.
    """
    spec = MODELS[model_name]
    directory = Path(directory)
    try:
        weights = directory / 'model.bin'
        if not weights.is_file() or weights.stat().st_size < spec['min_bytes']:
            return False, '模型权重不存在或尚未下载完整'
        for name in REQUIRED_FILES[model_name]:
            accepted, reason = validate_sidecar_file(directory / name, name)
            if not accepted:
                return False, reason
        if _file_sha256(weights, progress_callback=progress_callback) != spec['sha256']:
            return False, '模型权重 SHA-256 校验失败，不能使用损坏或来源不明的权重'
        return True, 'verified'
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        return False, '模型文件无法读取或解析：' + type(exc).__name__


def candidate_model_directories(model_name: str, root: Path) -> list[tuple[Path, str]]:
    """Find only expected local CT2 directories, including older HF caches.

    Never trust a model merely because a directory exists. Every candidate
    must still pass the official SHA-256 check before being loaded.
    """
    if model_name not in MODELS:
        raise ValueError('Unsupported multilingual model')
    root = Path(root)
    folder = 'models--' + MODELS[model_name]['repo'].replace('/', '--')
    groups = [(root, 'plugin')]
    # A previous faster-whisper version may have fetched weights to the
    # Hugging Face hub cache instead of the plugin's persistent model folder.
    # Reuse it offline if (and only if) the complete official files verify.
    cache_candidates = []
    if os.environ.get('HF_HUB_CACHE'):
        cache_candidates.append(Path(os.environ['HF_HUB_CACHE']))
    if os.environ.get('HF_HOME'):
        cache_candidates.append(Path(os.environ['HF_HOME']) / 'hub')
    try:
        cache_candidates.append(Path.home() / '.cache' / 'huggingface' / 'hub')
    except (RuntimeError, OSError):
        pass
    groups.extend((p, 'huggingface_cache') for p in cache_candidates)
    found = []
    seen = set()
    for base, origin in groups:
        options = [base / model_name, base / folder]
        snapshot_dir = base / folder / 'snapshots'
        try:
            if snapshot_dir.is_dir():
                options.extend(sorted((p for p in snapshot_dir.iterdir()
                                       if p.is_dir()), key=lambda p: p.name, reverse=True))
        except OSError:
            pass
        for candidate in options:
            text = str(candidate).casefold()
            if text not in seen:
                seen.add(text)
                found.append((candidate, origin))
    return found


def find_local_model(model_name: str, root: Path) -> Path | None:
    """Reuse validated plugin or standard HF cache snapshots without HTTP."""
    for location, _origin in candidate_model_directories(model_name, root):
        if location.is_dir() and check_local_model(location, model_name)[0]:
            return location
    return None


def _safe_sources(allow_public_mirror: bool) -> tuple[str, ...]:
    """Ordered, allowlisted sources; never forward credentials to mirrors.

    The independent prepare worker defaults to domestic-mirror-first because
    Hugging Face storage CDNs are frequently blocked without a proxy in China.
    Direct callers retain official-first ordering for backward compatibility.
    """
    if not allow_public_mirror:
        return (PUBLIC_SOURCES[0],)
    priority = (os.environ.get('NEKO_LYRICS_DOWNLOAD_PRIORITY') or 'official').lower()
    if priority == 'mirror':
        return (MODELSCOPE, PUBLIC_SOURCES[1], PUBLIC_SOURCES[2], PUBLIC_SOURCES[0])
    return PUBLIC_SOURCES + (MODELSCOPE,)


def _notify(callback, phase: str, endpoint: str, filename: str = ''):
    if callback:
        callback({'phase': phase, 'source': endpoint, 'source_name': SOURCE_TITLES[endpoint],
                  'filename': filename})


def _direct_file_url(endpoint: str, model_name: str, filename: str) -> str:
    """Use a pinned, public HTTPS model URL; no untrusted repo/file inputs."""
    if endpoint not in PUBLIC_SOURCES + (MODELSCOPE,) or filename not in ('model.bin',) + ALLOWED_FILES:
        raise ValueError('Source or model file not allowlisted')
    spec = MODELS[model_name]
    if endpoint == MODELSCOPE:
        # 魔搭没有 HF 的 commit 号；靠 model.bin 固定 SHA-256 保证是同一份权重。
        return (MODELSCOPE + '/models/' + MODELSCOPE_REPOS[model_name]
                + '/resolve/master/' + quote(filename, safe=''))
    revision = spec.get('revision', 'main')
    return endpoint + '/' + spec['repo'] + '/resolve/' + quote(revision, safe='') + '/' + quote(filename, safe='') + '?download=true'


def _file_sha256(path: Path, progress_callback=None) -> str:
    digest = hashlib.sha256()
    total = Path(path).stat().st_size
    done = 0
    if progress_callback:
        progress_callback(0, total)
    with Path(path).open('rb') as inp:
        for block in iter(lambda: inp.read(4 * 1024 * 1024), b''):
            digest.update(block)
            done += len(block)
            if progress_callback and (done == total or done % (64 * 1024 * 1024) < len(block)):
                progress_callback(done, total)
    return digest.hexdigest()


def _stream_to_file(url: str, destination: Path, *, min_bytes=20,
                    max_bytes=2_300_000_000, expected_sha256: str | None = None,
                    verification_progress=None, transfer_progress=None,
                    _force_fresh: bool = False, file_validator=None):
    """Resumable HTTPS transfer with strict EOF validation.

    A partially written weight is NEVER promoted merely because read() gave
    EOF and a loose min_bytes threshold was passed. Check Content-Length /
    Content-Range and the official SHA before os.replace. On interruption
    retain the .partial file so another permitted source can resume it.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + '.partial')
    restart = destination.with_name(destination.name + '.restart')
    try:
        offset = temp.stat().st_size if temp.is_file() and not _force_fresh else 0
    except OSError:
        offset = 0
    # A fully validated fresh restart left by a process interruption can be
    # promoted offline without touching the previous near-complete .partial.
    if expected_sha256 and restart.is_file() and not _force_fresh:
        if restart.stat().st_size >= min_bytes and _file_sha256(restart) == expected_sha256:
            os.replace(restart, destination)
            return
    if offset > max_bytes:
        raise LyricsModelDownloadError('已下载的临时文件大小异常；未删除该文件，请检查磁盘')
    # If a previous download reached EOF but was cancelled before activation,
    # finish it locally rather than issuing an unnecessary 1.6 GB request.
    if expected_sha256 and offset >= min_bytes and _file_sha256(temp, verification_progress) == expected_sha256:
        os.replace(temp, destination)
        return
    headers = {'User-Agent': 'N.E.K.O-WeiTaSuoTing/0.8.3',
               'Accept-Encoding': 'identity'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=25)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset:
            # A mirror may consider an existing .partial fully downloaded.
            # First trust it only if it passes SHA. Otherwise try a fresh
            # full response in .restart without destroying the old bytes.
            if expected_sha256 and _file_sha256(temp) == expected_sha256:
                os.replace(temp, destination)
                return
            headers.pop('Range', None)
            response = urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=25)
        else:
            raise
    with response:
        final = urlparse(response.geturl())
        if final.scheme.lower() != 'https':
            raise LyricsModelDownloadError('模型下载发生非 HTTPS 重定向')
        mime = str(response.headers.get('Content-Type', '')).split(';', 1)[0].strip().lower()
        if file_validator and mime in {'text/html', 'application/xhtml+xml',
                                       'application/xml', 'text/xml'}:
            raise LyricsModelDownloadError('镜像返回了网页/错误页面而不是模型文件；尝试下一下载源')
        code = getattr(response, 'status', response.getcode())
        content_range = response.headers.get('Content-Range', '')
        length_header = response.headers.get('Content-Length', '')
        expected_part = int(length_header) if str(length_header).isdigit() else None
        total_size = None
        if code == 206:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)', content_range.strip())
            if not match or int(match[1]) != offset or int(match[2]) < int(match[1]):
                raise LyricsModelDownloadError('服务器返回的断点位置不匹配；已保留原下载数据')
            first, last = int(match[1]), int(match[2])
            total_size = int(match[3]) if match[3] != '*' else None
            if total_size is not None and (last >= total_size or total_size > max_bytes):
                raise LyricsModelDownloadError('服务器返回了无效的模型文件总大小')
            range_length = last - first + 1
            if expected_part is not None and expected_part != range_length:
                raise LyricsModelDownloadError('服务器 Content-Length 与断点响应范围不一致')
            expected_part = range_length
            out_file = restart if _force_fresh else temp
            out_mode, prefix = ('wb', 0) if _force_fresh else ('ab', offset)
        elif code == 200:
            if expected_part is not None and expected_part > max_bytes:
                raise LyricsModelDownloadError('模型文件超过安全大小上限')
            # If a CDN ignores Range, download its full response to a separate
            # file. A broken restart must not overwrite the previous partial.
            out_file = restart if (offset or _force_fresh) else temp
            out_mode, prefix = 'wb', 0
        else:
            raise LyricsModelDownloadError('服务器返回不支持的 HTTP 状态码')
        transferred = 0
        # Report the *active transfer* instead of the largest stale file. A
        # corrupt 99%-complete .partial may force a fresh .restart; showing
        # the old .partial's 99% for this new stream is misleading.
        fresh = out_file == restart
        if transfer_progress:
            transfer_progress(prefix, total_size or (prefix + expected_part if expected_part is not None else None), fresh)
        last_report_bytes = 0
        with out_file.open(out_mode) as out:
            while True:
                block = response.read(256 * 1024)
                if not block:
                    break
                transferred += len(block)
                if prefix + transferred > max_bytes:
                    raise LyricsModelDownloadError('远程模型文件超出安全大小限制')
                out.write(block)
                if transfer_progress and transferred - last_report_bytes >= 4 * 1024 * 1024:
                    transfer_progress(prefix + transferred,
                        total_size or (prefix + expected_part if expected_part is not None else None), fresh)
                    last_report_bytes = transferred
        if transfer_progress:
            transfer_progress(prefix + transferred,
                total_size or (prefix + expected_part if expected_part is not None else None), fresh)
        # The old implementation accepted EOF after only min_bytes and then
        # discarded ~1.5 GB on SHA failure. Detect premature EOF FIRST.
        if expected_part is not None and transferred != expected_part:
            raise LyricsModelDownloadError(
                f'模型流提前结束：本次仅收到 {transferred} / {expected_part} 字节；已保留断点，可继续下载')
        size = prefix + transferred
        if total_size is not None and size < total_size:
            raise LyricsModelDownloadError(
                f'模型断点尚未完成：已有 {size} / {total_size} 字节；已保留断点')
        if size < min_bytes:
            raise LyricsModelDownloadError('远程文件太短或下载中断；已保留下载数据')
    # The downloaded candidate has passed transport checks; for model.bin
    # require the pinned published checksum before treating bytes as final.
    candidate = out_file
    if expected_sha256 and _file_sha256(candidate, verification_progress) != expected_sha256:
        # A corrupt old prefix cannot be fixed merely by resuming its last
        # 1%; use ONE fresh HTTPS response in .restart. Preserve .partial for
        # diagnosis and never promote anything without the pinned checksum.
        if prefix > 0 and not _force_fresh:
            return _stream_to_file(url, destination, min_bytes=min_bytes,
                max_bytes=max_bytes, expected_sha256=expected_sha256,
                verification_progress=verification_progress,
                transfer_progress=transfer_progress, _force_fresh=True,
                file_validator=file_validator)
        raise LyricsModelDownloadError('传输结束但模型权重 SHA-256 不匹配；已保存未验证的断点，未启用')
    if file_validator is not None:
        valid, reason = file_validator(candidate)
        if not valid:
            # Incomplete JSON or mirror HTML must NEVER replace a good file.
            # Nor can it become the Range prefix when retrying another host.
            # Save one rejected sidecar for troubleshooting (no private data).
            rejected = destination.with_name(destination.name + '.rejected')
            os.replace(candidate, rejected)
            raise LyricsModelDownloadError(reason + '；保留原有模型并切换下载源')
    if candidate != temp:
        os.replace(candidate, temp)
    os.replace(temp, destination)


def _download_direct_model(model_name: str, root: Path, endpoint: str, callback=None) -> Path:
    """Fallback to pinned /resolve/ files when snapshot/Xet cannot complete.

    The manual downloader's .partial is observed by the progress panel. Only
    the model.bin of the exact revision/size/hash is ever accepted.
    """
    spec = MODELS[model_name]
    def verify_progress(done, total):
        if callback:
            callback({'phase':'verifying_sha', 'source':endpoint,
                      'source_name':SOURCE_TITLES[endpoint], 'filename':'model.bin',
                      'verification_bytes':int(done), 'verification_total':int(total)})
    def transfer_progress(received, expected, is_restart):
        if callback:
            callback({'phase':'downloading', 'source':endpoint,
                      'source_name':SOURCE_TITLES[endpoint], 'filename':'model.bin',
                      'received_bytes':int(received),
                      'expected_bytes':int(expected) if expected is not None else None,
                      'restarting':bool(is_restart)})
    folder = root / model_name
    folder.mkdir(parents=True,exist_ok=True)
    for filename in REQUIRED_FILES[model_name]:
        existing = folder / filename
        if validate_sidecar_file(existing, filename)[0]:
            continue
        _notify(callback,'downloading',endpoint,filename)
        _stream_to_file(_direct_file_url(endpoint,model_name,filename),existing,
                        max_bytes=12_000_000,
                        file_validator=lambda candidate, name=filename:
                            validate_sidecar_file(candidate,name))
    weight = folder / 'model.bin'
    reuse = False
    if weight.is_file() and weight.stat().st_size >= spec['min_bytes']:
        digest=hashlib.sha256()
        with weight.open('rb') as input_file:
            for block in iter(lambda:input_file.read(1024*1024),b''):
                digest.update(block)
        reuse=(digest.hexdigest()==spec['sha256'])
    if not reuse:
        # Older builds promoted an incomplete stream to model.bin, then failed
        # its SHA. Salvage those ~99%-complete bytes as a resumable .partial
        # instead of starting at zero. A correct file is never moved because
        # `reuse` above already proved its digest.
        partial=folder/'model.bin.partial'
        if weight.is_file():
            try:
                if (not partial.exists() or
                        weight.stat().st_size > partial.stat().st_size):
                    os.replace(weight, partial)
            except OSError:
                # No destructive fallback if Windows is still holding it open.
                raise LyricsModelDownloadError('旧版未验证权重被占用，无法迁移到断点；请暂停插件再重试')
        _notify(callback,'downloading',endpoint,'model.bin')
        _stream_to_file(_direct_file_url(endpoint,model_name,'model.bin'),weight,
                        min_bytes=spec['min_bytes'],max_bytes=int(spec['min_bytes']*1.25+20_000_000),
                        expected_sha256=spec['sha256'],
                        verification_progress=verify_progress,
                        transfer_progress=transfer_progress)
    _notify(callback,'verifying_sha',endpoint,'model.bin')
    verified,reason=check_local_model(folder,model_name,progress_callback=verify_progress)
    if not verified:
        # Keep a partial/candidate for explicit retry and audit, but never
        # present a failed SHA as a usable model or delete 1.6 GB silently.
        raise LyricsModelDownloadError(reason)
    return folder


def ensure_local_model(model_name: str, root: Path, *,
                       allow_public_mirror: bool = True,
                       downloader: Callable | None = None,
                       progress_callback: Callable | None = None) -> Path:
    """Use a locally verified CT2 model or fetch it from official/mirror source.

    Uses huggingface_hub's resumable local_dir download. `token=False` prevents
    transmitting any cached HF access token to a third-party model mirror.
    """
    if model_name not in MODELS:
        raise ValueError('Unsupported multilingual Whisper model')
    root = Path(root)
    existing = find_local_model(model_name, root)
    if existing is not None:
        return existing
    root.mkdir(parents=True, exist_ok=True)
    # Neither downloads nor model verification touch private audio.
    # In mainland networks, the plugin prepare worker prefers a public mirror,
    # then another mirror and finally the official host. Each attempt is
    # checked against the SAME pinned official digest.
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','25')
    os.environ.setdefault('HF_HUB_ETAG_TIMEOUT','8')
    # Prefer direct HTTPS + resumable Range here: older hf-hub may follow
    # Xet/CDN endpoints that are inaccessible even if the Hub API is reachable.
    # The standard snapshot downloader remains a compatibility fallback.
    injected_downloader = downloader is not None
    if downloader is None:
        try:
            from huggingface_hub import snapshot_download as downloader
        except ImportError:
            downloader = None
    target = root / model_name
    errors = []
    sources = _safe_sources(allow_public_mirror)
    # Try the pinned direct HTTPS route on EVERY source before trying any
    # snapshot download. If a mirror sends an error page instead of the small
    # vocabulary.json, this must not trigger a giant Hub/Xet transfer before
    # the next public source has been tried. Existing partial weights survive.
    if not injected_downloader:
        for endpoint in sources:
            _notify(progress_callback,'connecting',endpoint)
            try:
                return _download_direct_model(model_name,root,endpoint,progress_callback)
            except (OSError, ValueError, LyricsModelDownloadError,
                    urllib.error.URLError, TimeoutError, http.client.IncompleteRead) as exc:
                safe_reason=(str(exc)[:115] if isinstance(exc,LyricsModelDownloadError)
                             else type(exc).__name__)
                errors.append(SOURCE_TITLES[endpoint]+'（直链）:'+safe_reason)
                cached=find_local_model(model_name,root)
                if cached is not None:
                    return cached
    # Compatibility fallback: pinned repository snapshot, with public tokens
    # disabled for every mirror. Mocked downloader tests only exercise this
    # branch; production uses it after direct sources are exhausted.
    for endpoint in sources:
        if endpoint == MODELSCOPE:
            continue  # HF Hub 客户端不支持魔搭；魔搭只走上面的直链
        if downloader is None:
            errors.append(SOURCE_TITLES[endpoint]+'：未安装 huggingface-hub')
            continue
        try:
            _notify(progress_callback,'downloading',endpoint,'snapshot')
            revision=MODELS[model_name].get('revision')
            downloaded=downloader(
                **({'revision':revision} if revision else {}),
                repo_id=MODELS[model_name]['repo'],local_dir=str(target),
                allow_patterns=list(ALLOWED_FILES),max_workers=2,
                etag_timeout=8,endpoint=endpoint,token=False)
            candidate=Path(downloaded)
            _notify(progress_callback,'verifying_sha',endpoint,'model.bin')
            valid,why=check_local_model(candidate,model_name)
            if valid:
                return candidate
            errors.append(SOURCE_TITLES[endpoint]+'（Hub）:'+why)
            # An interrupted snapshot stays on disk for retry, but never
            # becomes an active model without the published pinned checksum.
        except Exception as exc:
            errors.append(SOURCE_TITLES[endpoint]+'（Hub）:'+type(exc).__name__)
        cached=find_local_model(model_name,root)
        if cached is not None:
            return cached
    detail=' / '.join(errors)[:320]
    raise LyricsModelDownloadError(
        '歌词模型权重未下载完成；已尝试的官方源与公共镜像均没有取得校验通过的文件。'
        +('（'+detail+'）' if detail else '')
        +'。可将官方 '+MODELS[model_name]['repo']
        +' 完整 CT2 文件放入插件数据目录 models/whisper_multilingual/'+model_name+'/。'
    )
