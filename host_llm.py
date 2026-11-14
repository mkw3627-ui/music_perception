"""Auxiliary app-category classification through the N.E.K.O host LLM.

Only the executable name is sent (never window titles, PCM or file contents).
Calls run on a background thread; the scanner never blocks on the model.
Results are cached per executable name, and failures are cooled down so a
broken model does not get retried on every scan.
"""
from __future__ import annotations

import asyncio
import threading
import time

CATEGORY_TIMEOUT_S = 20.0
FAILURE_COOLDOWN_S = 300.0
VALID = frozenset({'music', 'video', 'other'})

_PROMPT = (
    '你是应用分类器。只根据进程可执行文件名判断它主要用于：\n'
    '- music：音乐播放器\n'
    '- video：视频播放器或视频网站客户端\n'
    '- other：其他（浏览器、游戏、通讯、系统等）\n'
    '进程名：{name}\n只输出一个单词：music、video 或 other。'
)


async def _host_completion(prompt: str, timeout: float) -> str | None:
    try:
        from utils.config_manager import get_config_manager
        from utils.llm_client import create_chat_llm_async
    except (ImportError, AttributeError):
        return None
    client = None
    try:
        manager = get_config_manager()
        api = None
        for role in ('summary', 'agent'):
            try:
                api = await asyncio.wait_for(manager.aget_model_api_config(role), timeout=4.0)
            except Exception:  # noqa: BLE001
                api = None
            if isinstance(api, dict) and str(api.get('model') or '').strip() \
                    and str(api.get('base_url') or '').strip():
                break
            api = None
        if api is None:
            return None
        client = await asyncio.wait_for(create_chat_llm_async(
            api['model'], api['base_url'], api.get('api_key', ''),
            provider_type=api.get('provider_type'), timeout=timeout,
            max_retries=0, max_completion_tokens=64,
        ), timeout=6.0)
        response = await asyncio.wait_for(
            client.ainvoke([{'role': 'user', 'content': prompt}]), timeout=timeout)
        content = getattr(response, 'content', None)
        return content if isinstance(content, str) else None
    except Exception:  # noqa: BLE001 - provider errors are arbitrary
        return None
    finally:
        if client is not None:
            try:
                await asyncio.wait_for(client.aclose(), timeout=3.0)
            except Exception:  # noqa: BLE001
                pass


def parse_category(text) -> str | None:
    word = str(text or '').strip().strip('`"\'.。 ').casefold()
    return word if word in VALID else None


class HostAppClassifier:
    """Cache + background worker. ``__call__`` never blocks on the model."""

    def __init__(self, completion=None, clock=time.monotonic):
        self._completion = completion or _host_completion
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: dict[str, str] = {}
        self._cooldown: dict[str, float] = {}
        self._pending: set[str] = set()

    def __call__(self, meta) -> str | None:
        name = str((meta or {}).get('name') or '').strip().casefold()
        if not name:
            return None
        with self._lock:
            if name in self._cache:
                return self._cache[name]
            if name in self._pending:
                return None
            if self._clock() < self._cooldown.get(name, 0.0):
                return None
            self._pending.add(name)
        threading.Thread(target=self._run, args=(name,), daemon=True,
                         name='music-perception-app-category').start()
        return None

    def _run(self, name: str) -> None:
        category = None
        try:
            text = asyncio.run(asyncio.wait_for(
                self._completion(_PROMPT.format(name=name), CATEGORY_TIMEOUT_S),
                timeout=CATEGORY_TIMEOUT_S + 5))
            category = parse_category(text)
        except Exception:  # noqa: BLE001 - classification is advisory only
            category = None
        with self._lock:
            self._pending.discard(name)
            if category is None:
                self._cooldown[name] = self._clock() + FAILURE_COOLDOWN_S
            else:
                self._cache[name] = category

    def cached(self, name: str) -> str | None:
        with self._lock:
            return self._cache.get(str(name or '').strip().casefold())
