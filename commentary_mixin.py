"""Glue between the 为她所听 plugin and the 吐槽 logic (commentary.py / music_moments.py).

Kept as a mixin so ``__init__.py`` only gains a few call sites:
  * ``_commentary_init()``       in startup
  * ``_commentary_on_clip()``    after each analyzed live music clip
  * ``_commentary_frame_wanted`` passed to the Live2D bridge server
  * ``_commentary_shutdown()``   in shutdown
plus the panel entries defined in __init__.py.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from .commentary import PROACTIVE_API, PROACTIVE_TAKEOVER, CommentaryState, music_prompt, roll, video_prompt
from .music_moments import MusicMomentTracker, describe_overall

# 音乐变化够明显也不是每次都说：保留一点随机，像人一样有时只是听着。
MUSIC_SPEAK_CHANCE = 0.85
VIDEO_SPEAK_CHANCE = 0.7
VIDEO_TICK_S = 10.0
# 屏幕分享帧多新才算「现在的画面」
FRAME_FRESH_S = 15.0
# 屏幕分享帧送给她看时附带的文字
FRAME_HINT = ('（这是主人屏幕分享的当前画面，你刚瞄到的。）像一起看的朋友那样挑一个你觉得有意思'
              '或想吐槽的点随口说一句，不用描述整个画面。')


class CommentaryMixin:
    # ------------------------------------------------------------------ setup
    def _commentary_init(self):
        self._commentary = CommentaryState(self.data_path('commentary.json'))
        self._moment_tracker = MusicMomentTracker()
        self._commentary_lock = threading.Lock()
        self._commentary_want_frame_until = 0.0
        self._commentary_restore_token = 0
        self._commentary_stop = threading.Event()
        self._commentary_thread = None
        if self._commentary.any_on:
            # 上次关插件时还开着：重新确认官方定时吐槽是关的（主人可能在别处又打开了）
            threading.Thread(target=lambda: asyncio.run(self._commentary_takeover()),
                             name='neko-commentary-takeover', daemon=True).start()
        self._commentary_start_thread()
        self._commentary_owner_flag()

    def _commentary_shutdown(self):
        stop = getattr(self, '_commentary_stop', None)
        if stop is not None:
            stop.set()
        # 插件关掉时把官方定时吐槽还给主人（下次启动、吐槽仍开着会再接管）
        state = getattr(self, '_commentary', None)
        if state is not None and state.saved_proactive is not None:
            try:
                self._commentary_restore_sync(keep_saved=True)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------- official proactive chat
    async def _commentary_takeover(self) -> dict:
        """Remember the owner's official proactive settings once, then switch them off."""
        state = self._commentary
        current = await asyncio.to_thread(_http_json, 'GET', '/api/proactive/settings')
        settings = (current or {}).get('settings') if isinstance(current, dict) else None
        if not isinstance(settings, dict):
            return {'ok': False, 'error': '主程序设置接口不可用'}
        if state.saved_proactive is None:
            state.saved_proactive = {k: settings.get(k) for k in PROACTIVE_TAKEOVER if k in settings}
            state.save()
        off = {k: False for k in PROACTIVE_TAKEOVER if settings.get(k)}
        if off:
            await asyncio.to_thread(_http_json, 'POST', '/api/proactive/settings', off)
        return {'ok': True, 'turned_off': sorted(off)}

    def _commentary_restore_sync(self, keep_saved: bool = False) -> dict:
        state = self._commentary
        saved = state.saved_proactive
        if not saved:
            return {'ok': True, 'restored': {}}
        body = {k: v for k, v in saved.items() if isinstance(v, bool)}
        result = _http_json('POST', '/api/proactive/settings', body) if body else {'success': True}
        if not keep_saved and result and result.get('success'):
            state.saved_proactive = None
            state.save()
        self._commentary_restore_token = int(time.time())
        self._commentary_restored_chat = bool(saved.get('proactiveChatEnabled'))
        return {'ok': bool(result and result.get('success')), 'restored': body}

    # ------------------------------------------------------ 为她所控 hand-off
    def _commentary_owner_flag(self):
        """Tell 为她所控 (via the shared media-mode file) that we own 吐槽 now."""
        store = getattr(self, '_media_mode_file', None)
        if store is None:
            return
        try:
            owner = 'music_perception' if self._commentary.any_on else ''
            if store.state.get('commentary_owner') != owner:
                store.state['commentary_owner'] = owner
                store._write()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------ page signals
    def _commentary_frame_wanted(self) -> dict:
        """Called by the bridge every 2 s; answers whether to grab a screen-share frame."""
        state = getattr(self, '_commentary', None)
        if state is None:
            return {'want_frame': False}
        out = {'want_frame': bool(state.video_on and time.time() < self._commentary_want_frame_until),
               'official_paused': bool(state.any_on and state.saved_proactive is not None)}
        if not out['official_paused'] and self._commentary_restore_token:
            out['official_restore'] = self._commentary_restore_token
            out['official_chat_enabled'] = bool(getattr(self, '_commentary_restored_chat', False))
        return out

    def _commentary_page(self) -> dict:
        try:
            from .live2d_stage_server import page_signals
            sig = page_signals()
        except Exception:  # noqa: BLE001
            return {}
        state = self._commentary
        if sig.get('user_input_at'):
            state.last_user_at = max(state.last_user_at, float(sig['user_input_at']))
        if sig.get('ai_speaking'):
            state.last_ai_voice_at = time.time()
        return sig

    # ------------------------------------------------------------------ speak
    def _commentary_can_speak(self) -> str:
        if (getattr(self, '_plugin_paused', False) or getattr(self, '_shutdown_requested', False)
                or not getattr(self, '_auto_consented', False)):
            return 'paused'
        if getattr(self, '_ai_voice_active', False):
            self._commentary.last_ai_voice_at = time.time()
        self._commentary_page()
        glance = getattr(self, '_glance', None)
        if glance is not None:
            from .glance import mood_refuses
            if mood_refuses(glance.mood):
                return 'mood_angry'
        return self._commentary.why_not_now()

    def _commentary_mood_factor(self) -> float:
        glance = getattr(self, '_glance', None)
        if glance is None:
            return 1.0
        from .glance import mood_factor
        return mood_factor(glance.mood)

    def _commentary_push(self, parts: list, source: str, summary: str) -> bool:
        """One background hint that starts an AI turn; she words it or stays quiet."""
        kwargs: dict[str, Any] = {'source': 'music_perception.commentary.' + source, 'visibility': [],
                                  'ai_behavior': 'respond', 'parts': parts, 'priority': 2,
                                  'coalesce_key': 'music_perception_commentary'}
        lanlan = ''
        try:
            from .live2d_stage_server import page_signals
            lanlan = page_signals().get('lanlan') or ''
        except Exception:  # noqa: BLE001
            lanlan = ''
        if lanlan:
            kwargs['target_lanlan'] = lanlan
        try:
            receipt = self.push_message(**kwargs)
        except Exception as exc:  # noqa: BLE001
            self._commentary.last = {'at': time.time(), 'kind': source, 'ok': False,
                                     'error': type(exc).__name__}
            return False
        ok = receipt is not False and not (isinstance(receipt, dict) and receipt.get('submitted') is False)
        state = self._commentary
        state.last_spoke_at = time.time()
        if ok:
            state.spoken += 1
        state.last = {'at': state.last_spoke_at, 'kind': source, 'ok': ok, 'what': summary[:80]}
        return ok

    # ------------------------------------------------------------------ music
    def _commentary_on_clip(self, report: dict) -> None:
        """After each analyzed live music clip: maybe remark on a felt change."""
        state = getattr(self, '_commentary', None)
        if state is None:
            return
        song = getattr(self, '_song_current_id', None) or getattr(self, '_track_display_name', None)
        moment, feel = self._moment_tracker.feed(song, report)
        if not state.music_on or moment is None:
            return
        why = self._commentary_can_speak()
        if why:
            state.last = {**state.last, 'skipped': why, 'pending': moment.kind}
            return
        if not roll(MUSIC_SPEAK_CHANCE, self._commentary_mood_factor()):
            self._moment_tracker.mark_said(moment)   # 这次选择安静听着，也算这一类说过了
            return
        title = str(getattr(self, '_track_display_name', '') or '')
        lyric = ''
        history = getattr(self, '_track_lyrics_history', None) or []
        if history:
            lyric = str(history[-1].get('text_preview') or '').strip()
        prompt = music_prompt(moment.feel, describe_overall(feel) if feel else '', title, lyric)
        if self._commentary_push([{'type': 'text', 'text': prompt}], 'music', moment.feel):
            self._moment_tracker.mark_said(moment)

    # ------------------------------------------------------------------ video
    def _commentary_start_thread(self):
        thread = getattr(self, '_commentary_thread', None)
        if thread is not None and thread.is_alive():
            return

        def work():
            while not self._commentary_stop.wait(VIDEO_TICK_S):
                try:
                    if self._commentary.video_on:
                        asyncio.run(self._commentary_video_tick())
                except Exception:  # noqa: BLE001
                    self.logger.exception('为她所听视频吐槽调度异常')

        self._commentary_thread = threading.Thread(target=work, name='neko-commentary', daemon=True)
        self._commentary_thread.start()

    def _commentary_video_now(self) -> bool:
        """She only comments on video when something is actually being watched."""
        scene = getattr(self, '_audio_scene', None) or {}
        kind = str((scene or {}).get('kind') or '')
        if kind in ('video', 'mixed'):
            return True
        last = getattr(self, '_media_mode_last', None) or {}
        return last.get('mode') in ('video', 'game')

    async def _commentary_video_tick(self) -> None:
        state = self._commentary
        now = time.time()
        if now - state.last_look_at < state.look_gap():
            return
        sig = self._commentary_page()
        sharing = bool(sig.get('screen_share'))
        if not (self._commentary_video_now() or sharing):
            return
        if self._commentary_can_speak():
            return
        state.last_look_at = now
        await self._read_companion_mood()
        if not roll(VIDEO_SPEAK_CHANCE, self._commentary_mood_factor()):
            return
        # 1) 为她所控在：让它看一眼当前视频窗口（文字描述）
        if await self._suokong_available():
            obs = await self._fetch_glance_observation()
            if isinstance(obs, dict) and obs.get('ok') and obs.get('description'):
                speech = ''
                transcript = getattr(self, '_video_transcript', None)
                if isinstance(transcript, dict):
                    from . import _trusted_speech_cues
                    cues = _trusted_speech_cues(transcript.get('cues'))
                    speech = cues[-1] if cues else ''
                prompt = video_prompt(str(obs['description']), str(obs.get('app') or ''), speech)
                self._commentary_push([{'type': 'text', 'text': prompt}], 'video', str(obs['description']))
                return
        # 2) 没有为她所控：用官方屏幕分享的画面（主窗口桥截一帧发过来）
        if not sharing:
            state.last = {**state.last, 'skipped': 'no_screen'}
            return
        from .live2d_stage_server import latest_screen_frame
        asked = time.time()
        self._commentary_want_frame_until = asked + 8.0
        frame = None
        for _ in range(16):
            await asyncio.sleep(0.5)
            frame = latest_screen_frame(FRAME_FRESH_S, newer_than=asked)
            if frame:
                break
        self._commentary_want_frame_until = 0.0
        if not frame:
            state.last = {**state.last, 'skipped': 'no_frame'}
            return
        data, mime = frame
        parts = [{'type': 'image', 'data': data, 'mime': mime}, {'type': 'text', 'text': FRAME_HINT}]
        self._commentary_push(parts, 'screen', '屏幕分享画面')


def _http_json(method: str, path: str, body: dict | None = None, timeout: float = 5.0):
    """Talk to the main program's settings API (local only, no proxies)."""
    import json
    import urllib.request
    req = urllib.request.Request(PROACTIVE_API + path, method=method,
                                 data=json.dumps(body).encode('utf-8') if body is not None else None,
                                 headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception:  # noqa: BLE001 - 主程序没开 / 版本不同
        return None
