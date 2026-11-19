"""吐槽 (spontaneous commentary) for 为她所听.

She remarks on what is playing like a friend sitting next to you:
  * music — when the song noticeably changes (gets louder, calmer, faster, a new
    style …), described in everyday feelings, never BPM / keys / numbers;
  * video — on the screen frames she is allowed to see (为她所控 window glance, or
    the host's own screen share), at a relaxed pace.

Delivery is a single ``push_message(ai_behavior='respond')`` whose text is a
*hint* — she words it herself, or stays quiet. Everything here is gated by an
explicit owner switch, a shared cooldown, a "don't talk over the user" guard
and the companion's mood.

While commentary is on, the host's own timed proactive chat (主动搭话) is turned
off so the two do not compete; the owner's previous settings are remembered and
restored when commentary is switched off or the plugin stops.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

PROACTIVE_API = 'http://127.0.0.1:48911'
# Fields we turn off while we own commentary (privacy switch proactiveVisionEnabled is never touched).
PROACTIVE_TAKEOVER = ('proactiveChatEnabled', 'proactiveVisionChatEnabled', 'proactiveMusicEnabled',
                      'proactiveVideoChatEnabled')

FREQUENCY_GAPS = {'rare': 600.0, 'normal': 240.0, 'lively': 90.0}
DEFAULT_FREQUENCY = 'lively'
# A video remark needs a fresh look; do not look more often than this.
VIDEO_LOOK_GAP_S = {'rare': 600.0, 'normal': 300.0, 'lively': 120.0}
# The user typed / talked recently: wait, don't cut in.
QUIET_AFTER_USER_S = 25.0
# She was speaking (TTS) recently: let her finish and breathe.
QUIET_AFTER_AI_S = 12.0

SPEAK_LIKE_A_PERSON = (
    '用你自己的口吻自然地说一句（可以带一点动作或神态），像坐在主人旁边一起听/看时随口说的话；'
    '说感受和联想就好，比如像什么、让你想到什么、想不想跟着动；'
    '不要报任何数字、速度、调性、频率、分数，不要用乐理术语，'
    '不要提插件、识别、检测、数据、系统；主人明显在忙或不想被打扰时可以不说。'
)


class CommentaryState:
    """Persisted owner choices + runtime pacing for 吐槽."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.music_on = False
        self.video_on = False
        self.frequency = DEFAULT_FREQUENCY
        self.saved_proactive: dict[str, Any] | None = None   # owner's settings before takeover
        self.last_spoke_at = 0.0
        self.last_look_at = 0.0
        self.last_user_at = 0.0
        self.last_ai_voice_at = 0.0
        self.spoken = 0
        self.last: dict[str, Any] = {}
        self.load()

    # ---- persistence -------------------------------------------------
    def load(self):
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        self.music_on = bool(data.get('music_on', False))
        self.video_on = bool(data.get('video_on', False))
        f = data.get('frequency')
        self.frequency = f if f in FREQUENCY_GAPS else DEFAULT_FREQUENCY
        sp = data.get('saved_proactive')
        self.saved_proactive = sp if isinstance(sp, dict) else None

    def save(self):
        payload = {'schema': 'music_perception_commentary_v1', 'music_on': self.music_on,
                   'video_on': self.video_on, 'frequency': self.frequency,
                   'saved_proactive': self.saved_proactive, 'updated_at': time.time()}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            tmp.replace(self.path)
        except OSError:
            pass

    # ---- pacing --------------------------------------------------------
    @property
    def any_on(self) -> bool:
        return self.music_on or self.video_on

    def gap(self) -> float:
        return FREQUENCY_GAPS.get(self.frequency, FREQUENCY_GAPS[DEFAULT_FREQUENCY])

    def look_gap(self) -> float:
        return VIDEO_LOOK_GAP_S.get(self.frequency, VIDEO_LOOK_GAP_S[DEFAULT_FREQUENCY])

    def why_not_now(self, now: float | None = None) -> str:
        """'' when a remark is allowed now, else a short reason (shown in the panel)."""
        now = time.time() if now is None else now
        if now - self.last_spoke_at < self.gap():
            return 'cooldown'
        if now - self.last_user_at < QUIET_AFTER_USER_S:
            return 'user_active'
        if now - self.last_ai_voice_at < QUIET_AFTER_AI_S:
            return 'ai_speaking'
        return ''

    def public(self) -> dict:
        now = time.time()
        return {'music_on': self.music_on, 'video_on': self.video_on, 'frequency': self.frequency,
                'took_over_official': self.saved_proactive is not None,
                'next_in_s': max(0, int(self.gap() - (now - self.last_spoke_at))) if self.last_spoke_at else 0,
                'spoken': self.spoken, 'last': dict(self.last)}


# ---- prompts -------------------------------------------------------------
def music_prompt(moment_feel: str, overall: str, title: str, lyric: str) -> str:
    """Hint for a music remark. Only feelings — no numbers reach the model."""
    lines = ['（你正和主人一起听歌，这是你自己耳朵里听到的变化，主人没有问你。）']
    if title:
        lines.append('正在放的歌：《' + title[:40] + '》。')
    lines.append('刚才这首歌：' + moment_feel + '。')
    if overall:
        lines.append('现在整体听起来：' + overall + '。')
    if lyric:
        lines.append('刚好唱到：「' + lyric[:40] + '」。')
    lines.append('如果你也有感觉，就随口说一句你的感受，比如像什么场景、什么天气、想不想跟着晃、'
                 '让你想起什么；不想说也可以不说。')
    return '\n'.join(lines) + '\n' + SPEAK_LIKE_A_PERSON


def video_prompt(seen: str, app: str, speech: str) -> str:
    lines = ['（你正和主人一起看东西，这是你刚瞄到的画面，主人没有问你。）']
    if app:
        lines.append('在看的是：' + app[:30] + '。')
    lines.append('画面里：' + seen[:300])
    if speech:
        lines.append('刚听到里面在说：「' + speech[:80] + '」。')
    lines.append('像一起看视频的朋友那样，挑一个你觉得有意思或想吐槽的点随口说一句；'
                 '可以接梗、惊讶、好奇或者调侃，不用描述整个画面。')
    return '\n'.join(lines) + '\n' + SPEAK_LIKE_A_PERSON


def roll(chance: float, mood_factor: float = 1.0) -> bool:
    return random.random() < max(0.0, min(1.0, chance * mood_factor))
