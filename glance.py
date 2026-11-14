"""为她所听·偷看/叫看调度（纯逻辑，不做截屏或投递）。

- 偷看：固定每 GLANCE_INTERVAL_S 秒判定一次，每日上限 GLANCE_DAILY_CAP 次。
- 叫看：主人触发，单独计数，不占偷看次数。
- 心情：读取 forever_companion 的只读心情维度，规则与陪看模块一致：
  高兴 ×1.3；心情差 ×0.6；极度生气拒绝（不截屏、不发言）；未知保持原值。
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable

GLANCE_INTERVAL_S = 300.0      # 固定每 5 分钟一次
GLANCE_DAILY_CAP = 20          # 偷看每日上限（可在配置中调整）
GLANCE_SCREEN_CHANCE = 0.40    # 主屏前台视频进入互动
GLANCE_SPEAK_CHANCE = 0.30     # 最后决定是否开口
MOOD_FRESH_S = 900.0
MOOD_HAPPY_V = 0.2
MOOD_LOW_V = -0.2
MOOD_ANGRY_V = -0.4
MOOD_ANGRY_AROUSAL_OVER = 0.18
MOOD_FACTOR_HAPPY = 1.3
MOOD_FACTOR_LOW = 0.6


def read_mood(raw: Any, *, now: float | None = None) -> dict[str, float] | None:
    """Parse forever_companion:get_affect_dimensions_v1. Unknown returns None."""
    if not isinstance(raw, dict) or raw.get("available") is not True or raw.get("mood_enabled") is not True:
        return None
    try:
        valence = float(raw["valence"])
        arousal = float(raw["arousal"])
        baseline = float(raw.get("baseline", 0.35))
    except (KeyError, TypeError, ValueError):
        return None
    if not (-1.0 <= valence <= 1.0 and 0.0 <= arousal <= 1.0):
        return None
    ts = raw.get("affect_updated_at")
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        now = time.time() if now is None else now
        if now - float(ts) > MOOD_FRESH_S:
            return None
    return {"valence": valence, "arousal": arousal, "baseline": baseline}


def mood_refuses(mood: dict[str, float] | None) -> bool:
    if not mood:
        return False
    return mood["valence"] <= MOOD_ANGRY_V and mood["arousal"] > mood["baseline"] + MOOD_ANGRY_AROUSAL_OVER


def mood_factor(mood: dict[str, float] | None) -> float:
    if not mood:
        return 1.0
    if mood["valence"] >= MOOD_HAPPY_V:
        return MOOD_FACTOR_HAPPY
    if mood["valence"] <= MOOD_LOW_V:
        return MOOD_FACTOR_LOW
    return 1.0


class GlanceScheduler:
    """Pure pacing state. The plugin supplies the clock, mood, and side effects."""

    def __init__(self, *, clock: Callable[[], float] = time.time,
                 rng: random.Random | None = None) -> None:
        self._clock = clock
        self._rng = rng or random.Random()
        self.day = ""
        self.glances_today = 0
        self.summons_today = 0
        self.last_glance_at = 0.0
        self.mood: dict[str, float] | None = None
        self.last: dict[str, Any] = {}

    def _roll_day(self) -> None:
        d = time.strftime("%Y-%m-%d", time.localtime(self._clock()))
        if d != self.day:
            self.day, self.glances_today, self.summons_today = d, 0, 0

    def set_mood(self, raw: Any) -> None:
        self.mood = read_mood(raw, now=self._clock())

    def decide_glance(self) -> str:
        """'glance' when a random glance should run now, otherwise 'none'."""
        self._roll_day()
        if mood_refuses(self.mood):
            return "none"
        now = self._clock()
        if now - self.last_glance_at < GLANCE_INTERVAL_S:
            return "none"
        if self.glances_today >= GLANCE_DAILY_CAP:
            return "none"
        return "glance"

    def mark_glance(self) -> None:
        self._roll_day()
        self.last_glance_at = self._clock()
        self.glances_today += 1

    def decide_summon(self) -> str:
        """Owner-triggered. Refused only for very angry mood; never uses glance budget."""
        self._roll_day()
        if mood_refuses(self.mood):
            return "refused"
        return "summon"

    def mark_summon(self) -> None:
        self._roll_day()
        self.summons_today += 1

    def speak_roll(self) -> bool:
        """Final gate before speaking. Mood factor applies; refusal already handled upstream."""
        p = GLANCE_SPEAK_CHANCE * mood_factor(self.mood)
        return self._rng.random() < min(0.95, p)

    def screen_roll(self) -> bool:
        p = GLANCE_SCREEN_CHANCE * mood_factor(self.mood)
        return self._rng.random() < min(0.95, p)

    def status(self) -> dict[str, Any]:
        self._roll_day()
        return {"glances_today": self.glances_today, "summons_today": self.summons_today,
                "daily_cap": GLANCE_DAILY_CAP, "interval_s": GLANCE_INTERVAL_S,
                "mood_refuses": mood_refuses(self.mood), "mood_factor": mood_factor(self.mood),
                "last": dict(self.last)}
