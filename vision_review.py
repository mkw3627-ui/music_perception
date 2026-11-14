"""为她所听·后台视觉复核（纯判定逻辑，不截屏、不调模型、不发消息）。

设计约定：
* 本插件从不声称"自己看到了画面"。画面判定由窗口插件（browser_history_insights）
  真实截帧后给出，这里只把它的结论登记成 ``external_review`` 标签。
* 候选标签（标题/进程名/模型提示）永远只是候选，只有外部复核才能解锁 ASR 与视频语音。
* 限流是为了控制视觉调用成本，而不是为了"看起来克制"：同一来源有最小间隔，
  每小时有上限，连续失败会退避。
"""
from __future__ import annotations

import time
from collections import deque

# 需要复核的候选标签：unknown 也要复核，浏览器标题看不出来时正是它最常见的样子。
REVIEWABLE_TAGS = frozenset({'video', 'music', 'unknown'})
MIN_ATTEMPT_GAP_S = 90.0
# Cost control, not a correctness limit. Every *new* source (a new tab, a new
# player) is a different identity and deserves one look; capping the whole hour
# at 12 meant a user who switched videos more than a dozen times was silently
# left at "音源待确认" for the rest of the hour.
MAX_ATTEMPTS_PER_HOUR = 40
FAILURE_BACKOFF_S = 600.0
FAILURE_STREAK_LIMIT = 3
# A window that simply is not media (login page, ad, a frame between cuts) is
# not a broken vision pipeline. It only cools THAT source down briefly; other
# sources must stay reviewable, otherwise three misses freeze the whole panel.
SOURCE_COOLDOWN_S = 120.0
SOURCE_STREAK_LIMIT = 2
# 复核标签的有效期。到期后重新变成候选，需要再核一次，避免"一次看错，整天都错"。
REVIEW_LABEL_TTL_S = 1800.0


def reviewable_tag(tag, tag_status) -> bool:
    """True when this source still needs an external window review."""
    if str(tag_status or 'candidate') == 'external_review':
        return False
    return str(tag or 'unknown') in REVIEWABLE_TAGS


def classify_observation(obs):
    """窗口插件的视觉结论 -> (tag, reason)。只认它的布尔标记，不猜文字。"""
    if not isinstance(obs, dict):
        return '', 'bad_result'
    if obs.get('ok') is not True:
        return '', str(obs.get('code') or obs.get('reason_code') or 'not_ok')[:60]
    video = obs.get('is_video') is True
    music = obs.get('is_music') is True
    if video and not music:
        return 'video', 'vision_is_video'
    if music and not video:
        return 'music', 'vision_is_music'
    return '', 'vision_not_media'


def normalize_app(name: str) -> str:
    text = str(name or '').strip().casefold()
    if text.endswith('.exe'):
        text = text[:-4]
    return ''.join(ch for ch in text if not ch.isspace())


def app_matches(selected_name, observed_app) -> bool:
    """同源校验：窗口插件看到的是不是我们正在采集的那个程序。

    它没报程序名时无法反驳，按通过处理；一旦报了别的程序，就必须拒绝——看错窗口
    绝不能让另一个程序的标签生效。
    """
    observed = normalize_app(observed_app)
    if not observed:
        return True
    selected = normalize_app(selected_name)
    if not selected:
        return False
    return observed == selected or observed in selected or selected in observed


def pick_review_target(selected_id: str, apps, *, make_id):
    """Which live source should the review slot look at this tick?

    The capture target (selected app) keeps the review slot only while it is
    actually audible. When it is silent or missing from the scan, the slot moves
    to the loudest audible, non-AI source that still needs a window review, so a
    silent pinned app cannot starve a playing one. Returns (row, reason).
    The capture itself is never switched here: only the *review* target moves.
    """
    rows = [a for a in (apps or []) if isinstance(a, dict)]
    by_id = {make_id(a): a for a in rows}
    selected = by_id.get(selected_id)
    # Keep the selected target only while it still NEEDS a review. A reviewed,
    # still-playing app must not hold the slot and starve a second unreviewed one.
    if (selected is not None and selected.get('audible') and not selected.get('host_ai')
            and reviewable_tag(selected.get('tag'), selected.get('tag_status'))):
        return selected, 'selected_audible'
    candidates = [a for a in rows
                  if a.get('audible') and not a.get('host_ai')
                  and reviewable_tag(a.get('tag'), a.get('tag_status'))]
    if not candidates:
        reason = 'selected_silent_no_candidate' if selected is not None else 'selected_not_in_scan'
        return None, reason
    # Loudest first: the most likely source of the sound a user is hearing now.
    best = max(candidates, key=lambda a: float(a.get('peak') or 0.0))
    return best, 'moved_to_audible_source'


def pick_capture_target(selected_id, apps, *, make_id, is_known_player=None,
                        selected_dead=False):
    """Should the *PCM capture* follow an audible source instead of the pinned app?

    ``pick_review_target`` only moves the review slot. That is not enough for
    lyrics: ASR can only ever hear the captured process, so a pinned-but-silent
    player yields a "successful" transcription of digital silence — zero words.
    This decides whether to retarget the capture itself, and is deliberately
    conservative:

    * the pinned app must NOT be audible (or have vanished from the scan);
    * exactly ONE non-AI app may be audible (never guess between two);
    * that app must already be externally reviewed as music/video, or be an
      exact known music player — the same rule the recognition gate applies.

    ``selected_dead`` relaxes only the "exactly one" rule, and only because the
    pinned process no longer exists: there is nothing left to be faithful to, so
    waiting forever is strictly worse than moving to the loudest *reviewed*
    source. A live-but-quiet pinned app keeps the strict rule (the owner may
    simply have paused).

    Returns ``(row, reason)``; ``row`` is ``None`` when the capture should stay
    where the user put it.
    """
    rows = [a for a in (apps or []) if isinstance(a, dict)]
    selected = next((a for a in rows if make_id(a) == selected_id), None)
    if (selected is not None and selected.get('audible')
            and not selected.get('host_ai')):
        return None, 'selected_audible'
    audible = [a for a in rows if a.get('audible') and not a.get('host_ai')]
    if not audible:
        return None, 'no_audible_source'
    if len(audible) > 1:
        if not selected_dead:
            # Two sources are playing: switching would be a guess. Leave it to the user.
            return None, 'multiple_audible_sources'
        reviewed = [a for a in audible
                    if str(a.get('tag_status') or '') == 'external_review'
                    and str(a.get('tag') or '') in {'music', 'video'}]
        if not reviewed:
            return None, 'multiple_audible_sources'
        audible = reviewed
    target = max(audible, key=lambda a: float(a.get('peak') or 0.0)) if len(audible) > 1 else audible[0]
    if make_id(target) == selected_id:
        return None, 'already_selected'
    reviewed = (str(target.get('tag_status') or '') == 'external_review'
                and str(target.get('tag') or '') in {'music', 'video'})
    known = bool(is_known_player(target)) if callable(is_known_player) else False
    if not (reviewed or known):
        return None, 'audible_source_unreviewed'
    return target, 'follow_audible_reviewed_source'


class VisionReviewPolicy:
    """Cost control: per-source gap, hourly cap, and failure backoff.

    Failure handling is deliberately split in two, because the old single
    global backoff turned three unremarkable "this window is not media" answers
    (a login page, an ad frame, a cut between scenes) into a ten-minute freeze
    of every source at once. That is exactly what a user reports as "切了三次
    之后就卡住了".

    * a *hard* failure (the cross-plugin vision call itself failed) means the
      pipeline is broken, so it still backs the whole subsystem off;
    * a *soft* miss (window plugin answered, just not video/music) only cools
      that one source down briefly. Other sources stay reviewable.
    """

    def __init__(self, *, clock=time.monotonic, gap_s: float = MIN_ATTEMPT_GAP_S,
                 hourly_cap: int = MAX_ATTEMPTS_PER_HOUR,
                 failure_backoff_s: float = FAILURE_BACKOFF_S,
                 failure_streak_limit: int = FAILURE_STREAK_LIMIT,
                 source_cooldown_s: float = SOURCE_COOLDOWN_S,
                 source_streak_limit: int = SOURCE_STREAK_LIMIT) -> None:
        self._clock = clock
        self.gap_s = float(gap_s)
        self.hourly_cap = int(hourly_cap)
        self.failure_backoff_s = float(failure_backoff_s)
        self.failure_streak_limit = int(failure_streak_limit)
        self.source_cooldown_s = float(source_cooldown_s)
        self.source_streak_limit = int(source_streak_limit)
        self._last_attempt: dict[str, float] = {}
        self._attempts: deque[float] = deque(maxlen=256)
        self._failures = 0
        self._blocked_until = 0.0
        self._source_failures: dict[str, int] = {}
        self._source_blocked_until: dict[str, float] = {}
        self.last: dict = {}
        self.labelled_total = 0

    def _trim(self, now: float) -> None:
        while self._attempts and now - self._attempts[0] > 3600.0:
            self._attempts.popleft()

    def allow(self, source_id: str):
        """(allowed, reason). Never mutates; call note() after a real attempt."""
        now = self._clock()
        self._trim(now)
        if now < self._blocked_until:
            return False, 'failure_backoff'
        if now < float(self._source_blocked_until.get(str(source_id), 0.0) or 0.0):
            return False, 'source_backoff'
        last = self._last_attempt.get(str(source_id))
        if last is not None and now - last < self.gap_s:
            return False, 'not_due'
        if len(self._attempts) >= self.hourly_cap:
            return False, 'hourly_cap'
        return True, 'due'

    @staticmethod
    def _is_hard_failure(outcome: str, reason: str) -> bool:
        if str(outcome) == 'labelled':
            return False
        text = (str(outcome) + ' ' + str(reason or '')).lower()
        return ('call_failed' in text or 'raised' in text or 'timeout' in text
                or 'vision_error' in text or 'error' in text)

    def note(self, source_id: str, outcome: str, *, tag: str = '', reason: str = '') -> None:
        now = self._clock()
        sid = str(source_id)
        self._last_attempt[sid] = now
        self._attempts.append(now)
        self._trim(now)
        ok = outcome == 'labelled'
        if ok:
            self._failures = 0
            self._source_failures.pop(sid, None)
            self._source_blocked_until.pop(sid, None)
            self.labelled_total += 1
        elif self._is_hard_failure(outcome, reason):
            self._failures += 1
            if self._failures >= self.failure_streak_limit:
                self._blocked_until = now + self.failure_backoff_s
                self._failures = 0
        else:
            # Soft miss: cool this source only, never its neighbours.
            streak = self._source_failures.get(sid, 0) + 1
            if streak >= self.source_streak_limit:
                self._source_blocked_until[sid] = now + self.source_cooldown_s
                streak = 0
            self._source_failures[sid] = streak
        self.last = {'at': round(time.time(), 3), 'source_id': str(source_id)[:64],
                     'outcome': str(outcome)[:40], 'tag': str(tag or '')[:16],
                     'reason': str(reason or '')[:80]}

    def status(self) -> dict:
        now = self._clock()
        self._trim(now)
        return {'gap_s': self.gap_s, 'hourly_cap': self.hourly_cap,
                'attempts_last_hour': len(self._attempts),
                'cooling_down_s': round(max(0.0, self._blocked_until - now), 1),
                'sources_cooling_down': sorted(
                    sid for sid, until in self._source_blocked_until.items()
                    if until > now)[:8],
                'source_cooldown_s': self.source_cooldown_s,
                'labelled_total': self.labelled_total,
                'label_ttl_s': REVIEW_LABEL_TTL_S,
                'last': dict(self.last)}
