"""为她所听·媒体模式：游戏 / 音乐 / 视频 / 一起看。

Pure logic + one small shared JSON file; no SDK imports (unit-testable).

* ``detect_game``: is this audible app a game?  exe table, then install path
  hints (steamapps / Epic / WeGame / Riot ...).  Returns (is_game, genre_key).
* ``GENRES``: genre table -> what she pays attention to on screen and in the
  sound, and how she tends to react.  Shared with 为她所控 through the mode file.
* ``MediaModeFile``: ``%LOCALAPPDATA%/N.E.K.O/media_mode_v1.json`` - what she is
  "doing with" the owner right now (game / music / video / none), plus a few
  recent lyric or narration lines.  为她所控 reads it to take a tier-throttled
  screen glance (kills, enemies, hints / picture + danmaku) and comment.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = "neko.media_mode.v1"
MODES = ("none", "game", "music", "video")
TOGETHER_MINUTES = (5, 240)

# genre_key -> (中文名, 画面关注点, 声音关注点, 她的反应方式)
GENRES: dict[str, tuple[str, str, str, str]] = {
    "fps": ("FPS 射击", "击杀提示、血量、弹药、小地图上的敌人、死亡回放", "枪声方向、脚步声、爆炸",
            "紧张时小声提醒，击杀时欢呼，被打死时心疼或吐槽"),
    "open_world": ("开放世界", "地图标记、任务提示、风景、角色对话", "环境音乐、对白",
                   "好奇地问去哪、夸风景、提醒任务"),
    "sandbox": ("沙盒建造", "正在建的东西、材料栏、生物", "挖掘放置声、怪物声",
                "夸建筑、给点子、听到怪物声提醒"),
    "horror": ("恐怖", "黑暗角落、突然出现的东西、提示文字", "心跳、尖叫、低频环境音",
               "害怕地缩一下、陪着壮胆、被吓到后埋怨"),
    "puzzle": ("解谜", "谜题机关、线索文字、提示", "机关声、提示音",
               "一起想办法、小心地给提示、解开时开心"),
    "moba": ("MOBA", "击杀播报、经济、小地图、团战", "击杀播报音、技能音效",
             "团战时紧张、拿人时欢呼、被抓时安慰"),
    "rpg": ("角色扮演 RPG", "剧情对话、角色、装备、等级", "剧情配乐、配音",
            "跟着剧情感慨、讨论角色、吐槽选择"),
    "action": ("动作", "连招、Boss 血条、闪避提示", "打击声、Boss 吼叫",
               "Boss 战时加油、死了陪着不甘心"),
    "racing": ("竞速", "名次、速度、弯道、对手", "引擎声、漂移声",
               "超车时兴奋、撞车时哎呀"),
    "fighting": ("格斗", "双方血条、连招、回合", "打击声、KO 播报",
                 "跟着喊、赢了欢呼、输了不服"),
    "rhythm": ("音游", "判定、连击数、谱面", "歌曲本身",
               "跟着节奏、夸手速、聊这首歌"),
    "management": ("模拟经营", "资金、建筑、人口、事件提示", "轻松的背景音乐",
                   "出主意、关心数字、发现问题提醒"),
    "strategy": ("策略", "地图、部队、资源、回合", "战斗音效、提示音",
                 "分析局势、紧张地看着、赢了得意"),
    "card": ("卡牌", "手牌、对手场面、血量", "出牌音效",
             "帮着算、好牌时激动、被翻盘时哀嚎"),
    "survival": ("生存", "饥饿口渴血量、夜晚、怪物", "怪物声、天气",
                 "担心他饿肚子、天黑提醒、陪着熬"),
    "game": ("游戏", "正在发生的事、提示文字、角色状态", "游戏音效与音乐",
             "像一起看的朋友那样偶尔说一句"),
}

# exe (casefold) -> genre.  Small curated table; unknown games fall to "game".
GAME_APPS: dict[str, str] = {
    "valorant.exe": "fps", "valorant-win64-shipping.exe": "fps", "cs2.exe": "fps", "csgo.exe": "fps",
    "r5apex.exe": "fps", "overwatch.exe": "fps", "cod.exe": "fps", "tslgame.exe": "fps",
    "deltaforceclient-win64-shipping.exe": "fps", "naraka.exe": "action",
    "genshinimpact.exe": "open_world", "yuanshen.exe": "open_world", "starrail.exe": "rpg",
    "zenlesszonezero.exe": "action", "wuthering waves.exe": "open_world", "client-win64-shipping.exe": "open_world",
    "eldenring.exe": "action", "sekiro.exe": "action", "witcher3.exe": "open_world", "gta5.exe": "open_world",
    "rdr2.exe": "open_world", "cyberpunk2077.exe": "open_world", "b1-win64-shipping.exe": "action",
    "minecraft.exe": "sandbox", "javaw.exe": "", "terraria.exe": "sandbox", "factorio.exe": "management",
    "phasmophobia.exe": "horror", "re4.exe": "horror", "re2.exe": "horror", "outlast2.exe": "horror",
    "portal2.exe": "puzzle", "thewitness.exe": "puzzle", "baba is you.exe": "puzzle",
    "league of legends.exe": "moba", "dota2.exe": "moba", "smoba.exe": "moba",
    "bg3.exe": "rpg", "persona5.exe": "rpg", "ffxv.exe": "rpg",
    "forzahorizon5.exe": "racing", "forzahorizon4.exe": "racing",
    "streetfighter6.exe": "fighting", "tekken8.exe": "fighting",
    "osu!.exe": "rhythm", "phigros.exe": "rhythm", "arcaea.exe": "rhythm",
    "stardew valley.exe": "management", "cities2.exe": "management", "citiesskylines.exe": "management",
    "civilizationvi.exe": "strategy", "eu4.exe": "strategy", "stellaris.exe": "strategy",
    "hearthstone.exe": "card", "slay the spire.exe": "card", "balatro.exe": "card",
    "dontstarve_steam.exe": "survival", "valheim.exe": "survival", "palworld-win64-shipping.exe": "survival",
}
_PATH_HINTS = ("\\steamapps\\common\\", "\\epic games\\", "\\wegame", "\\riot games\\",
               "\\ubisoft game launcher\\games\\", "\\gog galaxy\\games\\", "\\battle.net\\",
               "\\genshin impact", "\\star rail", "\\miHoYo".casefold(), "\\hoyoplay\\", "\\xboxgames\\")
_NOT_GAMES = frozenset({"steam.exe", "steamwebhelper.exe", "epicgameslauncher.exe", "wegame.exe",
                        "n.e.k.o.exe", "obs64.exe", "discord.exe", "qq.exe", "wechat.exe"})


def detect_game(app: dict | None) -> tuple[bool, str]:
    """(is_game, genre_key).  Never guesses from audio alone."""
    if not isinstance(app, dict):
        return False, ""
    name = str(app.get("name") or "").casefold()
    if not name or name in _NOT_GAMES or app.get("host_ai"):
        return False, ""
    genre = GAME_APPS.get(name)
    if genre:
        return True, genre
    path = str(app.get("path") or "").casefold().replace("/", "\\")
    if path and any(h in path for h in _PATH_HINTS):
        return True, "game"
    return False, ""


def genre_info(key: str) -> dict[str, str]:
    name, screen, sound, react = GENRES.get(key or "game", GENRES["game"])
    return {"key": key if key in GENRES else "game", "name": name, "screen_focus": screen,
            "sound_focus": sound, "reaction": react}


def genre_table() -> list[dict[str, str]]:
    return [genre_info(k) for k in GENRES]


def mode_path() -> Path:
    forced = os.environ.get("NEKO_MEDIA_MODE_FILE")
    if forced:
        return Path(forced)
    base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
    return base / "N.E.K.O" / "media_mode_v1.json"


_SAFE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clip(text: Any, limit: int) -> str:
    return " ".join(_SAFE.sub("", str(text or "")).split())[:limit]


class MediaModeFile:
    """Writer side (为她所听).  Atomic replace; never raises to callers."""

    def __init__(self, path: Path | None = None, clock=time.time):
        self._path = path or mode_path()
        self._clock = clock
        self._lock = threading.Lock()
        self.state: dict[str, Any] = {"schema": SCHEMA, "mode": "none", "app": "", "genre": "",
                                      "title": "", "lines": [], "together_until": 0.0,
                                      "together_kind": "", "updated_at": 0.0, "bpm": None}
        old = read_mode(self._path, max_age_s=10 ** 9)
        if old:
            self.state["together_until"] = float(old.get("together_until") or 0)
            self.state["together_kind"] = str(old.get("together_kind") or "")

    def together_active(self) -> bool:
        return self._clock() < float(self.state.get("together_until") or 0)

    def start_together(self, kind: str, minutes: float) -> dict[str, Any]:
        kind = kind if kind in ("video", "music", "game") else "video"
        minutes = max(TOGETHER_MINUTES[0], min(TOGETHER_MINUTES[1], float(minutes or 60)))
        self.state.update(together_until=self._clock() + minutes * 60, together_kind=kind)
        self._write()
        return {"kind": kind, "minutes": int(minutes)}

    def stop_together(self) -> None:
        self.state.update(together_until=0.0, together_kind="")
        self._write()

    def update(self, *, mode: str, app: str = "", genre: str = "", title: str = "",
               lines: list[str] | None = None, bpm: Any = None) -> bool:
        """Write only when something meaningful changed or every 20 s (heartbeat)."""
        mode = mode if mode in MODES else "none"
        new_lines = [_clip(x, 120) for x in (lines or []) if _clip(x, 120)][-8:]
        changed = (mode != self.state["mode"] or _clip(app, 80) != self.state["app"]
                   or genre != self.state["genre"] or _clip(title, 120) != self.state["title"]
                   or new_lines != self.state["lines"])
        stale = self._clock() - float(self.state.get("updated_at") or 0) > 20
        if not (changed or stale):
            return False
        self.state.update(mode=mode, app=_clip(app, 80), genre=genre if genre in GENRES else "",
                          title=_clip(title, 120), lines=new_lines,
                          bpm=(round(float(bpm), 1) if isinstance(bpm, (int, float)) else None))
        self._write()
        return changed

    def _write(self) -> None:
        self.state["updated_at"] = round(self._clock(), 1)
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_name(self._path.name + "." + secrets.token_hex(6) + ".tmp")
                tmp.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, self._path)
        except OSError:
            pass


def read_mode(path: Path | None = None, *, max_age_s: float = 45.0,
              now: float | None = None) -> dict[str, Any] | None:
    """Reader side (为她所控).  Stale file == no media mode."""
    try:
        data = json.loads((path or mode_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None
    now = time.time() if now is None else now
    if now - float(data.get("updated_at") or 0) > max_age_s:
        return None
    return data


def read_her_mood(*, max_age_s: float = 900.0) -> dict[str, Any]:
    """为她所想 publishes her mood + willingness here; missing or stale → {}."""
    forced = os.environ.get("NEKO_HER_MOOD_FILE")
    p = Path(forced) if forced else Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()) / "N.E.K.O" / "her_mood_v1.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("schema") != "neko.her_mood.v1":
        return {}
    if time.time() - float(data.get("updated_at") or 0) > max_age_s:
        return {}
    return data
