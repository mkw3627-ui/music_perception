"""Shared learning inbox: perception plugins append, 为她所想 ingests.

One JSON object per line at %LOCALAPPDATA%/N.E.K.O/shared_learning_inbox_v1.jsonl.
Identical copies live in 为她所听 / 为她所控 / 为她所想 (plugins never import
each other). Writers only append; the reader keeps its own byte cursor.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = "neko.shared_learning_inbox.v1"
MAX_BYTES = 4 * 1024 * 1024
KINDS = frozenset({
    "song_heard", "lyrics_crosscheck", "hot_word", "bili_video", "bili_comments",
    "bili_danmaku", "web_note", "curiosity", "owner_activity", "interaction",
    # v032
    "presence", "owner_pref", "media_moment", "game_moment", "mood_event",
})
_LOCK = threading.Lock()


def inbox_path() -> Path:
    forced = os.environ.get("NEKO_SHARED_INBOX")
    if forced:
        return Path(forced)
    base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
    return base / "N.E.K.O" / "shared_learning_inbox_v1.jsonl"


def _clean(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def post(source: str, kind: str, *, title: str = "", text: str = "",
         tags: list[str] | None = None, url: str = "", extra: dict | None = None,
         path: Path | None = None) -> bool:
    """Append one observation. Never raises; returns False on failure."""
    if kind not in KINDS:
        return False
    item = {
        "schema": SCHEMA, "id": uuid.uuid4().hex[:16], "ts": time.time(),
        "source": _clean(source, 40), "kind": kind,
        "title": _clean(title, 160), "text": _clean(text, 1200),
        "tags": [_clean(t, 24) for t in (tags or []) if _clean(t, 24)][:12],
        "url": _clean(url, 400),
    }
    if isinstance(extra, dict) and extra:
        try:
            blob = json.loads(json.dumps(extra, ensure_ascii=False, default=str))
            if len(json.dumps(blob, ensure_ascii=False)) <= 4000:
                item["extra"] = blob
        except (TypeError, ValueError):
            pass
    p = path or inbox_path()
    line = json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _LOCK:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                if p.stat().st_size > MAX_BYTES:
                    _rotate(p)
            except OSError:
                pass
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(line)
            return True
        except OSError:
            return False


def _rotate(p: Path) -> None:
    old = p.with_suffix(".old.jsonl")
    try:
        if old.exists():
            old.unlink()
    except OSError:
        pass
    try:
        os.replace(p, old)
    except OSError:
        pass


def read_since(cursor: dict | None, *, limit: int = 200,
               path: Path | None = None) -> tuple[list[dict], dict]:
    """Return (items, new_cursor). Cursor = {"offset": int, "size_id": float}.

    Rotation is detected when the file shrinks below the cursor offset.
    """
    p = path or inbox_path()
    cur = dict(cursor or {})
    offset = int(cur.get("offset") or 0)
    items: list[dict] = []
    try:
        size = p.stat().st_size
    except OSError:
        return [], {"offset": 0}
    if size < offset:
        offset = 0
    try:
        with open(p, "rb") as fh:
            fh.seek(offset)
            while len(items) < limit:
                raw = fh.readline()
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    break  # partial write; retry next time
                offset += len(raw)
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
                if isinstance(obj, dict) and obj.get("schema") == SCHEMA and obj.get("kind") in KINDS:
                    items.append(obj)
    except OSError:
        return [], {"offset": int(cur.get("offset") or 0)}
    return items, {"offset": offset}
