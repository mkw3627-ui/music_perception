"""Stable cognition bridge v1 for ``为她所听``.

The facade returns only authorised, bounded, structured observations.  It does
not export PCM/audio files and does not create chat messages.

PUBLIC CONTRACT FREEZE -- DO NOT MODIFY IN NORMAL FEATURE UPDATES
================================================================
Stable entry IDs: ``cog_describe_v1``, ``cog_poll_v1``, ``cog_observe_v1``.
Existing request fields and response semantics must remain compatible.
Internal ASR/source-separation/music-analysis implementations may change.
Breaking changes require a new v2 facade while v1 remains available.
================================================================
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any

SCHEMA_CAPABILITIES = "cognition.capabilities.v1"
SCHEMA_OBSERVATION = "cognition.observation.v1"
SOURCE_PLUGIN = "music_perception"
_ALLOWED_STATUS = {"ok", "empty", "denied", "unavailable", "stale", "error"}


def _utc(ts: float | int | None = None) -> str:
    value = time.time() if ts is None else float(ts)
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _text(value: Any, limit: int = 320) -> str:
    return " ".join(str(value or "").split()).strip()[:limit]


def _compact(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return "[truncated]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, str):
        return _text(value, 800)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in list(value.items())[:64]:
            name = _text(key, 80)
            if name:
                out[name] = _compact(item, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        return [_compact(x, depth=depth + 1) for x in list(value)[:32]]
    return _text(value, 160)


def _digest(value: Any) -> str:
    raw = json.dumps(_compact(value), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def _unwrap(value: Any) -> dict[str, Any]:
    data = getattr(value, "value", value)
    return data if isinstance(data, dict) else {}


def _event(*, kind: str, subject_type: str, subject_id: str, summary: str,
           epistemic: str = "model_inferred", features: dict[str, Any] | None = None,
           observed_at: float | int | None = None, ttl_seconds: int = 600) -> dict[str, Any]:
    safe_features = _compact(features or {})
    identity = {"kind": kind, "subject_type": subject_type,
                "subject_id": subject_id, "features": safe_features}
    stamp = time.time() if observed_at is None else float(observed_at)
    return {
        "event_id": f"{SOURCE_PLUGIN}:{kind}:{_digest(identity)}",
        "modality": "audio",
        "kind": kind,
        "observed_at": _utc(stamp),
        "time_basis": "observed" if observed_at else "received",
        "subject": {"type": subject_type, "id": _text(subject_id, 160)},
        "summary": _text(summary, 320),
        "epistemic_status": epistemic,
        "evidence_refs": [f"{SOURCE_PLUGIN}:{_digest(identity)}"],
        "features": safe_features,
        "expires_at": _utc(stamp + max(60, min(86400, int(ttl_seconds)))),
    }


def _batch(status: str, observations: list[dict[str, Any]] | None = None, *,
           cursor: str = "", reason_code: str = "", message: str = "") -> dict[str, Any]:
    if status not in _ALLOWED_STATUS:
        status = "error"
    return {
        "schema_version": SCHEMA_OBSERVATION,
        "source_plugin": SOURCE_PLUGIN,
        "status": status,
        "reason_code": _text(reason_code, 96),
        "message": _text(message, 240),
        "observations": list(observations or [])[:32],
        "next_cursor": _text(cursor, 128),
    }


class HearingCognitiveBridgeV1:
    def __init__(self, plugin: Any):
        self.plugin = plugin

    def describe(self) -> dict[str, Any]:
        consent = bool(getattr(self.plugin, "_auto_consented", False))
        paused = bool(getattr(self.plugin, "_plugin_paused", False))
        shutting_down = bool(getattr(self.plugin, "_shutdown_requested", False))
        active = consent and not paused and not shutting_down
        ai_share = bool(getattr(self.plugin, "_ai_delivery", False))
        channel_share = bool(getattr(self.plugin, "_external_channel_share", False))
        selected = bool(getattr(self.plugin, "_selected_application", None))
        return {
            "schema_version": SCHEMA_CAPABILITIES,
            "source_plugin": SOURCE_PLUGIN,
            "status": "ok",
            "contract": "stable-v1-do-not-modify",
            "capabilities": [
                {"capability_id": "audio.current_media", "mode": "read_only",
                 "requires_owner_grant": True, "available": active and ai_share,
                 "description": "读取当前已验证音乐或视频声音摘要；不复用上一首。"},
                {"capability_id": "audio.auditory_context", "mode": "read_only",
                 "requires_owner_grant": True, "available": active and ai_share,
                 "description": "读取带时间的音乐结构、歌词或视频语音候选；不返回 PCM。"},
                {"capability_id": "audio.source_list", "mode": "read_only",
                 "requires_owner_grant": True, "available": active and ai_share,
                 "description": "读取当前有声应用与候选标签；候选不等于已验证来源。"},
                {"capability_id": "audio.selected_app", "mode": "read_only",
                 "requires_owner_grant": True,
                 "available": active and ai_share and channel_share and selected,
                 "description": "读取已授权单应用声道的结构化文字或音乐分析。"},
                {"capability_id": "audio.scene", "mode": "read_only",
                 "requires_owner_grant": False, "available": True,
                 "description": "读取音乐/视频/混音/AI 输出的有限来源状态，不含正文。"},
                {"capability_id": "audio.current_lyrics", "mode": "read_only",
                 "requires_owner_grant": True, "available": active and ai_share,
                 "description": "读取当前歌曲已识别的歌词候选；不是官方歌词事实。"},
            ],
            "runtime_status": {
                "monitoring_authorized": consent,
                "plugin_paused": paused,
                "shutting_down": shutting_down,
                "ai_share_authorized": ai_share,
                "external_channel_share_authorized": channel_share,
                "selected_application_present": selected,
            },
            "forbidden_through_v1": [
                "raw_pcm", "audio_file_export", "microphone_capture", "permission.modify",
                "model.download", "source.route_change",
            ],
        }

    def _status_from_payload(self, data: dict[str, Any]) -> tuple[str, str, str]:
        if data.get("ok") is False:
            reason = _text(data.get("reason") or data.get("error"), 200)
            denied = any(word in reason.casefold() for word in
                         ("授权", "允许", "permission", "consent", "share disabled"))
            return ("denied" if denied else "unavailable",
                    "audio_permission_denied" if denied else "audio_unavailable", reason)
        if data.get("available") is False:
            # A verified source-state (video/mixed/AI/unknown) is still useful
            # cognition evidence even when no transcript/music body may be shared.
            scene = _text(data.get("scene_kind"), 40)
            if scene in {"video", "mixed", "ai_output", "unknown", "temporarily_unknown"}:
                return "ok", "", ""
            reason = _text(data.get("reason"), 200)
            return "empty", "no_current_audio_observation", reason
        return "ok", "", ""

    def _media_event(self, data: dict[str, Any]) -> dict[str, Any]:
        scene = _text(data.get("scene_kind") or ((data.get("context") or {}).get("scene_kind")), 40)
        if not scene:
            scene = "music" if data.get("context") or data.get("current_song") else "unknown"
        current_song = data.get("current_song") if isinstance(data.get("current_song"), dict) else {}
        current_id = _text(data.get("current_song_id") or current_song.get("id") or scene, 160)
        if scene == "video":
            video = data.get("video_audio_context") if isinstance(data.get("video_audio_context"), dict) else {}
            if not video:
                return _event(kind="audio.source_state", subject_type="audio_scene",
                              subject_id="video", epistemic="system_verified",
                              summary="当前音源已识别为单一视频，但尚无可共享的本地语音候选。",
                              features={"scene_kind": scene, "reason": data.get("reason"),
                                        "transcript_available": False}, ttl_seconds=120)
            return _event(kind="audio.video_speech_observed", subject_type="video_audio",
                          subject_id=current_id or "current-video",
                          summary="当前已验证单一视频来源存在本地语音候选；未读取画面。",
                          features={"scene_kind": scene, "video_audio_context": video,
                                    "transcript_is_candidate": True}, ttl_seconds=300)
        if scene in {"mixed", "ai_output", "unknown"}:
            return _event(kind="audio.source_state", subject_type="audio_scene",
                          subject_id=scene, epistemic="system_verified",
                          summary="当前音频来源无法安全作为独立外部内容读取。",
                          features={"scene_kind": scene, "reason": data.get("reason")},
                          ttl_seconds=120)
        if scene == "temporarily_unknown" and not (data.get("current_song_id") or data.get("current_lyrics")):
            return _event(kind="audio.source_state", subject_type="audio_scene",
                          subject_id=scene, epistemic="system_verified",
                          summary="当前音源暂时无法确认，且没有可归属于本首的安全历史候选。",
                          features={"scene_kind": scene, "reason": data.get("reason")},
                          ttl_seconds=120)
        features = {
            "scene_kind": scene,
            "current_song_id": data.get("current_song_id"),
            "current_song": current_song,
            "context": data.get("context") or {},
            "current_lyrics": data.get("current_lyrics") or {},
            "recent_music_parts": data.get("recent_music_parts") or [],
            "music_analysis_is_candidate": True,
        }
        return _event(kind="audio.music_observed", subject_type="music_track",
                      subject_id=current_id or "current-music",
                      summary="当前音乐有可用的结构化节拍、旋律、声层或歌词候选；审美评价仍由角色自己形成。",
                      features=features, ttl_seconds=600)

    async def poll(self, *, cursor: str = "", limit: int = 20) -> dict[str, Any]:
        try:
            media = _unwrap(await self.plugin.get_current_media_for_ai())
            status, code, message = self._status_from_payload(media)
            snapshot = _compact(media)
            next_cursor = _digest(snapshot)
            if cursor and cursor == next_cursor:
                return _batch("empty", cursor=next_cursor, reason_code="unchanged",
                              message="自上次游标后没有新的授权音频摘要。")
            if status != "ok":
                return _batch(status, cursor=next_cursor, reason_code=code, message=message)
            event = self._media_event(media)
            return _batch("ok", [event][:max(1, min(32, int(limit or 20)))], cursor=next_cursor)
        except Exception as exc:
            return _batch("error", reason_code="hearing_bridge_failure",
                          message=f"听觉感知桥读取失败：{type(exc).__name__}")

    async def observe(self, *, capability_id: str, subject_ref: str = "",
                      options: dict[str, Any] | None = None) -> dict[str, Any]:
        capability_id = _text(capability_id, 96)
        opts = options if isinstance(options, dict) else {}
        cursor = _digest({"capability": capability_id, "subject": subject_ref,
                          "options": _compact(opts), "bucket": int(time.time() // 5)})
        try:
            if capability_id in {"audio.current_media", "audio.auditory_context"}:
                history = bool(opts.get("include_recent_history", capability_id == "audio.auditory_context"))
                try:
                    parts = max(1, min(12, int(opts.get("max_parts", 6))))
                except (TypeError, ValueError):
                    parts = 6
                data = _unwrap(await self.plugin.get_ai_auditory_context(
                    include_recent_history=history, max_parts=parts,
                    include_raw_spectrum=False, song_scope="current"))
                status, code, message = self._status_from_payload(data)
                if status != "ok":
                    return _batch(status, cursor=cursor, reason_code=code, message=message)
                return _batch("ok", [self._media_event(data)], cursor=cursor)
            if capability_id == "audio.source_list":
                data = _unwrap(await self.plugin.get_current_audio_sources_for_ai())
                status, code, message = self._status_from_payload(data)
                if data.get("ok") is False:
                    return _batch(status, cursor=cursor, reason_code=code, message=message)
                return _batch("ok", [_event(
                    kind="audio.source_list", subject_type="audio_sources",
                    subject_id="current-session", epistemic="heuristic",
                    summary="读取当前有声应用与候选来源标签；候选标签需要视觉或用户复核。",
                    features={**data, "source_labels_are_candidates": True}, ttl_seconds=120)], cursor=cursor)
            if capability_id == "audio.selected_app":
                data = _unwrap(await self.plugin.get_selected_app_outputs())
                status, code, message = self._status_from_payload(data)
                if status != "ok":
                    return _batch(status, cursor=cursor, reason_code=code, message=message)
                source_id = _text(data.get("source_id"), 160) or "selected-app"
                return _batch("ok", [_event(
                    kind="audio.selected_app_observation", subject_type="application_audio",
                    subject_id=source_id, summary="读取已授权单应用声道的结构化声音摘要；没有导出音轨。",
                    features={**data, "text_candidates_are_unverified": True}, ttl_seconds=300)], cursor=cursor)
            if capability_id == "audio.scene":
                data = _unwrap(await self.plugin.get_media_scene())
                return _batch("ok", [_event(
                    kind="audio.scene", subject_type="audio_scene", subject_id="current-session",
                    epistemic="system_verified", summary="读取当前音频来源仲裁状态；此接口不含歌词或语音正文。",
                    features=data, ttl_seconds=120)], cursor=cursor)
            if capability_id == "audio.current_lyrics":
                # get_current_lyrics_for_ai() never existed in this plugin; the
                # only real read-only lyrics feed is get_live_lyrics(). It stays
                # local unless the owner separately authorised AI sharing.
                if not getattr(self.plugin, "_ai_delivery", False):
                    return _batch("denied", cursor=cursor,
                                  reason_code="ai_delivery_not_authorized",
                                  message="未授权对话 AI 读取歌词候选。")
                live = _unwrap(await self.plugin.get_live_lyrics())
                rows = [s for s in (live.get("songs") or []) if isinstance(s, dict)]
                current = next((s for s in rows if s.get("current")), rows[-1] if rows else {})
                data = {"ok": True,
                        "song_id": live.get("current_song_id"),
                        "cues": list(current.get("cues") or [])[:40],
                        "display_lines": list(current.get("display_lines") or [])[:40],
                        "recognition_status": live.get("recognition_status"),
                        "selected_language": live.get("selected_language"),
                        "recognized_language": live.get("recognized_language")}
                status, code, message = self._status_from_payload(data)
                if data.get("ok") is False:
                    return _batch(status, cursor=cursor, reason_code=code, message=message)
                return _batch("ok", [_event(
                    kind="audio.current_lyrics", subject_type="music_track",
                    subject_id=_text(data.get("song_id"), 160) or "current-song",
                    summary="读取当前歌曲的本地歌词识别候选；可能存在错字、语言切换或时间偏差。",
                    features={**data, "lyrics_are_candidates": True}, ttl_seconds=600)], cursor=cursor)
            return _batch("unavailable", cursor=cursor, reason_code="unknown_capability",
                          message="为她所听未声明该 v1 认知能力。")
        except Exception as exc:
            return _batch("error", cursor=cursor, reason_code="hearing_observe_failure",
                          message=f"按需听觉观察失败：{type(exc).__name__}")
