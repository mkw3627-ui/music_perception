"""Conservative, metadata-assisted classification of a *mixed* output stream.

No acoustic-only detector can tell whether a song was emitted by a music player
or by a video. Windows SMTC is used as *evidence*, not proof of audible PCM:
playback sessions can be silent and some renderers have no SMTC integration.

Policy: two distinct active media sources or active AI TTS with unisolated
output => block. An unverified 'video-like' speech window is NOT called video.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

VALID_ROLES = {'music', 'video', 'ai', 'unknown'}


def normal_role(value):
    name = str(value or '').strip().lower()
    if name in {'video', 'movie'}:
        return 'video'
    if name in {'music', 'audio'}:
        return 'music'
    if name in {'ai', 'tts', 'assistant'}:
        return 'ai'
    return 'unknown'


def _playback_role(session, overrides):
    app = str(session.get('app') or session.get('player') or '').strip().casefold()
    override = normal_role((overrides or {}).get(app))
    if override != 'unknown':
        return override, 'user_marked_app'
    playback_type = str(session.get('playback_type') or '').strip().lower()
    # Windows MediaPlaybackType can be returned as 'Music'/'Video' or enum
    # numeric 1/2. Only use the explicit media type provided by Windows.
    if playback_type in {'music', '1', 'mediaplaybacktype.music'}:
        return 'music', 'windows_playback_type'
    if playback_type in {'video', '2', 'mediaplaybacktype.video'}:
        return 'video', 'windows_playback_type'
    return 'unknown', 'unclassified_application'


@dataclass(frozen=True)
class SceneDecision:
    kind: str
    verified: bool
    reason: str
    source_count: int
    source_roles: tuple[str, ...]
    music_score: float = 0.0
    speech_score: float = 0.0

    def public(self):
        value = asdict(self)
        value['source_roles'] = list(self.source_roles)
        return value


def classify_scene(decision, *, snapshot=None, source_roles=None,
                   ai_speaking=False, ai_routed_elsewhere=False,
                   require_verified_source=False):
    """Classify one window. Do not use AI output as recognition input.

    ``ai_routed_elsewhere`` can be true ONLY when the user explicitly selected
    a DIFFERENT physical/virtual output capture endpoint and confirms TTS is
    routed away. The plugin cannot infer it by monitoring PCM alone.
    """
    music_score = float(getattr(decision, 'music_score', 0) or 0)
    speech_score = float(getattr(decision, 'speech_score', 0) or 0)
    has_music = bool(getattr(decision, 'music', False))
    if ai_speaking and not ai_routed_elsewhere:
        return SceneDecision('ai_output', False, 'neko_ai_tts_active_on_mixed_output',
                             0, (), music_score, speech_score)
    sessions = []
    for item in (snapshot or {}).get('sessions', []):
        if isinstance(item, dict) and item.get('playing') is True:
            app = str(item.get('app') or item.get('player') or '').strip()[:160]
            if app:
                # Multiple sessions from the SAME browser are still distinct
                # playing sources. The OS enumerator should supply one entry
                # per actual session, not multiple entries for 'current'.
                sessions.append({**item, 'app': app})
    roles = [(_playback_role(s, source_roles)[0], s) for s in sessions]
    if (snapshot or {}).get('incomplete_source_probe') and len(sessions) < 2:
        return SceneDecision('unknown', False, 'media_session_probe_incomplete',
                             len(sessions), tuple(role for role, _ in roles),
                             music_score, speech_score)
    if any(role == 'ai' for role, _ in roles) and not ai_routed_elsewhere:
        return SceneDecision('ai_output', False, 'ai_app_audio_not_isolated',
                             len(sessions), tuple(role for role, _ in roles), music_score, speech_score)
    # A source manually labeled AI can safely be ignored only if it is known
    # to be outside the selected monitoring endpoint.
    non_ai = [(role, item) for role, item in roles if role != 'ai']
    if len(non_ai) > 1:
        return SceneDecision('mixed', False, 'multiple_simultaneous_media_sessions',
                             len(non_ai), tuple(role for role, _ in non_ai), music_score, speech_score)
    if len(non_ai) == 1:
        role, session = non_ai[0]
        if role == 'video':
            return SceneDecision('video', True, 'verified_video_media_session',
                                 1, ('video',), music_score, speech_score)
        if role == 'music':
            if has_music:
                return SceneDecision('music', True, 'verified_music_media_session',
                                     1, ('music',), music_score, speech_score)
            return SceneDecision('unknown', False, 'music_session_not_acoustically_confirmed',
                                 1, ('music',), music_score, speech_score)
        # A browser's title is NOT proof of video; require a deliberate app
        # marking or Windows playback type, otherwise leave speech unclassified.
        if has_music:
            # Helpful in an open/acoustic-only mode, but unsafe to call this a
            # single source: an untyped player may be a video with soundtrack.
            return SceneDecision('unknown' if require_verified_source else 'music',
                                 False, 'unknown_player_type_not_verified',
                                 1, ('unknown',), music_score, speech_score)
        return SceneDecision('unknown', False, 'media_session_type_not_verified',
                             1, ('unknown',), music_score, speech_score)
    if has_music:
        return SceneDecision('unknown' if require_verified_source else 'music',
                             False, 'music_yamnet_only_no_source_proof',
                             0, (), music_score, speech_score)
    return SceneDecision('unknown', False, 'no_verified_single_audio_source',
                         0, (), music_score, speech_score)
