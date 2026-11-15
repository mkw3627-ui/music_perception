from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source():
    return (ROOT / '__init__.py').read_text(encoding='utf8')


def test_glance_has_listen_branch_when_screen_has_no_media():
    text = _source()
    assert 'async def _glance_listen_branch(self' in text
    # Passive glance falls through to the listen branch when screen is not video/music.
    assert 'return await self._glance_listen_branch()' in text


def test_listen_branch_reads_volume_only_and_never_records_pcm():
    body = _source().split('async def _glance_listen_branch(', 1)[1].split('async def _summon_once(', 1)[0]
    assert '_refresh_audible_cache' in body
    assert 'publish=False' in body
    # Passive listen only labels; it must not open a PCM capture stream here.
    assert 'start_capture' not in body
    assert 'record_pcm' not in body and 'capture_stream' not in body


def test_summon_accepts_optional_target_and_reports_missing_honestly():
    text = _source()
    assert 'async def summon_glance_now(self, target: str = ""' in text
    assert 'target_not_found' in text
    assert '如实告诉你' in text


def test_summon_vague_path_uses_process_lock_branch():
    body = _source().split('async def _summon_once(', 1)[1].split('@plugin_entry', 1)[0]
    assert "_summon_vague_by_process()" in body
    vague = _source().split('async def _summon_vague_by_process(', 1)[1].split('async def _summon_once(', 1)[0]
    assert "verify_selected_process(pid, created)" in vague
    assert "[:5]" in vague
    assert '"comment_engagement": "read_only"' in vague


def test_vague_window_filter_matches_app_name_not_pid():
    # list_open_windows exposes no PID; the filter must match the locked app name
    # and skip windows that are not capturable.
    vague = _source().split('async def _summon_vague_by_process(', 1)[1].split('async def _summon_once(', 1)[0]
    assert 'w.get("pid") == pid' not in vague
    assert 'app_name' in vague
    assert 'capture_available' in vague
    assert 'not w.get("minimized")' in vague


def test_summon_skips_speak_roll_and_respects_mood_refusal():
    body = _source().split('async def _summon_once(', 1)[1].split('@plugin_entry', 1)[0]
    assert 'decide_summon() == "refused"' in body
    assert 'speak_roll' not in body


def test_summon_outcomes_are_reported_ok():
    text = _source()
    for outcome in ('nothing_audible', 'silent', 'target_not_found'):
        assert f'"{outcome}"' in text


def test_browser_window_filter_excludes_hidden_and_minimized():
    # Source tree: ROOT.parents[1] is the N.E.K.O root. Installed copy: it is
    # .neko-plugin-installations itself, so the sibling plugin is one level up.
    if ROOT.parents[1].name == '.neko-plugin-installations':
        browser = ROOT.parents[0] / 'browser_history_insights' / '__init__.py'
    else:
        browser = ROOT.parents[1] / '.neko-plugin-installations' / 'plugins' / 'browser_history_insights' / '__init__.py'
    if not browser.exists():
        import pytest
        pytest.skip('browser_history_insights installation copy not present in this layout')
    text = browser.read_text(encoding='utf8')
    body = text.split('if not window_ref:', 1)[1].split('timeline, app, video_n', 1)[0]
    assert 'not w.get("minimized")' in body
    assert 'display_state' in body
    assert 'capture_available' in body
