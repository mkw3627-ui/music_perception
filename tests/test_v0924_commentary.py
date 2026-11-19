"""吐槽：音乐变化 → 人话；节奏/冷却；官方定时吐槽接管与恢复。"""
from __future__ import annotations

import json
import time

from music_perception.commentary import CommentaryState, music_prompt, video_prompt
from music_perception.music_moments import MusicMomentTracker, clip_feel, compare, describe_overall


def report(loud=0.05, bright=1800, bass=0.3, bpm=110, mode='major', mood='情绪不确定', segs=2, jump=1.0):
    seg = []
    for i in range(segs):
        seg.append({'loudness_rms': loud * (jump if i == segs - 1 else 1.0), 'spectral_centroid_hz': bright,
                    'bass_share': bass, 'local_bpm': bpm})
    return {'segments': seg, 'rhythm': {'bpm_estimate': bpm, 'beat_regularity': 0.8},
            'harmony': {'estimated_key': {'mode': mode, 'reliable': True}}, 'emotion': {'label': mood}}


def kinds(a, b):
    return [m.kind for m in compare(clip_feel(a), clip_feel(b))]


def test_loudness_burst_and_calm():
    assert kinds(report(loud=0.02), report(loud=0.06))[0] == 'burst'
    assert kinds(report(loud=0.06), report(loud=0.015))[0] == 'calm'


def test_burst_inside_clip_is_caught():
    assert 'burst' in kinds(report(loud=0.03), report(loud=0.03, jump=2.6))


def test_half_double_time_is_not_a_tempo_change():
    assert not {'faster', 'slower'} & set(kinds(report(bpm=80), report(bpm=160)))
    assert 'faster' in kinds(report(bpm=90), report(bpm=125))


def test_many_changes_read_as_new_style():
    k = kinds(report(loud=0.02, bright=1200, bass=0.2, bpm=80, mode='minor', mood='可能偏忧伤'),
              report(loud=0.07, bright=3000, bass=0.5, bpm=128, mode='major', mood='可能偏欢快'))
    assert k[0] == 'style'


def test_small_wobble_is_ignored():
    assert compare(clip_feel(report()), clip_feel(report(loud=0.053, bright=1900, bpm=112))) == []


def test_key_flip_alone_is_not_a_remark():
    # 实测调式估计相邻片段经常来回翻，单独不能触发
    ms = compare(clip_feel(report(mode='minor')), clip_feel(report(mode='major')))
    assert all(m.strength < 0.45 for m in ms)


def test_real_song_bridge_drop_numbers():
    # 2026-11 实测（Remember 130s→160s）：间奏低音撤掉、响度降，然后整段炸回来
    from music_perception.music_moments import ClipFeel
    a = ClipFeel(0.1681, 0.18, 1.18, 1916, 0.473, 110.3, None, '可能偏欢快', .8)
    b = ClipFeel(0.1366, 0.15, 1.11, 1908, 0.187, 110.3, None, '可能偏欢快', .8)
    c = ClipFeel(0.1675, 0.18, 1.04, 1993, 0.493, 110.3, True, '可能偏欢快', .8)
    assert {m.kind for m in compare(a, b) if m.strength >= .45} >= {'lighter', 'calm'}
    assert {m.kind for m in compare(b, c) if m.strength >= .45} >= {'heavier', 'burst'}


def test_feelings_never_contain_numbers_or_jargon():
    a = report(loud=0.02, bright=1200, bass=0.2, bpm=80, mode='minor', mood='可能偏忧伤')
    b = report(loud=0.07, bright=3000, bass=0.5, bpm=128, mode='major', mood='可能偏欢快')
    texts = [m.feel for m in compare(clip_feel(a), clip_feel(b))] + [describe_overall(clip_feel(b))]
    prompt = music_prompt(texts[0], texts[-1], '某首歌', '')
    for t in texts + [prompt.split('\n')[1]]:
        assert not any(ch.isdigit() for ch in t), t
        for word in ('BPM', 'bpm', 'Hz', '调', '大调', '小调', '节拍器'):
            assert word not in t, (word, t)


def test_tracker_warms_up_and_does_not_repeat_kind():
    tr = MusicMomentTracker()
    assert tr.feed('s1', report(loud=0.02)) [0] is None          # first clip: nothing to compare
    m, _ = tr.feed('s1', report(loud=0.07))
    assert m and m.kind == 'burst'
    tr.mark_said(m)
    tr.feed('s1', report(loud=0.02))                              # calm
    m2, _ = tr.feed('s1', report(loud=0.08))
    assert m2 is None or m2.kind != 'burst'
    m3, _ = tr.feed('s2', report(loud=0.08))                      # new song resets
    assert m3 is None


def test_state_pacing_and_persistence(tmp_path):
    st = CommentaryState(tmp_path / 'c.json')
    assert not st.any_on and st.frequency == 'lively'
    st.music_on = True
    st.saved_proactive = {'proactiveChatEnabled': True}
    st.save()
    again = CommentaryState(tmp_path / 'c.json')
    assert again.music_on and again.saved_proactive == {'proactiveChatEnabled': True}
    now = time.time()
    assert again.why_not_now(now) == ''
    again.last_spoke_at = now
    assert again.why_not_now(now + 10) == 'cooldown'
    assert again.why_not_now(now + again.gap() + 1) == ''
    again.last_user_at = now + again.gap()
    assert again.why_not_now(now + again.gap() + 1) == 'user_active'


def test_video_prompt_asks_for_a_remark_not_a_description():
    p = video_prompt('一只猫从桌子上跳下来', 'bilibili', '哈哈')
    assert '猫' in p and '不用描述整个画面' in p


def test_takeover_saves_and_restore_returns_owner_settings(tmp_path, monkeypatch):
    from music_perception import commentary_mixin as cm
    server = {'proactiveChatEnabled': True, 'proactiveVisionChatEnabled': True,
              'proactiveMusicEnabled': False, 'proactiveVideoChatEnabled': False,
              'proactiveVisionEnabled': True}
    calls = []

    def fake_http(method, path, body=None, timeout=5.0):
        calls.append((method, path, dict(body or {})))
        if method == 'GET':
            return {'success': True, 'settings': dict(server)}
        server.update(body or {})
        return {'success': True}
    monkeypatch.setattr(cm, '_http_json', fake_http)

    class P(cm.CommentaryMixin):
        pass
    p = P()
    p._commentary = CommentaryState(tmp_path / 'c.json')
    import asyncio
    asyncio.run(p._commentary_takeover())
    assert server['proactiveChatEnabled'] is False and server['proactiveVisionChatEnabled'] is False
    assert server['proactiveVisionEnabled'] is True               # privacy switch never touched
    assert all('proactiveVisionEnabled' not in b for _, _, b in calls)
    asyncio.run(p._commentary_takeover())                          # second takeover keeps ORIGINAL values
    assert p._commentary.saved_proactive['proactiveChatEnabled'] is True
    p._commentary_restore_sync()
    assert server['proactiveChatEnabled'] is True and server['proactiveVisionChatEnabled'] is True
    assert p._commentary.saved_proactive is None
    assert json.loads((tmp_path / 'c.json').read_text('utf-8'))['saved_proactive'] is None


def test_page_answer_pauses_official_only_while_owning(tmp_path):
    from music_perception import commentary_mixin as cm

    class P(cm.CommentaryMixin):
        pass
    p = P()
    p._commentary = CommentaryState(tmp_path / 'c.json')
    p._commentary_want_frame_until = 0.0
    p._commentary_restore_token = 0
    assert p._commentary_frame_wanted()['official_paused'] is False
    p._commentary.video_on = True
    p._commentary.saved_proactive = {'proactiveChatEnabled': True}
    ans = p._commentary_frame_wanted()
    assert ans['official_paused'] is True and ans['want_frame'] is False
    p._commentary_want_frame_until = time.time() + 5
    assert p._commentary_frame_wanted()['want_frame'] is True
