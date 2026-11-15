import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from video_window import decide_video_window  # noqa: E402


def test_position_known_processes_from_position_bounded_by_duration():
    w = decide_video_window(position_s=100, duration_s=1000, max_window_s=600)
    assert w['basis'] == 'playback_position'
    assert w['start_s'] == 100 and w['end_s'] == 700


def test_position_near_end_is_clamped_to_duration():
    w = decide_video_window(position_s=900, duration_s=1000, max_window_s=600)
    assert w['end_s'] == 1000


def test_unknown_position_uses_first_window():
    w = decide_video_window(position_s=None, duration_s=3000, max_window_s=600)
    assert w['basis'] == 'duration_bounded_start'
    assert w['start_s'] == 0 and w['end_s'] == 600


def test_unknown_position_and_duration_falls_back_to_first_window():
    w = decide_video_window(None, None, 600)
    assert w['basis'] == 'start_fallback'
    assert w['start_s'] == 0 and w['end_s'] == 600


def test_bad_values_never_raise():
    w = decide_video_window('nan', 'x', None)
    assert w['start_s'] == 0 and w['end_s'] == 600
    assert w['inferred'] is True


def test_position_past_end_is_ignored():
    w = decide_video_window(position_s=2000, duration_s=1000, max_window_s=600)
    assert w['basis'] == 'duration_bounded_start'


def test_output_contains_only_numbers_and_labels():
    w = decide_video_window(position_s=5, duration_s=50)
    assert set(w) == {'start_s', 'end_s', 'basis', 'inferred'}
    assert all(isinstance(w[k], (int, float)) for k in ('start_s', 'end_s'))
