from __future__ import annotations

from pathlib import Path

from video_series import infer_video_position, parse_part

ROOT = Path(__file__).resolve().parents[1]


def test_part_markers_are_parsed():
    assert parse_part('Some Show P2')[1] == 2
    assert parse_part('Some Show 第3集')[1] == 3
    assert parse_part('Some Show (4)')[1] == 4
    assert parse_part('Some Show - 5')[1] == 5


def test_next_and_previous_within_same_series():
    assert infer_video_position('Show P1', 'Show P2')['position'] == 'next'
    assert infer_video_position('Show P3', 'Show P2')['position'] == 'previous'


def test_first_part_and_standalone_labels():
    assert infer_video_position('', 'Show P1')['position'] == 'first'
    assert infer_video_position('', 'Show P3')['position'] == 'middle'
    assert infer_video_position('Other video', 'Unrelated clip')['position'] == 'standalone'


def test_non_consecutive_parts_are_not_claimed_as_next():
    result = infer_video_position('Show P1', 'Show P5')
    assert result['position'] == 'same_series'
    assert result['confidence'] == 'low'


def test_unknown_when_title_missing():
    assert infer_video_position('Show P1', '')['position'] == 'unknown'


def test_output_is_labels_only_never_titles():
    secret = 'Very Private Title P2'
    result = infer_video_position('Very Private Title P1', secret)
    assert secret not in repr(result)
    assert set(result) == {'position', 'inferred', 'confidence'}
    assert result['inferred'] is True


def test_plugin_payload_uses_safe_wrapper_only():
    text = (ROOT / '__init__.py').read_text(encoding='utf8')
    assert "'series_position':infer_video_position(" in text
    assert "'video_title'" not in text
