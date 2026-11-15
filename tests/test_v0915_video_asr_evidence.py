"""Text-only ASR hallucination screen for video speech candidates.

Loads speech_timeline.py by path so the test does not import the plugin
package (which needs numpy/soundfile).
"""
import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / 'speech_timeline.py'
_spec = importlib.util.spec_from_file_location('speech_timeline_under_test', _PATH)
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)


def _cue(at, text):
    return {'at_s': at, 'end_s': at + 1.0, 'text_candidate': text}


def test_repeated_text_marked_suspect():
    cues = [_cue(0, '你好'), _cue(2, '你好'), _cue(4, '你好'), _cue(6, '今天天气不错')]
    rows, suspects = st.annotate_video_evidence(cues)
    assert suspects == 3
    assert [r['evidence_status'] for r in rows] == [
        'repeated_suspect', 'repeated_suspect', 'repeated_suspect', 'candidate']
    assert rows[0]['evidence_reason'] == 'repeated_text'
    assert rows[3]['evidence_reason'] is None


def test_two_repeats_below_threshold_kept():
    cues = [_cue(0, '好的'), _cue(2, '好的')]
    rows, suspects = st.annotate_video_evidence(cues)
    assert suspects == 0
    assert all(r['evidence_status'] == 'candidate' for r in rows)


def test_template_phrase_marked_suspect():
    cues = [_cue(0, '字幕by索兰娅'), _cue(2, '正常的一句话')]
    rows, suspects = st.annotate_video_evidence(cues)
    assert rows[0]['evidence_status'] == 'repeated_suspect'
    assert rows[0]['evidence_reason'] == 'template_phrase'
    assert rows[1]['evidence_status'] == 'candidate'
    assert suspects == 1


def test_text_and_timestamps_not_rewritten():
    cues = [_cue(1.5, '谢谢观看')]
    rows, _ = st.annotate_video_evidence(cues)
    assert rows[0]['text_candidate'] == '谢谢观看'
    assert rows[0]['at_s'] == 1.5


def test_non_dict_entries_ignored():
    rows, suspects = st.annotate_video_evidence([None, 'x', _cue(0, 'ok')])
    assert len(rows) == 1 and suspects == 0
