"""Tag order: window title -> process name -> auxiliary classifier -> unknown."""
from music_perception import app_channels


def test_window_title_beats_process_name():
    app = {'name': 'chrome.exe', 'title': 'Some Video - bilibili'}
    assert app_channels.tag_application(app)['tag'] == 'video'
    assert app_channels.tag_application(app)['confidence'] == 'window_title_hint'


def test_process_name_used_when_title_empty():
    app = {'name': 'cloudmusic.exe', 'title': ''}
    info = app_channels.tag_application(app)
    assert info['tag'] == 'music'
    assert info['confidence'] == 'known_player_app_hint'


def test_unknown_app_consults_auxiliary_classifier():
    seen = []

    def classifier(meta):
        seen.append(meta['name'])
        return 'music'

    info = app_channels.tag_application({'name': 'mystery.exe', 'title': ''}, classifier=classifier)
    assert info['tag'] == 'music'
    assert info['confidence'] == 'auxiliary_model_hint'
    assert seen == ['mystery.exe']


def test_classifier_not_called_when_earlier_step_decides():
    def classifier(meta):
        raise AssertionError('auxiliary model must not run')

    info = app_channels.tag_application({'name': 'spotify.exe', 'title': ''}, classifier=classifier)
    assert info['tag'] == 'music'


def test_classifier_failure_or_unsure_yields_unknown():
    def boom(meta):
        raise RuntimeError('model down')

    assert app_channels.tag_application({'name': 'x.exe'}, classifier=boom)['tag'] == 'unknown'
    assert app_channels.tag_application({'name': 'x.exe'}, classifier=lambda m: 'other')['tag'] == 'unknown'


def test_host_ai_still_wins_over_title():
    info = app_channels.tag_application({'name': 'python.exe', 'title': '视频', 'host_ai': True})
    assert info['tag'] == 'ai'
