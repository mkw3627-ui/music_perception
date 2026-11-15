"""Host-LLM app classifier: cache, cooldown, never blocks, never raises."""
import time

from music_perception import host_llm
from music_perception.host_llm import HostAppClassifier, parse_category


def _wait(pred, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_parse_category_accepts_only_known_words():
    assert parse_category(' Music. ') == 'music'
    assert parse_category('`video`') == 'video'
    assert parse_category('maybe') is None
    assert parse_category(None) is None


def test_first_call_returns_none_then_cached_answer_is_used():
    calls = []

    async def completion(prompt, timeout):
        calls.append(prompt)
        return 'music'

    clf = HostAppClassifier(completion=completion)
    assert clf({'name': 'Mystery.exe'}) is None  # never blocks on the model
    assert _wait(lambda: clf.cached('mystery.exe') == 'music')
    assert clf({'name': 'mystery.exe'}) == 'music'
    assert len(calls) == 1
    assert 'mystery.exe' in calls[0]


def test_failure_enters_cooldown_and_is_not_retried_immediately():
    calls = []
    now = [1000.0]

    async def completion(prompt, timeout):
        calls.append(1)
        raise RuntimeError('provider down')

    clf = HostAppClassifier(completion=completion, clock=lambda: now[0])
    clf({'name': 'x.exe'})
    assert _wait(lambda: len(calls) == 1 and not clf._pending)
    clf({'name': 'x.exe'})
    time.sleep(0.05)
    assert len(calls) == 1  # still inside cooldown
    now[0] += host_llm.FAILURE_COOLDOWN_S + 1
    clf({'name': 'x.exe'})
    assert _wait(lambda: len(calls) == 2)


def test_invalid_answer_is_treated_as_failure():
    async def completion(prompt, timeout):
        return 'banana'

    clf = HostAppClassifier(completion=completion)
    clf({'name': 'y.exe'})
    assert _wait(lambda: not clf._pending)
    assert clf.cached('y.exe') is None


def test_empty_name_never_calls_model():
    def completion(prompt, timeout):
        raise AssertionError('must not call')

    assert HostAppClassifier(completion=completion)({'name': ''}) is None
