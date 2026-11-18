"""Small N.E.K.O SDK double, so local tests don't require the N.E.K.O host."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# Allow `python -m pytest tests -q` directly inside the plugin folder.
_SOURCE_PARENT = str(Path(__file__).resolve().parents[2])
if _SOURCE_PARENT not in sys.path:
    sys.path.insert(0, _SOURCE_PARENT)


def identity_plugin(cls):
    return cls


def decorator(*args, **kwargs):
    def apply(fn):
        return fn
    return apply


class Ok:
    def __init__(self, value):
        self.value = value


def _real_sdk_available() -> bool:
    """在 N.E.K.O 仓库里跑（插件市场 CI 的 check --release）时用真的 SDK。

    只有单独克隆插件仓库、没有宿主时才装下面的替身；否则替身会把真的
    `plugin` 包整个盖掉，pytest 就找不到 `plugin.plugins`。
    """
    import importlib.util

    try:
        return importlib.util.find_spec("plugin.sdk.plugin") is not None
    except ImportError:
        return False


if "plugin.sdk.plugin" not in sys.modules and not _real_sdk_available():
    plugin = types.ModuleType("plugin")
    plugin.__path__ = []
    sdk = types.ModuleType("plugin.sdk")
    sdk.__path__ = []
    module = types.ModuleType("plugin.sdk.plugin")
    module.NekoPluginBase = type("NekoPluginBase", (), {})
    module.Ok = Ok
    module.lifecycle = decorator
    module.llm_tool = decorator
    module.neko_plugin = identity_plugin
    module.plugin_entry = decorator
    module.ui = types.SimpleNamespace(context=decorator, action=decorator)
    sys.modules["plugin"] = plugin
    sys.modules["plugin.sdk"] = sdk
    sys.modules["plugin.sdk.plugin"] = module


# Legacy audio-analysis tests exercise business logic, not package downloads.
# Isolate them from the network and from the runtime bootstrap.


class ReadyTestEnvironment:
    def __init__(self, *, source, data, models, download_source='mirror'):
        self.download_source = download_source
        self.state = {"phase": "ready", "ready": True, "model_ready": True,
                      "stage": "测试环境已提供", "missing": [], "error": None,
                      "repair_attempts": 1}
    def snapshot(self):
        return dict(self.state)
    def ensure(self):
        return self.snapshot()
    def stop(self):
        pass
    def worker_runtime(self):
        # Mirrors runtime_environment.worker_runtime(): (python, vendor_dir).
        import sys
        from pathlib import Path
        return sys.executable, Path(sys.prefix)


@pytest.fixture(autouse=True)
def _isolate_bootstrap_in_legacy_tests(monkeypatch, request):
    if request.node.path.name == 'test_runtime_environment.py':
        return
    import music_perception
    monkeypatch.setattr(music_perception, 'MusicEnvironment', ReadyTestEnvironment)


@pytest.fixture(autouse=True)
def _no_real_audible_scanner(monkeypatch, request):
    # The Windows audible-app scanner runs a real background thread that pushes
    # host-visible messages. Tests that exercise it call it explicitly.
    if request.node.path.name in ('test_v0913_auto_sense_lyric_lock.py',):
        return
    import music_perception
    monkeypatch.setattr(music_perception.MusicPerceptionPlugin,
                        '_start_audible_scanner', lambda self: None)



# 许多旧测试共用 test_plugin.py 里的插件夹具。放在 conftest 里统一提供，
# 这样各测试文件不必再 `from test_plugin import plugin_fixture`（ruff 会把那种
# 只按名字注入的导入当成未使用删掉）。
@pytest.fixture
def plugin_fixture(tmp_path):
    from test_plugin import make_plugin_fixture

    yield from make_plugin_fixture(tmp_path)
