"""The visual installer must remain opt-in, limited to this source tree,
fail-closed and separate from the official install-plan transaction."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from music_perception import MusicPerceptionPlugin
from music_perception import install_prepare as builder


def _make_source(tmp_path):
    source = tmp_path / "music_perception"
    source.mkdir()
    original = Path(__file__).resolve().parents[1]
    (source / "plugin.toml").write_bytes((original / "plugin.toml").read_bytes())
    (source / "pyproject.toml").write_bytes((original / "pyproject.toml").read_bytes())
    return source


def test_accepts_external_visual_development_source(tmp_path):
    source = _make_source(tmp_path)
    assert builder.plugin_checkout(source) == source
    assert len(builder._dependencies(source)) == 10  # adds multilingual ASR to prior 8 audio deps


def test_canonical_checkout_recognized(tmp_path):
    root = tmp_path / "N.E.K.O"
    source = root / "plugin" / "plugins" / "music_perception"
    source.mkdir(parents=True)
    (root / "plugin" / "neko_plugin_cli").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='neko'\n", encoding="utf8")
    original = Path(__file__).resolve().parents[1]
    for f in ("plugin.toml", "pyproject.toml"):
        (source / f).write_bytes((original / f).read_bytes())
    assert builder.plugin_checkout(source) == root


def test_rejects_runtime_config_or_broken_source(tmp_path):
    source = tmp_path / "plugins" / "music_perception"
    source.mkdir(parents=True)
    with pytest.raises(builder.PreparationError):
        builder.plugin_checkout(source)


def test_rejects_remote_and_modified_dependencies(tmp_path):
    source = _make_source(tmp_path)
    file = source / "pyproject.toml"
    file.write_text(file.read_text(encoding="utf8").replace('"numpy>=1.24",', '"numpy @ https://fake.example/malware.whl",'), encoding="utf8")
    with pytest.raises(builder.PreparationError):
        builder._dependencies(source)


def test_rejects_model_html_and_accepts_tflite_header(tmp_path):
    file = tmp_path / "model.tflite"
    file.write_bytes(b"<html>" * 300_000)
    assert not builder.valid_model(file)
    file.write_bytes(b"1234TFL3" + b"0" * 1_100_000)
    assert builder.valid_model(file)


def test_preparation_external_uses_isolated_vendor_and_no_host_install(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    events = []
    monkeypatch.setattr(builder.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(builder, "download_yamnet", lambda src, cb: "mock-google")

    def fake_run(uv, cwd, args, *, timeout):
        events.append((uv, cwd, args))
        if args[:2] == ["pip", "install"]:
            target = Path(args[args.index("--target") + 1])
            (target / "mock_dist.txt").write_text("vendor", encoding="utf8")

    monkeypatch.setattr(builder, "_run_uv", fake_run)
    monkeypatch.setattr(builder, "_run_command", lambda root, cmd, *, timeout: events.append(("python", root, cmd)))
    result = builder.prepare(source, lambda _: None)
    assert result["ready"] is True and result["installed"] is False
    assert (source / "vendor" / "mock_dist.txt").is_file()
    assert len(events) == 2
    assert any("--only-binary" in x[2] for x in events)
    assert all("install-plan" not in x[2] and "plugin-cli/install" not in x[2] for x in events)


def test_vendor_preserved_on_pip_failure(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    (source / "vendor").mkdir()
    (source / "vendor" / "existing.txt").write_text("keep", encoding="utf8")
    monkeypatch.setattr(builder.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(builder, "download_yamnet", lambda src, cb: "mock-google")

    def reject(*args, **kwargs):
        raise builder.PreparationError("pip failed")

    monkeypatch.setattr(builder, "_run_uv", reject)
    monkeypatch.setattr(builder, "_run_command", reject)  # both resolvers intentionally fail, no external network
    with pytest.raises(builder.PreparationError):
        builder.prepare(source)
    assert (source / "vendor" / "existing.txt").read_text() == "keep"
    assert not list(source.glob(".vendor-build-*"))


def test_user_must_confirm_before_prepare(monkeypatch):
    call_count = []
    monkeypatch.setattr(builder, "prepare", lambda *a, **k: call_count.append(1))
    class FakeLogger:
        def exception(self, *args, **kwargs):
            raise AssertionError("must not run")
    fake = object.__new__(MusicPerceptionPlugin)
    result = asyncio.run(fake.prepare_visual_install(user_confirmed=False))
    assert result.value["ok"] is False
    assert call_count == []


def test_confirmed_user_can_prepare_but_no_automatic_listening(monkeypatch):
    seen = []
    monkeypatch.setattr(builder, "prepare", lambda src, progress: (seen.append(src) or {"ready": True, "installed": False}))
    fake = object.__new__(MusicPerceptionPlugin)
    result = asyncio.run(fake.prepare_visual_install(user_confirmed=True))
    assert result.value["ok"] and result.value["ready"]
    assert len(seen) == 1
    assert not getattr(fake, "_auto_consented", False)


def test_fallback_to_pip_when_uv_is_missing(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    calls = []
    monkeypatch.setattr(builder.shutil, "which", lambda name: None)
    monkeypatch.setattr(builder, "download_yamnet", lambda src, cb: "mock-google")

    def fake_command(root, cmd, *, timeout):
        calls.append(cmd)
        if "--target" in cmd:
            target = Path(cmd[cmd.index("--target") + 1])
            (target / "pip_fallback.txt").write_text("created", encoding="utf8")

    monkeypatch.setattr(builder, "_run_command", fake_command)
    result = builder.prepare(source)
    assert result["ready"] is True
    # 解析器返回 resolve() 后的真实解释器；Linux venv 的 python 是符号链接。
    real = str(Path(sys.executable).resolve())
    assert calls[0][:3] == [real, "-m", "pip"]
    assert calls[1][:2] == [real, "-c"]
    assert (source / "vendor/pip_fallback.txt").exists()



def test_visual_builder_falls_back_when_uv_fails(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    monkeypatch.setattr(builder.shutil, 'which', lambda name: '/usr/bin/uv' if name == 'uv' else None)
    monkeypatch.setattr(builder, 'download_yamnet', lambda src, progress: 'cached-test-model')
    events = []

    def bad_uv(*args, **kw):
        events.append('uv')
        raise builder.PreparationError('unknown option --target')

    def fake_pip(repo, cmd, *, timeout):
        if '-m' in cmd and 'pip' in cmd:
            events.append('pip')
            target = Path(cmd[cmd.index('--target') + 1])
            (target / 'ready').write_text('yes')
        else:
            events.append('native-probe')

    monkeypatch.setattr(builder, '_run_uv', bad_uv)
    monkeypatch.setattr(builder, '_run_command', fake_pip)
    events_progress = []
    result = builder.prepare(source, events_progress.append)
    assert result['ready'] is True
    assert events == ['uv', 'pip', 'native-probe']
    assert (source / 'vendor' / 'ready').exists()
    assert any('自动切换' in event for event in events_progress)


def test_visual_command_retains_sanitized_error(monkeypatch, tmp_path):
    from types import SimpleNamespace
    def fail(*args, **kwargs):
        return SimpleNamespace(returncode=2, stdout='unknown uv flag https://alice:pass@host.invalid/simple?token=SECRET')
    monkeypatch.setattr(builder.subprocess, 'run', fail)
    with pytest.raises(builder.PreparationError) as exc:
        builder._run_command(tmp_path, ['uv', 'pip', 'install'], timeout=2)
    text = str(exc.value)
    assert 'unknown uv flag' in text
    assert 'alice' not in text and 'SECRET' not in text and 'pass' not in text


def test_visual_builder_retries_pip_when_uv_installs_but_native_probe_fails(tmp_path, monkeypatch):
    """A successfully resolved wheel set may still have a DLL load error."""
    source = _make_source(tmp_path)
    monkeypatch.setattr(builder.shutil, 'which', lambda name: 'uv' if name == 'uv' else None)
    monkeypatch.setattr(builder, 'download_yamnet', lambda source, cb: 'cached')
    events = []
    def uv_install(cmd, cwd, params, *, timeout):
        assert params[:2] == ['pip','install']
        events.append('uv-install')
        (Path(params[params.index('--target')+1]) / 'uv_partial').write_text('broken')
    def run(source_path, cmd, *, timeout):
        if cmd[1:3] == ['-m', 'pip']:
            events.append('pip-install')
            stage = Path(cmd[cmd.index('--target')+1])
            assert not (stage / 'uv_partial').exists()
            (stage / 'pip_ok').write_text('verified')
        elif cmd[1] == '-c':
            events.append('native-probe')
            if (Path(cmd[-2]) / 'uv_partial').exists():
                raise builder.PreparationError('DLL load failed')
        else:
            raise AssertionError(cmd)
    monkeypatch.setattr(builder, '_run_uv', uv_install)
    monkeypatch.setattr(builder, '_run_command', run)
    details = []
    result = builder.prepare(source, details.append)
    assert result['ready'] is True
    assert events == ['uv-install', 'native-probe', 'pip-install', 'native-probe']
    assert (source / 'vendor/pip_ok').exists()
    assert not (source / 'vendor/uv_partial').exists()
    assert any('自动切换' in item for item in details)
