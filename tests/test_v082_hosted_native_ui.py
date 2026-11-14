"""v0.8.2 Hosted TSX, action permissions and model guard smoke tests.

These tests use the N.E.K.O SDK double and offline fake model data; a Windows
host/RTX card is still required for integration validation.
"""
from __future__ import annotations
import asyncio
import ast
import tomllib
from pathlib import Path

from music_perception import MusicPerceptionPlugin
from music_perception.lyrics_model_store import MODELS, PUBLIC_SOURCES, _direct_file_url
from test_plugin import plugin_fixture
from _version import ui_build, panel_ui_build, plugin_version  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_panel_declared_as_hosted_and_includes_native_kit():
    doc = tomllib.loads((ROOT / 'plugin.toml').read_text('utf8'))
    panels = doc['plugin']['ui']['panel']
    assert doc['plugin']['version'] == plugin_version()
    assert len(panels) == 1
    assert panels[0]['entry'] == 'ui/panel.tsx'
    assert panels[0]['context'] == 'dashboard'
    assert panels[0]['permissions'] == ['state:read', 'action:call']
    text=(ROOT / 'ui/panel.tsx').read_text('utf8')
    assert 'from "@neko/plugin-ui"' in text
    assert 'props.state' in text and 'props.api.refresh()' in text
    assert '<SegmentedControl' in text and 'function ModelSpotlight' in text  # v0.9.8 replaces plain StatCard
    assert 'setInterval(' in text
    assert 'javascript:' not in text and 'fetch(' not in text


def test_every_frontend_action_has_explicit_backend_permission():
    source = (ROOT / '__init__.py').read_text('utf8')
    tree=ast.parse(source)
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef)
             and node.name=='MusicPerceptionPlugin')
    exposed=set()
    context=set()
    for node in cls.body:
        if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
        for decorator in node.decorator_list:
            if not isinstance(decorator,ast.Call) or not isinstance(decorator.func,ast.Attribute):continue
            if not isinstance(decorator.func.value,ast.Name) or decorator.func.value.id!='ui':continue
            if decorator.func.attr=='action':exposed.add(node.name)
            if decorator.func.attr=='context':
                context.update(k.value.value for k in decorator.keywords if k.arg=='id' and isinstance(k.value,ast.Constant))
    assert context=={'dashboard'}
    needed={'start_auto_listen','stop_auto_listen','pause_plugin','resume_plugin',
            'set_ai_delivery','set_lyrics_language','stage_lyrics_model',
            'activate_verified_lyrics_model','get_lyrics_diagnostics',
            'run_lyrics_diagnostics','repair_environment','check_lyrics_gpu','fallback_verified_lyrics_model'}
    assert needed<=exposed
    tsx=(ROOT/'ui/panel.tsx').read_text('utf8')
    # One-click auto-check supersedes separate low-level repair/GPU controls.
    # Private AI sharing is the default; its manual switch was intentionally removed.
    # Automatic health runs at startup, so the one-click check is no longer user-facing.
    for action in needed-{'run_lyrics_diagnostics','repair_environment','check_lyrics_gpu','set_ai_delivery'}:
        assert '"'+action+'"' in tsx
    assert '"automatic_audio_check"' in tsx
    assert '"repair_environment"' in source


def test_download_visibility_and_unverified_bar_never_100():
    ui=(ROOT/'ui/panel.tsx').read_text('utf8')
    assert 'const loading=item.state==="downloading" || item.state==="verifying"' in ui
    assert '{loading?<DownloadProgress' in ui
    assert '{failed?<Alert tone="warning">' in ui
    assert 'Math.min(99,' in ui
    assert 'disabled={Boolean(busy)||!verified||paused||switchBusy}' in ui
    assert 'verification_pct' in ui
    assert '继续下载' in ui


def test_catalog_context_does_not_trigger_download(plugin_fixture):
    plugin,*_ = plugin_fixture
    async def scenario():
        await plugin.startup()
        output=await plugin.music_dashboard()
        assert set(output)=={'status','lyrics','audio','catalog','singing','live2d','onboarding'}
        assert output['catalog']['ok']
        assert len(output['catalog']['models'])==4
        assert output['status']['ai_context_policy']=='private_read_only_ai_decides_output_deduplicated'
        await plugin.shutdown()
    asyncio.run(scenario())


def test_all_four_official_model_links_are_pinned_and_https():
    assert PUBLIC_SOURCES[0]=='https://huggingface.co'
    assert len(set(PUBLIC_SOURCES))==3
    for model,spec in MODELS.items():
        assert len(spec['sha256'])==64
        assert spec['repo'] and spec['repo'].count('/')==1
        for endpoint in PUBLIC_SOURCES:
            path=_direct_file_url(endpoint,model,'model.bin')
            assert path.startswith(endpoint+'/')
            assert spec['repo']+'/resolve/' in path
            assert '/model.bin?' in path
    assert all('revision' in spec for spec in MODELS.values())
    assert MODELS['large-v3-turbo']['revision']=='44cbbd1adefe7387c83df88963a6d9ac4c9adea5'
    assert MODELS['large-v3-turbo']['sha256']=='e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da'


def test_failed_99_percent_restart_exposes_real_reset_bytes(tmp_path, monkeypatch):
    """A corrupt resumable prefix must not make a brand-new stream appear 99%."""
    import hashlib
    import io
    from music_perception import lyrics_model_store as store
    dest = tmp_path/'model.bin'
    good = b'SIGNED-MODEL-WEIGHT'*500
    corrupt = b'X'*(len(good)-25)
    (tmp_path/'model.bin.partial').write_bytes(corrupt)
    transfers = []

    class Response:
        status=200
        headers={'Content-Length':str(len(good))}
        def __enter__(self): return self
        def __exit__(self,*_): return False
        def read(self,n): return self.stream.read(n)
        def geturl(self): return 'https://hf-mirror.com/signed'
        def getcode(self): return 200
        def __init__(self): self.stream=io.BytesIO(good)

    monkeypatch.setattr(store.urllib.request,'urlopen',lambda *a,**kw:Response())
    store._stream_to_file('https://hf-mirror.com/file',dest,min_bytes=100,
        expected_sha256=hashlib.sha256(good).hexdigest(),
        transfer_progress=lambda size,total,restart:transfers.append((size,total,restart)))
    assert dest.read_bytes()==good
    assert any(restart and size==0 for size,_,restart in transfers)
    assert transfers[-1]==(len(good),len(good),True)


def test_offline_fallback_chooses_verified_small_without_network(plugin_fixture,monkeypatch):
    plugin,*_=plugin_fixture
    calls=[]
    from music_perception import lyrics_model_store
    def only_small(model,root):
        calls.append((model,str(root)))
        return Path(root)/model if model=='small' else None
    monkeypatch.setattr(lyrics_model_store,'find_local_model',only_small)
    async def fake_activation(*,quality,device):
        return {'ok':True,'quality':quality,'device':device}
    # Verify the selection policy without launching the GPU or changing files.
    async def run():
        await plugin.startup()
        plugin._lyrics_profile='ultra'
        async def mock_activate(**kwargs):
            from music_perception import Ok
            return Ok(await fake_activation(**kwargs))
        monkeypatch.setattr(plugin,'activate_verified_lyrics_model',mock_activate)
        result=(await plugin.fallback_verified_lyrics_model()).value
        assert result=={'ok':True,'quality':'high','device':'cpu'}
        assert [name for name,_ in calls][:2]==['large-v3-turbo','small']
        await plugin.shutdown()
    asyncio.run(run())
