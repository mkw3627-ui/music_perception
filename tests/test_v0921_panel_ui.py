# -*- coding: utf-8 -*-
"""面板 UI 静态断言（2026-11-13 新手引导改造）。

阶段 1 先写好；后端合入后如果个别断言依赖后端 action，会随阶段 2 转绿。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "ui" / "panel.tsx"
ONBOARD = ROOT / "ui" / "onboarding.tsx"
INIT = ROOT / "__init__.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def test_groove_page_has_no_dead_actions():
    """律动页不再出现那 4 个 action 名和 stageUrl。"""
    text = _read(PANEL)
    for name in ("install_live2d_bridge", "uninstall_live2d_bridge",
                 "close_live2d_stage", "open_live2d_stage", "stageUrl"):
        assert name not in text, "%s 仍出现在 panel.tsx" % name


def test_groove_switch_only_in_switch_callback():
    """set_live2d_groove 只出现在 Switch 的回调里。"""
    text = _read(PANEL)
    hits = [m.start() for m in re.finditer(r"set_live2d_groove", text)]
    assert len(hits) == 1, "set_live2d_groove 应只出现 1 次，实际 %d" % len(hits)
    around = text[max(0, hits[0] - 200):hits[0] + 60]
    assert "onChange" in around, "set_live2d_groove 应在 onChange 回调里"


def test_onboarding_has_all_actions():
    text = _read(ONBOARD)
    for action in ("onboarding_scan", "onboarding_browse_start", "onboarding_browse_wait",
                   "onboarding_browse_cancel", "onboarding_set_storage_root",
                   "onboarding_start_install", "onboarding_cancel_install",
                   "onboarding_finish", "set_download_source"):
        assert action in text, "onboarding.tsx 缺少 %s" % action
    # onboarding_reopen 由 panel.tsx 的设置页按钮调用（派工单 F-2.4）
    assert "onboarding_reopen" in _read(PANEL)


def test_no_frame_budget_ui():
    text = _read(PANEL)
    assert "set_video_frame_budget" not in text
    assert "每次视频最多抽帧数" not in text


def test_ui_build_version_updated():
    text = _read(PANEL)
    # 版本号会随迭代递增；只断言「两端一致」，不写死具体号。
    from _version import panel_ui_build, ui_build
    assert panel_ui_build() in text
    assert panel_ui_build() == ui_build(), "前后端 UI build 不一致"


def test_panel_imports_onboarding():
    text = _read(PANEL)
    assert 'from "./onboarding"' in text


def test_all_ui_actions_registered_in_backend():
    """panel/onboarding 里调用的 ui action 必须在 __init__.py 里注册。"""
    backend = _read(INIT)
    registered = set(re.findall(r"@ui\.action\(\s*id\s*=\s*['\"]([^'\"]+)['\"]", backend))
    called = set()
    for f in (PANEL, ONBOARD):
        text = _read(f)
        called |= set(re.findall(r'\b(?:call|perform)\(\s*["\']([a-z0-9_]+)["\']', text))
    missing = sorted(a for a in called if a not in registered)
    # 阶段 1 允许 onboarding_* 尚未注册（后端归 v4pro），但其它必须已注册
    unexpected = [a for a in missing if not a.startswith("onboarding_")]
    assert not unexpected, "调用了未注册的 action：%s" % unexpected[:8]


# ── 返工补充（2026-11-13 复核发现）──────────────────────────
def test_onboarding_imports_all_used_hooks():
    """用了 useState/useEffect 就必须导入——否则引导页一打开就白屏。"""
    text = _read(ONBOARD)
    header = text.split('from "@neko/plugin-ui"')[0]
    for hook in ("useState", "useEffect"):
        used = text.count(hook) > 0
        imported = hook in header
        if used:
            assert imported, "%s 被使用但没在导入里（会导致引导页白屏）" % hook


def test_download_items_use_contract_names():
    """下载项必须是契约名 lyrics_model / singing / stem_model，不能自造。"""
    text = _read(ONBOARD)
    for bad in ("video_asr", "singing_asr"):
        assert bad not in text, "自造名 %s 仍在 onboarding.tsx" % bad
    for good in ('"lyrics_model"', '"singing"', '"stem_model"'):
        assert good in text, "契约名 %s 缺失" % good
    # push("stem") 不合法，必须是 stem_model
    assert 'push("stem")' not in text


def test_guide_banner_jumps_to_step_two():
    text = _read(PANEL)
    assert "setGuideStep(2)" in text, "「去补齐」应跳到第 2 步"


def test_recheck_button_exists():
    text = _read(PANEL)
    assert "automatic_audio_check" in text, "缺少「重新检查」按钮"


def test_score_display_handles_null():
    text = _read(PANEL)
    assert "function displayScore" in text
    assert "displayScore(scene.speech_score)" in text


# —— 复核建议项落实（2026-11-13 返工） ——

def test_step5_failure_reason_uses_attempts():
    """失败原因取 attempts 里最后一条失败的 error，不是顶层 job.error。"""
    text = _read(ONBOARD)
    assert "function failureReason" in text
    assert "short(failureReason(job)" in text


def test_browse_cancel_follows_picker_state():
    """等待选择期间取消按钮必须在：跟着 browsing/picker.state 显示。"""
    text = _read(ONBOARD)
    assert 'picker.state==="open"' in text
    assert "setBrowsing(true)" in text


def test_step4_can_return_to_location():
    """必需项全 ok 时第 2 步直跳第 4 步，第 4 步要能回第 3 步改位置。"""
    text = _read(ONBOARD)
    assert "修改位置" in text
    assert "setStep(3)" in text


def test_total_bytes_only_counts_planned_items():
    """合计只算要装的那几项，不是把所有扫描项都加起来。"""
    text = _read(ONBOARD)
    assert "plannedIds" in text
    assert "planned.indexOf" in text
