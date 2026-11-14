#!/usr/bin/env python3
"""Standard-library first structural preflight for N.E.K.O music_perception source.

This is not a replacement for the official `neko-plugin check` or a test of the
user's N.E.K.O runtime. It catches common manifest, extraction and SDK errors
before the user opens Plugin Manager.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        sys.stderr.write("需要 Python 3.11+ 的 tomllib，或安装 tomli 后重试。")
        raise SystemExit(2)

PLUGIN_ID = 'music_perception'
ENTRY = f'plugin.plugins.{PLUGIN_ID}:MusicPerceptionPlugin'


def check(source: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    source = source.resolve()
    manifest = source / 'plugin.toml'
    if not manifest.is_file():
        errors.append(f'没有找到 {manifest}；要传入真正包含 plugin.toml 的 music_perception 文件夹')
        if (source / PLUGIN_ID / 'plugin.toml').is_file():
            errors.append(f'检测到多套了一层目录：正确文件夹是 {source / PLUGIN_ID}')
        return errors, warnings
    try:
        with manifest.open('rb') as fh:
            doc = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        errors.append(f'plugin.toml 的 TOML 语法错误：{exc}')
        return errors, warnings
    meta = doc.get('plugin')
    if not isinstance(meta, dict):
        errors.append('缺少 [plugin] 配置表')
        return errors, warnings
    for required in ('id','name','version','entry'):
        if not isinstance(meta.get(required), str) or not meta[required].strip():
            errors.append(f'[plugin].{required} 缺失或不是非空字符串')
    if meta.get('id') != PLUGIN_ID:
        errors.append(f'插件 ID 应为 {PLUGIN_ID}，实际 {meta.get("id")!r}')
    if source.name != PLUGIN_ID:
        errors.append(f'文件夹名应为 {PLUGIN_ID}，实际 {source.name!r}')
    if meta.get('entry') != ENTRY:
        errors.append(f'加载入口应为 {ENTRY}，实际 {meta.get("entry")!r}')
    if meta.get('type','plugin') != 'plugin':
        errors.append('[plugin].type 必须为 plugin')
    sdk = meta.get('sdk', {})
    if not isinstance(sdk, dict) or not isinstance(sdk.get('supported'), str):
        errors.append('缺少合法的 [plugin.sdk].supported')
    else:
        try:
            from packaging.specifiers import SpecifierSet
            if '0.1.0' not in SpecifierSet(sdk['supported']):
                errors.append('supported 不接受 SDK 0.1.0；旧版本会被宿主兼容性检查拒绝')
            rec = sdk.get('recommended')
            if isinstance(rec, str) and '0.1.0' not in SpecifierSet(rec):
                warnings.append('recommended 未包含 SDK 0.1.0（可能导致兼容性警告）')
        except ImportError:
            warnings.append('当前检查环境未安装 packaging，跳过 SDK 版本范围的精确计算')
        except Exception as exc:
            errors.append(f'SDK 版本范围无效：{exc}')
    runtime = doc.get('plugin_runtime')
    if not isinstance(runtime, dict):
        errors.append('缺少 [plugin_runtime] 表')
    ui = meta.get('ui', {})
    if isinstance(ui, dict) and ui.get('enabled'):
        panels = ui.get('panel', [])
        if not isinstance(panels, list) or not panels:
            errors.append('启用了 [plugin.ui]，但没有 [[plugin.ui.panel]]')
        else:
            for panel in panels:
                entry = panel.get('entry', '') if isinstance(panel,dict) else ''
                if not isinstance(entry,str) or not entry or not (source / entry).is_file():
                    errors.append(f'未找到 UI 入口 {entry!r}')
    init_file = source / '__init__.py'
    if not init_file.is_file():
        errors.append('缺少 __init__.py Python 入口模块')
    else:
        try:
            module = ast.parse(init_file.read_text(encoding='utf-8'))
            if not any(isinstance(node,ast.ClassDef) and node.name == 'MusicPerceptionPlugin'
                       for node in module.body):
                errors.append('Python 入口中缺少 MusicPerceptionPlugin 类')
        except (SyntaxError, OSError, UnicodeError) as exc:
            errors.append(f'入口文件无效：{exc}')
    project = source / 'pyproject.toml'
    if not project.is_file():
        errors.append('缺少 pyproject.toml（独立依赖构建所需）')
    else:
        try:
            with project.open('rb') as fh:
                pkg = tomllib.load(fh)
            if meta.get('version') != pkg.get('project', {}).get('version'):
                errors.append('plugin.toml 和 pyproject.toml 的版本号不一致')
        except (OSError,tomllib.TOMLDecodeError) as exc:
            errors.append(f'pyproject.toml 无法解析：{exc}')
    # The common Windows user-data path is NOT the N.E.K.O checkout's
    # plugin/plugins/ source path. We can detect this mistake without loading
    # or modifying any user directories.
    if source.parent.name == 'plugins' and source.parent.parent.name == 'N.E.K.O':
        warnings.append('看起来放进了 N.E.K.O/plugins（运行数据目录）；官方源码目录是 N.E.K.O/plugin/plugins/')
    if '.neko-plugin-installations' in source.parts:
        warnings.append('这是插件安装器管理的目录，不能直接覆盖源码；请在「开发」页面载入源目录')
    # Official mandatory dependencies require build-time vendor. The
    # bootstrap has no mandatory dependencies; it manages *optional* audio
    # packs only after N.E.K.O loads the standard-library SDK shell.
    try:
        with (source / 'pyproject.toml').open('rb') as fh:
            dependencies = tomllib.load(fh).get('project', {}).get('dependencies', [])
    except (OSError, tomllib.TOMLDecodeError):
        dependencies = []
    if dependencies and not (source / 'vendor').is_dir():
        warnings.append('缺少官方必需依赖 vendor：在 N.E.K.O 插件中心构建前应先运行官方同步')
    if not (source / 'models' / 'yamnet.tflite').is_file():
        warnings.append('YAMNet 模型没有预置：插件首次启动时会在自己的数据目录自动下载并验证')
    return errors, warnings



def extract_entry_ids(init_py: Path) -> set[str]:
    """用 AST 从 __init__.py 提取所有 @plugin_entry(id=...) 的 id。"""
    ids: set[str] = set()
    tree = ast.parse(init_py.read_text(encoding='utf-8', errors='replace'))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            fn = dec.func
            name = getattr(fn, 'id', None) or getattr(fn, 'attr', None)
            if name != 'plugin_entry':
                continue
            for kw in dec.keywords:
                if kw.arg == 'id' and isinstance(kw.value, ast.Constant):
                    ids.add(str(kw.value.value))
    return ids


def extract_manifest_entries(toml_path: Path) -> set[str]:
    """从 plugin.toml 的静态入口段读出 id 列表。"""
    data = tomllib.loads(toml_path.read_text(encoding='utf-8'))
    out: set[str] = set()
    for key in ('entry', 'entries'):
        seq = data.get(key)
        if isinstance(seq, list):
            for item in seq:
                if isinstance(item, dict) and item.get('id'):
                    out.add(str(item['id']))
    # 也支持 [[entry]] 表数组
    raw = data.get('plugin', {})
    for key in ('entry', 'entries'):
        seq = raw.get(key)
        if isinstance(seq, list):
            for item in seq:
                if isinstance(item, dict) and item.get('id'):
                    out.add(str(item['id']))
    return out


def compare_entries(source: Path) -> tuple[list[str], list[str]]:
    """__init__.py 的 @plugin_entry 与 plugin.toml 静态入口双向比对。"""
    errors: list[str] = []
    warnings: list[str] = []
    init_py = source / '__init__.py'
    toml_path = source / 'plugin.toml'
    if not init_py.is_file() or not toml_path.is_file():
        return errors, warnings
    try:
        code_ids = extract_entry_ids(init_py)
        toml_ids = extract_manifest_entries(toml_path)
    except (SyntaxError, ValueError) as exc:
        errors.append('入口比对失败：%s' % type(exc).__name__)
        return errors, warnings
    only_code = sorted(code_ids - toml_ids)
    only_toml = sorted(toml_ids - code_ids)
    if only_code:
        errors.append('plugin.toml 缺少这些入口：' + ', '.join(only_code[:12]))
    if only_toml:
        errors.append('plugin.toml 多出这些入口（__init__.py 里没有）：' + ', '.join(only_toml[:12]))
    # meta 只警告
    meta = source / 'plugin.meta.json'
    if meta.is_file():
        try:
            import hashlib, json
            data = json.loads(meta.read_text(encoding='utf-8'))
            want = str(data.get('source_sha256') or '')
            got = hashlib.sha256(init_py.read_bytes()).hexdigest()
            if want and want != got:
                warnings.append('plugin.meta.json 的 source_sha256 与当前 __init__.py 不一致（本轮不手改 meta）')
        except (OSError, ValueError):
            warnings.append('plugin.meta.json 读取失败')
    # catalog 只警告
    catalog = source / 'ai_entry_catalog.json'
    if catalog.is_file() and code_ids:
        try:
            import json
            data = json.loads(catalog.read_text(encoding='utf-8'))
            known = set()
            for v in (data.get('entries') or data.get('actions') or []):
                if isinstance(v, dict) and v.get('id'):
                    known.add(str(v['id']))
            missing = sorted(code_ids - known)
            if missing:
                warnings.append('ai_entry_catalog.json 缺少入口说明：' + ', '.join(missing[:10]))
        except (OSError, ValueError):
            warnings.append('ai_entry_catalog.json 读取失败')
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description='为她所听：N.E.K.O 源码结构和清单静态检查')
    parser.add_argument('source', nargs='?', type=Path, default=Path(__file__).resolve().parents[1], help='music_perception 文件夹')
    opts = parser.parse_args()
    errors, warnings = check(opts.source)
    e2, w2 = compare_entries(opts.source)
    errors += e2
    warnings += w2
    print('检查目录：', opts.source.resolve())
    for warning in warnings:
        print('提醒：', warning)
    for error in errors:
        print('错误：', error)
    if errors:
        print(f'未通过：{len(errors)} 个错误，请先修改再刷新插件列表')
        return 1
    print('通过：TOML 语法、SDK 范围、插件身份、Python 入口与 UI 路径静态检查。')
    print('仍须在 N.E.K.O 实机运行官方 check、构建和插件加载验证。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
