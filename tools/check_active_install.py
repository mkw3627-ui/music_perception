"""Read-only verifier for the actual installed music_perception code.

Usage (Windows): python tools/check_active_install.py "D:\\...\\plugins\\music_perception"
Does not use N.E.K.O SDK, download models, access audio, or change files.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

EXPECTED_VERSION = '0.8.4'
EXPECTED_PROTOCOL = '0.8.4-native-r1'


def examine(folder: Path) -> tuple[bool, list[str]]:
    folder=Path(folder)
    out=[]
    toml_path=folder/'plugin.toml'
    ui=folder/'ui'/'panel.tsx'
    backend=folder/'__init__.py'
    downloader=folder/'lyrics_model_store.py'
    asr=folder/'lyrics_backend.py'
    try:
        version=str(tomllib.loads(toml_path.read_text(encoding='utf8'))['plugin']['version'])
    except (OSError,KeyError,ValueError,UnicodeError) as err:
        return False,[f'无法读取当前插件清单：{type(err).__name__}']
    out.append('当前磁盘插件版本：'+version)
    if version!=EXPECTED_VERSION:
        out.append('错误：插件中心安装目录仍是旧版（预期 '+EXPECTED_VERSION+'）')
    ok=(version==EXPECTED_VERSION)
    checks=(
        (ui, '原生 TSX UI',EXPECTED_PROTOCOL),
        (backend,'后端版本标识',EXPECTED_PROTOCOL),
        (asr,'歌词子进程只读本地权重','识别子进程不会自动联网'),
        (downloader,'模型断点+SHA校验','expected_sha256'),
    )
    for file,label,marker in checks:
        try:
            content=file.read_text(encoding='utf8')
            passed=marker in content
        except (OSError,UnicodeError):
            passed=False
        out.append(('通过' if passed else '失败')+'：'+label)
        ok=ok and passed
    return ok,out


def main(argv=None):
    arg=argparse.ArgumentParser(description='只读核查 N.E.K.O 当前安装的「为她所听」插件版本')
    arg.add_argument('installed_plugin_dir',nargs='?',default=None,
                     help='实际安装目录：.neko-plugin-installations/plugins/music_perception')
    opts=arg.parse_args(argv)
    folder=(Path(opts.installed_plugin_dir) if opts.installed_plugin_dir
            else Path(__file__).resolve().parents[1])
    passed,lines=examine(folder)
    print('「为她所听」安装文件只读核验')
    print('检查目录：'+str(folder))
    for line in lines:
        print(line)
    print('结论：'+('磁盘源码与 v0.8.4 相符；仍须重启宿主并实机验证'
                  if passed else '安装文件版本不一致，请重新使用 N.E.K.O 插件中心升级'))
    return 0 if passed else 1

if __name__=='__main__':
    sys.exit(main())
