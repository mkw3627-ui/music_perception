"""Offline standalone demo: python cli.py path/to/music.wav [--json output.json]."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine import AudioAnalysisError, analyze_audio


def main():
    parser = argparse.ArgumentParser(description="本地整首歌节奏/频谱/近似旋律/情绪分析")
    parser.add_argument("audio", help="本地音频文件")
    parser.add_argument("--json", dest="output", help="把分析结果保存到指定 JSON")
    parser.add_argument("--segment-seconds", type=int, default=10)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--emotion-mode", choices=["fast", "essentia"], default="fast")
    parser.add_argument("--emotion-models-dir", default="")
    args = parser.parse_args()
    try:
        result = analyze_audio(args.audio, segment_seconds=args.segment_seconds,
                               max_seconds=args.max_seconds, emotion_mode=args.emotion_mode,
                               emotion_models_dir=args.emotion_models_dir)
    except (AudioAnalysisError, OSError) as exc:
        parser.exit(2, f"分析失败：{exc}\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON 已保存：{args.output}")


if __name__ == "__main__":
    main()
