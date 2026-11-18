"""Explicit sung-vocal transcription acceptance evaluation.

Input: records with language zh/ja/en, reference, predicted, source='separated_vocals'.
Chinese/Japanese: character error rate (CER). English: word error rate (WER).
A passing English clean-speech benchmark is NOT evidence about sung vocals.

Usage: python singing_quality_eval.py separated_vocals_gold.jsonl
This script never downloads a model, guesses a ground truth or reads system audio.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

TARGET = 0.05
# A single perfect phrase is not meaningful evidence. These limits are only a
# minimum coverage gate, not a statistical promise for songs outside the set.
MIN_CLIPS_PER_LANGUAGE = 10
MIN_REFERENCE_UNITS_PER_LANGUAGE = 500


def tokens(text: str, language: str) -> list[str]:
    if language not in {'zh','ja','en'}:
        raise ValueError('Only zh/ja/en are supported')
    normalized=unicodedata.normalize('NFKC',text).lower()
    if language=='en':
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?",normalized)
    return [c for c in normalized if (c.isalnum() or '\u3040'<=c<='\u30ff')]


def edit_distance(a:list[str],b:list[str]) -> int:
    # O(min(n,m)) RAM, no third-party imports
    if len(a)<len(b):
        a,b=b,a
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):
            row.append(min(previous[j]+1,row[-1]+1,previous[j-1]+(x!=y)))
        previous=row
    return previous[-1]


def measure(rows: list[dict]) -> dict:
    """Counts are aggregated across samples; never average per-clip rates."""
    per={lang:{'errors':0,'units':0,'clips':0} for lang in ('zh','ja','en')}
    if not rows:
        raise ValueError('No test clips supplied')
    for row in rows:
        if not isinstance(row,dict) or row.get('source')!='separated_vocals':
            raise ValueError('Evaluation requires captured separated vocal tracks')
        lang=row.get('language')
        if lang not in per or not isinstance(row.get('reference'),str) or not isinstance(row.get('predicted'),str):
            raise ValueError('Invalid reference/prediction entry')
        ref=tokens(row['reference'],lang)
        hyp=tokens(row['predicted'],lang)
        if not ref:
            raise ValueError('Reference text must not be empty')
        per[lang]['errors']+=edit_distance(ref,hyp)
        per[lang]['units']+=len(ref)
        per[lang]['clips']+=1
    complete=all(data['clips']>=MIN_CLIPS_PER_LANGUAGE and
                 data['units']>=MIN_REFERENCE_UNITS_PER_LANGUAGE for data in per.values())
    metrics={k:{**v,'error_rate_pct':round(100*v['errors']/v['units'],3) if v['units'] else None,
                 'metric':'WER' if k=='en' else 'CER',
                 'passed':v['units']>0 and v['errors']/v['units']<=TARGET}
             for k,v in per.items()}
    return {'target_error_rate_pct':5.0,'validated':complete,
            'minimum_coverage':{'clips_per_language':MIN_CLIPS_PER_LANGUAGE,
                                'reference_units_per_language':MIN_REFERENCE_UNITS_PER_LANGUAGE},
            'passed':complete and all(x['passed'] for x in metrics.values()),
            'languages':metrics,
            'warning':'Only user-supplied, human-verified separated-vocal clips; not a universal error guarantee'}


def main():
    if len(sys.argv)!=2:
        print('usage: python singing_quality_eval.py separated_vocals_gold.jsonl')
        return 2
    path=Path(sys.argv[1])
    if not path.is_file() or path.stat().st_size>5_000_000:
        print('invalid evaluation file')
        return 2
    try:
        rows=[json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]
        result=measure(rows)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if result['passed'] else 1
    except (OSError,UnicodeError,ValueError,TypeError) as e:
        print(type(e).__name__+': '+str(e)[:140])
        return 2
if __name__=='__main__':
    raise SystemExit(main())
