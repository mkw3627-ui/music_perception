"""Dedicated opt-in download subprocess for specialist song-vocal checkpoints."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import types
import uuid
from pathlib import Path


def main():
    if len(sys.argv)!=5:
        return 2
    vendor,root,quality,source=sys.argv[1:]
    here=Path(__file__).resolve().parent
    module=types.ModuleType('music_perception')
    module.__path__=[str(here)]
    sys.modules['music_perception']=module
    if quality not in {'low','medium','high','ultra'} or source not in {'mirror','official'}:
        return 2
    from music_perception.singing_catalog import acquire_model
    status=Path(root) / f'singing_download_{quality}.json'
    # Keep the last genuine byte counters during the separate SHA stage.
    # A stage-only update must not make a 99% progress bar disappear.
    last_transfer={}
    def publish(stage,error='',state=None,**details):
        allowed=('bytes_received','total_bytes','progress_pct','progress_estimated',
                 'speed_bytes_s','eta_seconds','stalled','source_name','last_progress_at')
        if stage=='starting' or str(stage).startswith('downloading:'):
            last_transfer.clear()
        for key in allowed:
            if key in details:
                last_transfer[key]=details[key]
        record={'state':state or ('failed' if error else 'running'),'stage':stage,
                'quality':quality,'updated_at':time.time(),'error':error[:160]}
        record.update(last_transfer)
        # Atomic publication prevents a Hosted UI refresh from reading a
        # truncated JSON file and failing the model download.
        status.parent.mkdir(parents=True,exist_ok=True)
        for attempt in range(4):
            tmp=status.with_name(status.name+'.'+uuid.uuid4().hex[:8]+'.new')
            try:
                tmp.write_text(json.dumps(record,ensure_ascii=False),encoding='utf-8')
                os.replace(tmp,status)
                return
            except OSError:
                time.sleep(0.04*(attempt+1))
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
    try:
        publish('starting')
        acquire_model(Path(root),quality,source=source,on_stage=publish,
                      on_progress=lambda stage,values:publish(stage,**values))
        publish('verified_and_ready',state='ready',progress_pct=100,
                progress_estimated=False,stalled=False,
                speed_bytes_s=None,eta_seconds=0)
        return 0
    except Exception as exc:
        # Parent launcher persists stderr to singing_asr_install.log.
        # Do not hide the real Python exception behind return code 1.
        traceback.print_exc(file=sys.stderr)
        publish('failed',type(exc).__name__+': '+str(exc)[:180])
        return 1
if __name__=='__main__':
    raise SystemExit(main())
