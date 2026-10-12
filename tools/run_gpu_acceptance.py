#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Bounded serial acceptance orchestration. No host/credentials/container defaults."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from musaopst.util import write_json


def validate_outcome(name, rc, expected, out):
    if rc != expected:
        return False
    try:
        if name == 'doctor':
            result=json.loads((out/name/'doctor.json').read_text())
            return result.get('gpu_available') is True
        result=json.loads((out/name/'results.json').read_text())
        cases=result.get('cases', [])
        if result.get('run_complete') is not True or not cases:
            return False
        if any(case.get('fatal_device_error') for case in cases):
            return False
        wanted='ACCURACY_FAILED' if name=='negative' else ('UNSUPPORTED' if name=='unsupported' else 'PASS')
        if any(case.get('status') != wanted for case in cases):
            return False
        if name=='negative' and any('benchmark' in case for case in cases):
            return False
        return True
    except (OSError,ValueError,TypeError,KeyError):
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',type=int,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('output must be new or empty')
    out.mkdir(parents=True,exist_ok=True)
    base=[sys.executable,'-m','musaopst']
    device=['--device',str(args.device)]
    stages=[
        ('doctor',0,['doctor',*device,'--output',str(out/'doctor')]),
        ('suite',0,['run',str(root/'examples/gpu_suite.json'),*device,'--benchmark','--output',str(out/'suite')]),
        ('negative',1,['run',str(root/'examples/gpu_negative.json'),*device,'--benchmark','--output',str(out/'negative')]),
        ('unsupported',1,['run',str(root/'examples/unsupported_layout.json'),*device,'--output',str(out/'unsupported')]),
        ('duration',0,['run',str(root/'examples/gpu_smoke.json'),*device,'--case','torch_softmax','--repeat','0','--duration','0.1','--output',str(out/'duration')]),
    ]
    records=[]
    for name,expected,arguments in stages:
        command=base+arguments
        begin=datetime.now(timezone.utc).isoformat()
        with (out/f'{name}.log').open('w') as log:
            rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=root).returncode
        outcome_ok=validate_outcome(name,rc,expected,out)
        records.append({'outcome_matches':outcome_ok,'stage':name,'command':command,'expected_exit':expected,'actual_exit':rc,
                        'started_utc':begin,'finished_utc':datetime.now(timezone.utc).isoformat()})
        write_json(out/'orchestration.json',{'records':records,'complete':len(records)==len(stages),
                                           'expected_outcomes_matched':all(r['outcome_matches'] for r in records),
                                           'expected_exit_codes_matched':all(r['actual_exit']==r['expected_exit'] for r in records)})
        print(f'{name}: actual_exit={rc} expected_exit={expected}',flush=True)
        if not outcome_ok:
            return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
