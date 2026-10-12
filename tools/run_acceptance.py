#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Serial bounded acceptance; records partial support without inventing passes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from musaopst.runner import source_manifest
from musaopst.util import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,required=True)
    p.add_argument('--stage',action='append',choices=['doctor','baseline','matrix','bindings','contracts','dtype','dynamic','profile','constants','negative','unsupported','duration'])
    a=p.parse_args();out=a.output.resolve()
    if out.exists() and any(out.iterdir()):p.error('output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    selected=a.stage or ['doctor','baseline','matrix','bindings','contracts','dtype','dynamic','profile','constants','negative','unsupported','duration']
    files={'baseline':'gpu_suite.json','matrix':'suites/backend_matrix.json','bindings':'suites/binding_matrix.json',
           'contracts':'suites/contracts_matrix.json','dtype':'suites/dtype_matrix.json','dynamic':'suites/dynamic_matrix.json',
           'profile':'suites/profile_matrix.json','constants':'suites/constants_matrix.json','negative':'gpu_negative.json','unsupported':'unsupported_layout.json','duration':'gpu_smoke.json'}
    initial=source_manifest();records=[];fatal=False
    for name in selected:
        command=[sys.executable,'-m','musaopst']
        if name=='doctor':command+=['doctor','--smoke','--output',str(out/name),'--device',str(a.device)]
        else:command+=['run',str(ROOT/'examples'/files[name]),'--output',str(out/name),'--device',str(a.device)]
        if name in ('baseline','negative'):command+=['--benchmark']
        if name=='duration':command+=['--case','torch_softmax','--repeat','0','--duration','0.1']
        begin=datetime.now(timezone.utc).isoformat()
        with (out/(name+'.log')).open('w') as log:rc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        row={'stage':name,'command':command,'returncode':rc,'started_utc':begin,'finished_utc':datetime.now(timezone.utc).isoformat()}
        path=out/name/('doctor.json' if name=='doctor' else 'results.json')
        if path.exists():
            report=json.loads(path.read_text());row['summary']=report.get('summary',report.get('smoke_summary'))
            if name=='doctor' and (out/name/'smoke/results.json').exists():
                report=json.loads((out/name/'smoke/results.json').read_text())
            fatal=any(c.get('fatal_device_error') or (c.get('status')=='TIMEOUT' and c.get('backend')!='cpu') for c in report.get('cases',[]))
        else:row['error']='structured report missing';fatal=True
        records.append(row);changed=source_manifest()['content_sha256']!=initial['content_sha256']
        summary={'source':initial,'source_changed':changed,'selected_stages':selected,'stages':records,'complete':len(records)==len(selected),'fatal_device_error':fatal,'interpretation':'Inspect every matrix cell; UNSUPPORTED/failed cells are never accuracy passes.'}
        write_json(out/'acceptance.json',summary)
        print(f'{name}: exit={rc} {row.get("summary",{})}',flush=True)
        if fatal or changed:break
    return 1 if fatal or summary['source_changed'] or not summary['complete'] else 0
if __name__=='__main__':raise SystemExit(main())
