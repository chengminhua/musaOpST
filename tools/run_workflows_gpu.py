#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Five-backend workflow acceptance, to run after frozen device matrices."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from musaopst.runner import source_manifest
from musaopst.util import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--arch',required=True);p.add_argument('--device',type=int,required=True);a=p.parse_args();out=a.output.resolve()
    if out.exists() and any(out.iterdir()):p.error('output must be empty')
    out.mkdir(parents=True,exist_ok=True);rows=[];source=source_manifest()
    def command(name,argv,expected=0):
        cmd=[sys.executable,'-m','musaopst',*map(str,argv)];log=out/(name+'.log')
        with log.open('w') as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT).returncode
        rows.append({'name':name,'command':cmd,'returncode':rc,'expected_returncode':expected,'passed':rc==expected,'log':log.name})
        write_json(out/'workflow.json',{'source':source,'source_unchanged':source_manifest()['content_sha256']==source['content_sha256'],'steps':rows,'all_steps_matched':all(r['passed'] for r in rows)})
        print(name,rc,flush=True);return rc
    for backend in ('torch_musa','musa_cpp','triton','tilelang','native'):
        project=out/('project-'+backend)
        if command('init-'+backend,['init',project,'--backend',backend]):continue
        if command('generated-run-'+backend,['run',project/'cases.json','--device',a.device,'--output',out/('generated-'+backend)]):continue
        if backend in ('musa_cpp','native'):
            prepared=out/('built-'+backend)
            if command('build-'+backend,['build',project/'cases.json','--arch',a.arch,'--device',a.device,'--output',prepared])==0:
                command('prebuilt-'+backend,['run','--prepared',prepared,'--device',a.device,'--output',out/('prebuilt-'+backend)])
    matrix=a.matrix.resolve()
    pattern='*_axpby_float32'
    command('resume',['run',ROOT/'examples/suites/backend_matrix.json','--case',pattern,'--resume',matrix,'--rerun-failed','--device',a.device,'--output',out/'resume'])
    resumed=json.loads((out/'resume/results.json').read_text())
    if resumed['summary']['reused']!=5 or resumed['summary']['fresh_gpu_passed']!=0:rows.append({'name':'resume-counts','passed':False,'reason':'five old records must be reused, none counted as fresh GPU'})
    command('replay',['replay',out/'resume','--case',pattern,'--device',a.device,'--output',out/'replay'])
    command('compare',['compare',out/'replay','--output',out/'recompare','--csv-limit',16])
    command('bundle',['bundle',out/'replay','--output',out/'reproduce.tar.gz'])
    command('merge',['report',out/'resume',out/'replay','--merge','--output',out/'merged'])
    merged=json.loads((out/'merged/results.json').read_text())
    if merged['summary']['fresh_gpu_passed']!=0:rows.append({'name':'merge-counts','passed':False,'reason':'imported results were counted as fresh'})
    # A new, equal workload with enough samples. Outcomes may be noisy rather
    # than OK; preserve every result without labelling incomparable data fast.
    config=json.loads((ROOT/'examples/suites/backend_matrix.json').read_text())
    config['cases']=[c for c in config['cases'] if c['name'].endswith('_axpby_float32_single_event')]
    for case in config['cases']:case['benchmark']['samples']=7
    cfg=out/'timing.json';write_json(cfg,config)
    for name in ('baseline','current'):command(name,['run',cfg,'--device',a.device,'--output',out/name])
    rc=command('baseline-compare',['report',out/'current','--baseline',out/'baseline','--output',out/'regression'],expected=0)
    rows[-1]['passed']=rc in (0,1,2) and (out/'regression/baseline.json').is_file();rows[-1]['interpretation']='0 comparable/no regression, 1 regression, 2 noisy/incomparable; inspect baseline.json'
    write_json(out/'workflow.json',{'source':source,'source_unchanged':source_manifest()['content_sha256']==source['content_sha256'],'steps':rows,'all_steps_matched':all(r['passed'] for r in rows),'completed_utc':datetime.now(timezone.utc).isoformat()})
    return 0 if all(r['passed'] for r in rows) else 1
if __name__=='__main__':raise SystemExit(main())
