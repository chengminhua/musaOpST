# SPDX-License-Identifier: MulanPSL-2.0
"""Report aggregation and explicitly comparable performance baselines."""
import copy
import json
from pathlib import Path
import shutil
import numpy as np
from .reporting import render_reports,validate_case_result
from .util import write_json


def _coverage(report, default_complete=True):
    """Recompute coverage from planned names, never from a cached summary."""
    complete=report.get('run_complete',default_complete)
    if type(complete) is not bool:raise ValueError('run_complete must be boolean')
    cases=report.get('cases')
    if not isinstance(cases,list):raise ValueError('report cases must be a list')
    names=[case.get('name') for case in cases if isinstance(case,dict)]
    if len(names)!=len(cases) or any(not isinstance(name,str) or not name for name in names):
        raise ValueError('report contains an invalid case name')
    if len(set(names))!=len(names):raise ValueError('report case names must be unique')
    planned=report.get('planned_cases',[
        {key:case[key] for key in ('name','backend','required') if key in case} for case in cases])
    if not isinstance(planned,list) or any(not isinstance(item,dict) or not isinstance(item.get('name'),str) or not item['name'] for item in planned):
        raise ValueError('planned_cases must contain named case objects')
    pending=[item['name'] for item in planned if item['name'] not in set(names)]
    return complete and not pending,planned,pending


def merge_reports(paths,output_dir):
    dest=Path(output_dir).resolve()
    if dest.exists() and any(dest.iterdir()):raise ValueError('merge output must be empty')
    dest.mkdir(parents=True,exist_ok=True)
    merged={'schema_version':2,'tool':'musaopst','cases':[],'planned_cases':[],
            'run_complete':True,'kind':'merged_evidence','sources':[],'source_coverage':[]}
    for n,path in enumerate(paths):
        path=Path(path).resolve();path=path/'results.json' if path.is_dir() else path
        report=json.loads(path.read_text());merged['sources'].append(str(path))
        complete,planned,pending=_coverage(report,default_complete=False)
        merged['run_complete']=merged['run_complete'] and complete
        merged['source_coverage'].append({'source':str(path),'run_complete':complete,'pending':pending})
        for item in planned:
            item=copy.deepcopy(item);item['name']=f'{n}:{item["name"]}'
            merged['planned_cases'].append(item)
        for i,item in enumerate(report['cases']):
            validate_case_result(item);case=copy.deepcopy(item)
            case['name']=f'{n}:{case["name"]}';case['provenance']='imported'
            source=path.parent/Path(item.get('case_result',f'case-{i:04d}/result.json')).parent
            sub=Path(f'source-{n}')/f'case-{i:04d}'
            if source.is_dir():shutil.copytree(source,dest/sub)
            case['case_result']=str(sub/'result.json');case['log']=str(sub/'worker.log')
            case['imported_from']=str(path);merged['cases'].append(case)
    render_reports(merged,dest)
    return merged


def _identity(case):
    cfg=case.get('config',{});meta=case.get('backend_metadata',{})
    bench=case.get('benchmark',{});policy=copy.deepcopy(bench.get('policy',{}))
    policy.pop('actual_stream',None)
    return {'backend':case.get('backend'),'inputs':cfg.get('inputs'),'outputs':cfg.get('outputs'),
            'attrs':cfg.get('attrs'),'seed':cfg.get('seed'),'entry':cfg.get('entry'),
            'device_name':meta.get('device_name',meta.get('runtime',{}).get('device_name')),
            'capability':meta.get('device_capability',meta.get('arch')),
            'versions':case.get('versions'),'toolchain_identity':case.get('toolchain_identity'),'policy':policy,'device_identity':case.get('device_identity'),'input_snapshot_sha256':case.get('input_snapshot_sha256')}


def compare_baseline(current,baseline,output_dir,threshold_percent=10.0):
    if not np.isfinite(threshold_percent) or threshold_percent<0:raise ValueError('threshold must be finite and nonnegative')
    def read(p):
        p=Path(p);return json.loads((p/'results.json' if p.is_dir() else p).read_text())
    now,prior=read(current),read(baseline)
    now_complete,_,_=_coverage(now)
    prior_complete,_,_=_coverage(prior)
    old={c['name']:c for c in prior['cases']};rows=[]
    for case in now['cases']:
        validate_case_result(case)
        row={'case':case['name'],'status':'INCOMPARABLE'};other=old.get(case['name'])
        baseline_error=None
        if other is not None:
            try:validate_case_result(other)
            except ValueError as exc:baseline_error=str(exc)
        if not now_complete or not prior_complete:row['reason']='one of the runs is incomplete'
        elif baseline_error is not None:row['reason']='baseline record is invalid: '+baseline_error
        elif not other or case.get('status')!='PASS' or other.get('status')!='PASS':row['reason']='matching passing baseline absent'
        elif not case.get('benchmark') or not other.get('benchmark'):row['reason']='benchmark samples absent'
        elif case.get('backend')!='cpu' and not (case.get('device_identity',{}) or {}).get('verified'):row['reason']='physical device identity is not verified'
        elif _identity(case)!=_identity(other):row['reason']='device/workload/software/timing policy differs'
        else:
            a=np.asarray(case['benchmark']['raw_samples_ms'],dtype=float);b=np.asarray(other['benchmark']['raw_samples_ms'],dtype=float)
            if min(a.size,b.size)<5:row['reason']='need at least five samples in both runs'
            elif not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any():row['reason']='invalid samples'
            else:
                am,bm=float(np.median(a)),float(np.median(b))
                noisy=any(float(np.percentile(x,75)-np.percentile(x,25))>max(float(np.median(x)),1e-12)*0.2 for x in (a,b))
                increase=(am/bm-1)*100 if bm else (0 if not am else float('inf'))
                row.update(current_median_ms=am,baseline_median_ms=bm,change_percent=increase,
                           status='NOISY' if noisy else ('REGRESSION' if increase>threshold_percent else 'OK'),
                           reason='IQR exceeds 20% of median' if noisy else 'matched declared workload and timing policy')
        rows.append(row)
    result={'schema_version':1,'kind':'performance_regression','threshold_percent':threshold_percent,
            'noise_policy':'minimum 5 samples; IQR/median <= 20% in both runs','cases':rows,
            'exit_code':1 if any(r['status']=='REGRESSION' for r in rows) else (0 if any(r['status']=='OK' for r in rows) and all(r['status']=='OK' for r in rows) else 2)}
    dest=Path(output_dir)
    if dest.exists() and any(dest.iterdir()):raise ValueError('baseline report output must be empty')
    dest.mkdir(parents=True,exist_ok=True);write_json(dest/'baseline.json',result)
    (dest/'baseline.md').write_text('| 用例 | 结论 | 原因 |\n|---|---|---|\n'+'\n'.join(f"| {r['case']} | {r['status']} | {r['reason']} |" for r in rows))
    return result
