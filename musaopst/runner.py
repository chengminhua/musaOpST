# SPDX-License-Identifier: MulanPSL-2.0
"""Serial fresh-interpreter supervision, including bounded dynamic groups."""
import copy
import fnmatch
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from . import __version__
from .config import load_config, ConfigError
from .reporting import render_reports, validate_case_result
from .util import write_json, sha256, entry_module
from .expectations import apply_expectation


def source_manifest():
    root=Path(__file__).resolve().parent
    files={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*'))
           if p.is_file() and p.suffix in {'.py','.mu','.cpp','.h','.hpp','.json'}}
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    return {'package_version':__version__,'content_sha256':digest,'files':files,'git_commit':'unknown'}


def terminate_worker(proc):
    try:
        os.killpg(proc.pid,signal.SIGTERM)
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL)
        proc.wait(timeout=5)
    except ProcessLookupError:
        pass


def clean_case(case):
    return {k:v for k,v in case.items() if not k.startswith('_') and k!='parameters_selected'}


def fingerprint(case,config_dir,device,source,device_identity=None,toolchain_identity=None):
    """Only declared sources are covered; all evidence states this scope."""
    versions={}
    for package in ('numpy','torch','torch_musa','triton','tilelang','tilelang-musa'):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]='unavailable'
    paths=[Path(config_dir),Path(__file__).resolve().parent.parent,*map(Path,sys.path)]
    declared={}
    entries={case['entry'],case['reference']}
    def custom_entries(value):
        if isinstance(value,dict):
            if isinstance(value.get('custom'),str):entries.add(value['custom'])
            for item in value.values():custom_entries(item)
        elif isinstance(value,list):
            for item in value:custom_entries(item)
    for value in (case.get('compare',{}),case.get('compare_by_dtype',{}),case.get('outputs',[])):
        custom_entries(value)
    for entry in sorted(entries):
        mod=entry_module(entry).replace('.','/')
        found=next((p/f'{mod}.py' for p in paths if (p/f'{mod}.py').is_file()),None)
        if found is None: return None
        declared[entry]=sha256(found)
    def files(value):
        if isinstance(value,dict):
            if isinstance(value.get('path'),str):
                p=Path(value['path']);p=p if p.is_absolute() else Path(config_dir)/p
                if p.is_file():declared[str(p.resolve())]=sha256(p)
            for item in value.values():files(item)
        elif isinstance(value,list):
            for item in value:files(item)
    files(case.get('inputs',[]))
    for raw in case.get('sources',[]):
        p=Path(raw);p=p if p.is_absolute() else Path(config_dir)/p
        if not p.is_file():return None
        declared[str(p.resolve())]=sha256(p)
    payload={'case':clean_case(case),'source':source,'declared':declared,'versions':versions,
             'device':device,'device_identity':device_identity,'toolchain_identity':toolchain_identity,'platform':platform.platform(),'python':sys.version,
             'visibility':{k:os.environ.get(k) for k in ('MUSA_VISIBLE_DEVICES','MTHREADS_VISIBLE_DEVICES')}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def reusable_snapshots(old, case_dir):
    """A manifest cannot make missing mandatory evidence optional."""
    required={'inputs.npz','actual.npz','reference.npz'}
    if old.get('config',{}).get('benchmark',{}).get('enabled') or old.get('benchmark'):
        required.add('post_benchmark.npz')
    if old.get('config',{}).get('profile',{}).get('enabled') or old.get('profile'):
        required.update({'profile_trace.json','profile_table.txt'})
    hashes=old.get('snapshot_hashes',{})
    root=Path(case_dir).resolve()
    if not isinstance(hashes,dict) or not required.issubset(hashes):return False
    try:
        for name,digest in hashes.items():
            path=(root/name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha256(path)!=digest:return False
        validate_case_result(old)
    except (OSError,ValueError,TypeError):return False
    return True


def run_suite(config_path,output_dir,device=0,benchmark=None,repeat=None,duration=None,pattern=None,
              tags=None,resume=None,rerun_failed=False,profile=None):
    from .config import revalidate_cases
    configs=[Path(p).resolve() for p in (config_path if isinstance(config_path,(list,tuple)) else [config_path])]
    output_dir=Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f'结果目录非空，拒绝覆盖：{output_dir}')
    output_dir.mkdir(parents=True,exist_ok=True)
    source=source_manifest()
    result={'schema_version':2,'tool':'musaopst','started_utc':datetime.now(timezone.utc).isoformat(),
            'environment':{'python':sys.version,'platform':platform.platform(),'hostname':platform.node(),'device_selection':device},
            'source':source,'cases':[],'config_file':str(configs[0]) if configs else 'unknown',
            'config_files':list(map(str,configs)),'reproduce':'unknown','run_complete':False}
    try:
        if not configs:raise ConfigError('必须提供配置或已有准备产物')
        originals=[];cases=[];origins=[]
        for config_index,path in enumerate(configs):
            originals.append(json.loads(path.read_text()))
            loaded=load_config(path)
            for case in loaded:
                if len(configs)>1:case['name']=str(config_index)+':'+path.stem+'::'+case['name']
                if pattern and not fnmatch.fnmatch(case['name'],pattern):continue
                if tags and not set(tags).issubset(case.get('tags',[])):continue
                cases.append(case);origins.append(str(path.parent))
        if not cases:raise ConfigError('筛选后没有用例')
        if len({c['name'] for c in cases})!=len(cases):raise ConfigError('配置展开后名称重复')
        result['original_config']=originals[0] if len(originals)==1 else originals
        result['config_sha256']=[sha256(p) for p in configs] if len(configs)>1 else sha256(configs[0])
        write_json(output_dir/'config.original.json',result['original_config'])
        selected={c['name']:c.get('parameters_selected',{}) for c in cases}
        for case in cases:
            if benchmark is not None:case['benchmark']['enabled']=benchmark
            if repeat is not None:case['repeat']['count']=repeat
            if duration is not None:case['repeat']['duration_s']=duration
            if profile is not None:
                case['_schema_version']=2
                case['profile']=dict(case.get('profile',{}),enabled=profile)
        cases=revalidate_cases(cases)
        for c,origin in zip(cases,origins):
            c['_config_dir']=origin
            c['parameters_selected']=selected.get(c['name'],{})
        groups={}
        for i,c in enumerate(cases):
            if c.get('shape_mode')=='runtime':groups.setdefault(c['compile_group'],[]).append(i)
        for group,indexes in groups.items():
            if indexes!=list(range(indexes[0],indexes[-1]+1)):raise ConfigError('动态组必须在配置中连续排列')
            first=cases[indexes[0]]
            for i in indexes:
                if any(cases[i][k]!=first[k] for k in ('backend','entry','attrs','_config_dir')):
                    raise ConfigError('动态组必须使用相同 backend、entry、attrs')
                if cases[i]['benchmark'].get('stream')!=first['benchmark'].get('stream'):raise ConfigError('动态组的stream政策必须一致')
                if cases[i]['skip']:raise ConfigError('动态组不支持跳过其中单例')
    except (ConfigError,ValueError,OSError) as exc:
        result['run_complete']=True
        result['cases'].append({'name':'configuration','backend':'config','required':True,'status':'CONFIG_ERROR','message':str(exc)})
        render_reports(result,output_dir);return result
    result['planned_cases']=[{k:c[k] for k in ('name','backend','required')} for c in cases]
    version=2 if any(c.get('_schema_version')==2 for c in cases) else 1
    write_json(output_dir/'config.resolved.json',{'schema_version':version,'cases':[clean_case(c) for c in cases]})
    replay_paths=[str(Path(__file__).resolve().parent.parent),*dict.fromkeys(origins)]
    if os.environ.get('PYTHONPATH'):replay_paths.append(os.environ['PYTHONPATH'])
    result['replay_module_search_paths']=replay_paths
    result['reproduce']=shlex.join(['env','PYTHONPATH='+os.pathsep.join(replay_paths),sys.executable,'-m','musaopst','replay',str(output_dir/'results.json'),'--output',str(output_dir)+'-replay','--device',str(device)])
    from .device_identity import probe, toolchain_probe
    physical=probe(device) if any(c['backend']!='cpu' for c in cases) else None
    result['device_identity']=physical
    toolchain=toolchain_probe() if physical is not None else None
    result['toolchain_identity']=toolchain
    if rerun_failed and not resume:raise ValueError('--rerun-failed requires --resume')
    result['selection_mode']='failed-and-unverified' if rerun_failed else 'all-selected'
    previous={};previous_root=None
    if resume:
        previous_path=Path(resume).resolve();previous_path=previous_path/'results.json' if previous_path.is_dir() else previous_path
        previous_root=previous_path.parent
        prior=json.loads(previous_path.read_text())
        previous={c['name']:c for c in prior.get('cases',[])}
        result['resume_source']=str(previous_path)
    stopped=False;completed_groups=set();group_aborted={};group_timeouts={}
    for index,case in enumerate(cases):
        path=output_dir/f'case-{index:04d}';path.mkdir(exist_ok=True)
        request={'case':case,'device':device,'config_dir':case['_config_dir'],'workdir':str(path)}
        write_json(path/'request.json',request)
        base={'name':case['name'],'backend':case['backend'],'required':case['required'],'config':case,
              'mode':'CPU_PLATFORM_TEST_ONLY' if case['backend']=='cpu' else 'MUSA_DEVICE',
              'log':str(path.relative_to(output_dir)/'worker.log'),'case_result':str(path.relative_to(output_dir)/'result.json'),
              'provenance':'fresh','fingerprint':fingerprint(case,case['_config_dir'],device,source['content_sha256'],physical if case['backend']!='cpu' else None,toolchain if case['backend']!='cpu' else None),'device_identity':physical if case['backend']!='cpu' else None,'toolchain_identity':toolchain if case['backend']!='cpu' else None}
        old=previous.get(case['name']);reused=False
        if old and old.get('status') in {'PASS','EXPECTED_FAILURE'} and base['fingerprint'] and old.get('fingerprint')==base['fingerprint'] and not case.get('compile_group'):
            # CPU is safe without a new device probe. GPU reuse remains conservative:
            # run fresh until physical-device identity can be independently revalidated.
            if case['backend']=='cpu' or (physical and physical.get('verified') and toolchain and toolchain.get('verified') and old.get('device_identity')==physical):
                old_case=previous_root/Path(old.get('case_result','')).parent
                if reusable_snapshots(old,old_case):
                    shutil.copytree(old_case,path,dirs_exist_ok=True)
                    kept={k:base[k] for k in ('log','case_result','config','fingerprint')}
                    base.update(old);base.update(kept,provenance='reused',reused_from=str(previous_root),elapsed_s=0)
                    reused=True
            else:base['reuse_note']='GPU physical identity not independently confirmed; execute fresh'
        if reused:pass
        elif index in group_timeouts:
            base.update(status='TIMEOUT',last_stage='timeout',exception_type='TimeoutExpired',message=group_timeouts[index],elapsed_s=0)
            apply_expectation(base,case)
        elif case['skip'] or (stopped and case['backend']!='cpu' and not (case.get('compile_group') in completed_groups and (path/'result.json').is_file())):
            base.update(status='SKIPPED',message='配置显式跳过' if case['skip'] else '先前 GPU 超时或严重设备错误，停止本轮设备验证',elapsed_s=0)
        else:
            start=time.monotonic();group=case.get('compile_group') if case.get('shape_mode')=='runtime' else None
            already=group in completed_groups if group else False
            cmd=[sys.executable,'-m','musaopst.worker',str(path/'request.json')]
            timeout=case['timeout_s']
            if group and not already:
                requests=[]
                for other in groups[group]:
                    dest=output_dir/f'case-{other:04d}';dest.mkdir(exist_ok=True)
                    requests.append({'case':cases[other],'device':device,'config_dir':cases[other]['_config_dir'],'workdir':str(dest)})
                write_json(path/'group.request.json',{'requests':requests})
                cmd[-1]=str(path/'group.request.json')
                timeout=sum(cases[i]['timeout_s'] for i in groups[group])
            env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([*replay_paths,env.get('PYTHONPATH','')])
            env.setdefault('OMP_NUM_THREADS','1');env.setdefault('MAX_JOBS','2')
            rc=0
            try:
                if not already:
                    with (path/'worker.log').open('w') as log:
                        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                        try:
                            if group:
                                active=0;active_start=time.monotonic()
                                while True:
                                    while active<len(requests) and (Path(requests[active]['workdir'])/'result.json').is_file():
                                        active+=1;active_start=time.monotonic()
                                    budget=requests[min(active,len(requests)-1)]['case']['timeout_s']
                                    remaining=budget-(time.monotonic()-active_start)
                                    if remaining<=0:
                                        exc=subprocess.TimeoutExpired(cmd,budget);exc.active_index=min(active,len(requests)-1);raise exc
                                    try:rc=proc.wait(timeout=min(.05,remaining));break
                                    except subprocess.TimeoutExpired:pass
                            else:rc=proc.wait(timeout=timeout)
                        except BaseException:
                            terminate_worker(proc);raise
                    if group:
                        completed_groups.add(group)
                        if not (path/'group.complete.json').is_file():
                            group_aborted[group]=rc
                if (path/'result.json').exists():
                    try:
                        record=json.loads((path/'result.json').read_text());validate_case_result(record)
                        if any(record.get(k)!=base[k] for k in ('name','backend','required','mode')):raise ValueError('worker结果标识不匹配')
                        base.update(record)
                        if rc!=0 and not group and base['status'] in {'PASS','EXPECTED_FAILURE'}:
                            base.update(status='EXECUTION_FAILED',message=f'worker退出{rc}，结果不能记为通过')
                            if case['backend']!='cpu':base['fatal_device_error']=True
                    except (ValueError,TypeError,OSError) as exc:
                        shutil.copyfile(path/'result.json',path/'worker_result.invalid.json')
                        base.update(status='EXECUTION_FAILED',message=f'worker结构化结果无效：{exc}')
                        if case['backend']!='cpu':base['fatal_device_error']=True
                elif already:
                    base.update(status='SKIPPED',message='动态组前序用例未完成，后续绑定未执行')
                else:
                    base.update(status='EXECUTION_FAILED',message=f'worker退出{rc}且没有结构化结果，见日志')
                    if case['backend']!='cpu':base['fatal_device_error']=True
            except subprocess.TimeoutExpired as exc:
                timed_index=groups[group][getattr(exc,'active_index',0)] if group else index
                if group:
                    group_timeouts[timed_index]=f'动态组当前用例超过 {exc.timeout} 秒，终止本测试进程组'
                base.update(status='TIMEOUT',last_stage='timeout',exception_type='TimeoutExpired',message=f'worker超过{timeout}秒；仅终止本测试进程组')
                if group:
                    completed_groups.add(group)
                    if timed_index!=index and (path/'result.json').is_file():
                        saved=json.loads((path/'result.json').read_text());validate_case_result(saved);base.update(saved)
                if case['backend']!='cpu':stopped=True
            if group in group_aborted:
                base.update(status='EXECUTION_FAILED',message=f'动态 worker 未正常完成，退出码 {group_aborted[group]}',fatal_device_error=case['backend']!='cpu')
            base['elapsed_s']=time.monotonic()-start
            apply_expectation(base,case)
            if base.get('fatal_device_error'):stopped=True
        write_json(path/'result.json',base);result['cases'].append(base)
        print(f"[{index+1}/{len(cases)}] {base['status']} {case['name']}",flush=True)
        render_reports(result,output_dir)
    result['run_complete']=True;result['finished_utc']=datetime.now(timezone.utc).isoformat()
    render_reports(result,output_dir);return result
