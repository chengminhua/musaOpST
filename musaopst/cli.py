# SPDX-License-Identifier: MulanPSL-2.0
"""SDK-free command parsing. Device execution always belongs to a fresh worker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from .util import write_json


def parser():
    from . import __version__
    p=argparse.ArgumentParser(prog='musaopst',description='MUSA 单算子测试、工程生成、回放与回归；CPU 模式仅验证平台。')
    p.add_argument('--version',action='version',version='musaopst '+__version__)
    sub=p.add_subparsers(dest='command',required=True)
    doctor=sub.add_parser('doctor',help='独立进程环境探测；--smoke 额外执行真实后端小算子')
    doctor.add_argument('--device',type=int,default=0);doctor.add_argument('--output',type=Path)
    doctor.add_argument('--timeout',type=float,default=60);doctor.add_argument('--smoke',action='store_true')
    doctor.add_argument('--backend',choices=['torch_musa','musa_cpp','triton','tilelang','native'],action='append')
    init=sub.add_parser('init',help='生成可编辑示例或指定后端工程')
    init.add_argument('directory',type=Path);init.add_argument('--backend',choices=['cpu','torch_musa','musa_cpp','triton','tilelang','native'])
    validate=sub.add_parser('validate',help='离线校验和展开 v1/v2 JSON');validate.add_argument('config',type=Path)
    for name in ('run','profile'):
        run=sub.add_parser(name,help='新进程串行执行用例' if name=='run' else '正确性通过后采集独立 GPU trace')
        run.add_argument('config',type=Path,nargs='*');run.add_argument('--prepared',type=Path)
        run.add_argument('--output',type=Path)
        run.add_argument('--device',type=int,default=0,help='可见设备列表内的序号（torch.musa.set_device 语义），不是物理 GPU 编号；默认 0')
        run.add_argument('--benchmark',action='store_true',default=None);run.add_argument('--repeat',type=int)
        run.add_argument('--duration',type=float)
        run.add_argument('--case',dest='pattern',help="按用例名筛选：通配符（'*softmax*'）或直接子串（'softmax'）")
        run.add_argument('--tag',action='append')
        run.add_argument('--resume',type=Path)
        run.add_argument('--rerun-failed',action='store_true')
        run.add_argument('--list-cases',action='store_true',help='只列出筛选后的 name/backend/tags，不执行用例')
    report=sub.add_parser('report',help='报告重绘、合并或可比性能基线检查')
    report.add_argument('results',type=Path,nargs='+');report.add_argument('--output',type=Path)
    report.add_argument('--merge',action='store_true');report.add_argument('--baseline',type=Path)
    report.add_argument('--threshold-percent',type=float,default=10)
    for name in ('prepare','data','build'):
        cmd=sub.add_parser(name,help='独立准备/造数/构建；不冒充精度验收')
        cmd.add_argument('config',type=Path);cmd.add_argument('--output',type=Path,required=True)
        cmd.add_argument('--device',type=int,default=0);cmd.add_argument('--arch')
    comp=sub.add_parser('compare',help='从快照重新比较，写新报告和CSV')
    comp.add_argument('results',type=Path);comp.add_argument('--output',type=Path,required=True)
    comp.add_argument('--policy',type=Path);comp.add_argument('--csv-limit',type=int,default=10000)
    replay=sub.add_parser('replay',help='使用保存的实际输入快照重新执行')
    replay.add_argument('results',type=Path);replay.add_argument('--output',type=Path,required=True)
    replay.add_argument('--device',type=int,default=0);replay.add_argument('--case',dest='pattern')
    bundle=sub.add_parser('bundle',help='打包声明源码、配置、快照和证据')
    bundle.add_argument('results',type=Path);bundle.add_argument('--output',type=Path,required=True)
    generate=sub.add_parser('generate',help='从算子描述生成工程、有限Fuzz配置')
    generate.add_argument('descriptor',type=Path);generate.add_argument('--output',type=Path,required=True)
    generate.add_argument('--seed',type=int,default=0);generate.add_argument('--count',type=int)
    proto=sub.add_parser('import-prototype',help='读取受支持的原型描述；缺失信息保留为候选')
    proto.add_argument('source',type=Path);proto.add_argument('--output',type=Path,required=True)
    model=sub.add_parser('extract-model',help='提取 ONNX/TF GraphDef 节点规格，不编造参考与真实输入')
    model.add_argument('model',type=Path);model.add_argument('--output',type=Path,required=True)
    model.add_argument('--format',choices=['onnx','tf'])
    cache=sub.add_parser('cache',help='管理本工具独立编译缓存')
    cache.add_argument('action',choices=['list','prune']);cache.add_argument('--root',type=Path)
    return p


def init_project(directory,backend=None):
    import shutil
    if backend:
        from .generation import generate
        with tempfile.TemporaryDirectory(prefix='musaopst-init-') as tmp:
            descriptor=Path(tmp)/'descriptor.json';write_json(descriptor,{'backend':backend,'name':'axpby','operation':'axpby'})
            result=generate(descriptor,directory)
        print(json.dumps(result,ensure_ascii=False));return
    root=Path(__file__).parent/'templates';directory.mkdir(parents=True,exist_ok=True)
    targets=[p for p in root.iterdir() if p.is_file() and p.suffix in {'.py','.json'}]
    conflicts=[directory/p.name for p in targets if (directory/p.name).exists()]
    if conflicts:raise ValueError(f'拒绝覆盖已有文件：{conflicts}')
    for source in targets:shutil.copyfile(source,directory/source.name)
    print(f'已生成模板：{directory.resolve()}')


def _module_context(directory):
    path=Path(directory)/'execution-context.json'
    if path.is_file():
        paths=json.loads(path.read_text()).get('module_search_paths',[])
        os.environ['PYTHONPATH']=os.pathsep.join([*paths,os.environ.get('PYTHONPATH','')])


def _config_backends(configs):
    from .config import ConfigError, load_config
    try:
        return [case['backend'] for path in configs for case in load_config(path)]
    except ConfigError:
        # An invalid configuration must still produce its CONFIG_ERROR report.
        return None


def _resolve_device(configs, requested):
    """Validate --device in the parent, before any worker or output is created."""
    from .device import resolve_device, announce
    backends=_config_backends(configs)
    if backends is None:return None
    if all(backend=='cpu' for backend in backends):
        print('device: 仅 CPU 平台用例，跳过 --device 校验',flush=True)
        return None
    selection=resolve_device(requested);announce(selection);return selection


def _list_cases(configs, pattern, tags):
    from .config import load_config, filter_cases, selection_error
    if not configs:raise ValueError('--list-cases 需要至少一个配置路径')
    entries=[];available=[]
    for index,path in enumerate(configs):
        for case in load_config(path):
            name=(str(index)+':'+path.stem+'::' if len(configs)>1 else '')+case['name']
            available.append(name)
            entries.append({'name':name,'backend':case['backend'],'tags':list(case.get('tags') or [])})
    selected=filter_cases(entries,pattern,tags)
    if not selected and (pattern or tags):
        raise ValueError(selection_error(available,pattern,tags))
    print('# name\tbackend\ttags')
    for case in selected:
        print(f"{case['name']}\t{case['backend']}\t{','.join(case['tags']) or '-'}")
    print(f'# {len(selected)}/{len(entries)} 个用例')
    return 0


def _doctor(args):
    from .runner import terminate_worker
    import math
    if not math.isfinite(args.timeout) or args.timeout<=0:raise ValueError('--timeout 必须是有限正数')
    with tempfile.TemporaryDirectory(prefix='musaopst-doctor-') as tmp:
        dest=args.output or Path(tmp);dest.mkdir(parents=True,exist_ok=True)
        if (dest/'doctor.json').exists():raise ValueError('doctor.json 已存在，拒绝覆盖')
        with (dest/'doctor.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m','musaopst.doctor',str(dest/'doctor.json'),str(args.device)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                terminate_worker(proc);write_json(dest/'doctor.json',{'gpu_available':False,'status':'TIMEOUT'});rc=1
        if not (dest/'doctor.json').exists():write_json(dest/'doctor.json',{'gpu_available':False,'status':'ENVIRONMENT_ERROR','message':f'worker exited {rc}'})
        record=json.loads((dest/'doctor.json').read_text())
        if record.get('device_resolution'):
            from .device import format_resolution
            print('device: '+format_resolution(record['device_resolution']),flush=True)
        if record.get('status')=='DEVICE_SELECTION_ERROR':
            print(record.get('device_selection_error','--device 越界'),file=sys.stderr)
            print(json.dumps(record,indent=2,ensure_ascii=False));return 2
        if args.smoke and record.get('gpu_available') and rc==0:
            from .runner import run_suite
            cases=[]
            entries={'torch_musa':'torch_ops:softmax','triton':'triton_ops:softmax','tilelang':'tilelang_ops:vector_add','musa_cpp':'cpp_ops:vector_add','native':'native_ops:vector_add'}
            for backend in (args.backend or list(entries)):
                soft=backend in {'torch_musa','triton'}
                spec={'shape':[2,17] if soft else [17],'dtype':'float32'}
                cases.append({'name':'smoke_'+backend,'backend':backend,'entry':'musaopst.examples.'+entries[backend],
                              'inputs':[spec] if soft else [spec,spec], 'outputs':[spec],
                              'reference':'musaopst.examples.references:'+('softmax' if soft else 'add'),
                              'attrs':{'dim':-1} if soft else {},'timeout_s':180})
            cfg=dest/'smoke.json';write_json(cfg,{'schema_version':1,'cases':cases})
            smoke=run_suite(cfg,dest/'smoke',device=record.get('device_selection',args.device),
                            device_selection=record.get('device_resolution'))
            record['smoke_summary']=smoke['summary'];rc=smoke['summary']['exit_code']
            write_json(dest/'doctor.json',record)
        print(json.dumps(record,indent=2,ensure_ascii=False));return 0 if record.get('gpu_available') and rc==0 else 1


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if hasattr(args,'device') and args.device<0:raise ValueError('--device 必须是非负整数')
        if args.command=='doctor':return _doctor(args)
        if args.command=='init':init_project(args.directory,args.backend);return 0
        if args.command=='validate':
            from .config import load_config
            cases=load_config(args.config);print(json.dumps({'valid':True,'expanded_count':len(cases),'names':[c['name'] for c in cases]},ensure_ascii=False));return 0
        if args.command in {'run','profile'}:
            from .runner import run_suite
            if args.list_cases and args.prepared:raise ValueError('--list-cases 不能与 --prepared 同时使用')
            configs=args.config
            if args.prepared:
                if configs:raise ValueError('配置与 --prepared 只能选择其一')
                from .workflows import prepared_config
                inputs=Path(str(args.output)+'-inputs');configs=[prepared_config(args.prepared,inputs)]
                _module_context(inputs)
            if args.list_cases:
                return _list_cases(configs,args.pattern,args.tag)
            if not args.output:raise ValueError(f'{args.command} 需要 --output')
            selection=_resolve_device(configs,args.device)
            result=run_suite(configs,args.output,device=selection.ordinal if selection else args.device,
                             device_selection=selection,benchmark=args.benchmark,repeat=args.repeat,
                             duration=args.duration,pattern=args.pattern,tags=args.tag,resume=args.resume,
                             rerun_failed=args.rerun_failed,profile=True if args.command=='profile' else None)
            if result.get('usage_error'):
                for case in result.get('cases',[]):
                    if case.get('status')=='CONFIG_ERROR':print('错误：'+case.get('message',''),file=sys.stderr)
            print(json.dumps(result['summary'],ensure_ascii=False));return result['summary']['exit_code']
        if args.command=='replay':
            from .runner import run_suite
            from .workflows import replay_config
            inputs=Path(str(args.output)+'-inputs');cfg=replay_config(args.results,inputs,case_pattern=args.pattern)
            _module_context(inputs);selection=_resolve_device([cfg],args.device)
            result=run_suite(cfg,args.output,device=selection.ordinal if selection else args.device,
                             device_selection=selection)
            print(json.dumps(result['summary'],ensure_ascii=False));return result['summary']['exit_code']
        if args.command=='report':
            if args.merge:
                if not args.output:raise ValueError('--merge requires --output')
                from .regression import merge_reports
                result=merge_reports(args.results,args.output);code=result['summary']['exit_code']
            elif args.baseline:
                if len(args.results)!=1 or not args.output:raise ValueError('baseline requires one current report and --output')
                from .regression import compare_baseline
                result=compare_baseline(args.results[0],args.baseline,args.output,args.threshold_percent);code=result['exit_code']
            else:
                if len(args.results)!=1:raise ValueError('multiple reports require --merge')
                from .reporting import render_reports
                path=args.results[0];path=path/'results.json' if path.is_dir() else path
                result=json.loads(path.read_text());code=render_reports(result,args.output or path.parent)
            print(json.dumps(result.get('summary',result),ensure_ascii=False));return code
        if args.command in {'prepare','data','build'}:
            from .workflows import prepare
            result=prepare(args.config,args.output,stage=args.command,device=args.device,arch=args.arch)
        elif args.command=='compare':
            from .workflows import recompare
            result=recompare(args.results,args.output,policy=json.loads(args.policy.read_text()) if args.policy else None,csv_limit=args.csv_limit)
        elif args.command=='bundle':
            from .artifacts import bundle
            result=bundle(args.results,args.output)
        elif args.command=='generate':
            from .generation import generate
            result=generate(args.descriptor,args.output,seed=args.seed,count=args.count)
        elif args.command=='import-prototype':
            from .generation import import_prototype
            result=import_prototype(args.source,args.output)
        elif args.command=='extract-model':
            from .models import extract
            result=extract(args.model,args.output,format=args.format)
        elif args.command=='cache':
            from .cache import cache_list,cache_prune
            result=cache_list(args.root) if args.action=='list' else cache_prune(args.root)
        else:raise ValueError('unknown command')
        print(json.dumps(result,ensure_ascii=False,indent=2));return result.get('exit_code',result.get('summary',{}).get('exit_code',0))
    except (ValueError,OSError,KeyError,TypeError,ImportError) as exc:
        print(f'错误：{exc}',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
