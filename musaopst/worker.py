# SPDX-License-Identifier: MulanPSL-2.0
"""Fresh-interpreter worker, never imported by a GPU-initialized parent."""
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from .compare import compare_outputs
from .data import make_inputs, reference_inputs, tensor_metadata
from .errors import BuildError, EnvironmentError, UnsupportedError, is_fatal_device_error
from .util import load_entry, as_outputs, write_json, sha256


def versions():
    found = {}
    for name in ['numpy','torch','torch_musa','triton','tilelang','tilelang-musa','apache-tvm-ffi','z3-solver']:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = 'unknown'
    return found


def entry_source(entry):
    fn = load_entry(entry)
    try:
        path = inspect.getsourcefile(fn)
    except (TypeError, OSError):
        path = None
    return {'entry':entry,'file':path or 'unknown','sha256':sha256(path) if path and Path(path).is_file() else 'unknown'}


def comparator_sources(case):
    """Best-effort provenance must not replace the comparator's own verdict."""
    policies=[case.get('compare',{}),*(output.get('compare',{}) for output in case.get('outputs',[]))]
    sources=[]
    for entry in sorted({policy['custom'] for policy in policies if policy.get('custom')}):
        try:sources.append(entry_source(entry))
        except Exception as exc:
            sources.append({'entry':entry,'file':'unknown','sha256':'unknown',
                            'source_error':f'{type(exc).__name__}: {exc}'})
    return sources


def execute(request, session=None):
    case = request['case']
    workdir = Path(request['workdir'])
    sys.path.insert(0, request['config_dir'])
    result = {'status':'EXECUTION_FAILED','name':case['name'],'backend':case['backend'],
              'required':case['required'],'versions':versions(),'seed':case['seed'],
              'mode':'CPU_PLATFORM_TEST_ONLY' if case['backend']=='cpu' else 'MUSA_DEVICE',
              'phase_times_s':{},'provenance':'fresh','worker_pid':os.getpid()}
    stage = 'environment'
    backend = None
    caught_error = None
    begin = time.monotonic()
    phase_begin = begin
    def phase(name):
        nonlocal stage, phase_begin
        now = time.monotonic()
        result['phase_times_s'][stage] = now-phase_begin
        stage, phase_begin = name, now
        write_json(workdir/'progress.json', {'stage':stage,'elapsed_s':now-begin})
    try:
        phase('inputs')
        inputs = make_inputs(case, base_dir=request['config_dir'])
        from .binding import flatten_tensors, tensor_specs, bind_inputs
        leaves, specs = flatten_tensors(inputs), tensor_specs(case['inputs'])
        result['inputs'] = [dict(tensor_metadata(x, spec['dtype']),name=spec.get('name',str(i))) for i,(x,spec) in enumerate(zip(leaves,specs))]
        np.savez(workdir/'inputs.npz', **{f'input_{i}':x for i,x in enumerate(leaves)})
        result['input_snapshot_sha256']=sha256(workdir/'inputs.npz')
        from .data import source_metadata
        result['input_sources']=[source_metadata(spec,request['config_dir']) for spec in specs]
        # Reference owns separate storage BEFORE the implementation can mutate inputs.
        ref_inputs = reference_inputs(inputs, policy=case.get('reference_policy','float64'))
        phase('environment')
        execution_case = dict(case, device=request['device'])
        if session is not None and session.get('backend') is not None:
            backend=session['backend']
            phase('bind')
            backend.rebind(execution_case, inputs)
        else:
            if case['backend']=='cpu':
                from .backends.cpu import CPUBackend
                backend = CPUBackend(execution_case, inputs, workdir)
            else:
                from .backends import create_backend
                backend = create_backend(execution_case, inputs, workdir)
            phase('build')
            backend.prepare()
            if session is not None:
                session['backend']=backend
                session['prepare_calls']=session.get('prepare_calls',0)+1
        if session is not None:
            result['dynamic_execution']={'group':case.get('compile_group'),
                'prepare_calls':session.get('prepare_calls',0),'same_worker':True,
                'artifact_evidence':backend.metadata().get('launch',{})}
        backend.synchronize()
        result['implementation_source'] = entry_source(case['entry'])
        result['backend_metadata'] = backend.metadata()
        observed_inputs=result['backend_metadata'].get('inputs')
        if isinstance(observed_inputs,list) and len(observed_inputs)==len(result['inputs']):
            checks=[]
            for wanted,got in zip(result['inputs'],observed_inputs):
                fields=['shape','strides','storage_offset']
                checked=[key for key in fields if key in got]
                match=all(got[key]==wanted[key] for key in checked)
                checks.append({'checked':checked,'passed':match})
                if not match:raise RuntimeError('设备输入实际布局与构造输入不一致')
            result['input_layout_checks']=checks
        result['memory_after_prepare'] = backend.memory()
        phase('execute')
        if hasattr(backend,'reset_inputs'):
            backend.reset_inputs()
        output = backend.run()
        backend.synchronize()
        actual_metadata=backend.output_metadata(output) if hasattr(backend,'output_metadata') else None
        actual, actual_dtypes = backend.snapshot(output)
        actual = [np.array(x,copy=True) for x in actual]
        result['execution_coverage']={'input_elements':sum(x.size for x in leaves),
            'output_elements':sum(x.size for x in actual),'no_tensor_inputs':not leaves,
            'empty_output_contract_only':not any(x.size for x in actual),
            'device_kernel_proof':'use a validated GPU profiler trace; nonempty tensors alone are not kernel tracing'}
        np.savez(workdir/'actual.npz', **{f'output_{i}':x for i,x in enumerate(actual)})
        result['actual_metadata'] = actual_metadata or [tensor_metadata(x,d) for x,d in zip(actual,actual_dtypes)]
        phase('reference')
        result['reference_source'] = entry_source(case['reference'])
        ref_args,ref_kwargs=bind_inputs(ref_inputs,case['inputs'],case['attrs'],binding=case.get('binding','positional'))
        expected = [np.array(x,copy=True) for x in as_outputs(load_entry(case['reference'])(*ref_args, **ref_kwargs))]
        if not expected:
            raise ValueError('参考计算返回空输出列表')
        np.savez(workdir/'reference.npz', **{f'output_{i}':x for i,x in enumerate(expected)})
        result['reference_policy'] = case.get('reference_policy','float64')
        result['snapshot_hashes']={name:sha256(workdir/name) for name in ['inputs.npz','actual.npz','reference.npz']}
        phase('compare')
        result['comparison'] = compare_outputs(actual, expected, case['outputs'], case['compare'], actual_dtypes=actual_dtypes,actual_metadata=result['actual_metadata'],input_metadata=result['inputs'])
        result['comparator_sources'] = comparator_sources(case)
        if not result['comparison']['passed']:
            result.update(status='ACCURACY_FAILED', exception_type='ComparisonFailure', message='输出契约或数值比较失败；未进入正式性能测试')
            return result
        phase('repeat')
        count, duration = case['repeat'].get('count',0), case['repeat'].get('duration_s',0)
        start = time.monotonic()
        repeat = {'requested_count':count,'requested_duration_s':duration,'completed':0,'anomalies':[],
                  'memory_before':backend.memory(),'interpretation':'有限条件重复测试，不代表长期稳定性'}
        while count > 0 or duration > 0:
            if count > 0 and repeat['completed'] >= count:
                break
            if duration > 0 and time.monotonic()-start >= duration:
                break
            if hasattr(backend,'reset_inputs'):
                backend.reset_inputs()
            output = backend.run()
            backend.synchronize()
            arrays, dtypes = backend.snapshot(output)
            checked = compare_outputs(arrays,expected,case['outputs'],case['compare'],actual_dtypes=dtypes,actual_metadata=backend.output_metadata(output) if hasattr(backend,'output_metadata') else None,input_metadata=result['inputs'])
            repeat['completed'] += 1
            if not checked['passed']:
                repeat['anomalies'].append({'iteration':repeat['completed'],'comparison':checked})
                np.savez(workdir/'repeat_failure.npz', **{f'output_{i}':x for i,x in enumerate(arrays)})
                break
        repeat.update(elapsed_s=time.monotonic()-start,memory_after=backend.memory())
        result['repeat'] = repeat
        if repeat['anomalies']:
            result.update(status='ACCURACY_FAILED',exception_type='ComparisonFailure',message='有限重复测试发现数值异常')
            return result
        negative_expected=case.get('expect',{}).get('status','PASS') != 'PASS'
        if case.get('profile',{}).get('enabled') and not negative_expected:
            phase('profile')
            if not hasattr(backend,'profile'):
                raise UnsupportedError('This backend does not implement device profiling')
            result['profile']=backend.profile(case['profile'])
            backend.synchronize()
        if case['benchmark']['enabled'] and not negative_expected:
            phase('benchmark')
            bench = backend.benchmark(case['benchmark'])
            backend.synchronize()
            samples = np.array(bench['raw_samples_ms'],dtype=np.float64)
            if samples.size != case['benchmark']['samples'] or not np.isfinite(samples).all() or (samples<0).any():
                raise RuntimeError('计时接口返回样本数不符或非法值')
            bench['statistics'] = {'median_ms':float(np.median(samples)), 'p90_ms':float(np.percentile(samples,90)),
                                   'min_ms':float(samples.min()),'max_ms':float(samples.max()),
                                   'percentile_method':'numpy linear','n':len(samples)}
            result['benchmark'] = bench
            phase('post_benchmark_compare')
            last = getattr(backend,'last_outputs',None)
            if last is None:
                last = backend.run()
                backend.synchronize()
            arrays,dtypes = backend.snapshot(last)
            result['post_benchmark_comparison'] = compare_outputs(arrays,expected,case['outputs'],case['compare'],actual_dtypes=dtypes,actual_metadata=backend.output_metadata(last) if hasattr(backend,'output_metadata') else None,input_metadata=result['inputs'])
            np.savez(workdir/'post_benchmark.npz', **{f'output_{i}':x for i,x in enumerate(arrays)})
            if not result['post_benchmark_comparison']['passed']:
                result.update(status='ACCURACY_FAILED',exception_type='ComparisonFailure',message='正式计时后输出比较失败，性能数据不可采信')
                return result
        result.update(status='PASS',message='独立参考及输出契约比较通过')
    except UnsupportedError as exc:
        caught_error = exc
        result.update(status='REFERENCE_FAILED' if stage=='reference' else 'UNSUPPORTED',message=str(exc),traceback=traceback.format_exc())
    except (EnvironmentError, ModuleNotFoundError) as exc:
        caught_error = exc
        result.update(status='REFERENCE_FAILED' if stage=='reference' else 'ENVIRONMENT_ERROR',message=str(exc),traceback=traceback.format_exc())
    except BuildError as exc:
        caught_error = exc
        result.update(status='REFERENCE_FAILED' if stage=='reference' else 'BUILD_FAILED',message=str(exc),traceback=traceback.format_exc())
    except Exception as exc:
        caught_error = exc
        status = {'environment':'ENVIRONMENT_ERROR','inputs':'CONFIG_ERROR','build':'BUILD_FAILED','reference':'REFERENCE_FAILED'}.get(stage,'EXECUTION_FAILED')
        result.update(status=status,message=str(exc),traceback=traceback.format_exc())
    finally:
        if caught_error is not None:
            result['exception_type']=type(caught_error).__name__
        if caught_error is not None and case['backend'] != 'cpu' and stage not in {'inputs', 'reference', 'compare'}:
            result['fatal_device_error'] = is_fatal_device_error(caught_error)
        result['phase_times_s'][stage] = time.monotonic()-phase_begin
        result['last_stage'] = stage
        if backend is not None:
            try:
                result['backend_metadata']=backend.metadata()
                if session is not None:
                    result['dynamic_execution']['artifact_evidence']=result['backend_metadata']
            except Exception as exc:
                result['metadata_error']=repr(exc)
        if backend is not None and session is None:
            try:
                result['memory_before_close'] = backend.memory()
                backend.close()
                result['memory_after_close'] = backend.memory()
            except Exception as exc:
                result['cleanup_error'] = repr(exc)
                if case['backend'] != 'cpu' and is_fatal_device_error(exc):
                    result['fatal_device_error'] = True
                if result['status']=='PASS':
                    result.update(status='EXECUTION_FAILED',message='清理阶段失败，见 cleanup_error')
        result['snapshot_hashes']={name:sha256(workdir/name) for name in
            ('inputs.npz','actual.npz','reference.npz','post_benchmark.npz',
             'profile_trace.json','profile_table.txt','repeat_failure.npz') if (workdir/name).is_file()}
        result['elapsed_s'] = time.monotonic()-begin
        from .expectations import apply_expectation
        apply_expectation(result,case)
    return result


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    if 'requests' in request:
        session={}
        results=[]
        try:
            for child in request['requests']:
                result=execute(child,session=session)
                write_json(Path(child['workdir'])/'result.json',result)
                results.append(result)
                if result.get('fatal_device_error') or result['status'] not in {'PASS','EXPECTED_FAILURE'}:
                    break
        finally:
            if session.get('backend') is not None:
                try: session['backend'].close()
                except Exception as exc:
                    for child,result in zip(request['requests'],results):
                        result.update(status='EXECUTION_FAILED',cleanup_error=repr(exc),message='动态组清理失败',fatal_device_error=is_fatal_device_error(exc))
                        write_json(Path(child['workdir'])/'result.json',result)
        write_json(Path(request['requests'][0]['workdir'])/'group.complete.json',{'completed':len(results),'total':len(request['requests'])})
        return 0 if len(results)==len(request['requests']) and all(r['status'] in {'PASS','EXPECTED_FAILURE'} for r in results) else 1
    result = execute(request)
    write_json(Path(request['workdir'])/'result.json',result)
    print(json.dumps({'name':result['name'],'status':result['status'],'message':result.get('message','')},ensure_ascii=False),flush=True)
    return 0 if result['status'] in {'PASS','EXPECTED_FAILURE'} else 1


if __name__=='__main__':
    raise SystemExit(main())
