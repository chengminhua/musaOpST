#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Validate evidence integrity separately from complete backend coverage.

Artifact hashes have explicit scopes: native/C++ binary SHA, Triton compiled
kernel hash, TileLang generated device-source SHA. Eager dispatch is never
certified as compile-once, even when one worker handles all shapes.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from musaopst.binding import tensor_specs
from musaopst.reporting import validate_case_result
from musaopst.util import sha256, write_json


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def check_snapshots(case, folder, errors):
    hashes = case.get('snapshot_hashes', {})
    required = set()
    if case.get('status') == 'PASS' or case.get('comparison'):
        required.update(('inputs.npz', 'actual.npz', 'reference.npz'))
    if case.get('benchmark'):
        required.add('post_benchmark.npz')
    if case.get('profile'):
        required.update(('profile_trace.json', 'profile_table.txt'))
    missing = required - hashes.keys()
    if missing:
        errors.append(case['name'] + ' missing mandatory snapshot hashes: ' + ', '.join(sorted(missing)))
    for name, expected in hashes.items():
        path = (folder / name).resolve()
        if path.parent != folder.resolve() or not _digest(expected) or not path.is_file() or sha256(path) != expected:
            errors.append(case['name'] + ' snapshot hash mismatch ' + name)
    if case.get('input_snapshot_sha256') and hashes.get('inputs.npz') != case['input_snapshot_sha256']:
        errors.append(case['name'] + ' conflicting input snapshot digest')


def check_timing(case, errors):
    config = case.get('config', {}).get('benchmark', {})
    benchmark = case.get('benchmark')
    if config.get('enabled') and case.get('status') == 'PASS' and not benchmark:
        errors.append(case['name'] + ' enabled benchmark missing')
        return
    if not benchmark:
        return
    name = case['name']
    policy = benchmark.get('policy', {})
    samples = benchmark.get('raw_samples_ms', [])
    mode = config.get('mode', 'batch_event')
    count = config.get('samples')
    iterations = config.get('iterations') if mode == 'batch_event' else 1
    if not isinstance(samples, list) or len(samples) != count or not samples or not all(_finite(x) for x in samples):
        errors.append(name + ' invalid benchmark samples')
        return
    expected = {'mode': mode, 'timer': 'host_wall_sync' if mode == 'wall_time' else 'musa_event',
                'iterations_per_sample': iterations, 'samples': count, 'warmup': config.get('warmup'),
                'includes_host_device_copy': False, 'includes_reference': False,
                'includes_compile_jit': False, 'includes_host_synchronization': mode == 'wall_time',
                'sampling': 'batch mean per invocation' if mode == 'batch_event' else 'single invocation'}
    for key, value in expected.items():
        if policy.get(key) != value:
            errors.append(name + ' timing policy mismatch ' + key)
    if 'includes_output_allocation' not in policy:
        errors.append(name + ' output allocation timing scope missing')
    meaning = policy.get('quantile_meaning', '')
    if (mode == 'batch_event' and ('batch means' not in meaning or 'NOT individual-call' not in meaning)) or (mode != 'batch_event' and 'individually measured' not in meaning):
        errors.append(name + ' quantile scope contradicts sampling method')
    requested_stream = config.get('stream', 'current')
    legacy_native = case.get('backend') == 'native' and case.get('backend_metadata', {}).get('protocol') != 'musaopst_native_v2'
    if not (legacy_native and requested_stream == 'current' and policy.get('stream') == 'native default stream (0)'):
        if policy.get('stream') != requested_stream or policy.get('actual_stream', {}).get('policy') != requested_stream:
            errors.append(name + ' actual stream policy not verified')
    if config.get('cache', 'warm') == 'interference':
        if policy.get('cache_bytes') != config.get('cache_bytes') or 'NOT guaranteed' not in policy.get('cache', ''):
            errors.append(name + ' interference cache policy invalid or overclaimed')
    if policy.get('input_mutation_detected'):
        if not config.get('allow_inplace') or mode == 'batch_event' or 'before each sample' not in policy.get('input_reset', ''):
            errors.append(name + ' invalid input mutation timing policy')
    raw = benchmark.get('raw_batch_elapsed_ms')
    if isinstance(raw, list):
        if len(raw) != count or not all(_finite(x) for x in raw) or not np.allclose(np.asarray(raw) / iterations, samples, rtol=1e-10, atol=1e-12):
            errors.append(name + ' batch elapsed times inconsistent with normalized samples')
    elif not legacy_native:
        errors.append(name + ' raw batch/event elapsed samples missing')
    statistics = benchmark.get('statistics', {})
    recomputed = {'median_ms': float(np.median(samples)), 'p90_ms': float(np.percentile(samples, 90)),
                  'min_ms': min(samples), 'max_ms': max(samples)}
    for key, value in recomputed.items():
        for record in (benchmark, statistics):
            observed = record.get(key)
            if not _finite(observed) or not math.isclose(observed, value, rel_tol=1e-10, abs_tol=1e-12):
                errors.append(name + ' incorrect timing statistic ' + key)
    if statistics.get('n') != count or statistics.get('percentile_method') != 'numpy linear':
        errors.append(name + ' percentile method/sample count missing')


def check_profile(case, folder, errors):
    config = case.get('config', {}).get('profile', {})
    profile = case.get('profile')
    if config.get('enabled') and case.get('status') == 'PASS' and not profile:
        errors.append(case['name'] + ' requested GPU profile missing')
    if not profile:
        return
    trace = folder / Path(profile.get('trace_path', '')).name
    table = folder / Path(profile.get('table_path', '')).name
    if not trace.is_file() or not table.is_file():
        errors.append(case['name'] + ' missing profile artifacts')
        return
    events = json.loads(trace.read_text()).get('traceEvents', [])
    kernels = [e for e in events if str(e.get('cat', '')).lower() in {'kernel', 'gpu_kernel'}]
    durations = [e.get('dur') for e in kernels]
    if (not kernels or len(kernels) != profile.get('gpu_kernel_count') or not all(_finite(x) for x in durations)
            or durations != profile.get('gpu_durations_us')
            or sorted({str(e.get('name', 'unknown')) for e in kernels}) != profile.get('kernel_names')):
        errors.append(case['name'] + ' GPU trace events disagree with profile metadata')
    if profile.get('separate_from_benchmark') is not True or profile.get('iterations') != config.get('iterations', 3):
        errors.append(case['name'] + ' profile isolation/iteration metadata invalid')


def check_dynamic(cases, errors):
    groups = defaultdict(list)
    for case in cases:
        group = case.get('config', {}).get('compile_group')
        if group:
            groups[(case['backend'], group)].append(case)
    evidence = []
    for (backend, group), members in groups.items():
        errors_before = len(errors)
        row = {'backend': backend, 'group': group, 'statuses': [c['status'] for c in members],
               'names': [c['name'] for c in members], 'same_artifact_verified': False}
        evidence.append(row)
        if not all(c['status'] == 'PASS' for c in members):
            row['classification'] = 'incomplete_or_unsupported'
            continue
        pids = {c.get('worker_pid') for c in members}
        shapes = {json.dumps([s['shape'] for s in tensor_specs(c['config']['inputs'])]) for c in members}
        if len(pids) != 1 or None in pids or len(shapes) < 2:
            errors.append(group + ' dynamic group lacks same-worker/multiple-shape evidence')
        ids = []
        kinds = set()
        for index, case in enumerate(members):
            dynamic = case.get('dynamic_execution', {})
            meta = case.get('backend_metadata', {})
            launch = meta.get('launch', {})
            kinds.add(meta.get('runtime_shape_kind'))
            if (dynamic.get('prepare_calls') != 1 or dynamic.get('same_worker') is not True or dynamic.get('group') != group
                    or meta.get('build_count') != 1 or meta.get('runtime_shape_rebindings', 0) != index):
                errors.append(case['name'] + ' dynamic preparation/rebinding evidence invalid')
            recorded = dynamic.get('artifact_evidence', {})
            if recorded.get('launch', recorded) != launch:
                errors.append(case['name'] + ' stale dynamic artifact evidence')
            if backend == 'torch_musa':
                continue
            if backend == 'triton':
                found = launch.get('compiled_artifact_ids')
                if not isinstance(found, list) or len(found) != 1 or not _digest(found[0]) or launch.get('observed_compiled_artifacts') != 1:
                    errors.append(case['name'] + ' missing single observed Triton artifact; re-specialization is not compile-once')
                else:
                    ids.extend(found)
            else:
                artifact = launch.get('artifact_id')
                if not _digest(artifact):
                    errors.append(case['name'] + ' missing actual dynamic artifact digest')
                else:
                    ids.append(artifact)
                if backend in ('native', 'musa_cpp'):
                    build = meta.get('build', launch.get('build', {}))
                    if build.get('artifact_sha256') != artifact:
                        errors.append(case['name'] + ' binary digest differs from dynamic artifact')
        row.update(worker_pids=sorted(pids, key=str), distinct_shapes=len(shapes), artifact_ids=sorted(set(ids)))
        if backend == 'torch_musa':
            row['classification'] = 'eager_dispatch_only; no compiled-artifact claim'
            if kinds != {'eager_framework_dispatch'}:
                errors.append(group + ' torch eager execution mislabeled as compilation')
        else:
            expected_kind = {'native': 'runtime_kernel_arguments_same_binary', 'musa_cpp': 'compiled_kernel_runtime_arguments',
                             'tilelang': 'compiled_symbolic_N'}
            if backend in expected_kind and kinds != {expected_kind[backend]}:
                errors.append(group + ' dynamic runtime-shape kind unsupported or per-shape specialization')
            row['classification'] = {'native': 'same_binary_fresh_native_process_per_call', 'musa_cpp': 'same_extension_binary',
                                     'triton': 'same_compiled_kernel', 'tilelang': 'same_symbolic_kernel_source'}[backend]
            row['identity_scope'] = ('generated_device_source_sha_and_retained_jit_object_in_same_worker; binary identity not measured'
                                     if backend == 'tilelang' else 'compiled_kernel_hash' if backend == 'triton' else 'binary_sha256')
            row['same_artifact_verified'] = len(ids) == len(members) and len(set(ids)) == 1 and len(errors) == errors_before
            if not row['same_artifact_verified']:
                errors.append(group + ' artifact changed across shapes or evidence missing')
    return evidence


def check_constants(cases_with_folders, errors):
    groups = defaultdict(list)
    for case, folder in cases_with_folders:
        if 'compiled_constant' not in case.get('config', {}).get('tags', []):
            continue
        groups[case['backend']].append((case, folder))
    evidence = []
    for backend, members in groups.items():
        row = {'backend': backend, 'statuses': dict(Counter(c['status'] for c, _ in members)), 'constants': []}
        evidence.append(row)
        fps = {}
        quantized = set()
        for case, folder in members:
            if case['status'] != 'PASS':
                continue
            launch = case.get('backend_metadata', {}).get('launch', {})
            records = launch.get('compiled_constants', [])
            if len(records) != 1:
                errors.append(case['name'] + ' compiled constant record missing')
                continue
            record = records[0]
            specs = tensor_specs(case['config']['inputs'])
            with np.load(folder / 'inputs.npz', allow_pickle=False) as inputs:
                actual = float(inputs['input_1'])
            value = record.get('quantized_value')
            if (record.get('input_index') != 1 or not specs[1].get('constant') or record.get('dtype') != specs[1]['dtype']
                    or record.get('source') != 'actual_quantized_device_input' or record.get('runtime_kernel_argument') is not False
                    or value != actual or record.get('value_hex') != actual.hex()):
                errors.append(case['name'] + ' embedded value does not match actual quantized scalar snapshot')
            digest = hashlib.sha256(json.dumps({k: v for k, v in record.items() if k != 'content_sha256'}, sort_keys=True).encode()).hexdigest()
            if record.get('content_sha256') != digest:
                errors.append(case['name'] + ' constant content digest invalid')
            artifact = launch.get('artifact_id')
            if not _digest(artifact) or launch.get('kernel_runtime_arguments') != ['vector_pointer', 'output_pointer']:
                errors.append(case['name'] + ' constant kernel artifact/runtime argument evidence invalid')
            if backend == 'triton':
                if (launch.get('artifact_kind') != 'triton_compiled_kernel_hash' or launch.get('constant_mechanism') != 'tl.constexpr'
                        or launch.get('compiled_artifact_ids') != [artifact]):
                    errors.append(case['name'] + ' Triton constexpr evidence invalid')
            elif backend == 'tilelang':
                source = folder / Path(launch.get('generated_source', '')).name
                if (launch.get('artifact_kind') != 'generated_device_source_sha256' or launch.get('constant_mechanism') != 'TileLang TIR embedded literal'
                        or not source.is_file() or sha256(source) != artifact):
                    errors.append(case['name'] + ' TileLang embedded source digest invalid')
            else:
                errors.append(case['name'] + ' unsupported backend claimed compiled constant success')
            row['constants'].append({'dtype': record.get('dtype'), 'value': value, 'artifact_id': artifact})
            if record.get('dtype') == 'float32':
                fps[value] = artifact
            if record.get('dtype') in ('float16', 'bfloat16'):
                literal = specs[1]['distribution']['values']
                wanted = 1.099609375 if record['dtype'] == 'float16' else 1.1015625
                if literal != 1.1 or actual != wanted or actual == literal:
                    errors.append(case['name'] + ' low-precision quantization evidence invalid')
                quantized.add(record['dtype'])
        if backend in ('triton', 'tilelang') and any(c['status'] == 'PASS' for c, _ in members):
            row['distinct_float32_artifacts'] = len(set(fps.values()))
            if set(fps) != {1.0, 2.0, 3.0} or len(set(fps.values())) != 3 or quantized != {'float16', 'bfloat16'}:
                errors.append(backend + ' constant acceptance lacks three specializations and both low-precision inputs')
    return evidence


def verify(root):
    root = Path(root)
    meta = json.loads((root / 'acceptance.json').read_text())
    errors, rows, all_cases, cases_with_folders = [], [], [], []
    digest = meta['source']['content_sha256']
    if not meta.get('complete') or meta.get('source_changed'):
        errors.append('orchestration incomplete or source changed')
    if not meta.get('stages'):
        errors.append('no acceptance stages')
    recorded_stages = [stage.get('stage') for stage in meta.get('stages', [])]
    if len(set(recorded_stages)) != len(recorded_stages) or recorded_stages != meta.get('selected_stages', recorded_stages):
        errors.append('selected acceptance stages missing, duplicated, or out of order')
    for stage in meta['stages']:
        path = root / stage['stage'] / ('smoke/results.json' if stage['stage'] == 'doctor' else 'results.json')
        if not path.is_file():
            errors.append(str(path) + ' missing')
            continue
        report = json.loads(path.read_text())
        if report.get('source', {}).get('content_sha256') != digest:
            errors.append(str(path) + ' source mismatch')
        if not report.get('run_complete') or not report.get('cases'):
            errors.append(str(path) + ' incomplete or empty')
        for case in report['cases']:
            all_cases.append(case)
            rows.append({'stage': stage['stage'], 'name': case['name'], 'backend': case['backend'], 'status': case['status'],
                         'required': case.get('required', True), 'message': case.get('message'), 'evidence': str(path.relative_to(root))})
            folder = path.parent / Path(case.get('case_result', 'case-0000/result.json')).parent
            cases_with_folders.append((case, folder))
            try:
                validate_case_result(case)
                if case.get('fatal_device_error'):
                    errors.append(case['name'] + ' fatal device error')
                if case.get('provenance') != 'fresh':
                    errors.append(case['name'] + ' unexpectedly reused during frozen acceptance')
                check_snapshots(case, folder, errors)
                if case['status'] == 'PASS':
                    if case.get('mode') != 'MUSA_DEVICE':
                        errors.append(case['name'] + ' not GPU mode')
                    check_timing(case, errors)
                    check_profile(case, folder, errors)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                errors.append(case['name'] + ': malformed evidence: ' + str(exc))
    try:
        dynamic = check_dynamic(all_cases, errors)
        constants = check_constants(cases_with_folders, errors)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        errors.append('dynamic/constant evidence malformed: ' + str(exc))
        dynamic, constants = [], []
    return {'evidence_valid': not errors, 'errors': errors, 'source_digest': digest,
            'counts': dict(Counter(r['status'] for r in rows)), 'rows': rows,
            'dynamic_groups': dynamic, 'compiled_constant_specializations': constants,
            'all_required_gpu_cases_passed': bool(rows) and all(r['status'] in {'PASS', 'EXPECTED_FAILURE'} for r in rows if r['stage'] != 'negative' and r['required']),
            'meaning': 'Evidence integrity is not complete backend coverage; inspect all failed/unsupported cells.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = verify(args.directory)
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False))
    return 0 if result['evidence_valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
