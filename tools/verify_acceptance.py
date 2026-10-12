#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Read-only audit of one final GPU suite and its separate negative/offline evidence.

This program never runs GPU code and never edits supplied evidence. Missing top-level
inputs produce exit 2 without an audit artifact; complete but invalid evidence produces
an explicit failed audit (exit 1). Only one final suite can supply positive GPU coverage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from musaopst.compare import compare_outputs
from musaopst.config import load_config
from musaopst.data import make_inputs, tensor_metadata
from musaopst.runner import source_manifest
from musaopst.util import write_json


class Audit:
    def __init__(self):
        self.failures = []
        self.checks = 0
        self.files = {}
        self.coverage = {}

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            self.failures.append(message)
        return bool(condition)

    def file(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f'missing evidence file: {path}')
        self.files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def json(self, path):
        return json.loads(self.file(path).read_text(encoding='utf-8'))

    def npz(self, path, prefix, count):
        with np.load(self.file(path), allow_pickle=False) as archive:
            wanted = [f'{prefix}_{index}' for index in range(count)]
            if set(archive.files) != set(wanted):
                raise ValueError(f'{path}: archive keys disagree with output/input count')
            return [archive[key].copy() for key in wanted]


def evidence_path(value, filename):
    path = Path(value).resolve()
    return path / filename if path.is_dir() else path


def check_source(audit, result, label):
    current = source_manifest()
    stored = result.get('source', {})
    audit.check(stored.get('files') == current['files'], f'{label}: package file hashes differ from current source')
    audit.check(stored.get('content_sha256') == current['content_sha256'], f'{label}: package content digest differs')
    audit.check(stored.get('package_version') == current['package_version'], f'{label}: package version differs')
    if isinstance(stored.get('files'), dict):
        digest = hashlib.sha256(json.dumps(stored['files'], sort_keys=True).encode()).hexdigest()
        audit.check(digest == stored.get('content_sha256'), f'{label}: manifest digest is internally inconsistent')


def check_suite_header(audit, path, expected, expected_pass):
    result = audit.json(path)
    label = str(path)
    audit.check(type(result.get('schema_version')) is int and result['schema_version'] == 1, f'{label}: wrong schema version')
    audit.check(result.get('tool') == 'musaopst', f'{label}: tool identifier missing')
    audit.check(result.get('run_complete') is True, f'{label}: run has not completed')
    audit.check(isinstance(result.get('finished_utc'), str), f'{label}: completion timestamp missing')
    cases = result.get('cases', [])
    names = [case.get('name') for case in cases]
    audit.check(names == [case['name'] for case in expected], f'{label}: case coverage/order does not equal current example configuration')
    planned = result.get('planned_cases', [])
    audit.check([case.get('name') for case in planned] == names, f'{label}: planned cases disagree with completed cases')
    summary = result.get('summary', {})
    audit.check(summary.get('passed') is expected_pass, f'{label}: unexpected overall pass claim')
    audit.check(summary.get('exit_code') == (0 if expected_pass else 1), f'{label}: wrong recorded suite exit code')
    audit.check(summary.get('pending') == 0 and summary.get('run_complete') is True, f'{label}: pending/uncompleted summary')
    audit.check(summary.get('total') == len(cases), f'{label}: incorrect summary total')
    audit.check(summary.get('counts') == dict(Counter(case.get('status') for case in cases)), f'{label}: incorrect status counts')
    audit.check(isinstance(result.get('reproduce'), str) and result['reproduce'] != 'unknown', f'{label}: reproduction command missing')
    check_source(audit, result, label)
    audit.file(path.parent / 'summary.md')
    junit = ET.parse(audit.file(path.parent / 'junit.xml')).getroot()
    audit.check(int(junit.get('tests', -1)) == len(expected), f'{label}: JUnit count differs')
    audit.check(int(junit.get('failures', -1)) == (0 if expected_pass else len(expected)), f'{label}: JUnit failure count differs')
    audit.check(int(junit.get('errors', -1)) == 0 and int(junit.get('skipped', -1)) == 0, f'{label}: unexpected JUnit error/skip')
    resolved = audit.json(path.parent / 'config.resolved.json')
    normalized = [{key: value for key, value in case.items() if key != 'parameters_selected'} for case in expected]
    audit.check(resolved == {'schema_version': 1, 'cases': normalized}, f'{label}: resolved configuration differs from expected invocation')
    original = audit.json(path.parent / 'config.original.json')
    audit.check(result.get('original_config') == original, f'{label}: original config snapshot differs')
    return result


def check_case_files(audit, root, case, expected, index, outputs=True):
    label = case['name']
    case_dir = root / f'case-{index:04d}'
    audit.check(case.get('mode') == 'MUSA_DEVICE', f'{label}: CPU/unknown execution mode')
    audit.check(case.get('backend') == expected['backend'], f'{label}: backend differs')
    audit.check(case.get('required') is True, f'{label}: required coverage was weakened')
    audit.check(case.get('config') == expected, f'{label}: effective configuration differs')
    audit.check(case.get('seed') == expected['seed'], f'{label}: seed metadata differs')
    stored = audit.json(case_dir / 'result.json')
    audit.check(stored == case, f'{label}: individual and suite records differ')
    request = audit.json(case_dir / 'request.json')
    audit.check(request.get('case') == expected, f'{label}: worker request differs from expected configuration')
    log = audit.file(case_dir / 'worker.log')
    audit.check(log.stat().st_size > 0, f'{label}: worker log empty')
    progress = audit.json(case_dir / 'progress.json')
    audit.check(isinstance(progress.get('stage'), str), f'{label}: progress record missing stage')
    inputs = audit.npz(case_dir / 'inputs.npz', 'input', len(expected['inputs']))
    generated = make_inputs(expected)
    for number, (snapshot, original) in enumerate(zip(inputs, generated)):
        audit.check(snapshot.dtype == original.dtype and np.array_equal(snapshot, original, equal_nan=True),
                    f'{label}: saved input[{number}] does not reproduce from seed and quantization')
    if outputs:
        actual = audit.npz(case_dir / 'actual.npz', 'output', len(expected['outputs']))
        reference = audit.npz(case_dir / 'reference.npz', 'output', len(expected['outputs']))
        actual_meta = case.get('actual_metadata', [])
        audit.check(len(actual_meta) == len(actual), f'{label}: actual dtype metadata count differs')
        checked = compare_outputs(actual, reference, expected['outputs'], expected['compare'],
                                  actual_dtypes=[value.get('dtype') for value in actual_meta])
        audit.check(checked['passed'] is (case['status'] == 'PASS'), f'{label}: saved arrays contradict recorded numerical outcome')
        recorded = case.get('comparison', {})
        audit.check(recorded.get('contract_errors') == checked['contract_errors'], f'{label}: contract errors contradict saved arrays')
        audit.check(recorded.get('policy') == expected['compare'], f'{label}: comparison policy differs')
        audit.check([value.get('mismatch_count') for value in recorded.get('outputs', [])] ==
                    [value.get('mismatch_count') for value in checked['outputs']], f'{label}: mismatch counts contradict saved arrays')
        for field in ('implementation_source', 'reference_source'):
            entry = expected['entry' if field == 'implementation_source' else 'reference']
            module = entry.split(':')[0]
            local_source = ROOT / (module.replace('.', '/') + '.py')
            digest = hashlib.sha256(audit.file(local_source).read_bytes()).hexdigest()
            audit.check(case.get(field, {}).get('sha256') == digest, f'{label}: {field} hash differs')
        return case_dir, generated, reference
    return case_dir, generated, None


def check_inputs(audit, case, expected, generated):
    label = case['name']
    host = case.get('inputs', [])
    backend = case.get('backend_metadata', {})
    device = backend.get('inputs', [])
    audit.check(backend.get('execution_device') == 'musa', f'{label}: missing MUSA execution metadata')
    audit.check(len(host) == len(generated) == len(device), f'{label}: input metadata counts differ')
    actual_noncontiguous = 0
    for i, (spec, array) in enumerate(zip(expected['inputs'], generated)):
        wanted = tensor_metadata(array, spec['dtype'])
        if i >= len(host) or i >= len(device):
            continue
        for key in ('shape', 'dtype', 'strides', 'contiguous'):
            audit.check(host[i].get(key) == wanted[key], f'{label}: host input[{i}] {key} differs')
            audit.check(device[i].get(key) == wanted[key], f'{label}: actual device input[{i}] {key} changed')
        if expected['backend'] != 'native':
            audit.check(str(device[i].get('device', '')).startswith('musa:'), f'{label}: input[{i}] is not on MUSA')
        if not wanted['contiguous']:
            actual_noncontiguous += 1
    empty = any(array.size == 0 for array in generated)
    audit.check(backend.get('empty_workload') is empty, f'{label}: empty workload classification missing/wrong')
    category = 'empty_input_contract_only' if empty else 'nonempty_input_workload'
    audit.check(backend.get('execution_classification') == category, f'{label}: numeric coverage classification missing/wrong')
    if expected['backend'] == 'native':
        runtime = backend.get('runtime', {})
        audit.check(runtime.get('empty_no_kernel_launch') is empty, f'{label}: native zero-launch evidence missing')
        audit.check(runtime.get('device_name') not in (None, '', 'unknown'), f'{label}: native device unknown')
    else:
        audit.check(backend.get('device_name') not in (None, '', 'unknown'), f'{label}: device unknown')
        if empty and expected['backend'] in ('triton', 'tilelang', 'musa_cpp'):
            audit.check(backend.get('launch', {}).get('empty_input') == 'no kernel launch', f'{label}: explicit empty no-op metadata missing')
    return empty, actual_noncontiguous


def check_benchmark(audit, case, expected):
    label = case['name']
    bench = case.get('benchmark', {})
    policy = bench.get('policy', {})
    config = expected['benchmark']
    required_policy = {'timer': 'musa_event', 'sampling': 'batch mean per invocation',
                       'warmup': config['warmup'], 'samples': config['samples'],
                       'iterations_per_sample': config['iterations'],
                       'includes_output_allocation': False, 'includes_host_device_copy': False,
                       'includes_reference': False, 'includes_compile_jit': False}
    for key, value in required_policy.items():
        audit.check(policy.get(key) == value and type(policy.get(key)) is type(value), f'{label}: timing policy {key} differs')
    audit.check(policy.get('cache') == 'warm; same input/output buffers; no explicit cache flush', f'{label}: cache policy differs')
    audit.check(policy.get('quantile_meaning') == 'quantiles of batch means, NOT individual-call tail latency', f'{label}: percentile meaning differs')
    audit.check(isinstance(policy.get('stream'), str) and bool(policy['stream']), f'{label}: stream not documented')
    if expected['backend'] == 'native':
        audit.check(policy.get('excludes_process_startup_and_file_io') is True, f'{label}: native timing scope not documented')
    else:
        audit.check(isinstance(policy.get('actual_stream'), dict), f'{label}: actual Event stream missing')
    values = np.asarray(bench.get('raw_samples_ms', []), dtype=np.float64)
    batches = np.asarray(bench.get('raw_batch_elapsed_ms', []), dtype=np.float64)
    if not audit.check(values.shape == (config['samples'],) and batches.shape == values.shape, f'{label}: Event sample count/shape differs'):
        return
    if not audit.check(bool(np.isfinite(values).all() and np.isfinite(batches).all() and (values >= 0).all() and (batches >= 0).all()),
                       f'{label}: Event samples nonfinite or negative'):
        return
    # Native C++ divides float Event milliseconds by an integer using float arithmetic.
    audit.check(bool(np.allclose(values, batches / config['iterations'], rtol=1e-6, atol=1e-9)), f'{label}: batch means do not match raw Event batches')
    stats = bench.get('statistics', {})
    for key, value in {'median_ms': np.median(values), 'p90_ms': np.percentile(values, 90),
                       'min_ms': values.min(), 'max_ms': values.max()}.items():
        audit.check(isinstance(stats.get(key), (int, float)) and np.isclose(stats[key], value, rtol=1e-12, atol=1e-15), f'{label}: statistic {key} differs')
    audit.check(stats.get('n') == config['samples'] and stats.get('percentile_method') == 'numpy linear', f'{label}: statistic sample definition differs')


def audit_gpu(audit, path):
    config_file = audit.file(ROOT / 'examples/gpu_suite.json')
    expected = load_config(config_file)
    audit.check(len(expected) == 57, 'current GPU example no longer defines the intended 57-case acceptance matrix')
    for case in expected:
        case['benchmark']['enabled'] = True
        case.setdefault('parameters_selected', {})
    result = check_suite_header(audit, path, expected, True)
    audit.check(result.get('config_sha256') == hashlib.sha256(config_file.read_bytes()).hexdigest(), 'GPU suite: original configuration hash differs')
    empty_count, nonempty_count, strided_count = 0, 0, 0
    backend_counts = Counter()
    for index, (case, wanted) in enumerate(zip(result.get('cases', []), expected)):
        before = len(audit.failures)
        try:
            audit.check(case.get('status') == 'PASS', f"{wanted['name']}: required positive case did not pass")
            directory, generated, reference = check_case_files(audit, path.parent, case, wanted, index)
            empty, strided = check_inputs(audit, case, wanted, generated)
            empty_count += int(empty)
            nonempty_count += int(not empty)
            strided_count += int(strided > 0)
            backend_counts[wanted['backend']] += 1
            check_benchmark(audit, case, wanted)
            repeat = case.get('repeat', {})
            audit.check(repeat.get('requested_count') == repeat.get('completed') == 3, f"{wanted['name']}: repeat did not complete exactly three iterations")
            audit.check(repeat.get('requested_duration_s') == 0 and repeat.get('anomalies') == [], f"{wanted['name']}: repeat policy/anomalies differ")
            audit.check(isinstance(repeat.get('elapsed_s'), (int, float)) and repeat['elapsed_s'] >= 0, f"{wanted['name']}: repeat duration missing")
            audit.check('memory_before' in repeat and 'memory_after' in repeat, f"{wanted['name']}: memory accounting missing")
            post = audit.npz(directory / 'post_benchmark.npz', 'output', len(wanted['outputs']))
            checked = compare_outputs(post, reference, wanted['outputs'], wanted['compare'], [spec['dtype'] for spec in wanted['outputs']])
            audit.check(checked['passed'] and case.get('post_benchmark_comparison', {}).get('passed') is True,
                        f"{wanted['name']}: post-benchmark comparison did not pass")
        except (KeyError, TypeError, ValueError, OSError, IndexError) as exc:
            audit.check(False, f"{wanted['name']}: unreadable or invalid case evidence: {exc}")
        audit.coverage[wanted['name']] = {'verified': len(audit.failures) == before, 'backend': wanted['backend']}
    audit.coverage['gpu_summary'] = {'expected_cases': len(expected), 'recorded_cases': len(result.get('cases', [])),
                                   'backend_counts': dict(backend_counts), 'empty_contract_only_cases': empty_count,
                                   'nonempty_workload_cases': nonempty_count, 'actual_noncontiguous_cases': strided_count}
    return result


def audit_negative(audit, path):
    expected = load_config(audit.file(ROOT / 'examples/gpu_negative.json'))
    for case in expected:
        case.setdefault('parameters_selected', {})
    result = check_suite_header(audit, path, expected, False)
    for index, (case, wanted) in enumerate(zip(result.get('cases', []), expected)):
        check_case_files(audit, path.parent, case, wanted, index)
        audit.check(case.get('status') == 'ACCURACY_FAILED', 'negative: did not report ACCURACY_FAILED')
        audit.check(case.get('comparison', {}).get('passed') is False, 'negative: numerical comparison did not fail')
        audit.check(sum(value.get('mismatch_count', 0) for value in case.get('comparison', {}).get('outputs', [])) > 0,
                    'negative: numerical mismatches not recorded')
        audit.check('benchmark' not in case and 'post_benchmark_comparison' not in case, 'negative: entered benchmark after failed correctness')
        audit.check(case.get('backend_metadata', {}).get('execution_device') == 'musa', 'negative: not real MUSA execution metadata')
        audit.check(case.get('backend_metadata', {}).get('launch', {}).get('safe_corruption') is True, 'negative: safe numerical fault injection marker missing')
    audit.coverage['negative'] = {'expected_status': 'ACCURACY_FAILED', 'cases': len(result.get('cases', [])),
                                  'meaning': 'tester detected safe corruption; tested operator itself did not pass'}


def audit_offline(audit, path):
    result = audit.json(path)
    audit.check(result.get('mode') == 'CPU_PLATFORM_UNIT_TESTS_ONLY', 'offline: test mode missing')
    audit.check(result.get('passed') is True and result.get('tests_run', 0) > 0, 'offline: unit tests unsuccessful/empty')
    cases = result.get('cases', [])
    audit.check(len(cases) == result.get('tests_run') and all(case.get('status') == 'PASS' for case in cases), 'offline: not every recorded test passed')
    hashes = result.get('source_sha256', {})
    for folder in ('musaopst', 'tests', 'tools'):
        for source in sorted((ROOT / folder).rglob('*.py')):
            relative = str(source.relative_to(ROOT))
            audit.check(hashes.get(relative) == hashlib.sha256(audit.file(source).read_bytes()).hexdigest(), f'offline: source evidence stale/missing for {relative}')
    audit.file(path.parent / 'unittest.log')
    junit = ET.parse(audit.file(path.parent / 'junit.xml')).getroot()
    audit.check(int(junit.get('tests', 0)) == len(cases), 'offline: JUnit test count differs')
    audit.check(all(int(junit.get(key, -1)) == 0 for key in ('failures', 'errors', 'skipped')), 'offline: JUnit has failures/errors/skips')
    audit.coverage['offline'] = {'tests_run': result.get('tests_run'), 'mode': result.get('mode')}


def audit_unsupported(audit, path):
    expected = load_config(audit.file(ROOT / 'examples/unsupported_layout.json'))
    for case in expected:
        case.setdefault('parameters_selected', {})
    result = check_suite_header(audit, path, expected, False)
    for index, (case, wanted) in enumerate(zip(result.get('cases', []), expected)):
        check_case_files(audit, path.parent, case, wanted, index, outputs=False)
        audit.check(case.get('status') == 'UNSUPPORTED', f"{wanted['name']}: unsupported layout not explicitly rejected")
        audit.check('benchmark' not in case and 'comparison' not in case, f"{wanted['name']}: unsupported layout was executed/compared")
    audit.coverage['unsupported'] = {'cases': len(result.get('cases', [])), 'expected_status': 'UNSUPPORTED'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True, help='final GPU suite directory or results.json')
    parser.add_argument('--negative', required=True, help='safe GPU corruption suite directory or results.json')
    parser.add_argument('--offline', required=True, help='offline unittest directory or unittest.json')
    parser.add_argument('--unsupported', help='explicit unsupported-layout directory or results.json')
    parser.add_argument('--output', type=Path, required=True, help='new audit JSON file; also writes a .sha256.json sidecar')
    args = parser.parse_args(argv)
    inputs = {'gpu': evidence_path(args.gpu, 'results.json'), 'negative': evidence_path(args.negative, 'results.json'),
              'offline': evidence_path(args.offline, 'unittest.json')}
    if args.unsupported:
        inputs['unsupported'] = evidence_path(args.unsupported, 'results.json')
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        parser.error('evidence not ready; no audit written: ' + ', '.join(missing))
    output = args.output.resolve()
    sidecar = output.with_suffix('.sha256.json')
    if output == sidecar or output.exists() or sidecar.exists():
        parser.error('output or hash sidecar already exists (or conflicts); refusing to overwrite')
    audit = Audit()
    for label, function in (('gpu', audit_gpu), ('negative', audit_negative), ('offline', audit_offline), ('unsupported', audit_unsupported)):
        if label not in inputs:
            continue
        try:
            function(audit, inputs[label])
        except (ValueError, TypeError, KeyError, OSError, IndexError, ET.ParseError) as exc:
            audit.check(False, f'{label}: evidence audit could not complete: {exc}')
    record = {'schema_version': 1, 'tool': 'musaopst acceptance evidence verifier',
              'checked_utc': datetime.now(timezone.utc).isoformat(), 'passed': not audit.failures,
              'checks': audit.checks, 'failures': audit.failures, 'coverage': audit.coverage,
              'inputs': {label: str(path) for label, path in inputs.items()},
              'current_package_source': source_manifest(), 'evidence_hashes_file': sidecar.name,
              'scope': 'One complete final positive GPU run only; negative, unsupported, and CPU evidence remain separate. Empty tensors are contract-only coverage. Three repetitions are not long-term stability certification.'}
    write_json(output, record)
    write_json(sidecar, {'algorithm': 'sha256', 'audit_file': str(output),
                        'audit_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'files': audit.files})
    print(json.dumps({'passed': record['passed'], 'checks': audit.checks, 'failures': audit.failures,
                      'audit': str(output), 'hashes': str(sidecar)}, ensure_ascii=False, indent=2))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
