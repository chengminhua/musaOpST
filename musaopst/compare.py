# SPDX-License-Identifier: MulanPSL-2.0
"""Contract-first, per-output comparisons with explicit numerical definitions."""
from __future__ import annotations

import copy
import math
import numpy as np
from .config import COMPARE_DEFAULTS


def _safe(value):
    value = float(value)
    if math.isnan(value):
        return 'NaN'
    if math.isinf(value):
        return 'Infinity' if value > 0 else '-Infinity'
    return value


def _scalar(value):
    if np.iscomplexobj(value):
        return {'real': _safe(np.real(value)), 'imag': _safe(np.imag(value))}
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return _safe(value)


def _policy(base, override):
    result = {**COMPARE_DEFAULTS, **base, **override}
    if 'metrics' in base or 'metrics' in override:
        result['metrics'] = {**base.get('metrics', {}), **override.get('metrics', {})}
    return result


def _components(array, ref, policy):
    integer_pair = array.dtype.kind in 'iub' and ref.dtype.kind in 'iub'
    a, r = array.astype(np.longdouble), ref.astype(np.longdouble)
    finite = np.isfinite(a) & np.isfinite(r)
    accepted = np.zeros(a.shape, dtype=bool)
    if policy['nan'] == 'equal':
        accepted |= np.isnan(a) & np.isnan(r)
    if policy['inf'] == 'same_sign':
        accepted |= np.isinf(a) & np.isinf(r) & (np.signbit(a) == np.signbit(r))
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        absolute = np.abs(a - r)
        if integer_pair:
            absolute = np.asarray(np.abs(array.astype(object) - ref.astype(object)), dtype=np.longdouble)
        threshold = np.longdouble(policy['atol']) + np.longdouble(policy['rtol']) * np.abs(r)
        if policy.get('exact', False):
            # Python int/float equality preserves the integer even for mixed
            # references; NumPy int64/uint64 == float64 can round unequal values
            # to the same float. This also covers complex real/imag components.
            integer_side = array.dtype.kind in 'iub' or ref.dtype.kind in 'iub'
            equality = array.astype(object) == ref.astype(object) if integer_side else array == ref
            matched = finite & equality | accepted
            threshold = np.zeros_like(absolute)
        else:
            matched = finite & (absolute <= threshold) | accepted
        relative = np.where(np.abs(r) == 0, np.where(absolute == 0, 0, np.inf), absolute / np.abs(r))
    return {'finite': finite, 'accepted': accepted, 'matched': matched, 'absolute': absolute,
            'relative': relative, 'threshold': threshold, 'nonfinite_bad': ~finite & ~accepted}


def _ulp_values(array, ref, dtype):
    """Ordered IEEE encodings after reference rounding to the output dtype."""
    if dtype == 'bfloat16':
        from .data import quantize_bfloat16
        actual = quantize_bfloat16(array).view(np.uint32) >> np.uint32(16)
        rounded = quantize_bfloat16(ref).view(np.uint32) >> np.uint32(16)
        width = 16
    else:
        target = np.dtype(dtype)
        if target.kind != 'f' or target.itemsize not in (2, 4, 8):
            raise ValueError('ULP only supports float16, bfloat16, float32 and float64')
        unsigned = np.dtype('uint' + str(target.itemsize * 8))
        with np.errstate(over='ignore', invalid='ignore'):
            actual = np.asarray(array, dtype=target).view(unsigned)
            rounded = np.asarray(ref, dtype=target).view(unsigned)
        width = target.itemsize * 8
    sign = 1 << (width - 1)
    mask = (1 << width) - 1
    def ordered(bits):
        raw = bits.astype(object)
        return np.where(raw & sign, mask - raw, raw + sign)
    distance = np.abs(ordered(actual) - ordered(rounded))
    return np.where((array == 0) & (ref == 0), 0, distance)


def compare_outputs(actual, expected, contracts, policy, actual_dtypes=None,
                    actual_metadata=None, input_metadata=None):
    """Validate count/shape/dtype/layout before comparing; never broadcast.

    Complex tolerance compares real and imaginary components independently. Error
    summaries use maximum component error; RMSE uses complex magnitude, cosine is
    the real Hermitian cosine. Metrics do not override nonfinite-policy violations.
    """
    policy = {**COMPARE_DEFAULTS, **policy}
    result = {'passed': False, 'outputs': [], 'contract_errors': [], 'policy': dict(policy),
              'definitions': {
                  'finite_match': 'abs(actual-reference) <= atol + rtol * abs(reference); exact=true requires equality',
                  'relative_error': 'abs(actual-reference)/abs(reference); 0/0=0; nonzero/0=Infinity',
                  'metric_population': 'finite actual/reference pairs only; empty population = null',
                  'mismatch_ratio': 'mismatched elements / total elements; empty output = 0',
                  'nonfinite_policy': 'violations always fail regardless of max_mismatch_ratio',
                  'complex': 'real and imaginary components each satisfy policy; element mismatch is their union',
                  'rmse': 'sqrt(mean(abs(actual-reference)**2)) on finite pairs',
                  'cosine': 'real(vdot(actual,reference))/(norm(actual)*norm(reference)); two zero vectors=1; one zero=0',
                  'ulp': 'ordered IEEE distance after reference rounds to declared output dtype; signed zero distance=0'}}
    errors = result['contract_errors']
    if not isinstance(actual, (list, tuple)) or not isinstance(expected, (list, tuple)):
        errors.append('actual and reference outputs must be lists or tuples')
        return result
    if len(actual) != len(contracts):
        errors.append(f'actual output count {len(actual)} != contract count {len(contracts)}')
    if len(expected) != len(contracts):
        errors.append(f'reference output count {len(expected)} != contract count {len(contracts)}')
    if actual_dtypes is not None and len(actual_dtypes) != len(actual):
        errors.append('actual_dtypes count does not match actual output count')
    if actual_metadata is not None and len(actual_metadata) != len(actual):
        errors.append('actual_metadata count does not match actual output count')
    for i, contract in enumerate(contracts):
        wanted_shape = contract['shape']
        if wanted_shape == 'reference':
            if i >= len(expected):
                continue
            ref = np.asarray(expected[i])
            wanted_shape = list(ref.shape)
            if ref.ndim != contract['rank']:
                errors.append(f'output[{i}] reference rank {ref.ndim} != {contract["rank"]}')
            if ref.size > contract['max_elements']:
                errors.append(f'output[{i}] reference elements {ref.size} exceed {contract["max_elements"]}')
        if i < len(actual):
            array = np.asarray(actual[i])
            dtype = (actual_dtypes[i] if actual_dtypes is not None and i < len(actual_dtypes)
                     else array.dtype.name)
            if list(array.shape) != wanted_shape:
                errors.append(f'output[{i}] actual shape {list(array.shape)} != {wanted_shape}')
            if dtype != contract['dtype']:
                errors.append(f'output[{i}] actual dtype {dtype} != {contract["dtype"]}')
            storage_dtype = 'float32' if dtype == 'bfloat16' else dtype
            if array.dtype.name != storage_dtype:
                errors.append(f'output[{i}] actual storage dtype {array.dtype.name} disagrees with reported dtype {dtype}')
            if array.dtype.kind not in 'iufbc':
                errors.append(f'output[{i}] actual is not a supported numeric array')
            if dtype == 'bfloat16' and array.dtype.name == 'float32':
                from .data import quantize_bfloat16
                if not np.array_equal(array, quantize_bfloat16(array), equal_nan=True):
                    errors.append(f'output[{i}] reported BF16 contains unquantized float32 values')
            meta = actual_metadata[i] if actual_metadata is not None and i < len(actual_metadata) else None
            if 'layout' in contract:
                if meta is None:
                    from .data import tensor_metadata
                    meta = tensor_metadata(array, dtype)
                for key, desired in contract['layout'].items():
                    if meta.get(key) != desired:
                        errors.append(f'output[{i}] layout {key} {meta.get(key)!r} != {desired!r}')
            if 'alias_input' in contract:
                target = contract['alias_input']
                if isinstance(target, str) and input_metadata is not None:
                    candidates = [j for j, item in enumerate(input_metadata) if item.get('name') == target]
                    target = candidates[0] if len(candidates) == 1 else target
                if meta is None or target not in meta.get('alias_inputs', []):
                    errors.append(f'output[{i}] missing proven alias to input {target!r}')
        if i < len(expected):
            ref = np.asarray(expected[i])
            if list(ref.shape) != wanted_shape:
                errors.append(f'output[{i}] reference shape {list(ref.shape)} != {wanted_shape}')
            if ref.dtype.kind not in 'iufbc':
                errors.append(f'output[{i}] reference is not a supported numeric array')
            if ref.dtype.kind == 'c' and not contract['dtype'].startswith('complex'):
                errors.append(f'output[{i}] complex reference cannot be compared to real output contract')
    if errors:
        return result
    for i, (array, ref) in enumerate(zip(actual, expected)):
        array, ref = np.asarray(array), np.asarray(ref)
        current = _policy(policy, contracts[i].get('compare', {}))
        complex_pair = np.iscomplexobj(array) or np.iscomplexobj(ref)
        components = [_components(array.real, ref.real, current)]
        if complex_pair:
            components.append(_components(array.imag, ref.imag, current))
        finite = np.logical_and.reduce([part['finite'] for part in components])
        matched = np.logical_and.reduce([part['matched'] for part in components])
        nonfinite_bad = np.logical_or.reduce([part['nonfinite_bad'] for part in components])
        absolute = np.maximum.reduce([part['absolute'] for part in components])
        relative = np.maximum.reduce([part['relative'] for part in components])
        threshold = np.maximum.reduce([part['threshold'] for part in components])
        mismatch = ~matched
        count, size = int(np.count_nonzero(mismatch)), int(array.size)
        violations = int(np.count_nonzero(nonfinite_bad))
        ratio = count / size if size else 0.0
        finite_absolute, finite_relative = absolute[finite], relative[finite]
        failures = []
        for flat_index in np.flatnonzero(mismatch.ravel())[:current['max_failures']]:
            index = tuple(int(x) for x in np.unravel_index(int(flat_index), array.shape))
            failure = {'index': list(index), 'actual': _scalar(array[index]), 'reference': _scalar(ref[index]),
                       'absolute_error': _safe(absolute[index]), 'relative_error': _safe(relative[index]),
                       'allowed_absolute_error': _safe(threshold[index]),
                       'reason': 'nonfinite_policy' if nonfinite_bad[index] else 'exact' if current.get('exact') else 'tolerance'}
            if complex_pair:
                failure['components'] = {name: {'passed': bool(part['matched'][index]),
                    'absolute_error': _safe(part['absolute'][index]), 'allowed_absolute_error': _safe(part['threshold'][index])}
                    for name, part in zip(('real', 'imag'), components)}
            failures.append(failure)
        output = {'index': i, 'passed': ratio <= current['max_mismatch_ratio'] and violations == 0,
                  'shape': list(array.shape), 'dtype': contracts[i]['dtype'], 'total_elements': size,
                  'policy': current, 'mismatch_count': count, 'mismatch_ratio': ratio,
                  'finite_pairs': int(np.count_nonzero(finite)), 'nonfinite_pairs': int(np.count_nonzero(~finite)),
                  'nonfinite_policy_violations': violations,
                  'max_absolute_error': _safe(np.max(finite_absolute)) if finite_absolute.size else None,
                  'max_relative_error': _safe(np.max(finite_relative)) if finite_relative.size else None,
                  'failures': failures, 'failure_details_truncated': count > len(failures)}
        metrics = {}
        if np.any(finite):
            metric_dtype = np.clongdouble if complex_pair else np.longdouble
            a, r = array[finite].astype(metric_dtype), ref[finite].astype(metric_dtype)
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                # Exact integer difference remains observable even if a platform's
                # longdouble equals float64.
                differences = (np.asarray(array[finite].astype(object) - ref[finite].astype(object), dtype=np.longdouble)
                               if array.dtype.kind in 'iub' and ref.dtype.kind in 'iub' else a - r)
                rmse = np.sqrt(np.mean(np.abs(differences) ** 2))
                scale_a, scale_r = np.max(np.abs(a)), np.max(np.abs(r))
                if scale_a == 0 or scale_r == 0:
                    cosine = np.longdouble(1 if scale_a == scale_r else 0)
                else:
                    na, nr = a / scale_a, r / scale_r
                    cosine = np.real(np.vdot(na, nr)) / np.sqrt(np.sum(np.abs(na)**2) * np.sum(np.abs(nr)**2))
                    cosine = np.clip(cosine, -1, 1)
            metrics.update(rmse=_safe(rmse), cosine=_safe(cosine))
            if 'ulp_max' in current.get('metrics', {}):
                distances = _ulp_values(array, ref, contracts[i]['dtype'])
                metrics['max_ulp'] = int(np.max(distances[finite]))
        else:
            metrics.update(rmse=None, cosine=None)
            if 'ulp_max' in current.get('metrics', {}):
                metrics['max_ulp'] = None
        checks = {}
        for key, limit in current.get('metrics', {}).items():
            value = metrics[{'rmse_max': 'rmse', 'cosine_min': 'cosine', 'ulp_max': 'max_ulp'}[key]]
            # Empty tensors satisfy vacuous metrics; nonempty all-nonfinite tensors
            # cannot claim a finite numerical metric was measured.
            ok = size == 0 if value is None else isinstance(value, (int, float)) and (value >= limit if key == 'cosine_min' else value <= limit)
            checks[key] = {'value': value, 'threshold': limit, 'passed': bool(ok)}
        output.update(metrics=metrics, metric_checks=checks)
        output['passed'] = output['passed'] and all(check['passed'] for check in checks.values())
        if 'custom' in current:
            from .util import load_entry, json_safe
            try:
                custom = load_entry(current['custom'])(array.copy(), ref.copy(), copy.deepcopy(contracts[i]), copy.deepcopy(current))
                if not isinstance(custom, dict) or type(custom.get('passed')) is not bool:
                    raise TypeError('custom comparator must return an object with a boolean passed field')
                output['custom'] = json_safe(custom)
                output['passed'] = output['passed'] and custom['passed']
            except Exception as exc:
                output['custom'] = {'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
                output['passed'] = False
        result['outputs'].append(output)
    result['passed'] = bool(result['outputs']) and all(output['passed'] for output in result['outputs'])
    return result
