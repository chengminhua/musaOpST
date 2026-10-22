# SPDX-License-Identifier: MulanPSL-2.0
"""Dependency-free, strict v1 configuration validation and parameter expansion."""
from __future__ import annotations

import copy
import fnmatch
import itertools
import json
import math
import re
from pathlib import Path

DTYPES = ("float16", "bfloat16", "float32", "float64", "int32", "int64")
BACKENDS = ("cpu", "torch_musa", "musa_cpp", "triton", "tilelang", "native")
COMPARE_DEFAULTS = {"atol": 1e-5, "rtol": 1e-5, "nan": "reject", "inf": "reject",
                    "max_mismatch_ratio": 0.0, "max_failures": 8}
BENCHMARK_DEFAULTS = {"enabled": False, "warmup": 3, "samples": 7, "iterations": 10}
REPEAT_DEFAULTS = {"count": 0, "duration_s": 0.0}
_ENTRY = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z_]\w*)\}$")


class ConfigError(ValueError):
    """A configuration is invalid; no device work has been launched."""


def _error(path, message):
    raise ConfigError(f"{path}: {message}")


def _mapping(value, path, allowed=None, required=()):
    if not isinstance(value, dict):
        _error(path, "must be an object")
    if any(not isinstance(key, str) for key in value):
        _error(path, "object keys must be strings")
    if allowed is not None:
        unknown = set(value) - set(allowed)
        if unknown:
            _error(path, f"unknown fields: {', '.join(sorted(unknown))}")
    missing = set(required) - set(value)
    if missing:
        _error(path, f"missing fields: {', '.join(sorted(missing))}")


def _number(value, path, minimum=None, integer=False, exclusive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(path, "must be a number" if not integer else "must be an integer")
    if integer and not isinstance(value, int):
        _error(path, "must be an integer")
    if isinstance(value, float) and not math.isfinite(value):
        _error(path, "must be finite")
    if minimum is not None and (value <= minimum if exclusive else value < minimum):
        _error(path, f"must be {'>' if exclusive else '>='} {minimum}")


def _boolean(value, path):
    if not isinstance(value, bool):
        _error(path, "must be a boolean")


def _json_value(value, path):
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        _number(value, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _error(path, "object keys must be strings")
            _json_value(item, f"{path}.{key}")
    else:
        _error(path, "must contain only JSON values")


def _shape(value, path):
    if not isinstance(value, list):
        _error(path, "must be a list of nonnegative integers")
    for i, dim in enumerate(value):
        _number(dim, f"{path}[{i}]", minimum=0, integer=True)


def _fixed_shape(value, path):
    """Validate numeric rectangular data and return its shape (scalar = [])."""
    if not isinstance(value, list):
        _number(value, path)
        return []
    child_shapes = [_fixed_shape(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if child_shapes and any(shape != child_shapes[0] for shape in child_shapes):
        _error(path, "fixed values must be a rectangular numeric array")
    return [len(value)] + (child_shapes[0] if child_shapes else [])


def _distribution(value, shape, path):
    _mapping(value, path, ("kind", "mean", "std", "low", "high", "values"), ("kind",))
    kind = value["kind"]
    if kind == "normal":
        _mapping(value, path, ("kind", "mean", "std"))
        value.setdefault("mean", 0.0)
        value.setdefault("std", 1.0)
        _number(value["mean"], path + ".mean")
        _number(value["std"], path + ".std", minimum=0)
    elif kind == "uniform":
        _mapping(value, path, ("kind", "low", "high"))
        value.setdefault("low", 0.0)
        value.setdefault("high", 1.0)
        _number(value["low"], path + ".low")
        _number(value["high"], path + ".high")
        if value["low"] >= value["high"]:
            _error(path, "uniform low must be less than high")
    elif kind == "fixed":
        _mapping(value, path, ("kind", "values"), ("values",))
        actual_shape = _fixed_shape(value["values"], path + ".values")
        # [] can express any empty tensor; there are no actual values to reshape.
        empty_data = actual_shape == [0] and 0 in shape
        if actual_shape and actual_shape != shape and not empty_data:
            _error(path, f"fixed values shape {actual_shape} does not match {shape}")
    else:
        _error(path + ".kind", "must be normal, uniform or fixed")


def _compare(value, path):
    _mapping(value, path, COMPARE_DEFAULTS)
    merged = {**COMPARE_DEFAULTS, **value}
    for key in ("atol", "rtol", "max_mismatch_ratio"):
        _number(merged[key], f"{path}.{key}", minimum=0)
    if merged["max_mismatch_ratio"] > 1:
        _error(path + ".max_mismatch_ratio", "must be <= 1")
    _number(merged["max_failures"], path + ".max_failures", minimum=0, integer=True)
    if merged["nan"] not in ("reject", "equal"):
        _error(path + ".nan", "must be reject or equal")
    if merged["inf"] not in ("reject", "same_sign"):
        _error(path + ".inf", "must be reject or same_sign")
    return merged


def _substitute(value, selected, path):
    if isinstance(value, str):
        match = _PLACEHOLDER.fullmatch(value)
        if match:
            name = match.group(1)
            if name not in selected:
                _error(path, f"unresolved parameter {name!r}")
            return copy.deepcopy(selected[name])
        if "${" in value:
            _error(path, "parameters must occupy the whole value, e.g. '${size}'")
        return value
    if isinstance(value, list):
        return [_substitute(item, selected, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, dict):
        return {key: _substitute(item, selected, f"{path}.{key}") for key, item in value.items()}
    return value


def _case(case, path):
    allowed = ("name", "backend", "entry", "inputs", "outputs", "reference", "attrs", "seed",
               "compare", "compare_by_dtype", "benchmark", "repeat", "timeout_s", "required", "skip")
    _mapping(case, path, allowed, ("name", "backend", "entry", "inputs", "outputs", "reference"))
    if not isinstance(case["name"], str) or not case["name"].strip():
        _error(path + ".name", "must be a nonempty string")
    if case["backend"] not in BACKENDS:
        _error(path + ".backend", f"must be one of {BACKENDS}")
    for field in ("entry", "reference"):
        if not isinstance(case[field], str) or not _ENTRY.fullmatch(case[field]):
            _error(path + "." + field, "must be a Python 'module:function' entry")
    if case["entry"] == case["reference"]:
        _error(path + ".reference", "reference must be independent from the tested entry")
    for field in ("inputs", "outputs"):
        if not isinstance(case[field], list) or not case[field]:
            _error(path + "." + field, "must be a nonempty list")
        for i, spec in enumerate(case[field]):
            loc = f"{path}.{field}[{i}]"
            keys = ("shape", "dtype", "layout", "distribution") if field == "inputs" else ("shape", "dtype")
            _mapping(spec, loc, keys, ("shape", "dtype"))
            _shape(spec["shape"], loc + ".shape")
            if spec["dtype"] not in DTYPES:
                _error(loc + ".dtype", f"must be one of {DTYPES}")
            if field == "inputs":
                spec.setdefault("layout", "contiguous")
                if spec["layout"] not in ("contiguous", "strided"):
                    _error(loc + ".layout", "must be contiguous or strided")
                if spec["layout"] == "strided" and not spec["shape"]:
                    _error(loc + ".layout", "strided scalar is not supported")
                spec.setdefault("distribution", {"kind": "normal"})
                _distribution(spec["distribution"], spec["shape"], loc + ".distribution")
    case.setdefault("attrs", {})
    _mapping(case["attrs"], path + ".attrs")
    _json_value(case["attrs"], path + ".attrs")
    case.setdefault("seed", 0)
    _number(case["seed"], path + ".seed", minimum=0, integer=True)
    if case["seed"] >= 2**64:
        _error(path + ".seed", "must be smaller than 2**64")
    case["compare"] = _compare(case.get("compare", {}), path + ".compare")
    if "compare_by_dtype" in case:
        policies = case["compare_by_dtype"]
        _mapping(policies, path + ".compare_by_dtype", DTYPES)
        for dtype, policy in policies.items():
            _compare(policy, path + ".compare_by_dtype." + dtype)
        case["compare"].update(policies.get(case["outputs"][0]["dtype"], {}))
    benchmark = case.setdefault("benchmark", {})
    _mapping(benchmark, path + ".benchmark", BENCHMARK_DEFAULTS)
    case["benchmark"] = benchmark = {**BENCHMARK_DEFAULTS, **benchmark}
    _boolean(benchmark["enabled"], path + ".benchmark.enabled")
    for key, minimum in (("warmup", 0), ("samples", 1), ("iterations", 1)):
        _number(benchmark[key], path + ".benchmark." + key, minimum=minimum, integer=True)
    repeat = case.setdefault("repeat", {})
    _mapping(repeat, path + ".repeat", REPEAT_DEFAULTS)
    case["repeat"] = repeat = {**REPEAT_DEFAULTS, **repeat}
    _number(repeat["count"], path + ".repeat.count", minimum=0, integer=True)
    _number(repeat["duration_s"], path + ".repeat.duration_s", minimum=0)
    case.setdefault("timeout_s", 120.0)
    _number(case["timeout_s"], path + ".timeout_s", minimum=0, exclusive=True)
    for key, default in (("required", True), ("skip", False)):
        case.setdefault(key, default)
        _boolean(case[key], path + "." + key)
    return case


def expand_config(config):
    """Return validated independent cases. Validation is complete before execution."""
    _mapping(config, "config", ("schema_version", "cases"), ("schema_version", "cases"))
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        _error("config.schema_version", "only integer version 1 is supported")
    if not isinstance(config["cases"], list) or not config["cases"]:
        _error("config.cases", "must be a nonempty list")
    result, names = [], set()
    for i, original in enumerate(config["cases"]):
        path = f"config.cases[{i}]"
        _mapping(original, path)
        original = copy.deepcopy(original)
        parameters = original.pop("parameters", {})
        _mapping(parameters, path + ".parameters")
        for name, values in parameters.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_]\w*", name):
                _error(path + ".parameters", "parameter names must be Python identifiers")
            if not isinstance(values, list) or not values:
                _error(path + ".parameters." + name, "must be a nonempty list")
            _json_value(values, path + ".parameters." + name)
        keys = list(parameters)
        combinations = itertools.product(*(parameters[key] for key in keys))
        for combination_index, values in enumerate(combinations):
            selected = dict(zip(keys, values))
            case = _case(_substitute(original, selected, path), path)
            if keys:
                # Index suffix remains compact even for list-valued shape parameters.
                case["name"] += f"[{combination_index}]"
                case["parameters_selected"] = selected
            if case["name"] in names:
                _error(path + ".name", f"duplicate expanded case name {case['name']!r}")
            names.add(case["name"])
            result.append(case)
    return result


def load_config(path):
    try:
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ConfigError(f"duplicate JSON key {key!r}")
                result[key] = value
            return result
        def nonfinite(value):
            raise ConfigError(f"non-standard JSON number {value!r} is not permitted")
        with Path(path).open(encoding="utf-8") as stream:
            config = json.load(stream, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    return expand_config(config)


def case_matches(name, pattern):
    """`--case` accepts a glob or a plain substring; no hidden glob requirement."""
    return not pattern or fnmatch.fnmatch(name, pattern) or pattern in name


def filter_cases(cases, pattern=None, tags=None):
    selected = []
    for case in cases:
        if not case_matches(case['name'], pattern):
            continue
        if tags and not set(tags).issubset(case.get('tags', [])):
            continue
        selected.append(case)
    return selected


def unique_names(names):
    return list(dict.fromkeys(names))


def selection_error(names, pattern=None, tags=None, limit=15):
    """Explain a zero-match selection with real names and both accepted forms."""
    names = unique_names(names)
    wanted = []
    if pattern:
        wanted.append(f'--case={pattern!r}')
    if tags:
        wanted.append('--tag=' + ','.join(tags))
    head = f"没有用例匹配 {' '.join(wanted) or '当前筛选'}（共 {len(names)} 个可用用例）。"
    if names:
        head += f"可用用例（前 {min(limit, len(names))} 个）：{'、'.join(names[:limit])}"
        if len(names) > limit:
            head += ' …'
        head += '。'
    if pattern:
        tail = (" --case 支持通配符（如 '*softmax*'）或子串（如 'softmax'），"
                "也可用 --list-cases 查看全部用例。")
    else:
        tail = ' 可用 --list-cases 查看全部用例名与标签。'
    return head + tail

# Version 1 is intentionally left intact above. Version 2 adds explicit tensor
# trees and richer policies without changing existing configurations' defaults.
_expand_config_v1 = expand_config
V2_DTYPES = DTYPES + ('bool', 'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64', 'complex64', 'complex128')
COMPARE_V2_KEYS = tuple(COMPARE_DEFAULTS) + ('exact', 'metrics', 'custom')
BENCHMARK_V2_DEFAULTS = {**BENCHMARK_DEFAULTS, 'mode': 'batch_event', 'stream': 'current',
                         'cache': 'warm', 'cache_bytes': 16777216, 'allow_inplace': False}
PROFILE_DEFAULTS = {'enabled': False, 'warmup': 1, 'iterations': 3,
                    'record_shapes': True, 'profile_memory': False}


def _entry(value, path):
    if not isinstance(value, str) or not _ENTRY.fullmatch(value):
        _error(path, "must be a Python 'module:function' entry")


def _compare_v2(value, path, full=True):
    _mapping(value, path, COMPARE_V2_KEYS)
    common = _compare({k: v for k, v in value.items() if k in COMPARE_DEFAULTS}, path)
    result = {**common, **value} if full else copy.deepcopy(value)
    if 'exact' in value:
        _boolean(value['exact'], path + '.exact')
    if 'custom' in value:
        _entry(value['custom'], path + '.custom')
    metrics = value.get('metrics', {})
    _mapping(metrics, path + '.metrics', ('rmse_max', 'cosine_min', 'ulp_max'))
    for key, threshold in metrics.items():
        _number(threshold, path + '.metrics.' + key, minimum=-1 if key == 'cosine_min' else 0,
                integer=key == 'ulp_max')
        if key == 'cosine_min' and threshold > 1:
            _error(path + '.metrics.cosine_min', 'must be <= 1')
    return result


def _policy_merge(*policies):
    merged = {}
    for policy in policies:
        _mapping(policy, 'comparison policy', COMPARE_V2_KEYS)
        if 'metrics' in policy:
            _mapping(policy['metrics'], 'comparison policy.metrics')
        for key, value in policy.items():
            if key == 'metrics':
                merged[key] = {**merged.get(key, {}), **value}
            else:
                merged[key] = copy.deepcopy(value)
    return merged


def _positive_strides(value, shape, path):
    if not isinstance(value, list) or len(value) != len(shape):
        _error(path, 'must contain one element stride per dimension')
    for i, stride in enumerate(value):
        _number(stride, f'{path}[{i}]', minimum=0, integer=True, exclusive=True)
    # A sufficient non-overlap condition, accepting permutations and padded views.
    span = 1
    for stride, dim in sorted(zip(value, shape)):
        if dim > 1 and stride < span:
            _error(path, 'overlapping input storage is not supported')
        span += max(dim - 1, 0) * stride


def _distribution_v2(value, shape, path):
    if not isinstance(value, dict):
        _error(path, 'must be an object')
    kind = value.get('kind')
    if kind in ('normal', 'uniform'):
        return _distribution(value, shape, path)
    if kind == 'fixed':
        _mapping(value, path, ('kind', 'values', 'imag'), ('values',))
        def check(data, at):
            if isinstance(data, bool):
                return []
            if not isinstance(data, list):
                _number(data, at)
                return []
            children = [check(item, f'{at}[{i}]') for i, item in enumerate(data)]
            if children and any(c != children[0] for c in children):
                _error(at, 'fixed values must be rectangular')
            return [len(data)] + (children[0] if children else [])
        for key in ('values', 'imag'):
            if key not in value:
                continue
            actual = check(value[key], path + '.' + key)
            if actual and actual != shape and not (actual == [0] and 0 in shape):
                _error(path + '.' + key, f'fixed values shape {actual} does not match {shape}')
        return
    choices = {'beta': ('alpha', 'beta'), 'laplace': ('loc', 'scale'),
               'triangular': ('left', 'mode', 'right'), 'integers': ('low', 'high'),
               'bernoulli': ('p',)}
    if not isinstance(kind, str) or kind not in choices:
        _error(path + '.kind', 'unsupported distribution')
    _mapping(value, path, ('kind',) + choices[kind], choices[kind])
    for key in choices[kind]:
        _number(value[key], path + '.' + key, integer=kind == 'integers')
    if kind == 'beta' and (value['alpha'] <= 0 or value['beta'] <= 0):
        _error(path, 'beta alpha and beta must be positive')
    if kind == 'laplace' and value['scale'] < 0:
        _error(path, 'laplace scale must be nonnegative')
    if kind == 'triangular' and not value['left'] <= value['mode'] <= value['right']:
        _error(path, 'must satisfy left <= mode <= right')
    if kind == 'triangular' and value['left'] == value['right']:
        _error(path, 'triangular left must be smaller than right')
    if kind == 'integers' and value['low'] >= value['high']:
        _error(path, 'integers low must be smaller than high (exclusive)')
    if kind == 'bernoulli' and not 0 <= value['p'] <= 1:
        _error(path, 'bernoulli p must be in [0,1]')


def _tensor_input_v2(spec, path, nested=False):
    _mapping(spec, path)
    kind = spec.get('kind', 'tensor')
    if kind not in ('tensor', 'absent', 'tensor_list'):
        _error(path + '.kind', 'must be tensor, absent or tensor_list')
    common = ('kind', 'name', 'binding')
    if 'name' in spec and (not isinstance(spec['name'], str) or not re.fullmatch(r'[A-Za-z_]\w*', spec['name'])):
        _error(path + '.name', 'must be a Python identifier')
    if spec.get('binding', 'positional') not in ('positional', 'named'):
        _error(path + '.binding', 'must be positional or named')
    if spec.get('binding') == 'named' and ('name' not in spec or nested):
        _error(path + '.binding', 'named binding requires a top-level name')
    if kind == 'absent':
        _mapping(spec, path, common)
        return
    if kind == 'tensor_list':
        _mapping(spec, path, common + ('items',), ('items',))
        if not isinstance(spec['items'], list):
            _error(path + '.items', 'must be a list')
        for i, item in enumerate(spec['items']):
            _tensor_input_v2(item, f'{path}.items[{i}]', nested=True)
        return
    _mapping(spec, path, common + ('shape', 'dtype', 'layout', 'distribution', 'source',
                                  'strides', 'storage_offset', 'base_shape', 'transforms', 'constant'),
             ('shape', 'dtype'))
    _shape(spec['shape'], path + '.shape')
    if spec['dtype'] not in V2_DTYPES:
        _error(path + '.dtype', f'must be one of {V2_DTYPES}')
    spec.setdefault('layout', 'contiguous')
    if spec.get('layout', 'contiguous') not in ('contiguous', 'strided', 'explicit'):
        _error(path + '.layout', 'must be contiguous, strided or explicit')
    if spec.get('layout') == 'strided' and not spec['shape']:
        _error(path, 'strided scalar is not supported')
    if 'strides' in spec:
        _positive_strides(spec['strides'], spec['shape'], path + '.strides')
        spec['layout'] = 'explicit'
    elif spec.get('layout') == 'explicit':
        _error(path, 'explicit layout requires strides')
    _number(spec.get('storage_offset', 0), path + '.storage_offset', minimum=0, integer=True)
    if 'constant' in spec:
        _boolean(spec['constant'], path + '.constant')
    generation_shape = spec.get('base_shape', spec['shape'])
    _shape(generation_shape, path + '.base_shape')
    if 'source' in spec:
        if 'distribution' in spec:
            _error(path, 'source and distribution are mutually exclusive')
        source = spec['source']
        _mapping(source, path + '.source', ('kind', 'path', 'format', 'key', 'byte_order', 'sha256'), ('path',))
        if source.get('kind', 'file') != 'file':
            _error(path + '.source.kind', 'must be file')
        if not isinstance(source['path'], str) or not source['path']:
            _error(path + '.source.path', 'must be a nonempty path')
        fmt = source.setdefault('format', Path(source['path']).suffix.lstrip('.').lower())
        if fmt not in ('npy', 'npz', 'bin'):
            _error(path + '.source.format', 'must be npy, npz or bin')
        if fmt == 'npz' and (not isinstance(source.get('key'), str) or not source['key']):
            _error(path + '.source.key', 'npz requires an explicit nonempty key')
        if fmt == 'bin' and source.get('byte_order') not in ('little', 'big'):
            _error(path + '.source.byte_order', 'bin requires little or big')
        if 'sha256' in source and (not isinstance(source['sha256'], str) or not re.fullmatch(r'[a-fA-F0-9]{64}', source['sha256'])):
            _error(path + '.source.sha256', 'must be a SHA256 hexadecimal digest')
    else:
        spec.setdefault('distribution', {'kind': 'normal'})
        _distribution_v2(spec['distribution'], generation_shape, path + '.distribution')
        if 'imag' in spec['distribution'] and not spec['dtype'].startswith('complex'):
            _error(path + '.distribution.imag', 'imaginary values require a complex dtype')
    transforms = spec.get('transforms', [])
    if not isinstance(transforms, list):
        _error(path + '.transforms', 'must be a list')
    shape = list(generation_shape)
    for i, transform in enumerate(transforms):
        loc = f'{path}.transforms[{i}]'
        _mapping(transform, loc)
        if transform.get('kind') == 'transpose':
            _mapping(transform, loc, ('kind', 'axes'), ('axes',))
            axes = transform['axes']
            if not isinstance(axes, list) or any(type(x) is not int for x in axes) or sorted(axes) != list(range(len(shape))):
                _error(loc + '.axes', 'must be a permutation of all dimensions')
            shape = [shape[axis] for axis in axes]
        elif transform.get('kind') == 'slice':
            _mapping(transform, loc, ('kind', 'start', 'stop', 'step'))
            slices = {}
            for key, default in (('start', None), ('stop', None), ('step', 1)):
                values = transform.get(key, [default] * len(shape))
                if not isinstance(values, list) or len(values) != len(shape):
                    _error(loc + '.' + key, 'must contain one value per dimension')
                for j, value in enumerate(values):
                    if value is not None:
                        _number(value, f'{loc}.{key}[{j}]', integer=True,
                                minimum=1 if key == 'step' else None)
                slices[key] = values
            shape = [len(range(*slice(a, b, c).indices(dim))) for dim, a, b, c in
                     zip(shape, slices['start'], slices['stop'], slices['step'])]
        else:
            _error(loc + '.kind', 'must be transpose or slice')
    if shape != spec['shape']:
        _error(path, f'base_shape/transforms produce {shape}, expected {spec["shape"]}')
    if transforms and ('strides' in spec or spec.get('layout') == 'strided' or spec.get('storage_offset', 0)):
        _error(path, 'transforms cannot be combined with explicit strides, strided layout or storage_offset')


def _output_v2(spec, path):
    _mapping(spec, path, ('name', 'shape', 'dtype', 'compare', 'rank', 'max_elements', 'layout', 'alias_input'),
             ('shape', 'dtype'))
    if spec['shape'] == 'reference':
        _number(spec.get('rank'), path + '.rank', minimum=0, integer=True)
        _number(spec.get('max_elements'), path + '.max_elements', minimum=0, integer=True)
    else:
        _shape(spec['shape'], path + '.shape')
    if spec['dtype'] not in V2_DTYPES:
        _error(path + '.dtype', f'must be one of {V2_DTYPES}')
    if 'name' in spec and (not isinstance(spec['name'], str) or not spec['name']):
        _error(path + '.name', 'must be nonempty')
    if 'compare' in spec:
        _compare_v2(spec['compare'], path + '.compare')
    if 'layout' in spec:
        layout = spec['layout']
        _mapping(layout, path + '.layout', ('strides', 'contiguous', 'storage_offset'))
        if 'strides' in layout:
            rank_shape = [1] * spec['rank'] if spec['shape'] == 'reference' else spec['shape']
            _positive_strides(layout['strides'], rank_shape, path + '.layout.strides')
        if 'contiguous' in layout:
            _boolean(layout['contiguous'], path + '.layout.contiguous')
        if 'storage_offset' in layout:
            _number(layout['storage_offset'], path + '.layout.storage_offset', minimum=0, integer=True)
    if 'alias_input' in spec and not (isinstance(spec['alias_input'], str) or
                                    type(spec['alias_input']) is int and spec['alias_input'] >= 0):
        _error(path + '.alias_input', 'must be a flattened input index or input name')


def _attribute_types(attrs, declarations, path):
    _mapping(declarations, path)
    predicates = {'bool': lambda x: type(x) is bool, 'int': lambda x: type(x) is int,
                  'float': lambda x: type(x) in (int, float), 'str': lambda x: isinstance(x, str)}
    for name, declaration in declarations.items():
        if name not in attrs:
            _error(path + '.' + name, 'declared attribute is missing')
        element = declaration[5:-1] if isinstance(declaration, str) and declaration.startswith('list[') and declaration.endswith(']') else None
        if isinstance(declaration, str) and declaration in predicates:
            valid = predicates[declaration](attrs[name])
        elif element in predicates:
            valid = isinstance(attrs[name], list) and all(predicates[element](x) for x in attrs[name])
        else:
            _error(path + '.' + name, 'type must be bool/int/float/str or list[type]')
        if not valid:
            _error(path + '.' + name, f'value does not match {declaration}')


def _constraint_value(value, case, path):
    if not isinstance(value, dict):
        _json_value(value, path)
        return value
    if set(value) == {'attr'}:
        if not isinstance(value['attr'], str) or value['attr'] not in case.get('attrs', {}):
            _error(path, 'unknown attribute')
        return case['attrs'][value['attr']]
    if set(value) == {'input', 'axis'}:
        from .binding import tensor_specs
        specs = tensor_specs(case['inputs'])
        if type(value['input']) is not int or value['input'] < 0 or type(value['axis']) is not int:
            _error(path, 'input and axis references must be integer indices; input must be nonnegative')
        try:
            return specs[value['input']]['shape'][value['axis']]
        except (IndexError, TypeError, KeyError):
            _error(path, 'invalid flattened input/axis reference')
    _error(path, 'constraint operand must be a value, {attr:name} or {input:index,axis:index}')


def _check_constraints(case, path):
    constraints = case.get('constraints', [])
    if not isinstance(constraints, list):
        _error(path + '.constraints', 'must be a list')
    for i, item in enumerate(constraints):
        loc = f'{path}.constraints[{i}]'
        _mapping(item, loc, ('left', 'op', 'right'), ('left', 'op', 'right'))
        left = _constraint_value(item['left'], case, loc + '.left')
        right = _constraint_value(item['right'], case, loc + '.right')
        op = item['op']
        try:
            if op == 'eq': accepted = left == right
            elif op == 'ne': accepted = left != right
            elif op == 'lt': accepted = left < right
            elif op == 'le': accepted = left <= right
            elif op == 'gt': accepted = left > right
            elif op == 'ge': accepted = left >= right
            elif op == 'divisible':
                accepted = type(left) is int and type(right) is int and right != 0 and left % right == 0
            else: _error(loc + '.op', 'must be eq/ne/lt/le/gt/ge/divisible')
        except TypeError as exc:
            _error(loc, f'incompatible operands: {exc}')
        if not accepted:
            _error(loc, f'parameter combination violates constraint: {left!r} {op} {right!r}')


def _case_v2(case, path, defaults=None):
    legacy_keys = ('name', 'backend', 'entry', 'inputs', 'outputs', 'reference', 'attrs', 'seed',
                   'compare', 'compare_by_dtype', 'benchmark', 'repeat', 'timeout_s', 'required', 'skip')
    extras = ('reference_policy', 'expect', 'tags', 'profile', 'operator', 'build', 'attr_types',
              'constraints', 'artifact', 'sources', 'description', 'binding', 'compile_group', 'shape_mode')
    _mapping(case, path, legacy_keys + extras, ('name', 'backend', 'entry', 'inputs', 'outputs', 'reference'))
    # Reuse the established scalar/entry/timeout/repeat validation, independently
    # of richer v2 inputs and policies.
    base = {key: copy.deepcopy(value) for key, value in case.items() if key in legacy_keys}
    for key in ('compare', 'compare_by_dtype', 'benchmark'):
        base.pop(key, None)
    base['inputs'] = [{'shape': [1], 'dtype': 'float32'}]
    base['outputs'] = [{'shape': [1], 'dtype': 'float32'}]
    normalized = _case(base, path)
    for key in ('inputs', 'outputs', 'compare', 'benchmark'):
        normalized.pop(key)
    validated_repeat = normalized['repeat']
    normalized.update(case)
    normalized['repeat'] = validated_repeat
    case = normalized
    if not isinstance(case['inputs'], list):
        _error(path + '.inputs', 'must be a list (zero inputs are allowed)')
    if not isinstance(case['outputs'], list) or not case['outputs']:
        _error(path + '.outputs', 'must be a nonempty list')
    names = set()
    for i, spec in enumerate(case['inputs']):
        _tensor_input_v2(spec, f'{path}.inputs[{i}]')
        if 'name' in spec:
            if spec['name'] in names or spec.get('binding', case.get('binding', 'positional')) == 'named' and spec['name'] in case['attrs']:
                _error(path + '.inputs', f'duplicate input/attribute binding {spec["name"]!r}')
            names.add(spec['name'])
    for i, spec in enumerate(case['outputs']):
        _output_v2(spec, f'{path}.outputs[{i}]')
    default_policy = (defaults or {}).get('compare', {})
    case['compare'] = _compare_v2(_policy_merge(default_policy, case.get('compare', {})), path + '.compare')
    by_dtype = case.get('compare_by_dtype', {})
    _mapping(by_dtype, path + '.compare_by_dtype', V2_DTYPES)
    for dtype, policy in by_dtype.items():
        _compare_v2(policy, path + '.compare_by_dtype.' + dtype)
    for i, output in enumerate(case['outputs']):
        output['compare'] = _compare_v2(_policy_merge(case['compare'], by_dtype.get(output['dtype'], {}),
                                                    output.get('compare', {})), f'{path}.outputs[{i}].compare')
        if 'ulp_max' in output['compare'].get('metrics', {}) and output['dtype'] not in ('float16', 'bfloat16', 'float32', 'float64'):
            _error(f'{path}.outputs[{i}].compare.metrics.ulp_max', 'ULP requires a real floating output')
    reference = case.setdefault('reference_policy', 'float64')
    if reference not in ('float64', 'float32', 'preserve'):
        _error(path + '.reference_policy', 'must be float64, float32 or preserve')
    benchmark = case.get('benchmark', {})
    _mapping(benchmark, path + '.benchmark', BENCHMARK_V2_DEFAULTS)
    supplied_iterations = 'iterations' in benchmark
    case['benchmark'] = benchmark = {**BENCHMARK_V2_DEFAULTS, **benchmark}
    for key in ('enabled', 'allow_inplace'):
        _boolean(benchmark[key], path + '.benchmark.' + key)
    if benchmark['mode'] not in ('batch_event', 'single_event', 'wall_time'):
        _error(path + '.benchmark.mode', 'must be batch_event, single_event or wall_time')
    if benchmark['stream'] not in ('current', 'independent'):
        _error(path + '.benchmark.stream', 'must be current or independent')
    if benchmark['cache'] not in ('warm', 'interference'):
        _error(path + '.benchmark.cache', 'must be warm or interference')
    if benchmark['mode'] == 'single_event':
        if supplied_iterations and benchmark['iterations'] != 1:
            _error(path + '.benchmark.iterations', 'single_event requires iterations=1')
        benchmark['iterations'] = 1
    for key, minimum in (('warmup', 0), ('samples', 1), ('iterations', 1), ('cache_bytes', 1)):
        _number(benchmark[key], path + '.benchmark.' + key, minimum=minimum, integer=True)
    if benchmark['allow_inplace'] and benchmark['mode'] == 'batch_event':
        _error(path + '.benchmark', 'allow_inplace requires single_event or wall_time')
    profile = case.get('profile', {})
    _mapping(profile, path + '.profile', PROFILE_DEFAULTS)
    case['profile'] = profile = {**PROFILE_DEFAULTS, **profile}
    for key in ('enabled', 'record_shapes', 'profile_memory'):
        _boolean(profile[key], path + '.profile.' + key)
    for key, minimum in (('warmup', 0), ('iterations', 1)):
        _number(profile[key], path + '.profile.' + key, minimum=minimum, integer=True)
    if case.get('binding', 'positional') not in ('positional', 'named'):
        _error(path + '.binding', 'must be positional or named')
    if case.get('binding') == 'named' and any('name' not in spec for spec in case['inputs']):
        _error(path + '.inputs', 'named binding requires all input names')
    if case.get('shape_mode', 'specialize') not in ('specialize', 'runtime'):
        _error(path + '.shape_mode', 'must be specialize or runtime')
    if 'compile_group' in case and (not isinstance(case['compile_group'], str) or not case['compile_group']):
        _error(path + '.compile_group', 'must be a nonempty string')
    if case.get('shape_mode') == 'runtime' and 'compile_group' not in case:
        _error(path + '.compile_group', 'runtime shape mode requires compile_group')
    tags = case.setdefault('tags', [])
    if not isinstance(tags, list) or any(not isinstance(x, str) or not x for x in tags) or len(tags) != len(set(tags)):
        _error(path + '.tags', 'must be a list of unique nonempty strings')
    if 'expect' in case:
        expect = case['expect']
        _mapping(expect, path + '.expect', ('status', 'stage', 'exception', 'message_contains'), ('status',))
        allowed = ('PASS', 'ACCURACY_FAILED', 'BUILD_FAILED', 'EXECUTION_FAILED', 'REFERENCE_FAILED', 'TIMEOUT', 'CONFIG_ERROR', 'UNSUPPORTED')
        if expect['status'] not in allowed:
            _error(path + '.expect.status', f'must be one of {allowed}; environment/fatal errors cannot be expected')
        if expect['status'] != 'PASS':
            if expect.get('stage') not in ('compare', 'build', 'execute', 'reference', 'inputs', 'timeout', 'repeat', 'benchmark', 'post_benchmark_compare'):
                _error(path + '.expect.stage', 'must explicitly name the expected failure stage')
            if not isinstance(expect.get('exception'), str) or not expect['exception']:
                _error(path + '.expect.exception', 'must explicitly name the expected exception type')
        if 'message_contains' in expect and not isinstance(expect['message_contains'], str):
            _error(path + '.expect.message_contains', 'must be a string')
    if 'attr_types' in case:
        _attribute_types(case['attrs'], case['attr_types'], path + '.attr_types')
    for field in ('operator', 'build', 'artifact'):
        if field in case:
            if not isinstance(case[field], (str, dict)):
                _error(path + '.' + field, 'must be a string or object')
            _json_value(case[field], path + '.' + field)
    if 'sources' in case:
        if not isinstance(case['sources'], list) or any(not isinstance(x, str) or not x for x in case['sources']):
            _error(path + '.sources', 'must be a list of source file paths')
    _check_constraints(case, path)
    case['_schema_version'] = 2
    return case


def expand_config(config):
    """Expand and validate v1/v2, with no GPU or NumPy dependency."""
    if isinstance(config, dict) and config.get('schema_version') == 1:
        return _expand_config_v1(config)
    _mapping(config, 'config', ('schema_version', 'defaults', 'cases'), ('schema_version', 'cases'))
    if type(config['schema_version']) is not int or config['schema_version'] != 2:
        _error('config.schema_version', 'only integer versions 1 and 2 are supported')
    defaults = config.get('defaults', {})
    _mapping(defaults, 'config.defaults', ('compare', 'benchmark', 'repeat', 'timeout_s', 'required',
                                          'reference_policy', 'profile', 'seed'))
    if not isinstance(config['cases'], list) or not config['cases']:
        _error('config.cases', 'must be a nonempty list')
    result, names = [], set()
    for i, original in enumerate(config['cases']):
        path = f'config.cases[{i}]'
        _mapping(original, path)
        original = copy.deepcopy(original)
        parameters = original.pop('parameters', {})
        symbols = original.pop('symbols', {})
        _mapping(parameters, path + '.parameters')
        _mapping(symbols, path + '.symbols')
        if set(parameters) & set(symbols):
            _error(path, 'parameters and symbols must use distinct names')
        parameters = {**parameters, **symbols}
        for name, values in parameters.items():
            if not re.fullmatch(r'[A-Za-z_]\w*', name):
                _error(path + '.parameters', 'parameter names must be Python identifiers')
            if not isinstance(values, list) or not values:
                _error(path + '.parameters.' + name, 'must be a nonempty finite list')
            _json_value(values, path + '.parameters.' + name)
            if name in symbols:
                for value in values:
                    _number(value, path + '.symbols.' + name, minimum=0, integer=True)
        keys = list(parameters)
        for index, values in enumerate(itertools.product(*(parameters[key] for key in keys))):
            selected = dict(zip(keys, values))
            source = _substitute(original, selected, path)
            for key, default in defaults.items():
                if key == 'compare':
                    continue
                if isinstance(default, dict) and isinstance(source.get(key), dict):
                    source[key] = {**copy.deepcopy(default), **source[key]}
                else:
                    source.setdefault(key, copy.deepcopy(default))
            case = _case_v2(source, path, defaults)
            if keys:
                case['name'] += f'[{index}]'
                case['parameters_selected'] = selected
            if case['name'] in names:
                _error(path + '.name', f'duplicate expanded case name {case["name"]!r}')
            names.add(case['name'])
            result.append(case)
    return result


def revalidate_cases(cases):
    """Revalidate normalized cases without passing internal fields into JSON schema."""
    result = []
    for original in cases:
        case = copy.deepcopy(original)
        version = case.pop('_schema_version', 1)
        selected = case.pop('parameters_selected', None)
        expanded = expand_config({'schema_version': version, 'cases': [case]})[0]
        if selected is not None:
            expanded['parameters_selected'] = selected
        result.append(expanded)
    return result
