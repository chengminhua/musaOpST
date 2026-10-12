# SPDX-License-Identifier: MulanPSL-2.0
"""Deterministic tensor trees, strict file input, quantization and layout snapshots."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import numpy as np
from .binding import map_tensors


def quantize_bfloat16(values):
    """Round float32 to bfloat16 (nearest, ties-to-even), stored as float32."""
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    with np.errstate(over='ignore'):
        rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    special = (bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    special_bits = bits & np.uint32(0xFFFF0000)
    special_bits = np.where(np.isnan(array), special_bits | np.uint32(0x00400000), special_bits)
    return np.asarray(np.where(special, special_bits, rounded), dtype=np.uint32).view(np.float32).copy()


def storage_offset(array):
    array = np.asarray(array)
    root = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    delta = array.__array_interface__['data'][0] - root.__array_interface__['data'][0]
    return int(delta // array.itemsize)


def _copy_strides(array, dtype=None):
    """Copy values and element strides/offset into independent storage."""
    array = np.asarray(array)
    dtype = np.dtype(dtype or array.dtype)
    strides = tuple(stride // array.itemsize * dtype.itemsize for stride in array.strides)
    low = sum(min(0, (dim - 1) * stride) for dim, stride in zip(array.shape, strides)) if array.size else 0
    high = sum(max(0, (dim - 1) * stride) for dim, stride in zip(array.shape, strides)) if array.size else 0
    offset = max(storage_offset(array) * dtype.itemsize, -low)
    storage = np.empty(max(offset + high + dtype.itemsize, dtype.itemsize), dtype=np.uint8)
    result = np.ndarray(array.shape, dtype=dtype, buffer=storage, offset=offset, strides=strides)
    result[...] = array
    return result


def _file_values(spec, base_dir):
    source = spec['source']
    path = Path(source['path'])
    if not path.is_absolute():
        path = Path(base_dir or '.') / path
    if not path.is_file():
        raise ValueError(f'input file does not exist: {path}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if 'sha256' in source and digest.lower() != source['sha256'].lower():
        raise ValueError(f'input file SHA256 mismatch: {path}')
    shape = tuple(spec.get('base_shape', spec['shape']))
    dtype = spec['dtype']
    fmt = source.get('format', path.suffix.lstrip('.').lower())
    if fmt == 'bin':
        if source.get('byte_order') not in ('little', 'big'):
            raise ValueError('raw binary input requires explicit byte_order')
        disk_dtype = np.dtype('uint16' if dtype == 'bfloat16' else dtype).newbyteorder(
            '<' if source['byte_order'] == 'little' else '>')
        expected_bytes = math.prod(shape) * disk_dtype.itemsize
        if path.stat().st_size != expected_bytes:
            raise ValueError(f'input binary length {path.stat().st_size} != {expected_bytes}')
        values = np.fromfile(path, dtype=disk_dtype).reshape(shape)
    elif fmt == 'npy':
        values = np.load(path, allow_pickle=False)
    elif fmt == 'npz':
        if not source.get('key'):
            raise ValueError('NPZ input requires an explicit key')
        with np.load(path, allow_pickle=False) as archive:
            if source['key'] not in archive.files:
                raise ValueError(f'NPZ input key {source["key"]!r} does not exist')
            values = archive[source['key']].copy()
    else:
        raise ValueError(f'unsupported input file format {fmt!r}')
    if not isinstance(values, np.ndarray) or values.dtype.hasobject:
        raise ValueError('input file must contain a numeric ndarray; pickle is forbidden')
    if tuple(values.shape) != shape:
        raise ValueError(f'input file shape {list(values.shape)} != {list(shape)}')
    if dtype == 'bfloat16':
        if values.dtype.name == 'uint16':
            values = (values.astype(np.uint32) << np.uint32(16)).view(np.float32)
        elif values.dtype.name == 'float32':
            rounded = quantize_bfloat16(values)
            if not np.array_equal(values, rounded, equal_nan=True):
                raise ValueError('BF16 float32 file contains values not exactly representable in BF16')
        else:
            raise ValueError('BF16 file storage must be uint16 bits or quantized float32')
    elif values.dtype.name != dtype:
        raise ValueError(f'input file dtype {values.dtype.name} != declared {dtype}')
    # Convert byte order, never numeric precision; framework input must be native endian.
    return values.astype(np.dtype('float32' if dtype == 'bfloat16' else dtype), copy=True)


def source_metadata(spec, base_dir=None):
    if 'source' not in spec:
        return None
    source = dict(spec['source'])
    path = Path(source['path'])
    if not path.is_absolute():
        path = Path(base_dir or '.') / path
    source.update(resolved_path=str(path.resolve()), actual_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  size_bytes=path.stat().st_size)
    return source


def _generated_values(spec, rng):
    shape = tuple(spec.get('base_shape', spec['shape']))
    distribution = spec.get('distribution', {'kind': 'normal'})
    kind = distribution['kind']
    if kind == 'normal':
        values = rng.normal(distribution.get('mean', 0), distribution.get('std', 1), size=shape)
    elif kind == 'uniform':
        values = rng.uniform(distribution.get('low', 0), distribution.get('high', 1), size=shape)
    elif kind == 'beta':
        values = rng.beta(distribution['alpha'], distribution['beta'], size=shape)
    elif kind == 'laplace':
        values = rng.laplace(distribution['loc'], distribution['scale'], size=shape)
    elif kind == 'triangular':
        values = rng.triangular(distribution['left'], distribution['mode'], distribution['right'], size=shape)
    elif kind == 'integers':
        dtype = np.dtype(spec['dtype'])
        target = dtype if dtype.kind in 'iu' else np.dtype('int64')
        values = rng.integers(distribution['low'], distribution['high'], size=shape, dtype=target)
    elif kind == 'bernoulli':
        values = rng.random(shape) < distribution['p']
    elif kind == 'fixed':
        def fixed(data):
            # Preserve Python integer precision before choosing the target dtype.
            # NumPy otherwise promotes [0, UINT64_MAX, UINT64_MAX-1] to float64.
            value = np.asarray(data, dtype=object)
            return np.full(shape, value) if value.ndim == 0 else value.reshape(shape).copy()
        values = fixed(distribution['values'])
        if 'imag' in distribution:
            values = values.astype(np.complex128) + 1j * fixed(distribution['imag'])
    else:
        raise ValueError(f'unsupported distribution {kind!r}')
    if spec['dtype'].startswith('complex') and kind != 'fixed':
        # Independent imaginary draws use the same configured real distribution.
        real_spec = dict(spec, dtype='float64')
        values = values.astype(np.complex128) + 1j * _generated_values(real_spec, rng)
    return values


def _layout(values, spec):
    array = np.array(values, copy=True, order='C')
    for transform in spec.get('transforms', []):
        if transform['kind'] == 'transpose':
            array = array.transpose(transform['axes'])
        elif transform['kind'] == 'slice':
            rank = array.ndim
            starts = transform.get('start', [None] * rank)
            stops = transform.get('stop', [None] * rank)
            steps = transform.get('step', [1] * rank)
            array = array[tuple(slice(a, b, c) for a, b, c in zip(starts, stops, steps))]
        else:
            raise ValueError('unknown input transform')
    if list(array.shape) != spec['shape']:
        raise ValueError(f'transformed input shape {list(array.shape)} != {spec["shape"]}')
    shape = tuple(spec['shape'])
    strides = spec.get('strides')
    if strides is None and spec.get('layout', 'contiguous') == 'strided':
        strides, stride = [], 2
        for dim in reversed(shape):
            strides.append(stride)
            stride *= max(dim, 1)
        strides.reverse()
    offset = spec.get('storage_offset', 0)
    if strides is None and offset:
        strides = [stride // array.itemsize for stride in array.strides]
    if strides is not None:
        span = 1 + sum(max(dim - 1, 0) * stride for dim, stride in zip(shape, strides))
        storage = np.empty(max(offset + span, 1), dtype=array.dtype)
        view = np.ndarray(shape, dtype=array.dtype, buffer=storage, offset=offset * array.itemsize,
                          strides=tuple(s * array.itemsize for s in strides))
        view[...] = array
        array = view
    return array


def make_inputs(case, base_dir=None):
    """Quantize once before device/reference clones, preserving the input tree."""
    rng = np.random.default_rng(case.get('seed', 0))
    def visit(spec):
        kind = spec.get('kind', 'tensor')
        if kind == 'absent':
            return None
        if kind == 'tensor_list':
            return [visit(item) for item in spec['items']]
        if 'source' in spec:
            values = _file_values(spec, base_dir)
        else:
            values = _generated_values(spec, rng)
            dtype = spec['dtype']
            if dtype == 'bfloat16':
                values = quantize_bfloat16(values)
            else:
                target = np.dtype(dtype)
                if case.get('_schema_version', 1) == 2 and target.kind in 'iu' and spec.get('distribution', {}).get('kind') == 'fixed':
                    info = np.iinfo(target)
                    if any(not info.min <= x <= info.max or int(x) != x for x in np.asarray(values).flat):
                        raise ValueError(f'fixed input is outside exact {dtype} range')
                values = np.asarray(values).astype(target)
        return _layout(values, spec)
    return [visit(spec) for spec in case['inputs']]


def clone_inputs(inputs):
    return map_tensors(inputs, _copy_strides)


def reference_inputs(inputs, policy='float64'):
    """Promote quantized values according to policy; integers stay exact."""
    if isinstance(policy, dict):
        policy = policy.get('dtype', 'float64')
    if policy not in ('float64', 'float32', 'preserve'):
        raise ValueError(f'unsupported reference precision {policy!r}')
    def convert(array):
        dtype = array.dtype
        if policy != 'preserve' and dtype.kind == 'f':
            dtype = np.dtype(policy)
        elif policy != 'preserve' and dtype.kind == 'c':
            dtype = np.dtype('complex128' if policy == 'float64' else 'complex64')
        return _copy_strides(array, dtype)
    return map_tensors(inputs, convert)


def tensor_metadata(array, dtype=None):
    array = np.asarray(array)
    return {'shape': list(array.shape), 'dtype': dtype or array.dtype.name,
            'strides': [stride // array.itemsize for stride in array.strides],
            'strides_unit': 'elements', 'storage_offset': storage_offset(array),
            'contiguous': bool(array.flags.c_contiguous)}
