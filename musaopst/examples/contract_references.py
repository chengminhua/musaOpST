# SPDX-License-Identifier: MulanPSL-2.0
"""Independent NumPy references for structured MUSA contract demonstrations."""
import numpy as np


def optional_add(a, b=None):
    return np.array(a, copy=True) if b is None else np.add(a, b)


def tensor_list_sum(xs):
    if not xs:
        raise ValueError('TensorList must contain at least one tensor')
    return np.sum(np.stack(xs, axis=0), axis=0)


def fill(shape, dtype='float32', value=0.0):
    # Floating reference values are computed independently in float64.
    target = np.dtype('float64' if dtype in ('float16','bfloat16','float32','float64') else dtype)
    result = np.empty(tuple(shape), dtype=target)
    result[...] = value
    return result


def max_with_index(a, axis=-1):
    return np.max(a, axis=axis), np.argmax(a, axis=axis).astype(np.int64)


def positive_coordinates(a):
    # Distinct implementation: collect each axis' nonzero coordinates then stack.
    return np.stack(np.nonzero(np.greater(a, 0)), axis=1).astype(np.int64)


def strided_view(a):
    selected = np.arange(1, a.shape[1], 2)
    return np.take(a, selected, axis=1)


def identity(a):
    return np.array(a, copy=True)


def inplace_add(a, value=1.0):
    return np.add(a, value)
