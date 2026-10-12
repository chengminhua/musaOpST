# SPDX-License-Identifier: MulanPSL-2.0
"""Independent NumPy references for advanced device examples."""
import numpy as np


def axpby(a, b, alpha=1.0, beta=1.0): return a * alpha + b * beta
def sum_max(a): return np.asarray(a.sum()), np.asarray(a.max() if a.size else 0, dtype=a.dtype)
def identity(a): return a.copy()
def matmul(a, b): return np.matmul(a, b)
def fill(shape, value=0.0, dtype="float32"): return np.full(tuple(shape), value, dtype=np.float64)
def inplace_add(a, value=1.0): return a + value
def optional_add(a, b=None): return a.copy() if b is None else a + b
def tensor_list_sum(xs): return np.sum(np.stack(xs), axis=0)
def nonzero(a): return np.argwhere(a != 0).astype(np.int64)
def softmax(a, dim=-1):
    if a.size == 0: return np.empty_like(a)
    exponent = np.exp(a - np.max(a, axis=dim, keepdims=True))
    return exponent / exponent.sum(axis=dim, keepdims=True)
