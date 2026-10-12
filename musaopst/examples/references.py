# SPDX-License-Identifier: MulanPSL-2.0
"""Independent CPU/NumPy references, never call a device implementation."""
import numpy as np


def add(a, b, **attrs):
    return np.add(a, b)


def softmax(x, dim=-1, **attrs):
    if x.size == 0:
        return np.empty(x.shape, dtype=np.float64)
    shifted = x - np.max(x, axis=dim, keepdims=True)
    ex = np.exp(shifted)
    return ex / np.sum(ex, axis=dim, keepdims=True)


def identity(x, **attrs):
    return x.copy()


def pair(a, b, **attrs):
    return [a + b, a - b]
