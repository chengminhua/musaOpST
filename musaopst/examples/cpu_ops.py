# SPDX-License-Identifier: MulanPSL-2.0
"""Explicit offline fixtures. These never count as GPU acceptance."""
import time
import numpy as np


def add(a, b, **attrs):
    return a + b


def identity(x, **attrs):
    return x.copy()


def pair(a, b, **attrs):
    return [a + b, a - b]


def wrong(a, b, **attrs):
    return a + b + 1


def wrong_shape(a, b, **attrs):
    return (a + b).reshape(-1)[:1]


def wrong_dtype(a, b, **attrs):
    return (a + b).astype(np.float64)


def raises(*args, **attrs):
    raise RuntimeError('controlled offline fixture failure')


def slow(*args, delay_s=5, **attrs):
    time.sleep(delay_s)
    return args[0].copy()


def inplace(a, b, **attrs):
    a += b
    return a
