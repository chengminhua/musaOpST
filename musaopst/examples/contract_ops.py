# SPDX-License-Identifier: MulanPSL-2.0
"""Real MUSA callable contract examples; importing this module needs no SDK."""


def optional_add(a, b=None):
    return a.clone() if b is None else a + b


def tensor_list_sum(xs):
    if not xs:
        raise ValueError('TensorList must contain at least one tensor')
    result = xs[0].clone()
    for value in xs[1:]:
        result = result + value
    return result


def fill(shape, dtype='float32', value=0.0):
    import torch
    return torch.full(tuple(shape), value, dtype=getattr(torch, dtype),
                      device=f'musa:{torch.musa.current_device()}')


def max_with_index(a, axis=-1):
    import torch
    value, index = torch.max(a, dim=axis)
    return value, index


def positive_coordinates(a):
    import torch
    return torch.nonzero(a > 0, as_tuple=False)


def strided_view(a):
    return a[:, 1::2]


def identity(a):
    return a.clone()


def safe_corrupt(a):
    # Deliberately wrong numerical output, without invalid access or unsafe launch.
    return a.clone() + 0.25


def inplace_add(a, value=1.0):
    return a.add_(value)
