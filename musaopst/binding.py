# SPDX-License-Identifier: MulanPSL-2.0
"""SDK-free tensor-tree traversal and explicit callable argument binding."""
from __future__ import annotations


def tensor_specs(specs):
    result = []
    for spec in specs:
        kind = spec.get('kind', 'tensor')
        if kind == 'tensor_list':
            result.extend(tensor_specs(spec['items']))
        elif kind != 'absent':
            result.append(spec)
    return result


def flatten_tensors(nodes):
    result = []
    for node in nodes:
        if isinstance(node, (list, tuple)):
            result.extend(flatten_tensors(node))
        elif node is not None:
            result.append(node)
    return result


def map_tensors(nodes, fn):
    return [map_tensors(node, fn) if isinstance(node, (list, tuple)) else
            None if node is None else fn(node) for node in nodes]


def map_inputs(nodes, specs, fn):
    if len(nodes) != len(specs):
        raise ValueError('input node count differs from input specification')
    result = []
    for node, spec in zip(nodes, specs):
        kind = spec.get('kind', 'tensor')
        if kind == 'absent':
            if node is not None:
                raise ValueError('absent input must be None')
            result.append(None)
        elif kind == 'tensor_list':
            if not isinstance(node, (list, tuple)):
                raise ValueError('TensorList input must be a list or tuple')
            result.append(map_inputs(node, spec['items'], fn))
        else:
            result.append(fn(node, spec))
    return result


def bind_inputs(nodes, specs, attrs=None, binding='positional'):
    """Bind top-level nodes; TensorList remains one argument and absent passes None."""
    if len(nodes) != len(specs):
        raise ValueError('input node count differs from input specification')
    args, kwargs = [], dict(attrs or {})
    for node, spec in zip(nodes, specs):
        if spec.get('binding', binding) == 'named':
            name = spec['name']
            if name in kwargs:
                raise ValueError(f'input name {name!r} collides with scalar attribute')
            kwargs[name] = node
        else:
            args.append(node)
    return args, kwargs


def tree_description(nodes):
    """JSON-safe shape of an input tree, with stable flattened tensor indices."""
    index = 0
    def visit(value):
        nonlocal index
        if value is None:
            return {'kind': 'absent'}
        if isinstance(value, (list, tuple)):
            return {'kind': 'tensor_list', 'items': [visit(item) for item in value]}
        current = index
        index += 1
        return {'kind': 'tensor', 'index': current}
    return [visit(node) for node in nodes]
