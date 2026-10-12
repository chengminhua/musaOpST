# SPDX-License-Identifier: MulanPSL-2.0
"""Small SDK-free utilities; user plugins are executable trusted code."""
import hashlib
import importlib
import json
import math
from pathlib import Path


def entry_module(entry):
    """Resolve the one renamed built-in module; leave user plugins untouched."""
    module = entry.split(':', 1)[0]
    return {'musaopst.examples.native_v2_ops':
            'musaopst.examples.native_protocol_ops'}.get(module, module)


def load_entry(entry):
    module, name = entry_module(entry), entry.split(':', 1)[1]
    value = importlib.import_module(module)
    for part in name.split('.'):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f'{entry} is not callable')
    return value


def as_outputs(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if hasattr(value, 'item'):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
