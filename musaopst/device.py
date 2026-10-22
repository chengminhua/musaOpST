# SPDX-License-Identifier: MulanPSL-2.0
"""Parent-side `--device` resolution.

`--device` keeps `torch.musa.set_device` semantics: it is an ordinal inside the
devices this process can see, not a physical GPU number. Container visibility
(`MUSA_VISIBLE_DEVICES` / `MTHREADS_VISIBLE_DEVICES`) renumbers the list, so a
physical id only reaches a worker through an explicit mapping. This module
performs that mapping once in the supervising parent, before any worker starts,
and reports an out-of-range request as a usage error instead of a broken device.

The parent only queries `torch.musa.device_count()`; it never allocates device
memory or launches a kernel. Without a MUSA SDK the selection is returned
unverified so pure CPU-platform runs keep working, and the worker text is
reclassified later by `is_invalid_ordinal`.
"""
from __future__ import annotations

from dataclasses import dataclass

from .device_identity import visibility_spec


class DeviceSelectionError(ValueError):
    """The requested ordinal cannot address a visible MUSA device (usage error)."""


@dataclass(frozen=True)
class DeviceSelection:
    requested: int
    ordinal: int
    physical_id: int | None
    visible_devices: list[int] | str | None
    method: str
    device_count: int | None = None
    visible_variable: str | None = None
    visible_setting: str | None = None
    visible_count: int | None = None
    note: str | None = None

    def as_dict(self):
        visible = list(self.visible_devices) if isinstance(self.visible_devices, list) else self.visible_devices
        return {'requested': self.requested, 'ordinal': self.ordinal,
                'physical_id': self.physical_id, 'visible_devices': visible,
                'method': self.method, 'device_count': self.device_count,
                'visible_variable': self.visible_variable,
                'visible_setting': self.visible_setting,
                'visible_count': self.visible_count, 'note': self.note}


def probe_device_count():
    """Logical device count of a fresh process; None when the SDK is unavailable."""
    try:
        import torch
        import torch_musa  # noqa: F401  registers the PrivateUse1 backend
        return int(torch.musa.device_count())
    except Exception:
        return None


def ordinal_hint(requested):
    return f'若想用物理 GPU {requested}，请设 MUSA_VISIBLE_DEVICES={requested} 并传 --device 0。'


def _range_error(requested, spec, visible_count, count):
    raw = spec['raw']
    count_text = count if count is not None else '不可用'
    if spec['indices'] is not None:
        visible = (f"{spec['variable']}={raw!r}，可见列表长度 {visible_count}，"
                   f"torch.musa.device_count()={count_text}")
        limit = visible_count
    elif raw is not None:
        visible = f"{spec['variable']}={raw!r}，torch.musa.device_count()={count_text}"
        limit = count
    else:
        visible = f'MUSA_VISIBLE_DEVICES 未设置，torch.musa.device_count()={count_text}'
        limit = count
    available = f'可用序号 0..{limit - 1}' if limit else '没有可用序号'
    return DeviceSelectionError(
        f'--device {requested} 越界：{visible}；{available}。\n'
        f'--device 是可见设备列表内的序号（torch.musa.set_device 语义），不是物理 GPU 编号。\n'
        f'{ordinal_hint(requested)}')


def resolve_device(requested, *, device_count=None, environ=None):
    """Map a requested ordinal to the physical card, rejecting silent mistakes."""
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
        raise DeviceSelectionError(f'--device 必须是非负整数，收到 {requested!r}。')
    spec = visibility_spec(environ)
    count = device_count if device_count is not None else probe_device_count()
    raw, indices = spec['raw'], spec['indices']
    if indices is not None:
        visible_count = len(indices)
        if requested >= visible_count:
            raise _range_error(requested, spec, visible_count, count)
        physical = indices[requested]
        note = None
        # Order-sensitive visibility (V="1,2,0") is valid but surprising: the
        # ordinal still wins, so say out loud which physical card was chosen.
        if physical != requested and requested in indices:
            note = (f'--device {requested} 也出现在可见列表 {raw} 中，但仍按序号解释为物理 GPU '
                    f'{physical}；若要使用物理 GPU {requested}，请改传 --device {indices.index(requested)}。')
        return DeviceSelection(requested=requested, ordinal=requested, physical_id=physical,
                               visible_devices=list(indices), method='visible-ordinal',
                               device_count=count, visible_variable=spec['variable'],
                               visible_setting=raw, visible_count=visible_count, note=note)
    if raw is not None:
        # UUID visibility has no host-side physical index we can state.
        if count is not None and requested >= count:
            raise _range_error(requested, spec, count, count)
        return DeviceSelection(requested=requested, ordinal=requested, physical_id=None,
                               visible_devices=raw, method='visible-ordinal-unknown-physical',
                               device_count=count, visible_variable=spec['variable'],
                               visible_setting=raw, visible_count=count)
    if count is not None and requested >= count:
        raise _range_error(requested, spec, count, count)
    return DeviceSelection(requested=requested, ordinal=requested,
                           physical_id=requested if count is not None else None,
                           visible_devices=None,
                           method='ordinal' if count is not None else 'ordinal-unverified',
                           device_count=count)


def format_resolution(resolution):
    """One-line `--device N → 物理 GPU P (...)` rendering from a plain dict."""
    requested = resolution.get('requested')
    ordinal = resolution.get('ordinal')
    physical = resolution.get('physical_id')
    raw = resolution.get('visible_setting')
    variable = resolution.get('visible_variable') or 'MUSA_VISIBLE_DEVICES'
    method = resolution.get('method')
    if method and method.startswith('visible'):
        count = resolution.get('visible_count', resolution.get('device_count'))
    else:
        count = resolution.get('device_count')
    where = '物理 GPU 未知' if physical is None else f'物理 GPU {physical}'
    visible = f'{variable}={raw}' if raw is not None else 'MUSA_VISIBLE_DEVICES 未设'
    if ordinal is not None and ordinal != requested:
        return f'--device {requested} → 序号 {ordinal} → {where} ({visible}, count={count})'
    return f'--device {requested} → {where} ({visible}, count={count})'


def info_line(selection):
    return 'device: ' + format_resolution(selection.as_dict())


def announce(selection):
    """Print the mapping (and any ambiguity note) in the supervising parent."""
    print(info_line(selection), flush=True)
    if selection.note:
        print('device: ' + selection.note, flush=True)


def is_invalid_ordinal(text):
    lowered = str(text).lower()
    return 'invalid device ordinal' in lowered or 'invalid ordinal' in lowered


def ordinal_error_message(message, requested):
    return (f'{message}\n--device 是可见设备列表内的序号，不是物理 GPU 编号。'
            f'{ordinal_hint(requested)} 这是参数或配置问题，不是硬件故障。')
