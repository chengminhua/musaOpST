# SPDX-License-Identifier: MulanPSL-2.0
class BuildError(RuntimeError):
    pass

class EnvironmentError(RuntimeError):
    pass

class UnsupportedError(RuntimeError):
    pass


class DeviceStateUnknownError(RuntimeError):
    """A device-owning process was lost without trustworthy completion evidence."""


def is_fatal_device_error(error):
    """Recognize severe runtime faults through wrappers, without importing GPU SDKs.

    Missing SDKs/devices, unsupported operators, compiler errors and OOM alone do
    not establish a poisoned device context. This is a conservative text-based
    fallback because backends expose different exception classes/runtime wrappers.
    """
    severe = (
        "illegal memory access", "illegal address", "device-side assert",
        "device side assert", "device lost", "device has been lost",
        "device is unavailable", "unspecified launch failure", "launch timeout",
        "launch timed out", "hardware error", "uncorrectable ecc", "ecc uncorrectable",
        "musaerrorillegaladdress", "musaerrorassert", "musaerrorlaunchfailure",
        "musaerrorlaunchtimeout", "musaerroreccuncorrectable",
    )
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, DeviceStateUnknownError):
            return True
        if any(marker in str(current).lower() for marker in severe):
            return True
        pending.extend([getattr(current, "__cause__", None), getattr(current, "__context__", None)])
    return False
