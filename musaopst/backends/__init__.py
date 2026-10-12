# SPDX-License-Identifier: MulanPSL-2.0
"""Lazy backend dispatch: importing the CLI never imports any GPU package."""
from musaopst.errors import UnsupportedError
from musaopst.util import load_entry


def create_backend(case, inputs, workdir):
    kind = case["backend"]
    if kind == "cpu":
        from .cpu import CPUBackend
        return CPUBackend(case, inputs, workdir)
    if kind == "native":
        from .native import NativeBackend
        return NativeBackend(case, inputs, workdir)
    if kind in ("torch_musa", "musa_cpp", "triton", "tilelang"):
        from .gpu import GPUBackend
        return GPUBackend(case, inputs, workdir)
    raise UnsupportedError(f"Unknown backend: {kind}")
