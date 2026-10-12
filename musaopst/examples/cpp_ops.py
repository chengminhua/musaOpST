# SPDX-License-Identifier: MulanPSL-2.0
"""Compiled MUSA extension using an explicit Tensor ABI and current MUSA stream."""
from hashlib import sha256
from pathlib import Path


def vector_add(case, inputs, workdir):
    import torch
    from torch_musa.utils.musa_extension import load
    from musaopst.errors import UnsupportedError
    if len(inputs) != 2:
        raise UnsupportedError("MUSA C++ vector_add expects two inputs")
    a, b = inputs
    if a.ndim != 1 or b.shape != a.shape or b.dtype != a.dtype:
        raise UnsupportedError("MUSA C++ vector_add requires same-shape/dtype 1-D inputs")
    if any(not x.is_contiguous() or (x.numel() and x.stride(0) != 1) for x in inputs):
        raise UnsupportedError("MUSA C++ example supports contiguous inputs only; no implicit copy")
    if a.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise UnsupportedError("MUSA C++ example supports float32, float16, bfloat16")
    if case["attrs"]:
        raise UnsupportedError("MUSA C++ example does not accept scalar attributes")
    source = Path(__file__).with_name("cpp_vector_add.mu")
    bindings = source.with_name("cpp_bindings.cpp")
    digest = sha256(source.read_bytes() + bindings.read_bytes()).hexdigest()[:12]
    build = Path(workdir).parent / ".build-cache" / ("cpp_" + digest)
    build.mkdir(parents=True, exist_ok=True)
    module = load(name="musaopst_vadd_" + digest, sources=[str(source), str(bindings)],
                  build_directory=str(build), extra_cflags=["-O2"],
                  extra_musa_cflags=["-O2"], with_musa=True, verbose=True)
    out = torch.empty_like(a)
    def launch():
        module.vector_add_out(a, b, out)
        return out
    launch.allocation_included = False
    launch.launch_metadata = {"source_sha256": sha256(source.read_bytes()).hexdigest(),
                              "bindings_sha256": sha256(bindings.read_bytes()).hexdigest(),
                              "block_threads": 256, "grid": [(a.numel() + 255) // 256],
                              "stream": "c10::musa::getCurrentMUSAStream()", "output": "preallocated",
                              "empty_input": "no kernel launch" if not a.numel() else False}
    return launch
