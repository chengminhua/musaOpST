# SPDX-License-Identifier: MulanPSL-2.0
"""Compile a real, quantized scalar input into Triton/TileLang device code.

These examples specialize at construction. Changing the scalar requires a new
construction; no runtime scalar pointer/value is passed to either kernel.
"""
from hashlib import sha256
import json
import math
from pathlib import Path

from musaopst.errors import UnsupportedError


def _constant_spec(case):
    specs = case["inputs"]
    if len(specs) != 2 or specs[0].get("constant", False) or not specs[1].get("constant", False):
        raise UnsupportedError("constant_scale requires input[0] runtime and input[1] constant:true")
    if specs[1].get("shape") != [] or specs[0].get("dtype") != specs[1].get("dtype"):
        raise UnsupportedError("constant_scale requires a scalar constant of the vector dtype")
    if specs[0].get("dtype") not in ("float32", "float16", "bfloat16"):
        raise UnsupportedError("constant_scale supports float32/float16/bfloat16")
    return specs[1]


def _extract(case, inputs):
    import torch
    spec = _constant_spec(case)
    if len(inputs) != 2 or inputs[0].ndim != 1 or not inputs[0].is_contiguous() or inputs[1].ndim != 0:
        raise UnsupportedError("constant_scale requires a contiguous vector and a scalar")
    if any(value.device.type != "musa" for value in inputs):
        raise UnsupportedError("constant_scale requires actual MUSA tensors")
    # The adapter has already quantized and uploaded both actual inputs. This
    # build-stage readback consumes the actual scalar, never its JSON literal.
    value = float(inputs[1].detach().to(device="cpu", dtype=torch.float32).item())
    if not math.isfinite(value):
        raise UnsupportedError("constant_scale currently embeds finite scalar constants only")
    record = {"input_index": 1, "dtype": spec["dtype"], "quantized_value": value,
              "value_hex": value.hex(), "source": "actual_quantized_device_input",
              "runtime_kernel_argument": False}
    record["content_sha256"] = sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    output = torch.empty(inputs[0].shape, dtype=inputs[0].dtype, device=inputs[0].device)
    return inputs[0], output, value, record


def triton_scale(case, inputs, workdir):
    global tl
    import torch
    import triton
    import triton.language as tl
    vector, output, value, constant = _extract(case, inputs)
    count = vector.numel()
    if not count:
        raise UnsupportedError("constant_scale compile evidence requires a nonempty vector")

    @triton.jit
    def kernel(A, Y, N: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
        indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(A + indices, indices < N, other=0).to(tl.float32)
        tl.store(Y + indices, values * SCALE, indices < N)

    def invoke():
        return kernel[(triton.cdiv(count, 256),)](vector, output, count, value, 256)

    compiled = invoke()
    torch.musa.synchronize()
    artifact = getattr(compiled, "hash", None)
    if artifact is None:
        raise UnsupportedError("Installed Triton exposes no compiled kernel hash for constant verification")
    def launch():
        invoke()
        return output
    launch.constants_supported = True
    launch.allocation_included = False
    launch.runtime_shape_kind = "specialize_per_shape_and_constant"
    launch.launch_metadata = {"operator": "constant_scale", "compiled_constants": [constant],
                              "constant_mechanism": "tl.constexpr", "artifact_id": str(artifact),
                              "artifact_kind": "triton_compiled_kernel_hash",
                              "compiled_artifact_ids": [str(artifact)], "output": "preallocated",
                              "kernel_runtime_arguments": ["vector_pointer", "output_pointer"]}
    return launch


def tilelang_scale(case, inputs, workdir):
    import torch
    import tilelang
    import tilelang.language as T
    vector, output, value, constant = _extract(case, inputs)
    count = vector.numel()
    if not count:
        raise UnsupportedError("constant_scale compile evidence requires a nonempty vector")
    dtype = str(vector.dtype).replace("torch.", "")

    @tilelang.jit(out_idx=[], target="musa")
    def make(N: int, SCALE: float, D: str):
        @T.prim_func
        def main(A: T.Tensor((N,), D), Y: T.Tensor((N,), D)):
            with T.Kernel(T.ceildiv(N, 128), threads=128) as block:
                for offset in T.Parallel(128):
                    index = block * 128 + offset
                    if index < N:
                        Y[index] = T.cast(A[index], "float32") * SCALE
        return main

    compiled = make(count, value, dtype)
    source = compiled.get_kernel_source() if hasattr(compiled, "get_kernel_source") else None
    if not isinstance(source, str):
        raise UnsupportedError("Installed TileLang exposes no generated source for constant verification")
    source_path = Path(workdir) / "constant_kernel.mu"
    source_path.write_text(source)
    def launch():
        compiled(vector, output)
        return output
    launch.constants_supported = True
    launch.allocation_included = False
    launch.runtime_shape_kind = "specialize_per_shape_and_constant"
    launch.launch_metadata = {"operator": "constant_scale", "compiled_constants": [constant],
                              "constant_mechanism": "TileLang TIR embedded literal",
                              "artifact_id": sha256(source.encode()).hexdigest(),
                              "artifact_kind": "generated_device_source_sha256",
                              "generated_source": str(source_path), "output": "preallocated",
                              "kernel_runtime_arguments": ["vector_pointer", "output_pointer"]}
    launch()
    torch.musa.synchronize()
    return launch


def unsupported(*args, **kwargs):
    raise UnsupportedError("This backend has no constant_scale compile-time input adapter; fixed runtime values are not compiled constants")


# The ordinary torch backend otherwise treats an entry as an eager callable.
unsupported.prepare = unsupported
