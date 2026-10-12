# SPDX-License-Identifier: MulanPSL-2.0
"""Stride-aware MUSA Triton softmax. Compiled only inside the GPU worker."""

def softmax(case, inputs, workdir):
    # Triton 3.2 resolves language symbols through function globals when compiling.
    # Populate lazily so offline CLI/module imports still require no GPU SDK.
    global tl
    import torch
    import triton
    import triton.language as tl
    from musaopst.errors import UnsupportedError
    if len(inputs) != 1 or inputs[0].ndim != 2:
        raise UnsupportedError("Triton example supports one 2-D input")
    x = inputs[0]
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise UnsupportedError("Triton softmax supports float32, float16, bfloat16")
    attrs = dict(case["attrs"])
    if attrs.pop("dim", -1) not in (-1, 1) or attrs:
        raise UnsupportedError("Triton example only supports dim=-1/1 and no other attributes")
    rows, cols = x.shape
    block = triton.next_power_of_2(max(cols, 1))
    if block > 65536:
        raise UnsupportedError("Triton example supports at most 65536 columns")
    y = torch.empty((rows, cols), dtype=x.dtype, device=x.device)

    @triton.jit
    def kernel(X, Y, N: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr, B: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, B)
        values = tl.load(X + row * S0 + col * S1, col < N, other=-float("inf")).to(tl.float32)
        ex = tl.exp(values - tl.max(values, axis=0))
        result = ex / tl.sum(ex, axis=0)
        tl.store(Y + row * N + col, result, col < N)

    def launch():
        if rows and cols:
            kernel[(rows,)](x, y, cols, x.stride(0), x.stride(1), block, num_warps=4)
        return y
    # Materialize JIT before run()/timing; errors are build failures, not accuracy.
    launch()
    torch.musa.synchronize()
    launch.allocation_included = False
    launch.launch_metadata = {"grid": [rows], "block_elements": block, "num_warps": 4,
                              "input_strides": list(x.stride()), "output": "preallocated", "jit_materialized_in_prepare": bool(rows and cols),
                              "empty_input": "no kernel launch" if not rows or not cols else False}
    return launch
