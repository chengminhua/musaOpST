# SPDX-License-Identifier: MulanPSL-2.0
"""TileLang vector-add with explicit output supplied to the compiled kernel."""

def vector_add(case, inputs, workdir):
    import torch
    import tilelang
    import tilelang.language as T
    from musaopst.errors import UnsupportedError
    if len(inputs) != 2:
        raise UnsupportedError("TileLang vector_add expects exactly two inputs")
    a, b = inputs
    if a.ndim != 1 or b.shape != a.shape or b.dtype != a.dtype:
        raise UnsupportedError("TileLang vector_add requires same-shape/dtype 1-D inputs")
    if not all(x.is_contiguous() and (x.numel() == 0 or x.stride(0) == 1) for x in inputs):
        raise UnsupportedError("TileLang vector_add supports contiguous inputs only; no implicit copy")
    if a.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise UnsupportedError("TileLang vector_add supports float32, float16, bfloat16")
    if case["attrs"]:
        raise UnsupportedError("TileLang vector_add does not accept scalar attributes")
    n = a.numel()
    dtype = str(a.dtype).replace("torch.", "")
    out = torch.empty_like(a)
    if n:
        @tilelang.jit(out_idx=[], target="musa")
        def make(size: int, scalar_type: str):
            @T.prim_func
            def main(A: T.Tensor((size,), scalar_type), B: T.Tensor((size,), scalar_type),
                     C: T.Tensor((size,), scalar_type)):
                with T.Kernel(T.ceildiv(size, 256), threads=128) as bx:
                    for j in T.Parallel(256):
                        i = bx * 256 + j
                        if i < size:
                            C[i] = A[i] + B[i]
            return main
        compiled = make(n, dtype)
        def launch():
            compiled(a, b, out)
            return out
        launch()
        torch.musa.synchronize()
    else:
        def launch():
            return out
    launch.allocation_included = False
    launch.launch_metadata = {"block_elements": 256, "threads": 128,
                              "grid": [(n + 255) // 256], "output": "preallocated", "jit_materialized_in_prepare": bool(n),
                              "empty_input": "no kernel launch" if n == 0 else False}
    return launch
