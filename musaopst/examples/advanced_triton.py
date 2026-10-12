# SPDX-License-Identifier: MulanPSL-2.0
"""Simple MUSA Triton examples with runtime sizes and actual artifact tracking."""

def _factory(case, inputs, workdir, operation):
    global tl
    import torch
    import triton
    import triton.language as tl
    from musaopst.backends.gpu import _flat
    from musaopst.errors import UnsupportedError

    @triton.jit(do_not_specialize=["N", "C", "A0", "A1", "B0", "B1", "ALPHA", "BETA"])
    def pointwise(A, B, Y, N, C, A0, A1, B0, B1, ALPHA, BETA, OP: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if OP == 3:
            value = tl.full((BLOCK,), 0, tl.float32) + ALPHA
        else:
            ai = i // C * A0 + i % C * A1
            value = tl.load(A + ai, i < N, other=0)
            if OP == 1:
                bi = i // C * B0 + i % C * B1
                second = tl.load(B + bi, i < N, other=0).to(tl.float32)
                value = value.to(tl.float32) * ALPHA + second * BETA
            elif OP == 2:
                value = value.to(tl.float32) + ALPHA
        if OP == 2:
            tl.store(Y + ai, value, i < N)
        else:
            tl.store(Y + i, value, i < N)

    @triton.jit(do_not_specialize=["N", "C", "A0", "A1"])
    def stats(A, SUM, MAX, N, C, A0, A1, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        offsets = i // C * A0 + i % C * A1
        value = tl.load(A + offsets, i < N, other=0).to(tl.float32)
        total = tl.sum(value, axis=0)
        maximum = tl.max(tl.where(i < N, value, -float("inf")), axis=0)
        tl.store(SUM, total)
        tl.store(MAX, tl.where(N > 0, maximum, 0.0))

    @triton.jit(do_not_specialize=["M", "N", "S0", "S1"])
    def softmax_kernel(A, Y, M, N, S0, S1, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        value = tl.load(A + row*S0 + col*S1, (row < M) & (col < N), other=-float("inf")).to(tl.float32)
        exponent = tl.exp(value - tl.max(value, axis=0))
        output = exponent / tl.sum(exponent, axis=0)
        tl.store(Y + row*N + col, output, (row < M) & (col < N))

    @triton.jit(do_not_specialize=["M", "N", "K", "A0", "A1", "B0", "B1"])
    def mm(A, B, Y, M, N, K, A0, A1, B0, B1, BLOCK: tl.constexpr):
        flat = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        row, col = flat // N, flat % N
        total = tl.full((BLOCK,), 0, tl.float32)
        for k in range(K):
            a = tl.load(A+row*A0+k*A1, row < M, other=0).to(tl.float32)
            b = tl.load(B+k*B0+col*B1, row < M, other=0).to(tl.float32)
            total += a*b
        tl.store(Y+flat, total, flat < M*N)

    state = {"artifact_ids": set()}
    initial_dtype = None
    def layout(tensor):
        if tensor.ndim > 2: raise UnsupportedError("Triton examples support rank <= 2")
        if tensor.ndim == 2: return max(tensor.shape[1], 1), tensor.stride(0), tensor.stride(1)
        return max(tensor.numel(), 1), 0, tensor.stride(0) if tensor.ndim else 1
    def bind(new_inputs, new_case):
        nonlocal initial_dtype
        values = list(_flat(new_inputs))
        attrs = new_case["attrs"]
        device = values[0].device if values else torch.device(f"musa:{torch.musa.current_device()}")
        if operation == "fill":
            if values: raise UnsupportedError("fill requires no tensor input")
            dtype = getattr(torch, attrs.get("dtype", new_case["outputs"][0]["dtype"]))
            shape = attrs.get("shape", new_case["outputs"][0]["shape"])
            outputs = [torch.empty(shape, dtype=dtype, device=device)]
        else:
            if not values: raise UnsupportedError("example requires tensor input")
            a = values[0]; layout(a); dtype = a.dtype
            if str(dtype) in ("torch.complex64", "torch.complex128"):
                raise UnsupportedError("Installed Triton MUSA cannot lower complex tensor pointers; this entry does not reinterpret complex storage as bytes")
            if operation != "identity" and dtype not in (torch.float32, torch.float16, torch.bfloat16): raise UnsupportedError("numerical examples support float32/float16/bfloat16")
            if operation in ("axpby", "matmul"):
                if len(values) != 2 or values[1].dtype != dtype: raise UnsupportedError("two same-dtype inputs required")
                b = values[1]; layout(b)
                if operation == "axpby":
                    if a.shape != b.shape: raise UnsupportedError("axpby requires identical shapes")
                    shapes = [a.shape]
                else:
                    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]: raise UnsupportedError("incompatible matrix shapes")
                    if a.shape[1] > 1024: raise UnsupportedError("demonstration matmul supports K<=1024")
                    shapes = [(a.shape[0], b.shape[1])]
            elif operation == "sum_max":
                if a.numel() > 4096: raise UnsupportedError("Triton sum_max example supports <=4096 elements")
                shapes = [(), ()]
            elif operation == "softmax":
                if a.ndim != 2 or a.shape[1] > 4096 or attrs.get("dim", -1) not in (-1, 1): raise UnsupportedError("Triton softmax requires rank2, last axis, cols<=4096")
                shapes = [a.shape]
            elif operation == "inplace_add":
                if any(n>1 and s==0 for n,s in zip(a.shape,a.stride())): raise UnsupportedError("Cannot mutate overlapping input")
                shapes = None
            else: shapes = [a.shape]
            outputs = [a] if shapes is None else [torch.empty(s, dtype=dtype, device=device) for s in shapes]
        if initial_dtype is not None and dtype != initial_dtype:
            raise UnsupportedError("Compile-once Triton rebinding cannot change dtype")
        initial_dtype = dtype
        state.update(inputs=values, outputs=outputs, attrs=attrs)
    bind(inputs, case)
    def launch():
        values, out, attrs = state["inputs"], state["outputs"], state["attrs"]
        a = values[0] if values else out[0]
        n = out[0].numel() if operation == "fill" else a.numel()
        c, s0, s1 = layout(a)
        compiled = None
        if operation in ("identity", "axpby", "inplace_add", "fill") and n:
            b = values[1] if operation == "axpby" else a
            _, b0, b1 = layout(b)
            op = {"identity":0,"axpby":1,"inplace_add":2,"fill":3}[operation]
            alpha = attrs.get("alpha", 1.0) if operation == "axpby" else attrs.get("value", 1.0 if operation == "inplace_add" else 0.0)
            compiled = pointwise[(triton.cdiv(n,256),)](a,b,out[0],n,c,s0,s1,b0,b1,float(alpha),float(attrs.get("beta",1.0)),op,256)
        elif operation == "sum_max":
            compiled = stats[(1,)](a,out[0],out[1],n,c,s0,s1,4096)
        elif operation == "softmax" and n:
            compiled = softmax_kernel[(a.shape[0],)](a,out[0],a.shape[0],a.shape[1],s0,s1,4096,num_warps=4)
        elif operation == "matmul" and out[0].numel():
            b = values[1]
            compiled = mm[(triton.cdiv(out[0].numel(),256),)](a,b,out[0],a.shape[0],b.shape[1],a.shape[1],a.stride(0),a.stride(1),b.stride(0),b.stride(1),256)
        if compiled is not None:
            artifact = getattr(compiled, "hash", None)
            if artifact is not None:
                state["artifact_ids"].add(str(artifact))
                launch.launch_metadata["compiled_artifact_ids"] = sorted(state["artifact_ids"])
                launch.launch_metadata["observed_compiled_artifacts"] = len(state["artifact_ids"])
            else:
                launch.launch_metadata["compiled_artifact_ids"] = "unknown"
        return out
    launch.rebind = bind
    launch.runtime_shape_kind = "runtime_arguments; verify compiled_artifact_ids for every binding"
    launch.allocation_included = False
    launch.launch_metadata = {"operator":operation,"output":"preallocated","compiled_artifact_ids":[],
                              "implementation":"simple demonstration kernels; not tuned"}
    launch()  # Materialize JIT during the build stage, never in measured samples.
    torch.musa.synchronize()
    return launch


def axpby(case, inputs, workdir): return _factory(case, inputs, workdir, "axpby")
def sum_max(case, inputs, workdir): return _factory(case, inputs, workdir, "sum_max")
def fill(case, inputs, workdir): return _factory(case, inputs, workdir, "fill")
def identity(case, inputs, workdir): return _factory(case, inputs, workdir, "identity")
def matmul(case, inputs, workdir): return _factory(case, inputs, workdir, "matmul")
def softmax(case, inputs, workdir): return _factory(case, inputs, workdir, "softmax")
def inplace_add(case, inputs, workdir): return _factory(case, inputs, workdir, "inplace_add")


def optional_add(case, inputs, workdir):
    from musaopst.backends.gpu import _flat
    from musaopst.errors import UnsupportedError
    if len(inputs) != 2 or inputs[0] is None:
        raise UnsupportedError("optional_add expects tensor a and tensor/absent b")
    return _factory(case, list(_flat(inputs)), workdir, "identity" if inputs[1] is None else "axpby")


def tensor_list_sum(case, inputs, workdir):
    from musaopst.errors import UnsupportedError
    if len(inputs) != 1 or not isinstance(inputs[0], list) or len(inputs[0]) != 2:
        raise UnsupportedError("This Triton TensorList example explicitly supports exactly two tensors")
    return _factory(case, inputs[0], workdir, "axpby")
