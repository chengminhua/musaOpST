# SPDX-License-Identifier: MulanPSL-2.0
"""Additional eager MUSA examples. Imports are deferred to device execution."""

def axpby(a, b, alpha=1.0, beta=1.0):
    return (a.float() * alpha + b.float() * beta).to(a.dtype)


def sum_max(a):
    import torch
    flat = a.float().reshape(-1)
    maximum = torch.amax(flat) if flat.numel() else torch.zeros((), dtype=torch.float32, device=a.device)
    return flat.sum().to(a.dtype), maximum.to(a.dtype)


def identity(a):
    from musaopst.errors import UnsupportedError
    if str(a.dtype) in ("torch.uint16", "torch.uint32"):
        raise UnsupportedError("This torch_musa identity entry uses clone; installed MUSA clone rejects uint16/uint32 (SetMUTensorDType Unsupported)")
    return a.clone()


def matmul(a, b):
    import torch
    from musaopst.errors import UnsupportedError
    mudnn = getattr(torch.backends, "mudnn", None)
    if mudnn is None or not hasattr(mudnn, "flags"):
        raise UnsupportedError("This strict FP32 matmul example needs the installed muDNN allow_tf32 control")
    # Documented torch_musa control. Keep full input mantissas for the common
    # accuracy policy; do not loosen tolerances to hide TF32 dispatch.
    mublas = getattr(getattr(torch.backends, "musa", None), "matmul", None)
    has_mublas_flag = mublas is not None and hasattr(mublas, "allow_tf32")
    previous_mublas = mublas.allow_tf32 if has_mublas_flag else None
    try:
        if has_mublas_flag:
            mublas.allow_tf32 = False
        with mudnn.flags(allow_tf32=False):
            return (a.float() @ b.float()).to(a.dtype)
    finally:
        if has_mublas_flag:
            mublas.allow_tf32 = previous_mublas


def softmax(a, dim=-1):
    import torch
    return torch.softmax(a.float(), dim=dim).to(a.dtype)


def fill(shape, value=0.0, dtype="float32"):
    import torch
    return torch.full(tuple(shape), value, dtype=getattr(torch, dtype), device=f"musa:{torch.musa.current_device()}")


def inplace_add(a, value=1.0):
    return a.add_(value)


def optional_add(a, b=None):
    return a.clone() if b is None else a + b


def tensor_list_sum(xs):
    if not xs:
        raise ValueError("tensor_list_sum needs at least one input tensor")
    result = xs[0].float()
    for value in xs[1:]:
        result = result + value.float()
    return result.to(xs[0].dtype)


def nonzero(a):
    import torch
    return torch.nonzero(a, as_tuple=False)
