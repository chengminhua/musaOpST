# SPDX-License-Identifier: MulanPSL-2.0
"""torch_musa callable examples. Importing this module requires no torch."""

def softmax(x, dim=-1):
    import torch
    return torch.softmax(x, dim=dim)


def _prepare_softmax(case, inputs, workdir, corrupt=False):
    import torch
    from musaopst.errors import UnsupportedError
    if len(inputs) != 1 or inputs[0].ndim != 2:
        raise UnsupportedError("torch softmax example requires one 2-D input")
    x = inputs[0]
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise UnsupportedError("torch softmax example supports float32, float16, bfloat16")
    attrs = dict(case["attrs"])
    dim = attrs.pop("dim", -1)
    if dim not in (-1, 1) or attrs:
        raise UnsupportedError("torch softmax example supports dim=-1/1 and no other attributes")
    out = torch.empty(tuple(x.shape), device=x.device, dtype=x.dtype)
    def launch():
        torch.softmax(x, dim=dim, out=out)
        if corrupt and out.numel():
            # Safe fault injection: numerical perturbation, no invalid memory access.
            out.add_(0.25)
        return out
    launch.allocation_included = False
    launch.launch_metadata = {"operator": "torch.softmax", "dim": dim,
                              "output": "preallocated", "safe_corruption": corrupt}
    return launch


softmax.prepare = _prepare_softmax


def softmax_corrupt(x, dim=-1):
    return softmax(x, dim=dim) + 0.25


softmax_corrupt.prepare = lambda case, inputs, workdir: _prepare_softmax(case, inputs, workdir, True)
