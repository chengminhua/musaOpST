# SPDX-License-Identifier: MulanPSL-2.0
"""Generic native descriptors; the host helper owns data/Events/synchronization."""
from pathlib import Path


def _descriptor(operator):
    return {"protocol": "musaopst_native_v2", "operator": operator,
            "source": str(Path(__file__).with_name("native_kernels.mu")),
            "compile_flags": ["-O2", "-std=c++17"]}


def axpby(case, inputs, workdir): return _descriptor("axpby")
def sum_max(case, inputs, workdir): return _descriptor("sum_max")
def fill(case, inputs, workdir): return _descriptor("fill")
def identity(case, inputs, workdir): return _descriptor("identity")
def matmul(case, inputs, workdir): return _descriptor("matmul")
def softmax(case, inputs, workdir): return _descriptor("softmax")
def inplace_add(case, inputs, workdir): return _descriptor("inplace_add")


def optional_add(case, inputs, workdir):
    from musaopst.errors import UnsupportedError
    if len(inputs) != 2 or inputs[0] is None:
        raise UnsupportedError("optional_add expects tensor a and tensor/absent b")
    return _descriptor("identity" if inputs[1] is None else "axpby")


def tensor_list_sum(case, inputs, workdir):
    from musaopst.errors import UnsupportedError
    if len(inputs) != 1 or not isinstance(inputs[0], list) or len(inputs[0]) != 2:
        raise UnsupportedError("This native TensorList example explicitly supports exactly two tensors")
    return _descriptor("axpby")
