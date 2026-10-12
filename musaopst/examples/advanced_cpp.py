# SPDX-License-Identifier: MulanPSL-2.0
"""MUSA C++ examples with preallocated outputs and compile-once rebind hooks."""
from pathlib import Path


def build_descriptor(case, workdir):
    root = Path(__file__).parent
    return {"sources": [str(root/"advanced_cpp.mu"), str(root/"advanced_cpp_bindings.cpp")],
            "headers": [str(root/name) for name in ("native_kernels.mu", "native_host.hpp", "native_json.hpp")],
            "extra_cflags": ["-O2"], "extra_musa_cflags": ["-O2", "-std=c++17"]}


def _factory(case, inputs, workdir, operation):
    import torch
    from musaopst.errors import UnsupportedError
    from musaopst.backends.gpu import _flat
    from musaopst.backends.build_cache import build_cpp_descriptor, load_cpp_artifact
    major, minor = torch.musa.get_device_capability()
    descriptor = case.get("artifact") or build_descriptor(case, workdir)
    built = build_cpp_descriptor(descriptor, workdir, arch=f"mp_{major}{minor}", isolated=False)
    module = load_cpp_artifact(built)
    state = {}
    def bind(new_inputs, new_case):
        values = list(_flat(new_inputs))
        attrs = new_case["attrs"]
        device = values[0].device if values else torch.device(f"musa:{torch.musa.current_device()}")
        if operation == "fill":
            if values: raise UnsupportedError("fill expects no tensor input")
            shape = tuple(attrs.get("shape", new_case["outputs"][0]["shape"]))
            dtype = getattr(torch, attrs.get("dtype", new_case["outputs"][0]["dtype"]))
            outputs = [torch.empty(shape, device=device, dtype=dtype)]
        else:
            if not values: raise UnsupportedError("example needs a tensor input")
            a = values[0]
            if a.ndim > 8: raise UnsupportedError("C++ examples support rank <= 8")
            if operation != "identity" and a.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise UnsupportedError("C++ numerical examples support float32/float16/bfloat16")
            if operation in ("axpby", "matmul"):
                if len(values) != 2 or values[1].dtype != a.dtype: raise UnsupportedError("two same-dtype inputs required")
                b = values[1]
                if operation == "axpby":
                    if b.shape != a.shape: raise UnsupportedError("axpby needs identical shapes")
                    shapes = [a.shape]
                else:
                    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]: raise UnsupportedError("incompatible matmul shapes")
                    shapes = [(a.shape[0], b.shape[1])]
            elif operation == "sum_max":
                if a.numel() > 65536: raise UnsupportedError("sum_max example supports <=65536 elements")
                shapes = [(), ()]
            elif operation == "softmax":
                if a.ndim != 2 or a.shape[1] > 65536 or attrs.get("dim", -1) not in (-1, 1): raise UnsupportedError("softmax requires rank2 last axis")
                shapes = [a.shape]
            elif operation == "inplace_add":
                if any(n > 1 and stride == 0 for n, stride in zip(a.shape, a.stride())): raise UnsupportedError("cannot mutate overlapping input")
                shapes = None
            else:
                shapes = [a.shape]
            outputs = [a] if shapes is None else [torch.empty(shape, dtype=a.dtype, device=device) for shape in shapes]
        state.update(inputs=values, outputs=outputs, attrs=attrs)
    bind(inputs, case)
    def launch():
        attrs = state["attrs"]
        first = attrs.get("alpha", 1.0) if operation == "axpby" else attrs.get("value", 1.0 if operation == "inplace_add" else 0.0)
        module.run_out(operation, state["inputs"], state["outputs"], first, attrs.get("beta", 1.0))
        return state["outputs"]
    launch.rebind = bind
    launch.runtime_shape_kind = "compiled_kernel_runtime_arguments"
    launch.allocation_included = False
    launch.launch_metadata = {"artifact_id": built["artifact_sha256"], "build": built,
                              "operator": operation, "runtime_shape": True, "output": "preallocated",
                              "implementation": "simple demonstration kernels; not tuned"}
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
        raise UnsupportedError("This C++ TensorList example explicitly supports exactly two tensors")
    return _factory(case, inputs[0], workdir, "axpby")


for _entry in (axpby, sum_max, fill, identity, matmul, softmax, inplace_add, optional_add, tensor_list_sum):
    _entry.build_descriptor = build_descriptor
