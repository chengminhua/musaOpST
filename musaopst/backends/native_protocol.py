# SPDX-License-Identifier: MulanPSL-2.0
"""Explicit, general JSON + tensor-storage protocol for native MUSA programs.

The platform never guesses an ABI. A developer supplies a program implementing
musaopst_native_v2, or uses the bundled host helper and implements its hooks.
"""
from __future__ import annotations

import json
import copy
import os
from pathlib import Path
import subprocess
import numpy as np

from musaopst.errors import DeviceStateUnknownError, EnvironmentError, UnsupportedError
from .build_cache import build_native_descriptor
from .gpu import event_statistics, _flat
from . import load_entry

PROTOCOL = "musaopst_native_v2"
DTYPES = {name: np.dtype(name).newbyteorder("<") for name in
          ("float16", "float32", "float64", "int8", "int16", "int32", "int64",
           "uint8", "uint16", "uint32", "uint64", "bool")}
DTYPES["bfloat16"] = np.dtype("<u2")


def encode_storage(array, spec):
    """Return the physical little-endian storage and logical view descriptor."""
    name = spec["dtype"]
    if name not in DTYPES:
        raise UnsupportedError(f"Native file protocol does not implement dtype {name}")
    array = np.asarray(array)
    strides = [s // array.itemsize for s in array.strides]
    if any(s < 0 for s in strides):
        raise UnsupportedError("Native protocol helper supports nonnegative strides only")
    from musaopst.data import storage_offset
    offset = storage_offset(array)
    if offset < 0:
        raise ValueError("Negative storage offset")
    span = offset + 1 + sum(max(dim-1, 0)*s for dim, s in zip(array.shape, strides))
    dtype = DTYPES[name]
    storage = np.zeros(max(span, 1), dtype=dtype)
    view = np.ndarray(array.shape, dtype=dtype, buffer=storage, offset=offset*dtype.itemsize,
                      strides=tuple(s*dtype.itemsize for s in strides))
    if name == "bfloat16":
        bits = np.asarray(array, dtype="<f4").view("<u4")
        view[...] = (bits >> 16).astype("<u2")
    else:
        view[...] = array
    return storage, {"shape": list(array.shape), "dtype": name, "strides": strides,
                     "storage_offset": offset, "storage_bytes": storage.nbytes, "byte_order": "little",
                     "strides_unit": "elements", "contiguous": bool(array.flags.c_contiguous)}


def decode_storage(path, meta):
    """Validate actual metadata before allocating or decoding any output view."""
    shape, strides = meta.get("shape"), meta.get("strides")
    dtype_name = meta.get("dtype")
    if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
        raise RuntimeError("Native output has malformed shape")
    if not isinstance(strides, list) or len(strides) != len(shape) or any(type(s) is not int or s < 0 for s in strides):
        raise RuntimeError("Native output has malformed element strides")
    if dtype_name not in DTYPES:
        raise UnsupportedError(f"Native output encoding unsupported: {dtype_name}")
    dtype = DTYPES[dtype_name]
    offset, size = meta.get("storage_offset", 0), meta.get("storage_bytes")
    if type(offset) is not int or offset < 0 or type(size) is not int or size < 0 or size % dtype.itemsize:
        raise RuntimeError("Native output has malformed offset/storage bytes")
    if meta.get("byte_order", "little") != "little":
        raise UnsupportedError("Native protocol requires little-endian tensor storage")
    if not path.is_file() or path.stat().st_size != size:
        raise RuntimeError("Native output file byte count differs from actual metadata")
    elements = int(np.prod(shape, dtype=object))
    span = offset + (1 + sum(max(n-1, 0)*s for n, s in zip(shape, strides)) if elements else 0)
    if span * dtype.itemsize > size:
        raise RuntimeError("Native output view exceeds its declared storage")
    storage = np.fromfile(path, dtype=dtype)
    array = np.ndarray(tuple(shape), dtype=dtype, buffer=storage,
                       offset=offset*dtype.itemsize, strides=tuple(s*dtype.itemsize for s in strides))
    if dtype_name == "bfloat16":
        # NumPy 1.x promotes a uint32 scalar shifted by a Python int to int64.
        # Keep BOTH operands uint32 so scalar and non-scalar views stay 4 bytes.
        array = np.left_shift(array.astype("<u4"), np.uint32(16), dtype=np.uint32).view("<f4")
    return np.array(array, copy=True)


class NativeProtocolBackend:
    def __init__(self, case, inputs, workdir, descriptor=None):
        self.case, self.inputs, self.workdir = case, inputs, Path(workdir).resolve()
        self.descriptor = descriptor
        self.sequence = 0
        self.binding_sequence = 0
        self.last_outputs, self.last_dtypes, self.actual_metadata = [], [], []
        self.last_result = {}
        self.info = {"backend": "native", "protocol": PROTOCOL, "execution_device": "musa",
                     "repeat_scope": "fresh native process/context per invocation; same compiled artifact",
                     "input_transfer": "explicit little-endian physical storage files"}

    def prepare(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.descriptor = self.descriptor or load_entry(self.case["entry"])(self.case, self.inputs, self.workdir)
        from musaopst.binding import tensor_specs
        if any(spec.get("constant", False) for spec in tensor_specs(self.case["inputs"])) and not self.descriptor.get("constants_supported", False):
            raise UnsupportedError("Native descriptor does not implement compile-time constant inputs")
        arch = self.descriptor.get("arch") or os.environ.get("MUSA_OPST_ARCH")
        if not arch and not self.descriptor.get("binary"):
            try:
                import torch
                import torch_musa
                if not torch.musa.is_available():
                    raise RuntimeError("MUSA device unavailable")
                major, minor = torch.musa.get_device_capability(int(self.case.get("device", 0)))
                arch = f"mp_{major}{minor}"
            except Exception as exc:
                raise EnvironmentError("Cannot discover native target; set MUSA_OPST_ARCH for SDK-only builds: " + str(exc)) from exc
        self.build = build_native_descriptor(self.descriptor, self.workdir, arch=arch, isolated=False)
        self.info.update(build=self.build, build_count=1, runtime_shape_kind="runtime_kernel_arguments_same_binary")
        self.info["launch"] = {"artifact_id": self.build["artifact_sha256"],
                               "build_kind": self.build["build_kind"],
                               "runtime_shape": "runtime arguments; same native binary; fresh native process per call"}
        self._write_inputs()

    def _write_inputs(self):
        from musaopst.binding import tensor_specs
        self.binding_sequence += 1
        self.input_meta = []
        for index, (array, spec) in enumerate(zip(_flat(self.inputs), tensor_specs(self.case["inputs"]))):
            storage, meta = encode_storage(array, spec)
            path = self.workdir / f"native_input_b{self.binding_sequence}_{index}.bin"
            storage.tofile(path)
            meta.update(file=path.name, name=spec.get("name", f"input_{index}"))
            self.input_meta.append(meta)
        indices = iter(range(len(self.input_meta)))
        def structure(node):
            if node is None:
                return None
            if isinstance(node, (list, tuple)):
                return [structure(x) for x in node]
            return next(indices)
        self.input_structure = [structure(node) for node in self.inputs]
        self.info["inputs"] = self.input_meta
        empty = any(np.asarray(x).size == 0 for x in _flat(self.inputs))
        self.info.update(empty_workload=empty, execution_classification="empty_input_contract_only" if empty else "nonempty_input_workload")

    def rebind(self, case, inputs):
        descriptor = load_entry(case["entry"])(case, inputs, self.workdir)
        if descriptor != self.descriptor:
            raise UnsupportedError("Native runtime-shape binding changed its build descriptor")
        self.case, self.inputs = case, inputs
        self._write_inputs()
        self.info["runtime_shape_rebindings"] = self.info.get("runtime_shape_rebindings", 0) + 1

    def _invoke(self, policy=None):
        self.sequence += 1
        prefix = f"native_{self.sequence}"
        request_path = self.workdir / (prefix + "_request.json")
        result_path = self.workdir / (prefix + "_result.json")
        request = {"protocol": PROTOCOL, "operator": self.descriptor.get("operator", "custom"),
                   "device": int(self.case.get("device", 0)), "inputs": self.input_meta,
                   "input_structure": self.input_structure, "attrs": self.case.get("attrs", {}),
                   "output_contracts": self.case["outputs"], "output_prefix": prefix,
                   "timing": policy or {"warmup": 0, "samples": 0, "iterations": 1,
                                         "mode": "batch_event", "stream": "current", "cache": "warm"}}
        request_path.write_text(json.dumps(request, ensure_ascii=False, allow_nan=False))
        process = subprocess.run([self.build["binary"], "--request", str(request_path), "--result", str(result_path)],
                                 cwd=self.workdir, capture_output=True, text=True, check=False)
        (self.workdir / (prefix + ".log")).write_text(process.stdout + process.stderr)
        if process.returncode < 0:
            raise DeviceStateUnknownError(f"Native process exited on signal {-process.returncode}: {process.stderr[-4000:]}")
        if not result_path.is_file():
            raise DeviceStateUnknownError(f"Native process exited {process.returncode} without its structured completion record; device state cannot be certified: {process.stderr[-4000:]}")
        try:
            result = json.loads(result_path.read_text())
        except (TypeError, ValueError) as exc:
            raise DeviceStateUnknownError("Native completion JSON is malformed") from exc
        if result.get("status") == "UNSUPPORTED":
            raise UnsupportedError(result.get("message", "Native program rejected the request"))
        if result.get("status") not in ("PASS", "UNSUPPORTED", "EXECUTION_FAILED"):
            raise DeviceStateUnknownError("Native completion status is invalid")
        if process.returncode and result.get("status") == "PASS":
            raise DeviceStateUnknownError("Native process exited nonzero despite a PASS completion record")
        if process.returncode or result.get("status") != "PASS":
            raise RuntimeError(result.get("message", "Native execution failed"))
        if result.get("protocol") != PROTOCOL or result.get("device_execution") != "musa":
            raise DeviceStateUnknownError("Native protocol/device completion is invalid")
        if result.get("device_index") != int(self.case.get("device", 0)):
            raise RuntimeError("Native program executed on a different device")
        metas = result.get("outputs")
        if not isinstance(metas, list):
            raise RuntimeError("Native output list is missing")
        self.last_outputs, self.last_dtypes = [], []
        for meta in metas:
            if not isinstance(meta, dict) or not isinstance(meta.get("file"), str):
                raise RuntimeError("Native output descriptor is malformed")
            path = (self.workdir / meta["file"]).resolve()
            if path.parent != self.workdir or not path.name.startswith(prefix + "_output_"):
                raise RuntimeError("Native output path is outside this invocation's output prefix")
            self.last_outputs.append(decode_storage(path, meta))
            self.last_dtypes.append(meta["dtype"])
        self.actual_metadata, self.last_result = metas, result
        self.info["runtime"] = result.get("runtime", "unknown")
        self.info["stream"] = result.get("stream", "unknown")
        return result

    def run(self):
        self._invoke()
        return self.last_outputs

    def synchronize(self):
        pass  # A successful native completion is emitted only after device sync.

    def reset_inputs(self):
        pass  # Each invocation reloads the immutable original physical storage.

    def snapshot(self, outputs):
        return [np.array(x, copy=True) for x in outputs], list(self.last_dtypes)

    def output_metadata(self, outputs):
        return [dict(meta, alias_inputs=meta.get("alias_inputs", []), strides_unit="elements") for meta in self.actual_metadata]

    def benchmark(self, policy):
        result = self._invoke(policy)
        samples = result.get("raw_samples_ms", [])
        if len(samples) != policy["samples"]:
            raise RuntimeError("Native benchmark sample count mismatch")
        summary = event_statistics(samples, policy, allocation_included=False)
        summary["raw_batch_elapsed_ms"] = result.get("raw_batch_elapsed_ms", [])
        summary["policy"].update(result.get("timing_policy", {}))
        summary["policy"].update(actual_stream=result.get("stream", "unknown"),
                                 excludes_process_startup_and_file_io=True)
        return summary

    def profile(self, policy):
        raise UnsupportedError("Native child GPU trace requires a native profiler provider; Python PrivateUse1 cannot certify it")

    def metadata(self):
        return copy.deepcopy(self.info)

    def memory(self):
        return {"memory_allocated_bytes": "unknown", "memory_reserved_bytes": "unknown",
                "runtime_free_total": self.last_result.get("runtime", {}).get("memory", "unknown"),
                "meaning": "Runtime free/total is not active tensor allocation or caching allocator reservation"}

    def close(self):
        self.last_outputs = []
