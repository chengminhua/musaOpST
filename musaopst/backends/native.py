# SPDX-License-Identifier: MulanPSL-2.0
"""A standalone native MUSA host program, invoked with a strict file protocol."""
import hashlib
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from musaopst.errors import BuildError, EnvironmentError, UnsupportedError, DeviceStateUnknownError
from . import load_entry
from .gpu import event_statistics


class NativeBackend:
    def __getattribute__(self, name):
        if name in {"run", "reset_inputs", "synchronize", "snapshot", "benchmark", "metadata", "memory", "close", "profile", "output_metadata", "rebind", "last_outputs", "last_dtypes", "last_result"}:
            delegate = object.__getattribute__(self, "__dict__").get("_delegate")
            if delegate is not None:
                return getattr(delegate, name)
        return object.__getattribute__(self, name)

    def __init__(self, case, inputs, workdir):
        self.case, self.inputs, self.workdir = case, inputs, Path(workdir)
        from .gpu import _flat
        empty = any(x.size == 0 for x in _flat(inputs))
        self.info = {"backend": "native", "execution_device": "musa",
                     "empty_workload": empty,
                     "execution_classification": "empty_input_contract_only" if empty else "nonempty_input_workload",
                     "input_transfer": "explicit float32 binary files",
                     "repeat_scope": "each invocation creates a fresh native process/context"}
        self.last_outputs = []
        self.last_dtypes = []
        self.sequence = 0
        self.last_result = {}

    def prepare(self):
        spec = load_entry(self.case["entry"])(self.case, self.inputs, self.workdir) if self.case.get("entry") else None
        if spec and spec.get("protocol") == "musaopst_native_v2":
            from .native_protocol import NativeProtocolBackend
            self._delegate = NativeProtocolBackend(self.case, self.inputs, self.workdir, spec)
            return self._delegate.prepare()
        if any(spec.get("constant",False) for spec in self.case["inputs"]):
            raise UnsupportedError("Legacy native v1 does not implement compile-time constant inputs")
        if len(self.inputs) != 2 or any(not isinstance(x,np.ndarray) or x.ndim != 1 for x in self.inputs):
            raise UnsupportedError("native vector_add_f32_v1 needs two 1-D inputs")
        if self.inputs[0].shape != self.inputs[1].shape:
            raise UnsupportedError("native vector_add_f32_v1 needs equal input shapes")
        for x, input_spec in zip(self.inputs, self.case["inputs"]):
            if input_spec["dtype"] != "float32":
                raise UnsupportedError("native vector_add_f32_v1 supports float32 only")
            if input_spec["layout"] != "contiguous" or not x.flags.c_contiguous:
                raise UnsupportedError("native vector_add_f32_v1 supports contiguous layout only")
        if self.case["attrs"]:
            raise UnsupportedError("native vector_add_f32_v1 has no scalar attributes")
        spec = spec or load_entry(self.case["entry"])(self.case, self.inputs, self.workdir)
        if spec.get("protocol") != "vector_add_f32_v1":
            raise UnsupportedError("native adapter requires explicit vector_add_f32_v1 protocol")
        source = Path(spec["source"]).resolve()
        if not source.is_file():
            raise BuildError(f"Native source does not exist: {source}")
        home = Path(os.environ.get("MUSA_HOME", "/usr/local/musa"))
        compiler = shutil.which(os.environ.get("MUSA_OPST_MCC", "mcc"))
        if not compiler and (home / "bin/mcc").is_file():
            compiler = str(home / "bin/mcc")
        if not compiler:
            raise EnvironmentError("mcc unavailable; set PATH/MUSA_HOME or MUSA_OPST_MCC")
        arch = os.environ.get("MUSA_OPST_ARCH")
        device = int(self.case.get("device", 0))
        if not arch:
            try:
                import torch
                import torch_musa
                if not torch.musa.is_available():
                    raise RuntimeError("MUSA device unavailable")
                major, minor = torch.musa.get_device_capability(device)
                arch = f"mp_{major}{minor}"
            except Exception as exc:
                raise EnvironmentError("Cannot discover native target; set MUSA_OPST_ARCH explicitly "
                                       f"or install working torch_musa: {exc}") from exc
        if not re.fullmatch(r"mp_[0-9]+", arch):
            raise EnvironmentError("MUSA_OPST_ARCH must have format mp_<digits>")
        self.workdir.mkdir(parents=True, exist_ok=True)
        compiler_version = subprocess.run([compiler, "--version"], capture_output=True, text=True,
                                          check=False).stdout.strip()
        digest = hashlib.sha256(source.read_bytes() + compiler_version.encode() + arch.encode()).hexdigest()
        cache = self.workdir.parent / ".build-cache" / ("native_" + digest[:12])
        cache.mkdir(parents=True, exist_ok=True)
        self.binary = cache / "native_vector_add"
        command = [compiler, "-O2", f"--offload-arch={arch}", str(source),
                   "-L" + str(home / "lib"), "-Wl,-rpath," + str(home / "lib"),
                   "-lmusart", "-o", str(self.binary)]
        if not self.binary.exists():
            process = subprocess.run(command, capture_output=True, text=True)
            (self.workdir / "native_build.log").write_text(process.stdout + process.stderr)
            if process.returncode:
                raise BuildError(f"mcc failed with {process.returncode}: {process.stderr[-4000:]}")
        self.input_files = []
        for i, array in enumerate(self.inputs):
            path = self.workdir / f"native_input_{i}.bin"
            np.asarray(array, dtype="<f4").tofile(path)
            self.input_files.append(path)
        self.info.update({"compiler": compiler, "compiler_version": compiler_version or "unknown",
                          "build_command": command, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "arch": arch, "device_index": device, "protocol": spec["protocol"],
                          "inputs": [{"shape": list(x.shape), "dtype": "float32",
                                      "strides": [s // x.itemsize for s in x.strides],
                                      "contiguous": bool(x.flags.c_contiguous)} for x in self.inputs]})

    def _invoke(self, policy=None):
        self.sequence += 1
        output = self.workdir / f"native_output_{self.sequence}.bin"
        p = policy or {"warmup": 0, "samples": 0, "iterations": 1}
        cmd = [str(self.binary), str(self.input_files[0]), str(self.input_files[1]), str(output),
               str(self.inputs[0].size), str(self.case.get("device", 0)),
               str(p["warmup"]), str(p["samples"]), str(p["iterations"])]
        process = subprocess.run(cmd, capture_output=True, text=True)
        (self.workdir / f"native_run_{self.sequence}.log").write_text(process.stdout + process.stderr)
        if process.returncode < 0:
            raise DeviceStateUnknownError(f"Native device process exited on signal {-process.returncode}: {process.stderr[-4000:]}")
        if process.returncode:
            raise RuntimeError(f"Native execution failed ({process.returncode}): {process.stderr[-4000:]}")
        try:
            result = json.loads(process.stdout)
        except (ValueError, TypeError) as exc:
            raise DeviceStateUnknownError("Native device process returned no valid JSON completion record") from exc
        if not isinstance(result, dict) or result.get("protocol") != "vector_add_f32_v1" or result.get("device_execution") != "musa":
            raise DeviceStateUnknownError("Invalid native completion protocol/device; device state unknown")
        contract = result.get("output", {})
        shape, dtype_name = contract.get("shape"), contract.get("dtype")
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise RuntimeError("Native program returned malformed shape metadata")
        dtype_map = {"float32": "<f4", "float16": "<f2", "float64": "<f8", "int32": "<i4", "int64": "<i8"}
        if dtype_name not in dtype_map:
            raise RuntimeError(f"Native protocol cannot decode dtype {dtype_name!r}")
        # Decode ACTUAL metadata. The shared comparator, not this adapter, decides
        # whether shape/dtype satisfies the user's requested output contract.
        dtype = np.dtype(dtype_map[dtype_name])
        elements = 1
        for extent in shape:
            elements *= extent
        expected_bytes = elements * dtype.itemsize
        if not output.exists() or output.stat().st_size != expected_bytes:
            raise RuntimeError("Native output file byte count does not match its metadata")
        self.last_outputs = [np.fromfile(output, dtype=dtype).reshape(shape).copy()]
        self.last_dtypes = [dtype_name]
        self.last_result = result
        self.info["runtime"] = {key: result.get(key, "unknown") for key in (
            "device_name", "driver_api_version", "runtime_api_version", "capability", "memory", "empty_no_kernel_launch")}
        return result

    def run(self):
        self._invoke()
        return self.last_outputs

    def reset_inputs(self):
        # Program inputs are immutable files; each invocation reads fresh host buffers.
        pass

    def profile(self, policy):
        raise UnsupportedError("Independent native process profiling is not provided by the Python PrivateUse1 profiler")

    def rebind(self, case, inputs):
        raise UnsupportedError("Legacy native v1 has no declared runtime-shape rebinding; use native protocol v2")

    def output_metadata(self, outputs):
        from musaopst.data import tensor_metadata
        return [tensor_metadata(x, d) for x, d in zip(outputs, self.last_dtypes)]

    def synchronize(self):
        # Child checks launch errors and musaDeviceSynchronize before returning.
        pass

    def snapshot(self, outputs):
        return [np.array(x, copy=True) for x in outputs], (self.last_dtypes or [str(x.dtype) for x in outputs])

    def benchmark(self, policy):
        if (policy.get("mode","batch_event") != "batch_event" or policy.get("stream","current") != "current"
                or policy.get("cache","warm") != "warm" or policy.get("allow_inplace",False)):
            raise UnsupportedError("Legacy native v1 timing only supports batch_event/current/warm without in-place inputs; use native protocol v2 for additional policies")
        result = self._invoke(policy)
        samples = result.get("raw_samples_ms", [])
        if len(samples) != policy["samples"]:
            raise RuntimeError("Native benchmark sample count mismatch")
        summary = event_statistics(samples, policy, allocation_included=False)
        summary["raw_batch_elapsed_ms"] = result.get("raw_batch_elapsed_ms", "unknown")
        summary["policy"]["stream"] = "native default stream (0)"
        summary["policy"]["excludes_process_startup_and_file_io"] = True
        return summary

    def metadata(self):
        return copy.deepcopy(self.info)

    def memory(self):
        return {"memory_allocated_bytes": "unknown", "memory_reserved_bytes": "unknown",
                "runtime_free_total": self.last_result.get("memory", "unknown"),
                "meaning": "runtime free/total is not active allocation or allocator reserved bytes"}

    def close(self):
        self.last_outputs = []
