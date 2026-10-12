# SPDX-License-Identifier: MulanPSL-2.0
"""Lazy MUSA adapters: structured inputs, explicit layout, timing and profiling."""
import importlib
import copy
import json
from pathlib import Path
import time

import numpy as np

from musaopst.errors import EnvironmentError, UnsupportedError, is_fatal_device_error
from . import load_entry


def _flat(nodes):
    for node in nodes:
        if node is None:
            continue
        if isinstance(node, (list, tuple)):
            yield from _flat(node)
        else:
            yield node


def event_statistics(samples, policy, *, allocation_included):
    values = np.asarray(samples, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("Device timer returned invalid timing samples")
    mode = policy.get("mode", "batch_event")
    single = mode != "batch_event"
    return {
        "raw_samples_ms": values.tolist(), "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)), "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "policy": {
            "timer": "host_wall_sync" if mode == "wall_time" else "musa_event",
            "mode": mode, "stream": policy.get("stream", "current"),
            "cache": "warm; same input/output buffers; no explicit cache flush" if policy.get("cache", "warm") == "warm"
                else "interference buffer writes; hardware cold-cache state is NOT guaranteed",
            "cache_bytes": policy.get("cache_bytes", 0) if policy.get("cache") == "interference" else 0,
            "sampling": "single invocation" if single else "batch mean per invocation",
            "quantile_meaning": "quantiles of individually measured invocation samples" if single else
                "quantiles of batch means, NOT individual-call tail latency",
            "warmup": policy["warmup"], "samples": policy["samples"],
            "iterations_per_sample": 1 if single else policy["iterations"],
            "includes_output_allocation": allocation_included, "includes_host_device_copy": False,
            "includes_reference": False, "includes_compile_jit": False,
                "includes_host_submission_gaps": True,
            "includes_host_synchronization": mode == "wall_time",
        },
    }


class GPUBackend:
    def __init__(self, case, inputs, workdir):
        self.case, self.host_inputs, self.workdir = case, inputs, Path(workdir)
        self.inputs, self.last_outputs = [], []
        self.launch = None
        flat = list(_flat(inputs))
        empty = any(x.size == 0 for x in flat)
        self.info = {"backend": case["backend"], "execution_device": "musa", "empty_workload": empty,
                     "execution_classification": "empty_input_contract_only" if empty else "nonempty_input_workload"}
        self.allocation_included = "unknown (callable owns allocation)"
        self.build_count = 0

    def _upload(self, inputs, case):
        from musaopst.binding import tensor_specs
        from musaopst.data import storage_offset
        specs = iter(tensor_specs(case["inputs"]))
        self._device_bases, self._host_storages = [], []
        def convert(node):
            if node is None:
                return None
            if isinstance(node, (list, tuple)):
                return [convert(item) for item in node]
            spec = next(specs)
            array = np.asarray(node)
            strides = tuple(s // array.itemsize for s in array.strides)
            offset = storage_offset(array)
            if any(s < 0 for s in strides):
                raise UnsupportedError("Negative strides are not supported by torch MUSA; input was not made contiguous")
            dtype = getattr(self.torch, spec["dtype"], None)
            if dtype is None:
                raise UnsupportedError(f"Installed torch has no dtype {spec['dtype']}")
            span = offset + 1 + sum(max(dim - 1, 0) * stride for dim, stride in zip(array.shape, strides))
            storage = np.zeros(max(span, 1), dtype=array.dtype)
            view = np.ndarray(array.shape, dtype=array.dtype, buffer=storage,
                              offset=offset * array.itemsize,
                              strides=tuple(s * array.itemsize for s in strides))
            view[...] = array
            base = self.torch.empty(storage.size, dtype=dtype, device=self.device)
            base.copy_(self.torch.from_numpy(storage).to(dtype=dtype))
            tensor = base.as_strided(tuple(array.shape), strides, offset)
            if tuple(tensor.stride()) != strides or tensor.storage_offset() != offset:
                raise RuntimeError("GPU input layout changed during preparation")
            self._device_bases.append(base)
            self._host_storages.append(storage)
            return tensor
        return [convert(node) for node in inputs]

    def prepare(self):
        try:
            self.torch = importlib.import_module("torch")
            self.torch_musa = importlib.import_module("torch_musa")
        except Exception as exc:
            raise EnvironmentError(f"torch_musa unavailable: {exc}") from exc
        torch = self.torch
        try:
            if not torch.musa.is_available():
                raise EnvironmentError("MUSA is not available; CPU fallback is forbidden")
            self.device_index = int(self.case.get("device", 0))
            torch.musa.set_device(self.device_index)
            self.device = torch.device(f"musa:{self.device_index}")
        except EnvironmentError:
            raise
        except Exception as exc:
            raise EnvironmentError(f"Cannot initialize selected MUSA device: {exc}") from exc
        if self.case["backend"] in ("triton", "tilelang"):
            try:
                package = importlib.import_module(self.case["backend"])
                self.info["backend_version"] = getattr(package, "__version__", "unknown")
            except Exception as exc:
                raise EnvironmentError(f"{self.case['backend']} unavailable: {exc}") from exc
        stream_mode = self.case.get("benchmark", {}).get("stream", "current")
        self.execution_stream = torch.musa.Stream(device=self.device_index) if stream_mode == "independent" else torch.musa.current_stream(self.device_index)
        with torch.musa.stream(self.execution_stream):
            self.inputs = self._upload(self.host_inputs, self.case)
            entry = load_entry(self.case["entry"])
            if self.case["backend"] == "torch_musa":
                if hasattr(entry, "prepare"):
                    self.launch = entry.prepare(self.case, self.inputs, self.workdir)
                else:
                    from musaopst.binding import bind_inputs
                    def invoke():
                        args, kwargs = bind_inputs(self.inputs, self.case["inputs"], self.case["attrs"], binding=self.case.get("binding", "positional"))
                        return entry(*args, **kwargs)
                    self.launch = invoke
                    # Eager torch callables can bind new runtime shapes without
                    # rebuilding this wrapper. Internal library dispatch is not a
                    # guarantee that a third-party compiled graph is shape-generic.
                    self.launch.rebind = lambda inputs, case: None
                    self.launch.runtime_shape_kind = "eager_framework_dispatch"
            else:
                self.launch = entry(self.case, self.inputs, self.workdir)
        self.synchronize()
        if not callable(self.launch):
            raise TypeError("GPU entry factory must return a zero-argument launch callable")
        from musaopst.binding import tensor_specs
        if any(spec.get("constant", False) for spec in tensor_specs(self.case["inputs"])) and not getattr(self.launch, "constants_supported", False):
            raise UnsupportedError("Entry does not declare/consume compile-time constant inputs; fixed runtime values are not compiled constants")
        self.build_count += 1
        self.allocation_included = getattr(self.launch, "allocation_included", self.allocation_included)
        self.info.update({"device_index": self.device_index,
                          "device_name": torch.musa.get_device_name(self.device_index),
                          "device_capability": list(torch.musa.get_device_capability(self.device_index)),
                          "torch_version": str(torch.__version__),
                          "torch_musa_version": getattr(self.torch_musa, "__version__", "unknown"),
                          "launch": getattr(self.launch, "launch_metadata", {}),
                          "build_count": self.build_count, "stream": self._stream_metadata(),
                          "runtime_shape_kind": getattr(self.launch, "runtime_shape_kind", "not_declared")})
        self.info["inputs"] = self.output_metadata(list(_flat(self.inputs)), include_aliases=False)

    def rebind(self, case, inputs):
        """Reuse a prepared kernel only when its factory explicitly supports it."""
        if not hasattr(self.launch, "rebind"):
            raise UnsupportedError("This entry has no runtime-shape rebind hook; per-shape JIT is not compile-once")
        self.synchronize()
        with self.torch.musa.stream(self.execution_stream):
            new_inputs = self._upload(inputs, case)
            self.launch.rebind(new_inputs, case)
            self.case, self.host_inputs, self.inputs = case, inputs, new_inputs
        self.synchronize()
        self.info["inputs"] = self.output_metadata(list(_flat(self.inputs)), include_aliases=False)
        self.info["runtime_shape_rebindings"] = self.info.get("runtime_shape_rebindings", 0) + 1
        self.info["build_count"] = self.build_count

    def _stream_metadata(self):
        stream = self.execution_stream
        return {"description": str(stream), "musa_stream_handle": getattr(stream, "musa_stream", "unknown"),
                "device": str(getattr(stream, "device", "unknown")),
                "policy": self.case.get("benchmark", {}).get("stream", "current")}

    def run(self):
        with self.torch.inference_mode(), self.torch.musa.stream(self.execution_stream):
            outputs = self.launch()
        artifacts = getattr(self.launch, "launch_metadata", {}).get("compiled_artifact_ids")
        if self.case.get("shape_mode") == "runtime" and isinstance(artifacts, list) and len(artifacts) > 1:
            raise UnsupportedError("Runtime-shape group compiled more than one artifact; per-shape JIT cannot pass compile-once validation")
        self.last_outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        for value in self.last_outputs:
            if not isinstance(value, self.torch.Tensor):
                raise TypeError("GPU entries must return actual MUSA tensors")
            if value.device.type != "musa":
                raise RuntimeError(f"GPU entry returned {value.device}; CPU fallback is forbidden")
            if value.device != self.device:
                raise RuntimeError(f"GPU entry returned {value.device}; selected timing device is {self.device}")
        return self.last_outputs

    def synchronize(self):
        self.torch.musa.synchronize(self.device_index)

    def output_metadata(self, outputs, include_aliases=True):
        def pointer(tensor):
            try:
                return tensor.untyped_storage().data_ptr()
            except AttributeError:
                return tensor.storage().data_ptr()
        inputs = list(_flat(self.inputs))
        result = []
        for out in outputs:
            aliases = [i for i, inp in enumerate(inputs) if out.numel() and inp.numel() and pointer(out) == pointer(inp)] if include_aliases else []
            result.append({"shape": list(out.shape), "dtype": str(out.dtype).replace("torch.", ""),
                           "strides": list(out.stride()), "strides_unit": "elements",
                           "storage_offset": out.storage_offset(), "contiguous": out.is_contiguous(),
                           "device": str(out.device), "alias_inputs": aliases})
        return result

    def snapshot(self, outputs):
        result, dtypes = [], []
        for output in outputs:
            dtype = str(output.dtype).replace("torch.", "")
            dtypes.append(dtype)
            cpu = output.detach().cpu()
            if dtype == "bfloat16":
                cpu = cpu.float()
            result.append(cpu.numpy().copy())
        return result, dtypes

    def reset_inputs(self):
        with self.torch.musa.stream(self.execution_stream):
            for base, storage in zip(self._device_bases, self._host_storages):
                base.copy_(self.torch.from_numpy(storage).to(dtype=base.dtype))
        self.synchronize()

    def _mutates_inputs(self):
        self.reset_inputs()
        self.run()
        self.synchronize()
        actual, _ = self.snapshot(list(_flat(self.inputs)))
        return any(not np.array_equal(a, b, equal_nan=True) for a, b in zip(_flat(self.host_inputs), actual))

    def benchmark(self, policy):
        mode = policy.get("mode", "batch_event")
        mutates = self._mutates_inputs()
        if mutates and (not policy.get("allow_inplace", False) or mode == "batch_event"):
            raise UnsupportedError("Input-mutating benchmark requires allow_inplace=true and single_event/wall_time")
        interference = None
        if policy.get("cache", "warm") == "interference":
            size = policy.get("cache_bytes", 0)
            if not size:
                raise UnsupportedError("Cache interference needs a positive explicit cache_bytes")
            interference = self.torch.empty(size, dtype=self.torch.uint8, device=self.device)
        with self.torch.musa.stream(self.execution_stream):
            for _ in range(policy["warmup"]):
                if mutates:
                    self.reset_inputs()
                self.run()
            self.synchronize()
            samples, raw = [], []
            iterations = policy["iterations"] if mode == "batch_event" else 1
            for _ in range(policy["samples"]):
                if mutates:
                    self.reset_inputs()
                if interference is not None:
                    interference.zero_()
                    self.synchronize()
                if mode == "wall_time":
                    self.synchronize()
                    start = time.perf_counter_ns()
                    self.run()
                    self.synchronize()
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                else:
                    start = self.torch.musa.Event(enable_timing=True)
                    stop = self.torch.musa.Event(enable_timing=True)
                    start.record()
                    for _ in range(iterations):
                        self.run()
                    stop.record()
                    stop.synchronize()
                    elapsed = start.elapsed_time(stop)
                raw.append(elapsed)
                samples.append(elapsed / iterations)
        self.synchronize()
        result = event_statistics(samples, policy, allocation_included=self.allocation_included)
        result["raw_batch_elapsed_ms"] = raw
        result["policy"].update(actual_stream=self._stream_metadata(), input_mutation_detected=mutates,
                                input_reset="before each sample, excluded from timer" if mutates else "not needed",
                                input_reset_changes_cache=mutates)
        return result

    def profile(self, policy):
        """Separate diagnostic launches. CPU-only traces are not GPU profiling."""
        if not hasattr(self.torch.profiler.ProfilerActivity, "PrivateUse1"):
            raise UnsupportedError("Installed profiler has no PrivateUse1 GPU activity")
        self.workdir.mkdir(parents=True, exist_ok=True)
        trace = self.workdir / "profile_trace.json"
        table = self.workdir / "profile_table.txt"
        self.reset_inputs()
        for _ in range(policy.get("warmup", 1)):
            self.run()
            self.synchronize()
            self.reset_inputs()
        activities = [self.torch.profiler.ProfilerActivity.CPU, self.torch.profiler.ProfilerActivity.PrivateUse1]
        with self.torch.profiler.profile(activities=activities,
                record_shapes=policy.get("record_shapes", True),
                profile_memory=policy.get("profile_memory", False)) as profile:
            for _ in range(policy.get("iterations", 3)):
                self.reset_inputs()
                with self.torch.profiler.record_function("musaopst::" + self.case["name"]):
                    self.run()
                    self.synchronize()
        profile.export_chrome_trace(str(trace))
        table.write_text(profile.key_averages().table(row_limit=100))
        data = json.loads(trace.read_text())
        events = [event for event in data.get("traceEvents", [])
                  if str(event.get("cat", "")).lower() in ("kernel", "gpu_kernel")]
        if not events:
            raise UnsupportedError("Profiler exported a trace but found no GPU kernel events; CPU-only trace is not accepted")
        return {"trace_path": str(trace), "table_path": str(table), "gpu_kernel_count": len(events),
                "kernel_names": sorted({str(event.get("name", "unknown")) for event in events}),
                "gpu_durations_us": [event.get("dur", "unknown") for event in events],
                "iterations": policy.get("iterations", 3), "separate_from_benchmark": True,
                "hardware_counters": "not collected"}

    def metadata(self):
        return copy.deepcopy(self.info)

    def memory(self):
        result = {}
        if not hasattr(self, "torch") or not hasattr(self, "device_index"):
            return {"memory_allocated_bytes": "unknown", "memory_reserved_bytes": "unknown"}
        for api in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
            function = getattr(self.torch.musa, api, None)
            try:
                result[api + "_bytes"] = int(function(self.device_index)) if function else "unknown"
            except Exception as exc:
                if is_fatal_device_error(exc):
                    raise
                result[api + "_bytes"] = "unknown"
        result["meaning"] = "allocated=active tensor bytes; reserved=caching allocator pool bytes"
        return result

    def close(self):
        self.last_outputs, self.inputs = [], []
        self._device_bases, self._host_storages = [], []
        self.launch = None
        self.execution_stream = None
