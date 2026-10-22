# SPDX-License-Identifier: MulanPSL-2.0
"""Explicit preparation and offline artifact workflows; GPU imports remain lazy."""
from __future__ import annotations

import copy
import csv
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from .artifacts import case_directory, content_digest, file_manifest, result_path, verify_manifest
from .compare import compare_outputs
from .config import load_config, case_matches, selection_error
from .data import make_inputs, tensor_metadata
from .util import load_entry, sha256, write_json


def _fresh(directory):
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _specs(specs):
    for spec in specs:
        if spec.get("kind") == "absent":
            continue
        if spec.get("kind") == "tensor_list":
            yield from _specs(spec["items"])
        else:
            yield spec


def _leaves(nodes):
    for node in nodes:
        if node is None:
            continue
        if isinstance(node, (list, tuple)):
            yield from _leaves(node)
        else:
            yield np.asarray(node)


def export_case(case):
    """Drop runtime-only fields when emitting a new editable config."""
    return copy.deepcopy({k: v for k, v in case.items()
                          if not k.startswith("_") and k not in {"parameters_selected", "device"}})


def _snapshot_v2_case(case, source_version):
    result = export_case(case)
    if source_version == 1:
        # File-backed inputs need v2, but replay must preserve v1's already
        # resolved whole-case dtype policy rather than silently change it.
        result.pop("compare_by_dtype", None)
        for output in result["outputs"]:
            output["compare"] = copy.deepcopy(result["compare"])
    return result


def _make_inputs(case, base_dir):
    # v1 remains usable while v2 input loading is optional to older installs.
    if "base_dir" in inspect.signature(make_inputs).parameters:
        return make_inputs(case, base_dir=base_dir)
    return make_inputs(case)


def save_inputs(case, inputs, directory):
    directory = Path(directory)
    arrays, specs = list(_leaves(inputs)), list(_specs(case["inputs"]))
    if len(arrays) != len(specs):
        raise ValueError("input tensor tree differs from specification")
    path = directory / "inputs.npz"
    np.savez(path, **{f"input_{i}": array for i, array in enumerate(arrays)})
    metadata = [tensor_metadata(array, spec["dtype"]) for array, spec in zip(arrays, specs)]
    write_json(directory / "inputs.meta.json", {"format": "musaopst-inputs-v2", "sha256": sha256(path),
               "tensors": metadata, "input_specs": case["inputs"],
               "storage": "logical values; strides/offset reconstructed from input_specs"})
    return {"file": "inputs.npz", "sha256": sha256(path), "tensors": metadata}


def prepare(config_path, output_dir, *, stage="prepare", device=0, arch=None):
    """Materialize inputs/config/build evidence without launching tested kernels."""
    if stage not in {"data", "prepare", "build"}:
        raise ValueError("stage must be data, prepare or build")
    config_path = Path(config_path).resolve()
    cases = load_config(config_path)
    output = _fresh(output_dir)
    original = json.loads(config_path.read_text(encoding="utf-8"))
    write_json(output / "config.original.json", original)
    version = original["schema_version"]
    resolved_cases = [export_case(c) for c in cases]
    for resolved in resolved_cases:
        if "sources" in resolved:
            resolved["sources"] = [str((config_path.parent / path).resolve()) for path in resolved["sources"]]
    write_json(output / "config.resolved.json", {"schema_version": version, "cases": resolved_cases})
    manifest = {"workflow_version": 1, "kind": stage, "config_sha256": sha256(config_path),
                "resolved_config_sha256": sha256(output / "config.resolved.json"),
                "device_execution": False, "accuracy_evaluated": False, "cases": [], "complete": False,
                "module_search_paths": [str(config_path.parent), str(Path(__file__).parent.parent),
                                        *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]]}
    old_path = list(sys.path)
    sys.path.insert(0, str(config_path.parent))
    try:
        for i, case in enumerate(cases):
            case_dir = output / f"case-{i:04d}"
            case_dir.mkdir()
            begin = time.monotonic()
            record = {"name": case["name"], "backend": case["backend"], "required": case["required"],
                      "directory": case_dir.name, "status": "PREPARED", "device_execution": False}
            phase = "inputs"
            try:
                inputs = _make_inputs(case, config_path.parent)
                record["inputs"] = save_inputs(case, inputs, case_dir)
                write_json(case_dir / "case.json", export_case(case))
                if stage == "build":
                    phase = "build"
                    if case["backend"] == "native":
                        from .backends.build_cache import build_native_descriptor
                        descriptor = load_entry(case["entry"])(dict(case, device=device), inputs, case_dir)
                        record["descriptor"] = descriptor
                        record["build"] = build_native_descriptor(descriptor, case_dir, arch=arch, timeout_s=case["timeout_s"])
                        record["status"] = "BUILT"
                    elif case["backend"] == "musa_cpp":
                        from .backends.build_cache import build_cpp_descriptor
                        entry = load_entry(case["entry"])
                        hook = getattr(entry, "build_descriptor", None)
                        descriptor = case.get("artifact")
                        if descriptor is None and not callable(hook):
                            record.update(status="UNSUPPORTED", message="C++ factory has no explicit build_descriptor hook; use runtime run or supply an artifact descriptor")
                        else:
                            descriptor = descriptor or hook(case, case_dir)
                            record["descriptor"] = descriptor
                            record["build"] = build_cpp_descriptor(descriptor, case_dir, arch=arch, timeout_s=case["timeout_s"])
                            record.update(status="BUILT", runtime_device_probe_possible=True)
                    elif case["backend"] in {"cpu", "torch_musa"}:
                        record.update(status="NOT_REQUIRED", message="callable backend has no standalone offline build step")
                    else:
                        record.update(status="UNSUPPORTED", message="factory requires runtime device tensors/JIT; compile by run in isolated worker")
                elif stage == "prepare":
                    record["execution_plan"] = {"entry": case["entry"], "reference": case["reference"],
                                                "backend": case["backend"], "build": "deferred to explicit build/run",
                                                "timeout_s": case["timeout_s"]}
            except Exception as exc:
                from .errors import BuildError, EnvironmentError, UnsupportedError
                status = ("TIMEOUT" if isinstance(exc, TimeoutError) else
                          "UNSUPPORTED" if isinstance(exc, UnsupportedError) else
                          "ENVIRONMENT_ERROR" if isinstance(exc, (EnvironmentError, ModuleNotFoundError)) else
                          "BUILD_FAILED" if isinstance(exc, BuildError) or phase == "build" else "CONFIG_ERROR")
                record.update(status=status, phase=phase, message=f"{type(exc).__name__}: {exc}")
            record["elapsed_s"] = time.monotonic() - begin
            record["artifacts"] = file_manifest(case_dir)
            write_json(case_dir / "preparation.json", record)
            manifest["cases"].append(record)
            write_json(output / "preparation.json", manifest)
    finally:
        sys.path[:] = old_path
    manifest["complete"] = True
    manifest["exit_code"] = int(any(c["status"] not in {"PREPARED", "BUILT", "NOT_REQUIRED"} and
                                    not (c["status"] == "UNSUPPORTED" and not c["required"])
                                    for c in manifest["cases"]))
    write_json(output / "preparation.json", manifest)
    return manifest


def _load_arrays(path, prefix):
    with np.load(path, allow_pickle=False) as archive:
        expected = [f"{prefix}_{i}" for i in range(len(archive.files))]
        if set(archive.files) != set(expected):
            raise ValueError(f"snapshot keys are not a contiguous {prefix} sequence")
        return [np.array(archive[key], copy=True) for key in expected]


def _csv_failures(path, actual, expected, comparison, contracts, default_policy, limit):
    rows = 0
    considered = sum(o.get("mismatch_count", 0) for o in comparison.get("outputs", []))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["output", "index", "actual", "reference", "absolute_error", "relative_error", "allowed_absolute_error", "reason"])
        if comparison.get("contract_errors"):
            return {"rows": 0, "limit": limit, "truncated": False, "mismatch_count": None,
                    "reason": "contract failed; no numerical comparison was performed"}
        for output, (a, r) in enumerate(zip(actual, expected)):
            current = {**default_policy, **contracts[output].get("compare", {})}
            # Reuse the authoritative comparator in bounded chunks. Global metrics
            # remain in the main comparison; CSV describes element mismatches only.
            for offset in range(0, a.size, 4096):
                if rows >= limit:
                    return {"rows": rows, "limit": limit, "truncated": rows < considered, "mismatch_count": considered}
                av, rv = a.reshape(-1)[offset:offset + 4096], r.reshape(-1)[offset:offset + 4096]
                policy = {**current, "max_failures": min(len(av), limit - rows)}
                policy.pop("metrics", None)
                policy.pop("custom", None)
                contract = {"shape": [len(av)], "dtype": contracts[output]["dtype"]}
                chunk = compare_outputs([av], [rv], [contract], policy, actual_dtypes=[contract["dtype"]])
                for failure in chunk["outputs"][0]["failures"]:
                    flat_index = offset + failure["index"][0]
                    index = [int(i) for i in np.unravel_index(flat_index, a.shape)]
                    writer.writerow([output, json.dumps(index), str(failure["actual"]), str(failure["reference"]),
                                     failure["absolute_error"], failure["relative_error"],
                                     failure["allowed_absolute_error"], failure["reason"]])
                    rows += 1
    return {"rows": rows, "limit": limit, "truncated": rows < considered, "mismatch_count": considered}


def recompare(results_path, output_dir, *, policy=None, csv_limit=10000):
    if type(csv_limit) is not int or csv_limit < 0:
        raise ValueError("csv_limit must be a nonnegative integer")
    if policy is not None:
        from . import config
        getattr(config, "_compare_v2", config._compare)(policy, "recompare.policy")
    source = result_path(results_path)
    previous = json.loads(source.read_text(encoding="utf-8"))
    output = _fresh(output_dir)
    records = []
    for i, record in enumerate(previous.get("cases", [])):
        directory = case_directory(source, record, i)
        target = output / f"case-{i:04d}"
        target.mkdir()
        fresh = {"name": record["name"], "backend": record.get("backend", "unknown"),
                 "required": record.get("required", True), "status": "RECOMPARE_UNAVAILABLE"}
        try:
            for filename in ("actual.npz", "reference.npz"):
                recorded = record.get("snapshot_hashes", {}).get(filename)
                if recorded and sha256(directory / filename) != recorded:
                    raise ValueError(f"snapshot digest mismatch: {filename}")
            actual = _load_arrays(directory / "actual.npz", "output")
            expected = _load_arrays(directory / "reference.npz", "output")
            case = record["config"]
            contracts = copy.deepcopy(case["outputs"])
            selected = {**case["compare"], **(policy or {})}
            if policy:
                for contract in contracts:
                    contract["compare"] = {**contract.get("compare", {}), **policy}
            metadata = record.get("actual_metadata")
            dtypes = [x["dtype"] for x in metadata] if metadata else [x.dtype.name for x in actual]
            kwargs = {"actual_dtypes": dtypes}
            if "actual_metadata" in inspect.signature(compare_outputs).parameters:
                kwargs.update(actual_metadata=metadata, input_metadata=record.get("inputs"))
            old_path = list(sys.path)
            try:
                sys.path[:0] = [case.get("_config_dir", str(source.parent)), *previous.get("replay_module_search_paths", [])]
                comparison = compare_outputs(actual, expected, contracts, selected, **kwargs)
            finally:
                sys.path[:] = old_path
            fresh.update(status="RECOMPARE_PASS" if comparison["passed"] else "RECOMPARE_FAILED", comparison=comparison,
                         original_status=record["status"], source_snapshots={"actual": sha256(directory / "actual.npz"),
                         "reference": sha256(directory / "reference.npz")})
            fresh["csv"] = _csv_failures(target / "failures.csv", actual, expected, comparison, contracts, selected, csv_limit)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            fresh["message"] = f"{type(exc).__name__}: {exc}"
        write_json(target / "comparison.json", fresh)
        records.append(fresh)
    passed = bool(records) and all(r["status"] == "RECOMPARE_PASS" for r in records if r["required"]) and not any(r["status"] == "RECOMPARE_FAILED" for r in records)
    passed = passed and any(r["status"] == "RECOMPARE_PASS" for r in records)
    result = {"workflow_version": 1, "kind": "offline_recompare", "source": str(source), "source_sha256": sha256(source),
              "device_execution": False, "new_gpu_passed": 0, "policy_override": policy,
              "cases": records, "exit_code": 0 if passed else 1}
    write_json(output / "recompare.json", result)
    lines = ["# 离线重新比较", "", "使用既有实际/参考快照，未执行设备；原始验收结论保持不变。", "",
             "| 用例 | 离线重比状态 |", "|---|---|"]
    lines += [f"| {r['name'].replace('|', '/')} | {r['status']} |" for r in records]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def replay_config(results_path, output_dir, *, case_pattern=None):
    source = result_path(results_path)
    prior = json.loads(source.read_text(encoding="utf-8"))
    output = _fresh(output_dir)
    selected = []
    provenance = []
    for i, record in enumerate(prior.get("cases", [])):
        if case_pattern and not case_matches(record["name"], case_pattern):
            continue
        if "config" not in record:
            raise ValueError(f"case has no resolved configuration: {record['name']}")
        case = _snapshot_v2_case(record["config"], record["config"].get("_schema_version", 1))
        if "sources" in case:
            origin = record["config"].get("_config_dir") or str(Path(prior.get("config_file", source)).resolve().parent)
            case["sources"] = [str((Path(origin) / path).resolve()) for path in case["sources"]]
        original = case_directory(source, record, i) / "inputs.npz"
        recorded = record.get("snapshot_hashes", {}).get("inputs.npz")
        if recorded and sha256(original) != recorded:
            raise ValueError("input snapshot digest mismatch")
        # Decode now: absent/corrupt snapshots must not produce a runnable-looking config.
        arrays = _load_arrays(original, "input")
        specs = list(_specs(case["inputs"]))
        if len(arrays) != len(specs):
            raise ValueError(f"input snapshot count mismatch: {record['name']}")
        destination = output / f"case-{i:04d}"
        destination.mkdir()
        copied = destination / "inputs.npz"
        shutil.copyfile(original, copied)
        metadata = record.get("inputs", [])
        for j, (spec, array) in enumerate(zip(specs, arrays)):
            if list(array.shape) != spec["shape"]:
                raise ValueError(f"snapshot shape differs from configured input: {record['name']} input {j}")
            spec.pop("distribution", None)
            spec.pop("source", None)
            if spec.get("transforms"):
                if len(metadata) != len(specs):
                    raise ValueError("transformed input replay requires original stride metadata")
                spec.pop("transforms", None)
                spec.pop("base_shape", None)
                spec.update(layout="explicit", strides=metadata[j]["strides"],
                            storage_offset=metadata[j].get("storage_offset", 0))
            spec["source"] = {"format": "npz", "path": str(copied.relative_to(output)),
                              "key": f"input_{j}", "sha256": sha256(copied)}
        # Reproduce exactly the selected concrete case, not its original parameter grid.
        case.pop("parameters", None)
        case.pop("symbols", None)
        selected.append(case)
        provenance.append({"case": case["name"], "snapshot_sha256": sha256(original)})
    if not selected:
        names = [record.get("name", "") for record in prior.get("cases", [])]
        raise ValueError(selection_error(names, case_pattern) if case_pattern else "no cases selected for replay")
    config = {"schema_version": 2, "cases": selected}
    path = output / "replay.json"
    write_json(path, config)
    # Validate file descriptors and contracts, while leaving device execution to run.
    load_config(path)
    write_json(output / "replay.provenance.json", {"source": str(source), "source_sha256": sha256(source),
               "kind": "snapshot_replay_configuration", "device_execution": False, "cases": provenance,
               "required_module_paths": prior.get("replay_module_search_paths", [])})
    write_json(output / "execution-context.json", {"module_search_paths": prior.get("replay_module_search_paths", [])})
    return path


def prepared_config(prepared_path, output_dir):
    """Verify a prepared project and emit snapshot inputs plus prebuilt factories."""
    root = Path(prepared_path).resolve()
    if root.is_file():
        root = root.parent
    manifest = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("exit_code"):
        raise ValueError("preparation is incomplete or failed; cannot execute its artifacts")
    config = json.loads((root / "config.resolved.json").read_text(encoding="utf-8"))
    if sha256(root / "config.resolved.json") != manifest.get("resolved_config_sha256"):
        raise ValueError("prepared resolved configuration digest mismatch")
    if len(config["cases"]) != len(manifest["cases"]):
        raise ValueError("prepared config and manifest case counts differ")
    for record in manifest["cases"]:
        directory = (root / record["directory"]).resolve()
        directory.relative_to(root)
        verified = verify_manifest(directory, record["artifacts"])
        if not verified["passed"]:
            raise ValueError("prepared artifacts failed integrity: " + "; ".join(verified["errors"]))
    output = _fresh(output_dir)
    cases = []
    for i, (case, record) in enumerate(zip(config["cases"], manifest["cases"])):
        if case["name"] != record["name"]:
            raise ValueError("prepared case identity differs from its manifest")
        case = _snapshot_v2_case(case, config["schema_version"])
        directory = root / record["directory"]
        destination = output / f"case-{i:04d}"
        destination.mkdir()
        copied = destination / "inputs.npz"
        shutil.copyfile(directory / "inputs.npz", copied)
        specs = list(_specs(case["inputs"]))
        for j, spec in enumerate(specs):
            spec.pop("distribution", None)
            spec["source"] = {"format": "npz", "path": str(copied.relative_to(output)), "key": f"input_{j}", "sha256": sha256(copied)}
            if spec.get("transforms"):
                spec.pop("transforms", None)
                spec.pop("base_shape", None)
                metadata = record["inputs"]["tensors"][j]
                spec.update(layout="explicit", strides=metadata["strides"], storage_offset=metadata.get("storage_offset", 0))
        if case["backend"] == "native" and record["status"] == "BUILT":
            descriptor = copy.deepcopy(record["descriptor"])
            binary = Path(record["build"]["binary"])
            if not binary.is_file() or sha256(binary) != record["build"]["artifact_sha256"]:
                raise ValueError("prepared native binary missing or changed")
            target_binary = destination / "native_operator"
            shutil.copyfile(binary, target_binary)
            target_binary.chmod(0o755)
            descriptor.update(binary=str(target_binary), artifact_sha256=sha256(target_binary), arch=record["build"].get("arch", "unknown"))
            module = f"prepared_native_{i}"
            source = "import json\n\ndef factory(case, inputs, workdir):\n    return json.loads(" + repr(json.dumps(descriptor)) + ")\n"
            (output / (module + ".py")).write_text(source, encoding="utf-8")
            case["entry"] = module + ":factory"
        elif case["backend"] == "musa_cpp" and record["status"] == "BUILT":
            artifact = record["build"]
            binary = Path(artifact["binary"])
            if not binary.is_file() or sha256(binary) != artifact["artifact_sha256"]:
                raise ValueError("prepared C++ extension missing or changed")
            target_binary = destination / ("cpp_extension" + binary.suffix)
            shutil.copyfile(binary, target_binary)
            case["artifact"] = {"binary": str(target_binary), "artifact_sha256": sha256(target_binary),
                                "module_name": artifact["module_name"], "arch": artifact.get("arch", "unknown"), "fingerprint": copy.deepcopy(artifact.get("fingerprint", {}))}
        cases.append(case)
    path = output / "prepared.json"
    write_json(path, {"schema_version": 2, "cases": cases})
    load_config(path)
    write_json(output / "execution-context.json", {"module_search_paths": manifest.get("module_search_paths", []),
               "source_preparation": str(root), "source_sha256": sha256(root / "preparation.json")})
    return path
