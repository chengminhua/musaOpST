# SPDX-License-Identifier: MulanPSL-2.0
"""Finite deterministic case generation and explicit editable operator projects."""
from __future__ import annotations

import ast
import configparser
import copy
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile

from .artifacts import content_digest
from .config import ConfigError, expand_config
from .util import write_json
from .workflows import _fresh, export_case


_EXAMPLES = {
    ("cpu", "vector_add"): ("cpu_ops.py", "add", "add"),
    ("torch_musa", "softmax"): ("torch_ops.py", "softmax", "softmax"),
    ("triton", "softmax"): ("triton_ops.py", "softmax", "softmax"),
    ("musa_cpp", "vector_add"): ("cpp_ops.py", "vector_add", "add"),
    ("tilelang", "vector_add"): ("tilelang_ops.py", "vector_add", "add"),
    ("native", "vector_add"): ("native_protocol_ops.py", "axpby", "add"),
    ("cpu", "axpby"): (None, "axpby", "axpby"),
    ("torch_musa", "axpby"): ("advanced_ops.py", "axpby", "axpby"),
    ("musa_cpp", "axpby"): ("advanced_cpp.py", "axpby", "axpby"),
    ("triton", "axpby"): ("advanced_triton.py", "axpby", "axpby"),
    ("tilelang", "axpby"): ("advanced_tilelang.py", "axpby", "axpby"),
    ("native", "axpby"): ("native_protocol_ops.py", "axpby", "axpby"),
}


def _attributes(case, descriptor):
    definitions = descriptor.get("attributes", {})
    if not isinstance(definitions, dict):
        raise ValueError("descriptor.attributes must be an object")
    attrs = case.setdefault("attrs", {})
    for name, spec in definitions.items():
        if name not in attrs:
            if "default" in spec:
                attrs[name] = copy.deepcopy(spec["default"])
            elif spec.get("required", True):
                raise ValueError(f"missing required scalar attribute: {name}")
        if name not in attrs:
            continue
        value, kind = attrs[name], spec.get("type")
        types = {"int": int, "float": (int, float), "bool": bool, "string": str, "list": list}
        if kind not in types:
            raise ValueError(f"unsupported attribute type {kind!r}: {name}")
        if not isinstance(value, types[kind]) or (kind in {"int", "float"} and isinstance(value, bool)):
            raise ValueError(f"attribute {name} must be {kind}")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"attribute {name} outside enum")


def _sample(rng, spec):
    if not isinstance(spec, dict):
        raise ValueError("fuzz parameter must be a generator object")
    kind = spec.get("kind", "choice")
    if kind == "choice":
        choices = spec.get("values")
        if not isinstance(choices, list) or not choices:
            raise ValueError("choice generator needs nonempty values")
        return copy.deepcopy(rng.choice(choices))
    if kind == "int":
        low, high = spec.get("min"), spec.get("max")
        if type(low) is not int or type(high) is not int or low > high:
            raise ValueError("int generator requires integer min <= max (inclusive)")
        step = spec.get("step", 1)
        if type(step) is not int or step <= 0:
            raise ValueError("int generator step must be positive")
        return low + step * rng.randrange((high - low) // step + 1)
    raise ValueError(f"unsupported fuzz generator kind: {kind}")


def _plugin_sample(entry, seed, index, timeout, base_dir):
    """A trusted generator still has a finite new-process wall-clock budget."""
    if not isinstance(entry, str) or ":" not in entry:
        raise ValueError("fuzz.generator must be module:function")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 60:
        raise ValueError("fuzz.timeout_s must be in (0, 60]")
    with tempfile.TemporaryDirectory(prefix="musaopst-fuzz-") as temporary:
        output = Path(temporary) / "sample.json"
        script = ("import sys\nfrom musaopst.util import load_entry,write_json\n"
                  "value=load_entry(sys.argv[1])(seed=int(sys.argv[2]), index=int(sys.argv[3]))\n"
                  "write_json(sys.argv[4],value)\n")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(base_dir or Path.cwd()), str(Path(__file__).parent.parent), env.get("PYTHONPATH", "")])
        process = subprocess.Popen([sys.executable, "-c", script, entry, str(seed), str(index), str(output)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   start_new_session=True, env=env)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise ValueError(f"custom fuzz generator exceeded {timeout}s") from None
        if process.returncode or not output.is_file():
            raise ValueError(f"custom fuzz generator failed ({process.returncode}): {stderr[-2000:]}")
        if output.stat().st_size > 1024 * 1024:
            raise ValueError("custom fuzz sample exceeds 1 MiB")
        value = json.loads(output.read_text())
        if not isinstance(value, dict) or any(not isinstance(k, str) or not k.isidentifier() for k in value):
            raise ValueError("custom fuzz generator must return a parameter-name/value object")
        return value


def finite_fuzz(template, fuzz, *, seed=0, count=None, schema_version=2, base_dir=None):
    requested = count if count is not None else fuzz.get("count", 16)
    attempts_limit = fuzz.get("max_attempts", max(100, requested * 20))
    if type(requested) is not int or not 1 <= requested <= 10000:
        raise ValueError("fuzz count must be an integer in [1, 10000]")
    if type(attempts_limit) is not int or not requested <= attempts_limit <= 100000:
        raise ValueError("max_attempts must be in [count, 100000]")
    parameters = fuzz.get("parameters", {})
    plugin = fuzz.get("generator")
    if not isinstance(parameters, dict) or (not parameters and not plugin):
        raise ValueError("fuzz requires nonempty parameters or an explicit generator entry")
    if parameters and plugin:
        raise ValueError("fuzz.parameters and fuzz.generator are mutually exclusive")
    rng = random.Random(seed)
    cases, rejected, seen = [], [], set()
    attempts = 0
    while len(cases) < requested and attempts < attempts_limit:
        attempts += 1
        selected = (_plugin_sample(plugin, seed, attempts - 1, fuzz.get("timeout_s", 5), base_dir)
                    if plugin else {name: _sample(rng, spec) for name, spec in parameters.items()})
        candidate = copy.deepcopy(template)
        candidate["parameters"] = {**candidate.get("parameters", {}), **{k: [v] for k, v in selected.items()}}
        candidate["seed"] = seed
        try:
            expanded = expand_config({"schema_version": schema_version, "cases": [candidate]})
            if len(expanded) != 1:
                raise ValueError("fuzz template must yield exactly one case per attempt")
            resolved = export_case(expanded[0])
            identity = content_digest({k: v for k, v in resolved.items() if k not in {"name", "seed"}})
            if identity in seen:
                rejected.append({"attempt": attempts, "reason": "duplicate", "selected": selected})
                continue
            seen.add(identity)
            resolved["name"] = f"{template['name']}-fuzz-{identity[:10]}"
            cases.append(resolved)
        except (ValueError, ConfigError) as exc:
            rejected.append({"attempt": attempts, "reason": str(exc), "selected": selected})
    return {"cases": cases, "requested": requested, "generated": len(cases), "attempts": attempts,
            "max_attempts": attempts_limit, "seed": seed, "complete": len(cases) == requested,
            "rejected": rejected, "determinism": "Python Random seed and frozen generated config"}


def _example_case(descriptor, output):
    backend, operation = descriptor.get("backend", "cpu"), descriptor.get("operation", "vector_add")
    pair = (backend, operation)
    if pair not in _EXAMPLES:
        raise ValueError(f"no built-in {operation} template for {backend}; provide an explicit case descriptor")
    filename, entry, reference = _EXAMPLES[pair]
    examples = Path(__file__).parent / "examples"
    if filename is None:
        (output / "operator_impl.py").write_text(
            '"""Explicit CPU platform-only implementation; not GPU emulation."""\nimport numpy as np\n\n'
            'def axpby(a, b, alpha=1.0, beta=1.0):\n    out = np.empty_like(a)\n'
            '    np.add(np.multiply(a, alpha), np.multiply(b, beta), out=out)\n    return out\n', encoding="utf-8")
    else:
        shutil.copyfile(examples / filename, output / "operator_impl.py")
    shutil.copyfile(examples / ("advanced_references.py" if operation == "axpby" else "references.py"),
                    output / "independent_reference.py")
    sources = ["operator_impl.py", "independent_reference.py"]
    if backend in {"native", "musa_cpp"}:
        for source in examples.iterdir():
            if source.suffix in {".mu", ".cpp", ".h", ".hpp"}:
                shutil.copyfile(source, output / source.name)
                sources.append(source.name)
    shape = descriptor.get("shape", [4, 129] if operation == "softmax" else [257])
    dtype = descriptor.get("dtype", "float32")
    tensor = {"shape": shape, "dtype": dtype, "layout": "contiguous", "distribution": {"kind": "normal"}}
    attrs = {"dim": -1} if operation == "softmax" else {"alpha": 2.0, "beta": -0.5} if operation == "axpby" else {}
    attrs.update(descriptor.get("attrs", {}))
    result = {"name": descriptor.get("name", operation), "backend": backend, "entry": "operator_impl:" + entry,
            "reference": "independent_reference:" + reference,
            "inputs": [copy.deepcopy(tensor) for _ in range(1 if operation == "softmax" else 2)],
            "outputs": [{"shape": shape, "dtype": dtype}], "attrs": attrs,
            "compare": {"atol": 0.02 if dtype == "bfloat16" else 0.002 if dtype == "float16" else 1e-5,
                        "rtol": 0.02 if dtype == "bfloat16" else 0.002 if dtype == "float16" else 1e-5}}
    if descriptor.get("schema_version", 2) == 2:
        result["sources"] = sources
    return result


def generate(descriptor_path, output_dir, *, seed=0, count=None):
    """Create a runnable project from a bounded descriptor, never infer an ABI."""
    descriptor_path = Path(descriptor_path).resolve()
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if "prototype_version" in descriptor:
        if not descriptor.get("runnable_descriptor"):
            raise ValueError("prototype has unresolved fields; complete its descriptor before generation")
        descriptor = descriptor["descriptor"]
    if descriptor.get("descriptor_version", 1) != 1:
        raise ValueError("only descriptor_version 1 is supported")
    output = _fresh(output_dir)
    write_json(output / "operator.descriptor.json", descriptor)
    case = copy.deepcopy(descriptor["case"]) if "case" in descriptor else _example_case(descriptor, output)
    _attributes(case, descriptor)
    case.setdefault("seed", seed)
    schema_version = descriptor.get("schema_version", 2)
    if "fuzz" in descriptor:
        generation = finite_fuzz(case, descriptor["fuzz"], seed=seed, count=count,
                                 schema_version=schema_version, base_dir=descriptor_path.parent)
        cases = generation["cases"]
    else:
        cases = [export_case(c) for c in expand_config({"schema_version": schema_version, "cases": [case]})]
        generation = {"complete": True, "generated": len(cases), "seed": seed}
    # Resolve external input files relative to the original descriptor, not output.
    from .workflows import _specs
    for generated in cases:
        if "case" in descriptor and "sources" in generated:
            generated["sources"] = [str((descriptor_path.parent / source).resolve()) for source in generated.get("sources", [])]
        for spec in _specs(generated["inputs"]):
            if "source" in spec:
                spec["source"]["path"] = str((descriptor_path.parent / spec["source"]["path"]).resolve())
    if cases:
        write_json(output / "cases.json", {"schema_version": schema_version, "cases": cases})
    write_json(output / "generation.json", generation)
    (output / "run.sh").write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\nexec python -m musaopst run cases.json --output "${1:-results}" "${@:2}"\n', encoding="utf-8")
    (output / "run.sh").chmod(0o755)
    (output / "README.md").write_text(
        "# 生成的算子测试工程\n\n先编辑 cases.json 和实现/独立参考，再运行 `python -m musaopst validate cases.json`。\n\n"
        "`bash run.sh results --device 0` 执行；`musaopst prepare cases.json --output prepared` 只生成输入和计划。\n\n"
        "现成模板仅证明明确协议下的接入方法，不会自动分析任意 .mu/.so；缺 SDK、动态 JIT 或后端不支持会明确报告。\n"
        "自定义 descriptor.case 的入口文件需在 Python 模块搜索路径可用；工程不会猜测或复制外部源码依赖。\n",
        encoding="utf-8")
    return {"directory": str(output), "config": str(output / "cases.json") if cases else None,
            "generated": len(cases), "complete": generation["complete"], "exit_code": 0 if generation["complete"] else 1}


def import_prototype(source_path, output_path):
    """Import explicit annotations or restricted INI, without executing source."""
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite prototype result: {output}")
    text = source.read_text(encoding="utf-8")
    unresolved = []
    descriptor = None
    if source.suffix == ".json":
        descriptor = json.loads(text)
    elif source.suffix == ".ini":
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(text)
        if "operator" not in parser:
            raise ValueError("restricted INI needs [operator] and [input.N]/[output.N] sections")
        base = dict(parser["operator"])
        case = {k: base[k] for k in ("name", "backend", "entry", "reference") if k in base}
        for kind in ("input", "output"):
            sections = sorted((s for s in parser.sections() if re.fullmatch(kind + r"\.\d+", s)),
                              key=lambda s: int(s.rsplit(".", 1)[1]))
            case[kind + "s"] = []
            for section in sections:
                values = dict(parser[section])
                item = {"shape": json.loads(values["shape"]), "dtype": values["dtype"]}
                if "name" in values:
                    item["name"] = values["name"]
                case[kind + "s"].append(item)
        case["attrs"] = json.loads(base.get("attrs", "{}"))
        descriptor = {"descriptor_version": 1, "schema_version": 2, "case": case}
    elif source.suffix == ".py":
        tree = ast.parse(text, filename=str(source))
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target, ast.Name) and target.id == "MUSA_OPST_DESCRIPTOR" for target in targets):
                    descriptor = ast.literal_eval(node.value)
        if descriptor is None:
            signatures = [{"name": node.name, "parameters": [arg.arg for arg in node.args.args]}
                          for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            descriptor = {"candidate_functions": signatures}
            unresolved = ["shape, dtype, backend and independent reference cannot be inferred from a Python signature"]
    elif source.suffix in {".cpp", ".cc", ".h", ".hpp", ".mu"}:
        found = re.search(r"/\*\s*MUSA_OPST_DESCRIPTOR\s*(\{.*?\})\s*\*/", text, re.S)
        if not found:
            raise ValueError("C++/MUSA import requires /* MUSA_OPST_DESCRIPTOR {JSON} */ annotation; arbitrary ABI inference is unsupported")
        descriptor = json.loads(found.group(1))
    else:
        raise ValueError("prototype format must be json, ini, py, cpp/cc/h/hpp/mu")
    result = {"prototype_version": 1, "source": str(source), "descriptor": descriptor,
              "unresolved": unresolved, "runnable_descriptor": not unresolved}
    if not unresolved and "case" in descriptor:
        expand_config({"schema_version": descriptor.get("schema_version", 2), "cases": [descriptor["case"]]})
    write_json(output, result)
    return result
