# SPDX-License-Identifier: MulanPSL-2.0
"""Optional ONNX/TF graph specification extraction, never model execution."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from .util import sha256, write_json
from .workflows import _fresh


def _plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"encoding": "hex", "value": value.hex()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "SerializeToString"):
        blob = value.SerializeToString()
        result = {"protobuf_type": value.DESCRIPTOR.full_name, "bytes": len(blob),
                  "sha256": hashlib.sha256(blob).hexdigest()}
        if hasattr(value, "dims"):
            result["shape"] = list(value.dims)
        if hasattr(value, "data_type"):
            result["data_type_code"] = value.data_type
        return result
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return {"unresolved_type": type(value).__name__}


def _onnx(path):
    onnx = importlib.import_module("onnx")
    # External tensor data is not loaded or followed outside the supplied model.
    model = onnx.load(str(path), load_external_data=False)
    inferred = False
    warnings = []
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
        inferred = True
    except Exception as exc:
        warnings.append(f"shape inference unavailable: {type(exc).__name__}: {exc}")
    dtype_map = {1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32", 7: "int64",
                 9: "bool", 10: "float16", 11: "float64", 12: "uint32", 13: "uint64", 16: "bfloat16"}
    tensors = {}
    for item in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        tensor = item.type.tensor_type
        if not item.type.HasField("tensor_type"):
            tensors[item.name] = {"name": item.name, "kind": "non_tensor", "type": str(item.type)}
            continue
        dims = ([dim.dim_value if dim.HasField("dim_value") else dim.dim_param if dim.dim_param else None
                 for dim in tensor.shape.dim] if tensor.HasField("shape") else None)
        tensors[item.name] = {"name": item.name, "shape": dims,
                              "dtype": dtype_map.get(tensor.elem_type, f"onnx_dtype_{tensor.elem_type}")}
    initializers = {}
    for tensor in model.graph.initializer:
        initializers[tensor.name] = {"name": tensor.name, "shape": list(tensor.dims),
                                     "dtype": dtype_map.get(tensor.data_type, f"onnx_dtype_{tensor.data_type}"),
                                     "initializer": True, "external_data": bool(tensor.external_data),
                                     "tensor_sha256": hashlib.sha256(tensor.SerializeToString()).hexdigest()}
        tensors[tensor.name] = initializers[tensor.name]
    candidates = []
    for index, node in enumerate(model.graph.node):
        inputs = [{"name": name, "kind": "absent"} if not name else
                  dict(tensors.get(name, {"name": name, "shape": None, "dtype": None})) for name in node.input]
        outputs = [dict(tensors.get(name, {"name": name, "shape": None, "dtype": None})) for name in node.output]
        unresolved = ["backend implementation entry", "independent reference entry", "runtime input values"]
        if any(spec.get("shape") is None or spec.get("dtype") is None or
               any(not isinstance(dim, int) for dim in (spec.get("shape") or []))
               for spec in inputs + outputs if spec.get("kind") != "absent"):
            unresolved.append("concrete shape/dtype bindings")
        candidates.append({"name": node.name or f"{node.op_type}_{index}", "op_type": node.op_type,
                           "domain": node.domain, "inputs": inputs, "outputs": outputs,
                           "attrs": {attr.name: _plain(onnx.helper.get_attribute_value(attr)) for attr in node.attribute},
                           "unresolved": unresolved})
    return {"format": "onnx", "extractor_version": onnx.__version__, "shape_inference": inferred,
            "opsets": {x.domain or "ai.onnx": x.version for x in model.opset_import},
            "candidates": candidates, "initializers": initializers, "warnings": warnings}


def _tensorflow(path):
    try:
        graph_pb2 = importlib.import_module("tensorflow.core.framework.graph_pb2")
        types_pb2 = importlib.import_module("tensorflow.core.framework.types_pb2")
        parser = "tensorflow GraphDef protobuf"
    except ImportError:
        # TensorBoard ships the same GraphDef schema without requiring the
        # TensorFlow executor or initializing its device runtime.
        graph_pb2 = importlib.import_module("tensorboard.compat.proto.graph_pb2")
        types_pb2 = importlib.import_module("tensorboard.compat.proto.types_pb2")
        parser = "tensorboard compatible GraphDef protobuf"
    graph = graph_pb2.GraphDef()
    graph.ParseFromString(path.read_bytes())
    names = {1: "float32", 2: "float64", 3: "int32", 4: "uint8", 5: "int16", 6: "int8", 9: "int64",
             10: "bool", 14: "bfloat16", 17: "uint16", 19: "float16", 22: "uint32", 23: "uint64"}

    def shape(value):
        return None if value.unknown_rank else [d.size if d.size >= 0 else d.name or None for d in value.dim]

    def dtype(code):
        return names.get(code, types_pb2.DataType.Name(code) if code else None)

    specifications = {}
    for node in graph.node:
        attr = node.attr
        shapes = [shape(s) for s in attr["_output_shapes"].list.shape] if "_output_shapes" in attr else []
        if not shapes and "shape" in attr:
            shapes = [shape(attr["shape"].shape)]
        if not shapes and "value" in attr:
            shapes = [shape(attr["value"].tensor.tensor_shape)]
        types = list(attr["Tout"].list.type) if "Tout" in attr else []
        if not types:
            types = [attr["dtype"].type if "dtype" in attr else attr["T"].type if "T" in attr else 0]
        count = max(len(shapes), len(types), 1)
        specifications[node.name] = [{"name": f"{node.name}:{i}", "shape": shapes[i] if i < len(shapes) else None,
                                       "dtype": dtype(types[i] if i < len(types) else 0)} for i in range(count)]
    candidates = []
    for node in graph.node:
        inputs, control_inputs = [], []
        for name in node.input:
            if name.startswith("^"):
                control_inputs.append(name[1:])
                continue
            parts = name.rsplit(":", 1)
            origin, index = (parts[0], int(parts[1])) if len(parts) == 2 and parts[1].isdigit() else (name, 0)
            known = specifications.get(origin, [])
            inputs.append(dict(known[index]) if index < len(known) else {"name": name, "shape": None, "dtype": None})
        attrs = {}
        for name, value in node.attr.items():
            field = value.WhichOneof("value")
            if field in {"s", "i", "f", "b"}:
                attrs[name] = _plain(getattr(value, field))
            elif field == "type":
                attrs[name] = dtype(value.type)
            elif field == "shape":
                attrs[name] = shape(value.shape)
            elif field == "list":
                attrs[name] = {key: [_plain(v) for v in getattr(value.list, key)] for key in
                               ("s", "i", "f", "b", "type", "shape", "tensor", "func") if len(getattr(value.list, key))}
            else:
                attrs[name] = _plain(getattr(value, field)) if field else None
        candidates.append({"name": node.name, "op_type": node.op, "inputs": inputs,
                           "outputs": specifications[node.name], "attrs": attrs, "control_inputs": control_inputs,
                           "unresolved": ["backend implementation entry", "independent reference entry", "runtime input values",
                                          "GraphDef may omit shapes or output count; verify signature against operator definition"]})
    return {"format": "tensorflow_graphdef", "parser": parser, "candidates": candidates,
            "warnings": ["GraphDef specifications only; SavedModel execution and runtime tensor capture are not implemented"]}


def extract(model_path, output_dir, *, format=None):
    path = Path(model_path).resolve()
    if not path.is_file():
        raise ValueError(f"model does not exist: {path}")
    selected = format or ("onnx" if path.suffix.lower() == ".onnx" else "tensorflow" if path.suffix.lower() == ".pb" else None)
    if selected not in {"onnx", "tensorflow", "tf"}:
        raise ValueError("model format must be onnx or tensorflow (.pb GraphDef)")
    output = _fresh(output_dir)
    try:
        extracted = _onnx(path) if selected == "onnx" else _tensorflow(path)
        extracted.update(status="EXTRACTED", exit_code=0)
    except ImportError as exc:
        extracted = {"status": "ENVIRONMENT_UNAVAILABLE", "format": selected, "candidates": [],
                     "message": f"optional model parser dependency unavailable: {exc}", "exit_code": 1}
    except Exception as exc:
        extracted = {"status": "EXTRACTION_FAILED", "format": selected, "candidates": [],
                     "message": f"{type(exc).__name__}: {exc}", "exit_code": 1}
    extracted.update(extraction_version=1, source=str(path), source_sha256=sha256(path),
                     device_execution=False, runnable_config=False,
                     meaning="operator specification candidates; not captured real inputs or verified reference implementations")
    write_json(output / "case_candidates.json", extracted)
    lines = ["# 模型算子规格抽取", "", f"状态：{extracted['status']}；候选数量：{len(extracted['candidates'])}。", "",
             "候选仍需实现入口、独立参考与实际数据/造数政策；不能直接作为GPU验收结果。", "",
             "| 节点 | 算子 | 输入数 | 输出数 |", "|---|---|---:|---:|"]
    for item in extracted["candidates"]:
        lines.append(f"| {item['name'].replace('|', '/')} | {item['op_type']} | {len(item['inputs'])} | {len(item['outputs'])} |")
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return extracted
