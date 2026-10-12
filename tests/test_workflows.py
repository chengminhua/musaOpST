# SPDX-License-Identifier: MulanPSL-2.0
import copy
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from musaopst.artifacts import (bundle, case_fingerprint, file_manifest, resume_identity,
                                 verified_reuse, verify_manifest)
from musaopst.config import expand_config, load_config
from musaopst.data import make_inputs, tensor_metadata
from musaopst.generation import finite_fuzz, generate, import_prototype
from musaopst.models import extract
from musaopst.util import sha256, write_json
from musaopst.workflows import prepare, prepared_config, recompare, replay_config


def base_case():
    return {"name": "identity", "backend": "cpu", "entry": "musaopst.examples.cpu_ops:identity",
            "reference": "musaopst.examples.references:identity",
            "inputs": [{"shape": [3], "dtype": "float32", "distribution": {"kind": "fixed", "values": [1, 2, 3]}}],
            "outputs": [{"shape": [3], "dtype": "float32"}]}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, case=None, version=2):
        path = self.root / "cases.json"
        write_json(path, {"schema_version": version, "cases": [case or base_case()]})
        return path

    def evidence(self, *, wrong=False, case=None):
        case = expand_config({"schema_version": 2, "cases": [case or base_case()]})[0]
        directory = self.root / "evidence" / "case-0000"
        directory.mkdir(parents=True)
        inputs = make_inputs(case)
        np.savez(directory / "inputs.npz", input_0=inputs[0])
        actual = np.array(inputs[0], copy=True)
        if wrong:
            actual += 1
        np.savez(directory / "actual.npz", output_0=actual)
        np.savez(directory / "reference.npz", output_0=inputs[0].astype(np.float64))
        record = {"name": case["name"], "backend": "cpu", "required": True, "config": case,
                  "status": "ACCURACY_FAILED" if wrong else "PASS", "case_result": "case-0000/result.json",
                  "inputs": [tensor_metadata(inputs[0], case["inputs"][0]["dtype"])],
                  "actual_metadata": [tensor_metadata(actual, case["outputs"][0]["dtype"])],
                  "snapshot_hashes": {name: sha256(directory / name) for name in ("inputs.npz", "actual.npz", "reference.npz")}}
        write_json(directory / "result.json", record)
        source = directory.parent / "results.json"
        write_json(source, {"schema_version": 1, "cases": [record], "environment": {}, "replay_module_search_paths": []})
        return source

    def test_prepare_cpu_and_frozen_inputs(self):
        path = self.config()
        result = prepare(path, self.root / "prepared", stage="data")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["device_execution"])
        frozen = prepared_config(self.root / "prepared", self.root / "frozen")
        self.assertEqual(load_config(frozen)[0]["_schema_version"], 2)
        np.testing.assert_array_equal(make_inputs(load_config(frozen)[0], frozen.parent)[0], [1, 2, 3])

    def test_preparation_integrity_rejects_input_tamper(self):
        prepare(self.config(), self.root / "prepared")
        (self.root / "prepared/case-0000/inputs.npz").write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "integrity"):
            prepared_config(self.root / "prepared", self.root / "frozen")

    def test_v1_prepared_replay_preserves_legacy_dtype_policy(self):
        case = base_case()
        case["outputs"].append({"shape": [3], "dtype": "float64"})
        case["compare_by_dtype"] = {"float32": {"atol": 0.1}, "float64": {"atol": 0.9}}
        prepare(self.config(case, version=1), self.root / "prepared")
        frozen = prepared_config(self.root / "prepared", self.root / "frozen")
        replay = load_config(frozen)[0]
        self.assertEqual(replay["outputs"][0]["compare"]["atol"], 0.1)
        self.assertEqual(replay["outputs"][1]["compare"]["atol"], 0.1)

    def test_preparation_integrity_rejects_config_tamper(self):
        prepare(self.config(), self.root / "prepared")
        (self.root / "prepared/config.resolved.json").write_text('{"cases": []}')
        with self.assertRaisesRegex(ValueError, "digest"):
            prepared_config(self.root / "prepared", self.root / "frozen")

    def test_prepare_runtime_jit_is_explicitly_unsupported(self):
        case = base_case()
        case["backend"] = "triton"
        result = prepare(self.config(case), self.root / "prepared", stage="build")
        self.assertEqual(result["cases"][0]["status"], "UNSUPPORTED")
        self.assertEqual(result["exit_code"], 1)

    def test_cpp_build_hook_and_prebuilt_metadata_flow(self):
        case = base_case()
        case.update(backend="musa_cpp", entry="musaopst.examples.advanced_cpp:identity")
        binary = self.root / "fixture.so"
        binary.write_bytes(b"unit-test-metadata-only-not-a-GPU-binary")
        manifest = {"binary": str(binary), "artifact_sha256": sha256(binary), "module_name": "fixture",
                    "build_kind": "musa_cpp_extension", "arch": "mp_31", "cache_hit": False}
        with patch("musaopst.backends.build_cache.build_cpp_descriptor", return_value=manifest):
            result = prepare(self.config(case), self.root / "prepared", stage="build", arch="mp_31")
        self.assertEqual(result["cases"][0]["status"], "BUILT")
        frozen = prepared_config(self.root / "prepared", self.root / "frozen")
        artifact = load_config(frozen)[0]["artifact"]
        self.assertEqual(artifact["module_name"], "fixture")
        self.assertEqual(sha256(artifact["binary"]), sha256(binary))
        self.assertFalse(result["device_execution"])

    def test_cpp_timeout_is_not_build_failure(self):
        case = base_case()
        case.update(backend="musa_cpp", entry="musaopst.examples.advanced_cpp:identity")
        with patch("musaopst.backends.build_cache.build_cpp_descriptor", side_effect=TimeoutError("controlled build timeout")):
            result = prepare(self.config(case), self.root / "prepared", stage="build", arch="mp_31")
        self.assertEqual(result["cases"][0]["status"], "TIMEOUT")
        self.assertEqual(result["exit_code"], 1)

    def test_cpp_factory_without_build_hook_explicitly_unsupported(self):
        case = base_case()
        case["backend"] = "musa_cpp"
        result = prepare(self.config(case), self.root / "prepared", stage="build")
        self.assertEqual(result["cases"][0]["status"], "UNSUPPORTED")

    def test_native_build_uses_case_timeout_and_reports_timeout(self):
        case = base_case()
        case.update(backend="native", entry="musaopst.examples.native_protocol_ops:identity", timeout_s=7)
        with patch("musaopst.backends.build_cache.build_native_descriptor", side_effect=TimeoutError("controlled native timeout")) as build:
            result = prepare(self.config(case), self.root / "prepared", stage="build", arch="mp_31")
        self.assertEqual(build.call_args.kwargs["timeout_s"], 7)
        self.assertEqual(result["cases"][0]["status"], "TIMEOUT")
        self.assertEqual(result["exit_code"], 1)

    def test_recompare_reports_mismatch_and_bounded_csv(self):
        source = self.evidence(wrong=True)
        original = source.read_bytes()
        result = recompare(source, self.root / "recompared", csv_limit=2)
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["new_gpu_passed"], 0)
        self.assertEqual(result["cases"][0]["csv"]["rows"], 2)
        self.assertTrue(result["cases"][0]["csv"]["truncated"])
        self.assertEqual(source.read_bytes(), original)
        tolerant = recompare(source, self.root / "tolerant", policy={"atol": 2, "rtol": 0})
        self.assertEqual(tolerant["exit_code"], 0)

    def test_recompare_detects_tampered_snapshot(self):
        source = self.evidence()
        np.savez(source.parent / "case-0000/actual.npz", output_0=np.ones(3, np.float32))
        result = recompare(source, self.root / "recompared")
        self.assertEqual(result["cases"][0]["status"], "RECOMPARE_UNAVAILABLE")
        self.assertIn("digest", result["cases"][0]["message"])

    def test_replay_uses_quantized_snapshot_not_new_random_data(self):
        source = self.evidence()
        config = replay_config(source, self.root / "replay")
        case = load_config(config)[0]
        self.assertNotIn("distribution", case["inputs"][0])
        np.testing.assert_array_equal(make_inputs(case, config.parent)[0], [1, 2, 3])

    def test_replay_transpose_preserves_real_strides(self):
        case = base_case()
        case["inputs"] = [{"shape": [3, 2], "base_shape": [2, 3], "dtype": "float32",
                           "transforms": [{"kind": "transpose", "axes": [1, 0]}]}]
        case["outputs"] = [{"shape": [3, 2], "dtype": "float32"}]
        source = self.evidence(case=case)
        config = replay_config(source, self.root / "replay")
        case = load_config(config)[0]
        actual = make_inputs(case, config.parent)[0]
        self.assertEqual(tuple(s // actual.itemsize for s in actual.strides), (1, 3))
        self.assertNotIn("transforms", case["inputs"][0])

    def test_bundle_manifest_and_no_fake_portable_environment(self):
        source = self.evidence()
        result = bundle(source, self.root / "evidence.tar.gz")
        self.assertFalse(result["portable_environment"])
        with tarfile.open(result["path"]) as archive:
            self.assertIn("evidence/results.json", archive.getnames())
            manifest = json.load(archive.extractfile("bundle.json"))
            self.assertEqual(manifest["files"]["evidence/results.json"]["sha256"], sha256(source))
        with self.assertRaisesRegex(ValueError, "overwrite"):
            bundle(source, self.root / "evidence.tar.gz")

    def test_bundle_declared_sources_preserve_paths_and_reject_credentials(self):
        source = self.evidence()
        helper = self.root / "helpers" / "math.py"
        helper.parent.mkdir()
        helper.write_text("def identity(x): return x\n")
        result = json.loads(source.read_text())
        result["cases"][0]["config"].update(_config_dir=str(self.root), sources=["helpers/math.py"])
        write_json(source, result)
        archive = bundle(source, self.root / "declared.tar.gz")
        with tarfile.open(archive["path"]) as tar:
            self.assertIn("source/helpers/math.py", tar.getnames())
            manifest = json.load(tar.extractfile("bundle.json"))
            self.assertFalse(manifest["declared_sources"][0]["relocation_required"])
        credential = self.root / ".env"
        credential.write_text("DUMMY_FIXTURE=not-a-secret\n")
        result["cases"][0]["config"]["sources"] = [".env"]
        write_json(source, result)
        with self.assertRaisesRegex(ValueError, "credential"):
            bundle(source, self.root / "must-not-exist.tar.gz")
        self.assertFalse((self.root / "must-not-exist.tar.gz").exists())

    def test_bundle_declared_source_directory_is_not_recursed(self):
        source = self.evidence()
        result = json.loads(source.read_text())
        result["cases"][0]["config"].update(_config_dir=str(self.root), sources=[str(self.root)])
        write_json(source, result)
        with self.assertRaisesRegex(ValueError, "regular file"):
            bundle(source, self.root / "must-not-exist.tar.gz")

    def test_generated_project_is_valid_and_sources_are_editable(self):
        descriptor = self.root / "operator.json"
        write_json(descriptor, {"descriptor_version": 1, "operation": "vector_add", "backend": "cpu"})
        result = generate(descriptor, self.root / "project")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(len(load_config(result["config"])), 1)
        self.assertTrue((self.root / "project/operator_impl.py").is_file())
        self.assertTrue((self.root / "project/independent_reference.py").is_file())

    def test_generated_axpby_contracts_cover_all_backend_templates(self):
        for backend in ("cpu", "torch_musa", "musa_cpp", "triton", "tilelang", "native"):
            descriptor = self.root / (backend + ".json")
            write_json(descriptor, {"backend": backend, "operation": "axpby"})
            result = generate(descriptor, self.root / backend)
            case = load_config(result["config"])[0]
            self.assertEqual(case["attrs"], {"alpha": 2.0, "beta": -0.5})
            self.assertEqual(case["backend"], backend)
            if backend == "native":
                self.assertTrue((self.root / backend / "native_host.hpp").is_file())

    def test_prebuilt_native_preparation_does_not_invoke_compiler(self):
        binary = self.root / "fixture-host"
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        plugin = self.root / "prebuilt_fixture.py"
        descriptor = {"protocol": "musaopst_native_v2", "binary": str(binary), "artifact_sha256": sha256(binary), "operator": "identity"}
        plugin.write_text("def factory(case, inputs, workdir):\n    return " + repr(descriptor) + "\n")
        case = base_case()
        case.update(backend="native", entry="prebuilt_fixture:factory")
        result = prepare(self.config(case), self.root / "prepared", stage="build")
        self.assertEqual(result["cases"][0]["status"], "BUILT")
        self.assertEqual(result["cases"][0]["build"]["build_kind"], "explicit_prebuilt")
        frozen = prepared_config(self.root / "prepared", self.root / "frozen")
        self.assertTrue((frozen.parent / "case-0000/native_operator").is_file())
        self.assertFalse(result["device_execution"])

    def test_fuzz_is_finite_deterministic_and_reports_rejections(self):
        case = base_case()
        case["inputs"][0] = {"shape": ["${n}"], "dtype": "float32"}
        case["outputs"][0]["shape"] = ["${n}"]
        case["constraints"] = [{"left": "${n}", "op": "divisible", "right": 2}]
        fuzz = {"count": 2, "max_attempts": 30, "parameters": {"n": {"kind": "int", "min": 1, "max": 4}}}
        a, b = finite_fuzz(case, fuzz, seed=4), finite_fuzz(case, fuzz, seed=4)
        self.assertEqual(a, b)
        self.assertTrue(a["complete"])
        self.assertTrue(a["rejected"])
        fuzz["count"] = 3
        incomplete = finite_fuzz(case, fuzz, seed=4)
        self.assertFalse(incomplete["complete"])
        self.assertEqual(incomplete["attempts"], 30)

    def test_prototype_import_does_not_execute_python(self):
        source = self.root / "prototype.py"
        source.write_text("raise RuntimeError('must not execute')\nMUSA_OPST_DESCRIPTOR = " + repr({"case": base_case()}) + "\n")
        result = import_prototype(source, self.root / "imported.json")
        self.assertTrue(result["runnable_descriptor"])
        generated = generate(self.root / "imported.json", self.root / "imported-project")
        self.assertEqual(generated["generated"], 1)

    def test_custom_fuzz_generator_has_finite_process_timeout(self):
        plugin = self.root / "custom_generator.py"
        plugin.write_text("def sample(seed, index):\n    return {'n': index + 1}\n\ndef slow(seed, index):\n    import time\n    time.sleep(3)\n    return {'n': 1}\n")
        case = base_case()
        case["inputs"][0] = {"shape": ["${n}"], "dtype": "float32"}
        case["outputs"][0]["shape"] = ["${n}"]
        generated = finite_fuzz(case, {"generator": "custom_generator:sample", "count": 2}, base_dir=self.root)
        self.assertTrue(generated["complete"])
        with self.assertRaisesRegex(ValueError, "exceeded"):
            finite_fuzz(case, {"generator": "custom_generator:slow", "count": 1, "timeout_s": 0.15}, base_dir=self.root)

    def test_arbitrary_cpp_abi_not_guessed(self):
        source = self.root / "anything.mu"
        source.write_text("__global__ void op(float* data) {}")
        with self.assertRaisesRegex(ValueError, "arbitrary ABI"):
            import_prototype(source, self.root / "imported.json")

    def test_missing_model_parser_is_explicit(self):
        model = self.root / "dummy.onnx"
        model.write_bytes(b"not parsed")
        with patch("musaopst.models.importlib.import_module", side_effect=ModuleNotFoundError("optional parser")):
            result = extract(model, self.root / "model-candidates")
        self.assertEqual(result["status"], "ENVIRONMENT_UNAVAILABLE")
        self.assertFalse(result["runnable_config"])
        self.assertFalse(result["device_execution"])

    def test_real_onnx_node_specifications_when_dependency_available(self):
        try:
            import onnx
        except ImportError:
            self.skipTest("optional ONNX parser not installed")
        helper, dtype = onnx.helper, onnx.TensorProto.FLOAT
        inputs = [helper.make_tensor_value_info(name, dtype, [2, 3]) for name in ("a", "b")]
        output = helper.make_tensor_value_info("out", dtype, [2, 3])
        graph = helper.make_graph([helper.make_node("Add", ["a", "b"], ["out"], name="add")], "model", inputs, [output])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        path = self.root / "real.onnx"
        onnx.save(model, str(path))
        result = extract(path, self.root / "extracted")
        self.assertEqual(result["status"], "EXTRACTED")
        self.assertEqual(result["candidates"][0]["outputs"][0]["shape"], [2, 3])
        self.assertFalse(result["runnable_config"])

    def test_real_graphdef_node_specifications_when_dependency_available(self):
        try:
            from tensorboard.compat.proto import graph_pb2, types_pb2
        except ImportError:
            self.skipTest("optional compatible GraphDef protobuf not installed")
        graph = graph_pb2.GraphDef()
        node = graph.node.add(name="input", op="Placeholder")
        node.attr["dtype"].type = types_pb2.DT_FLOAT
        node.attr["shape"].shape.dim.add(size=7)
        path = self.root / "real.pb"
        path.write_bytes(graph.SerializeToString())
        result = extract(path, self.root / "extracted")
        self.assertEqual(result["status"], "EXTRACTED")
        self.assertEqual(result["candidates"][0]["outputs"][0]["shape"], [7])
        self.assertFalse(result["device_execution"])

    def test_artifact_manifest_detects_escape_and_changed_bytes(self):
        file = self.root / "input.bin"
        file.write_bytes(b"abc")
        manifest = file_manifest(self.root)
        self.assertTrue(verify_manifest(self.root, manifest)["passed"])
        file.write_bytes(b"abd")
        self.assertFalse(verify_manifest(self.root, manifest)["passed"])
        self.assertFalse(verify_manifest(self.root, {"../outside": {"sha256": "", "bytes": 0}})["passed"])

    def test_resume_does_not_reuse_unknown_or_corrupt_evidence(self):
        case = base_case()
        unknown = resume_identity(case, source={}, environment={}, entries=[])
        self.assertFalse(unknown["reusable"])
        identity = resume_identity(case, source={"content_sha256": "abc"}, environment={"cpu": "aarch64"},
                                   entries=[{"sha256": "impl"}, {"sha256": "ref"}])
        evidence = self.root / "actual.npz"
        evidence.write_bytes(b"fixture")
        record = {"status": "PASS", "comparison": {"passed": True}, "resume_identity": identity,
                  "artifact_manifest": file_manifest(self.root)}
        self.assertTrue(verified_reuse(record, identity["fingerprint"], self.root)["reusable"])
        evidence.write_bytes(b"changed")
        self.assertFalse(verified_reuse(record, identity["fingerprint"], self.root)["reusable"])


if __name__ == "__main__":
    unittest.main()
