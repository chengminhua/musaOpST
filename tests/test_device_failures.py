# SPDX-License-Identifier: MulanPSL-2.0
"""Offline routing tests only; fault objects never execute or simulate a GPU kernel."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from musaopst.config import expand_config
from musaopst.errors import BuildError, EnvironmentError, is_fatal_device_error
from musaopst.runner import run_suite
from musaopst.worker import execute


def base_case(name="fault", backend="torch_musa"):
    return {"name": name, "backend": backend,
            "entry": "musaopst.examples.cpu_ops:add",
            "inputs": [{"shape": [3], "dtype": "float32"}] * 2,
            "outputs": [{"shape": [3], "dtype": "float32"}],
            "reference": "musaopst.examples.references:add",
            "compare": {"atol": 1e-6, "rtol": 1e-5}}


class ControlledBackend:
    """CPU arrays sufficient to test exception routing; not acceptance evidence."""
    def __init__(self, inputs, error=None, cleanup_error=None):
        self.inputs, self.error, self.cleanup_error = inputs, error, cleanup_error

    def prepare(self):
        if self.error:
            raise self.error

    def run(self):
        return [self.inputs[0] + self.inputs[1]]

    def synchronize(self):
        pass

    def snapshot(self, values):
        return [x.copy() for x in values], [str(x.dtype) for x in values]

    def metadata(self):
        return {"offline_exception_routing_fixture": True}

    def memory(self):
        return {}

    def close(self):
        if self.cleanup_error:
            raise self.cleanup_error


class FatalDeviceTests(unittest.TestCase):
    def test_precise_runtime_faults_and_normal_errors(self):
        for message in ("MUSA error: an illegal memory access was encountered", "device-side assert triggered",
                        "musaErrorIllegalAddress", "musaErrorLaunchTimeout", "uncorrectable ECC",
                        "device is unavailable", "unspecified launch failure"):
            self.assertTrue(is_fatal_device_error(RuntimeError(message)), message)
        for message in ("No module named torch_musa", "MUSA is not available", "out of memory",
                        "mcc compilation failed", "worker timeout", "unsupported dtype", "no device"):
            self.assertFalse(is_fatal_device_error(RuntimeError(message)), message)

    def test_wrapped_causes_and_context_cycles(self):
        error = EnvironmentError("initialization failed")
        error.__cause__ = RuntimeError("musaErrorIllegalAddress")
        self.assertTrue(is_fatal_device_error(error))
        first, second = BuildError("plain build error"), RuntimeError("no hardware fault")
        first.__context__, second.__context__ = second, first
        self.assertFalse(is_fatal_device_error(first))

    def execute_fixture(self, error=None, cleanup_error=None, reference_error=False, backend="torch_musa"):
        case = expand_config({"schema_version": 1, "cases": [base_case(backend=backend)]})[0]
        with tempfile.TemporaryDirectory(prefix="opst-routing-") as tmp:
            def make_backend(case, inputs, workdir):
                return ControlledBackend(inputs, error, cleanup_error)
            target = "musaopst.backends.cpu.CPUBackend" if backend == "cpu" else "musaopst.backends.create_backend"
            with patch(target, side_effect=make_backend):
                if reference_error:
                    with patch("musaopst.examples.references.add", side_effect=RuntimeError("device-side assert")):
                        return execute({"case": case, "workdir": tmp, "config_dir": tmp, "device": 0})
                return execute({"case": case, "workdir": tmp, "config_dir": tmp, "device": 0})

    def test_typed_backend_exceptions_preserve_fatal_marker(self):
        for exception, status in ((EnvironmentError, "ENVIRONMENT_ERROR"), (BuildError, "BUILD_FAILED")):
            error = exception("wrapped backend failure")
            error.__cause__ = RuntimeError("musaErrorIllegalAddress")
            result = self.execute_fixture(error=error)
            self.assertEqual(result["status"], status)
            self.assertTrue(result["fatal_device_error"])

    def test_cleanup_device_fault_marks_result(self):
        result = self.execute_fixture(cleanup_error=RuntimeError("musaErrorLaunchFailure"))
        self.assertEqual(result["status"], "EXECUTION_FAILED")
        self.assertTrue(result["fatal_device_error"])

    def test_cpu_and_reference_only_messages_are_not_device_faults(self):
        result = self.execute_fixture(error=RuntimeError("device-side assert"), backend="cpu")
        self.assertFalse(result.get("fatal_device_error", False))
        result = self.execute_fixture(reference_error=True)
        self.assertEqual(result["status"], "REFERENCE_FAILED")
        self.assertFalse(result.get("fatal_device_error", False))

    def test_lost_gpu_worker_stops_gpu_but_allows_following_cpu(self):
        real_popen = subprocess.Popen
        class LostWorker:
            def wait(self, timeout=None):
                return -11
        calls = []
        def controlled_popen(command, **kwargs):
            if "musaopst.worker" not in command:
                return real_popen(command, **kwargs)
            calls.append(command)
            if len(calls) == 1:
                return LostWorker()
            return real_popen(command, **kwargs)
        with tempfile.TemporaryDirectory(prefix="opst-lost-worker-") as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text(json.dumps({"schema_version": 1, "cases": [
                base_case("lost"), base_case("later_gpu"), base_case("later_cpu", "cpu")]}))
            with patch("musaopst.runner.subprocess.Popen", side_effect=controlled_popen):
                result = run_suite(config, root / "result")
            self.assertEqual([c["status"] for c in result["cases"]], ["EXECUTION_FAILED", "SKIPPED", "PASS"])
            self.assertEqual(len(calls), 2)
            self.assertTrue(result["cases"][0]["fatal_device_error"])
            self.assertTrue(result["run_complete"])
            self.assertEqual(len(result["planned_cases"]), 3)
            self.assertFalse(result["summary"]["passed"])

    def test_incomplete_run_never_renders_green(self):
        from musaopst.reporting import render_reports
        import xml.etree.ElementTree as ET
        passed_case = self.execute_fixture(backend="cpu")
        self.assertEqual(passed_case["status"], "PASS")
        for complete in (False, True):
            report = {"schema_version": 1, "run_complete": complete,
                      "cases": [passed_case], "planned_cases": [
                          {"name": "fault", "backend": "cpu", "required": True},
                          {"name": "not_executed", "backend": "cpu", "required": True}]}
            with tempfile.TemporaryDirectory(prefix="opst-incomplete-") as tmp:
                self.assertEqual(render_reports(report, tmp), 1)
                self.assertFalse(report["summary"]["passed"])
                self.assertEqual(report["summary"]["pending"], 1)
                self.assertGreater(int(ET.parse(Path(tmp) / "junit.xml").getroot().get("failures")), 0)

    def test_native_lost_completion_is_unsafe_to_continue(self):
        from types import SimpleNamespace
        from musaopst.backends.native import NativeBackend
        from musaopst.errors import DeviceStateUnknownError
        for code, stdout in ((-11, ""), (0, "not JSON"), (0, "{}")):
            with tempfile.TemporaryDirectory(prefix="opst-native-routing-") as tmp:
                backend = NativeBackend({}, [np.ones(3, np.float32)] * 2, tmp)
                backend.binary = "never-executed"
                backend.input_files = [Path(tmp) / "a", Path(tmp) / "b"]
                process = SimpleNamespace(returncode=code, stdout=stdout, stderr="controlled offline test")
                with patch("musaopst.backends.native.subprocess.run", return_value=process):
                    with self.assertRaises(DeviceStateUnknownError) as caught:
                        backend.run()
                    self.assertTrue(is_fatal_device_error(caught.exception))
