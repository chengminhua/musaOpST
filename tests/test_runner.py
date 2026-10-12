# SPDX-License-Identifier: MulanPSL-2.0
"""Offline end-to-end checks of the tester, using fresh CPU workers only."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET

from musaopst.runner import run_suite
from musaopst.reporting import render_reports, summarize

ROOT = Path(__file__).resolve().parents[1]


def case(name="add", **updates):
    value = {
        "name": name, "backend": "cpu", "entry": "musaopst.examples.cpu_ops:add",
        "inputs": [{"shape": [17], "dtype": "float32", "layout": "strided"}] * 2,
        "outputs": [{"shape": [17], "dtype": "float32"}],
        "reference": "musaopst.examples.references:add",
        "seed": 11, "timeout_s": 10,
        "compare": {"atol": 1e-6, "rtol": 1e-5},
    }
    value.update(updates)
    return value


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="musaopst-offline-")
        self.root = Path(self.tmp.name)
        self.counter = 0

    def tearDown(self):
        self.tmp.cleanup()

    def run_cases(self, cases, **options):
        self.counter += 1
        config = self.root / f"config-{self.counter}.json"
        config.write_text(json.dumps({"schema_version": 1, "cases": cases}))
        out = self.root / f"result-{self.counter}"
        return run_suite(config, out, **options), out

    def test_correct_inplace_and_multiple_outputs(self):
        pair = case("pair", entry="musaopst.examples.cpu_ops:pair",
                    reference="musaopst.examples.references:pair")
        pair["outputs"] *= 2
        values, output = self.run_cases([
            case(), case("inplace", entry="musaopst.examples.cpu_ops:inplace"), pair
        ])
        self.assertTrue(values["summary"]["passed"])
        self.assertEqual(values["summary"]["gpu_passed"], 0)
        self.assertEqual(values["summary"]["cpu_platform_passed"], 3)
        self.assertTrue(all(c["mode"] == "CPU_PLATFORM_TEST_ONLY" for c in values["cases"]))
        self.assertTrue(all(c["comparison"]["passed"] for c in values["cases"]))
        self.assertTrue((output / "junit.xml").is_file())

    def test_numeric_shape_dtype_and_execution_errors(self):
        cases = [case(entry=entry, name=name) for name, entry in [
            ("value", "musaopst.examples.cpu_ops:wrong"),
            ("shape", "musaopst.examples.cpu_ops:wrong_shape"),
            ("dtype", "musaopst.examples.cpu_ops:wrong_dtype"),
            ("execution", "musaopst.examples.cpu_ops:raises"),
        ]]
        result, output = self.run_cases(cases, benchmark=True)
        self.assertEqual([c["status"] for c in result["cases"]],
                         ["ACCURACY_FAILED", "ACCURACY_FAILED", "ACCURACY_FAILED", "EXECUTION_FAILED"])
        self.assertFalse(result["summary"]["passed"])
        self.assertTrue(all(not c.get("benchmark") for c in result["cases"]))
        self.assertEqual(ET.parse(output / "junit.xml").getroot().get("failures"), "4")

    def test_reference_exception_cannot_pass(self):
        result, _ = self.run_cases([case(reference="musaopst.examples.cpu_ops:raises")])
        self.assertEqual(result["cases"][0]["status"], "REFERENCE_FAILED")
        self.assertFalse(result["summary"]["passed"])

    def test_worker_timeout_is_controlled_and_following_cpu_case_runs(self):
        slow = case("timeout", entry="musaopst.examples.cpu_ops:slow",
                    attrs={"delay_s": 5}, timeout_s=0.5)
        start = time.monotonic()
        result, _ = self.run_cases([slow, case("following")])
        self.assertLess(time.monotonic() - start, 6)
        self.assertEqual([c["status"] for c in result["cases"]], ["TIMEOUT", "PASS"])
        self.assertEqual(result["summary"]["exit_code"], 1)

    def test_optional_skips_retained_but_all_skipped_fails(self):
        skip = case("optional", skip=True, required=False)
        result, output = self.run_cases([skip])
        self.assertFalse(result["summary"]["passed"])
        self.assertEqual(result["summary"]["optional_unexecuted"], 1)
        junit = ET.parse(output / "junit.xml").getroot()
        self.assertEqual(junit.get("skipped"), "1")
        self.assertEqual(junit.get("failures"), "1")
        result, _ = self.run_cases([skip, case()])
        self.assertTrue(result["summary"]["passed"])

    def test_required_skip_fails_even_with_passing_case(self):
        result, _ = self.run_cases([case("skip", skip=True), case()])
        self.assertFalse(result["summary"]["passed"])
        self.assertEqual(result["summary"]["required_nonpass"], 1)

    def test_missing_entry_classification_is_nonpass(self):
        result, _ = self.run_cases([case(entry="missing_test_module:no_such_function")])
        self.assertNotEqual(result["cases"][0]["status"], "PASS")
        self.assertEqual(result["summary"]["exit_code"], 1)

    def test_finite_repeat_records_actual_count(self):
        result, _ = self.run_cases([case(repeat={"count": 3})])
        self.assertEqual(result["cases"][0]["status"], "PASS")
        repeat = result["cases"][0]["repeat"]
        self.assertEqual(repeat["completed"], 3)
        self.assertGreaterEqual(repeat["elapsed_s"], 0)

    def test_nonfinite_cpu_worker_policy(self):
        (self.root / "nonfinite_fixture.py").write_text(
            "import numpy as np\n"
            "def dut(x, **attrs): return np.array([np.nan,np.inf,-np.inf,0],dtype=np.float32)\n"
            "def ref(x, **attrs): return np.array([np.nan,np.inf,-np.inf,0],dtype=np.float64)\n"
        )
        c = case(entry="nonfinite_fixture:dut", reference="nonfinite_fixture:ref",
                 inputs=[{"shape": [4], "dtype": "float32"}],
                 outputs=[{"shape": [4], "dtype": "float32"}])
        reject, _ = self.run_cases([c])
        self.assertEqual(reject["cases"][0]["status"], "ACCURACY_FAILED")
        c["compare"] = {"nan": "equal", "inf": "same_sign"}
        allow, _ = self.run_cases([c])
        self.assertEqual(allow["cases"][0]["status"], "PASS")

    def test_invalid_config_emits_all_report_formats(self):
        result, out = self.run_cases([case(inputs=[{"shape": [-1], "dtype": "float32"}])])
        self.assertEqual(result["cases"][0]["status"], "CONFIG_ERROR")
        for name in ("results.json", "summary.md", "junit.xml"):
            self.assertTrue((out / name).is_file())

    def test_cli_exit_code_and_report_rebuild(self):
        config = self.root / "cli.json"
        config.write_text(json.dumps({"schema_version": 1, "cases": [case(entry="musaopst.examples.cpu_ops:wrong")]}))
        out = self.root / "cli-out"
        command = [sys.executable, "-m", "musaopst", "run", str(config), "--output", str(out)]
        failed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)
        rebuilt = self.root / "rebuilt"
        report = subprocess.run([sys.executable, "-m", "musaopst", "report", str(out / "results.json"), "--output", str(rebuilt)],
                                cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertTrue((rebuilt / "summary.md").is_file(), report.stdout + report.stderr)
        self.assertFalse(json.loads((rebuilt / "results.json").read_text())["summary"]["passed"])

    def test_output_directory_is_not_overwritten(self):
        _, out = self.run_cases([case()])
        saved = (out / "results.json").read_bytes()
        with self.assertRaises(ValueError):
            run_suite(self.root / "config-1.json", out)
        self.assertEqual((out / "results.json").read_bytes(), saved)

    def test_config_expand_all_examples_without_gpu(self):
        from musaopst.config import load_config
        for config in sorted((ROOT / "examples").glob("*.json")):
            with self.subTest(config=config.name):
                self.assertTrue(load_config(config))

    def test_help_validation_and_reports_need_no_gpu_imports(self):
        result, output = self.run_cases([case()])
        commands = [
            ["--help"],
            ["validate", str(ROOT / "examples/gpu_suite.json")],
            ["report", str(output / "results.json"), "--output", str(self.root / "sdk-free-report")],
        ]
        script = (
            "import importlib.abc,sys,json\n"
            "class NoGPU(importlib.abc.MetaPathFinder):\n"
            " def find_spec(self, fullname, path=None, target=None):\n"
            "  if fullname.split('.')[0] in {'torch','torch_musa','triton','tilelang'}: "
            "raise AssertionError('unexpected GPU import: '+fullname)\n"
            "sys.meta_path.insert(0,NoGPU())\n"
            "from musaopst.cli import main\n"
            "raise SystemExit(main(json.loads(sys.argv[1])))\n"
        )
        for command in commands:
            with self.subTest(command=command[0]):
                completed = subprocess.run([sys.executable, "-c", script, json.dumps(command)],
                                           cwd=ROOT, capture_output=True, text=True, timeout=20)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_init_template_is_editable_valid_and_does_not_overwrite(self):
        target = self.root / "new-op"
        first = subprocess.run([sys.executable, "-m", "musaopst", "init", str(target)],
                               cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        files = list(target.glob("*.json"))
        self.assertTrue(files)
        from musaopst.config import load_config
        for path in files:
            self.assertTrue(load_config(path))
        originals = {path.name: path.read_bytes() for path in target.iterdir() if path.is_file()}
        second = subprocess.run([sys.executable, "-m", "musaopst", "init", str(target)],
                                cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual({path.name: path.read_bytes() for path in target.iterdir() if path.is_file()}, originals)

    def test_empty_summary_never_passes(self):
        self.assertFalse(summarize([])["passed"])
        self.assertEqual(summarize([])["exit_code"], 1)


if __name__ == "__main__":
    unittest.main()
