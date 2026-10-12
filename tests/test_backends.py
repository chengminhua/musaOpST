# SPDX-License-Identifier: MulanPSL-2.0
"""Offline adapter policy checks; these do not claim GPU verification."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from musaopst.backends.gpu import event_statistics
from musaopst.backends.native import NativeBackend
from musaopst.errors import UnsupportedError


class BackendPolicyTests(unittest.TestCase):
    def test_renamed_builtin_entry_keeps_legacy_configuration_and_provenance(self):
        from musaopst import backends
        from musaopst.runner import fingerprint, source_manifest
        from musaopst.util import entry_module, load_entry, sha256
        from musaopst.worker import entry_source
        old = 'musaopst.examples.native_v2_ops:identity'
        current = 'musaopst.examples.native_protocol_ops:identity'
        self.assertIs(load_entry(old), load_entry(current))
        self.assertIs(backends.load_entry(old), load_entry(current))
        descriptor = load_entry(old)({}, [], '.')
        self.assertTrue(Path(descriptor['source']).is_file())
        self.assertEqual(Path(descriptor['source']).name, 'native_kernels.mu')
        source = entry_source(old)
        self.assertEqual(source['entry'], old)
        self.assertEqual(Path(source['file']).name, 'native_protocol_ops.py')
        self.assertEqual(source['sha256'], sha256(source['file']))
        case = {'entry': old, 'reference': 'musaopst.examples.references:identity'}
        self.assertIsNotNone(fingerprint(case, '.', 0, source_manifest()))
        self.assertEqual(entry_module('user.native_v2_ops:identity'), 'user.native_v2_ops')

    def test_gpu_example_imports_are_lazy(self):
        program = """
import sys
from musaopst.backends import create_backend
from musaopst.examples import torch_ops, cpp_ops, triton_ops, tilelang_ops, native_ops
assert not any(name in sys.modules for name in ('torch','torch_musa','triton','tilelang'))
"""
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_event_quantiles_are_batch_mean_quantiles(self):
        policy = {"warmup": 2, "samples": 3, "iterations": 10}
        output = event_statistics([1., 2., 3.], policy, allocation_included=False)
        self.assertEqual(output["median_ms"], 2.)
        self.assertAlmostEqual(output["p90_ms"], 2.8)
        self.assertEqual(output["raw_samples_ms"], [1., 2., 3.])
        self.assertIn("NOT individual-call tail latency", output["policy"]["quantile_meaning"])
        self.assertFalse(output["policy"]["includes_output_allocation"])
        self.assertFalse(output["policy"]["includes_host_device_copy"])

    def test_event_invalid_samples_rejected(self):
        for samples in ([], [float("nan")], [float("inf")], [-1.]):
            with self.subTest(samples=samples), self.assertRaises(RuntimeError):
                event_statistics(samples, {}, allocation_included=False)

    def test_native_never_silently_materializes_strides(self):
        values = np.arange(12, dtype=np.float32)[::2]
        case = {"backend": "native", "inputs": [{"dtype": "float32", "layout": "strided"}] * 2,
                "attrs": {}}
        native = NativeBackend(case, [values, values.copy()], Path("unused"))
        with self.assertRaisesRegex(UnsupportedError, "contiguous"):
            native.prepare()

    def test_native_snapshot_is_independent(self):
        original = np.arange(4, dtype=np.float32)
        native = NativeBackend({}, [], Path("unused"))
        snapshots, dtypes = native.snapshot([original])
        original.fill(99)
        self.assertEqual(snapshots[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(dtypes, ["float32"])
