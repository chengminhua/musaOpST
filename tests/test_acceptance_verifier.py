# SPDX-License-Identifier: MulanPSL-2.0
"""Offline tests of evidence rejection; fixtures are never GPU acceptance evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.verify_acceptance import Audit, check_benchmark, check_suite_header, main


class AcceptanceVerifierTests(unittest.TestCase):
    def test_missing_evidence_writes_no_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'audit.json'
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(['--gpu', str(root / 'missing-gpu'), '--negative', str(root / 'missing-negative'),
                      '--offline', str(root / 'missing-offline'), '--output', str(output)])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix('.sha256.json').exists())

    def test_existing_audit_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / 'fixture.json'
            evidence.write_text('{}')
            output = root / 'audit.json'
            output.write_text('preserved audit')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(['--gpu', str(evidence), '--negative', str(evidence), '--offline', str(evidence), '--output', str(output)])
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(output.read_text(), 'preserved audit')

    def test_wall_time_and_missing_event_samples_are_rejected(self):
        audit = Audit()
        check_benchmark(audit,
                        {'name': 'offline-invalid-metadata', 'benchmark': {'policy': {'timer': 'python_wall_time'}}},
                        {'backend': 'torch_musa', 'benchmark': {'warmup': 3, 'samples': 5, 'iterations': 10}})
        self.assertTrue(any('timer' in failure for failure in audit.failures))
        self.assertTrue(any('sample count' in failure for failure in audit.failures))

    def test_partial_report_is_rejected_before_artifacts_can_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            path.write_text(json.dumps({'schema_version': 1, 'tool': 'musaopst', 'run_complete': False,
                                        'cases': [], 'summary': {'passed': True}}))
            audit = Audit()
            with self.assertRaisesRegex(ValueError, 'missing evidence file'):
                check_suite_header(audit, path, [], True)
            self.assertTrue(any('not completed' in failure for failure in audit.failures))


if __name__ == '__main__':
    unittest.main()
