# SPDX-License-Identifier: MulanPSL-2.0
"""Fail-closed and isolation regressions; every worker in this file is CPU only."""
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from musaopst.config import ConfigError, expand_config
from musaopst.reporting import render_reports, summarize
from musaopst.runner import run_suite


def case(**updates):
    value = {'name': 'safety', 'backend': 'cpu', 'entry': 'musaopst.examples.cpu_ops:add',
             'reference': 'musaopst.examples.references:add',
             'inputs': [{'shape': [3], 'dtype': 'float32'} for _ in range(2)],
             'outputs': [{'shape': [3], 'dtype': 'float32'}],
             'seed': 45, 'timeout_s': 10}
    value.update(updates)
    return value


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='musaopst-regression-')
        self.root = Path(self.temporary.name)
        self.output = self.root / 'result'
        self.config = self.root / 'config.json'

    def tearDown(self):
        self.temporary.cleanup()

    def run_case(self, value):
        self.config.write_text(json.dumps({'schema_version': 1, 'cases': [value]}))
        return run_suite(self.config, self.output)

    def test_missing_reference_module_classifies_reference_failure(self):
        result = self.run_case(case(reference='missing_reference_module_for_test:reference'))
        self.assertEqual(result['cases'][0]['status'], 'REFERENCE_FAILED')
        self.assertFalse(result['summary']['passed'])

    def test_builtin_reference_can_have_unknown_source(self):
        result = self.run_case(case(reference='numpy:add'))
        self.assertTrue(result['summary']['passed'])
        self.assertEqual(result['cases'][0]['reference_source']['file'], 'unknown')

    def test_reference_cannot_reclassify_its_own_error_as_skip(self):
        (self.root / 'reference_fault.py').write_text(
            'from musaopst.errors import UnsupportedError\n'
            'def reference(*args, **attrs): raise UnsupportedError("controlled reference exception")\n')
        result = self.run_case(case(reference='reference_fault:reference', required=False))
        self.assertEqual(result['cases'][0]['status'], 'REFERENCE_FAILED')
        self.assertEqual(result['summary']['optional_unexecuted'], 0)

    def test_output_snapshot_survives_reference_overwriting_shared_buffer(self):
        (self.root / 'shared_buffer.py').write_text(
            'import numpy as np\n'
            'buffer = np.empty(3,dtype=np.float32)\n'
            'def implementation(a,b):\n'
            ' np.add(a,b,out=buffer)\n'
            ' return buffer\n'
            'def reference(a,b):\n'
            ' buffer.fill(123456)\n'
            ' return np.add(a,b)\n')
        result = self.run_case(case(entry='shared_buffer:implementation', reference='shared_buffer:reference', repeat={'count': 2}))
        self.assertTrue(result['summary']['passed'])
        actual = np.load(self.output / 'case-0000' / 'actual.npz')['output_0']
        expected = np.load(self.output / 'case-0000' / 'reference.npz')['output_0']
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_repeat_anomaly_records_completed_count_and_blocks_benchmark(self):
        (self.root / 'repeat_fault.py').write_text(
            'counter=0\n'
            'def implementation(a,b):\n'
            ' global counter\n'
            ' counter += 1\n'
            ' return a+b+(0 if counter==1 else 2)\n')
        result = self.run_case(case(entry='repeat_fault:implementation', repeat={'count': 3}, benchmark={'enabled': True}))
        tested = result['cases'][0]
        self.assertEqual(tested['status'], 'ACCURACY_FAILED')
        self.assertEqual(tested['repeat']['completed'], 1)
        self.assertEqual(len(tested['repeat']['anomalies']), 1)
        self.assertNotIn('benchmark', tested)

    def test_malformed_worker_result_becomes_execution_failure(self):
        (self.root / 'corrupt_worker.py').write_text(
            'import os\nfrom pathlib import Path\n'
            'def implementation(*args,result_path,**attrs):\n'
            ' Path(result_path).write_text("not json")\n'
            ' os._exit(4)\n')
        result_path = self.output / 'case-0000' / 'result.json'
        result = self.run_case(case(entry='corrupt_worker:implementation', attrs={'result_path': str(result_path)}))
        self.assertEqual(result['cases'][0]['status'], 'EXECUTION_FAILED')
        self.assertIn('结构化结果无效', result['cases'][0]['message'])
        self.assertEqual(result['summary']['exit_code'], 1)
        self.assertTrue((self.output / 'junit.xml').exists())

    def test_parameters_retained_and_resolved_replay_has_no_internal_fields(self):
        value = case(parameters={'n': [1, 3]})
        for spec in value['inputs'] + value['outputs']:
            spec['shape'] = ['${n}']
        result = self.run_case(value)
        self.assertEqual([c['config']['parameters_selected'] for c in result['cases']], [{'n': 1}, {'n': 3}])
        resolved = json.loads((self.output / 'config.resolved.json').read_text())
        self.assertTrue(all('parameters_selected' not in c for c in resolved['cases']))
        self.assertEqual(len(expand_config(resolved)), 2)

    def test_replay_preserves_custom_module_directory_from_unrelated_cwd(self):
        (self.root / 'custom_replay.py').write_text('def implementation(a,b): return a+b\n')
        result = self.run_case(case(entry='custom_replay:implementation'))
        completed = subprocess.run(shlex.split(result['reproduce']), cwd='/', capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        replay = json.loads(Path(str(self.output) + '-replay', 'results.json').read_text())
        self.assertTrue(replay['summary']['passed'])

    def test_report_rejects_unknown_status_required_type_and_unsupported_pass(self):
        original = {'name': 'bad', 'backend': 'cpu', 'mode': 'CPU_PLATFORM_TEST_ONLY', 'required': True, 'status': 'PASS'}
        records = [original, dict(original, status='NEW_UNKNOWN_STATUS'), dict(original, status='SKIPPED', required='false')]
        for record in records:
            with self.subTest(record=record), self.assertRaises(ValueError):
                summarize([record])

    def test_report_rejects_bool_schema_and_preserves_existing_file(self):
        self.output.mkdir()
        saved = self.output / 'results.json'
        saved.write_text('original evidence')
        with self.assertRaises(ValueError):
            render_reports({'schema_version': True, 'cases': []}, self.output)
        self.assertEqual(saved.read_text(), 'original evidence')

    def test_nested_config_boolean_types_are_strict(self):
        for updates in ({'required': 0}, {'skip': 'false'}, {'benchmark': {'enabled': 1}},
                        {'repeat': {'count': True}}, {'benchmark': {'samples': True}}):
            with self.subTest(updates=updates), self.assertRaises(ConfigError):
                expand_config({'schema_version': 1, 'cases': [case(**updates)]})


if __name__ == '__main__':
    unittest.main()
