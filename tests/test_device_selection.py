# SPDX-License-Identifier: MulanPSL-2.0
"""`--device` ordinal resolution and `--case` selection usability checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from musaopst.config import filter_cases, selection_error
from musaopst.device import (DeviceSelectionError, format_resolution,
                             is_invalid_ordinal, ordinal_error_message, resolve_device)
from musaopst.runner import run_suite

ROOT = Path(__file__).resolve().parents[1]


def cpu_case(name='add'):
    return {'name': name, 'backend': 'cpu', 'entry': 'musaopst.examples.cpu_ops:add',
            'inputs': [{'shape': [4], 'dtype': 'float32'}] * 2,
            'outputs': [{'shape': [4], 'dtype': 'float32'}],
            'reference': 'musaopst.examples.references:add', 'seed': 3, 'timeout_s': 10,
            'compare': {'atol': 1e-6, 'rtol': 1e-5}}


def gpu_case(name='gpu'):
    value = cpu_case(name)
    value.update(backend='torch_musa', entry='musaopst.examples.torch_ops:softmax',
                 reference='musaopst.examples.references:softmax', attrs={'dim': -1})
    return value


class DeviceResolutionTests(unittest.TestCase):
    def test_unset_visibility_is_a_plain_ordinal(self):
        selection = resolve_device(1, device_count=4, environ={})
        self.assertEqual((selection.ordinal, selection.physical_id, selection.method),
                         (1, 1, 'ordinal'))
        self.assertIn('--device 1 → 物理 GPU 1', format_resolution(selection.as_dict()))

    def test_single_visible_card_maps_to_its_physical_id(self):
        selection = resolve_device(0, device_count=1, environ={'MUSA_VISIBLE_DEVICES': '6'})
        self.assertEqual((selection.ordinal, selection.physical_id), (0, 6))
        self.assertEqual(selection.method, 'visible-ordinal')
        line = format_resolution(selection.as_dict())
        self.assertIn('--device 0 → 物理 GPU 6', line)
        self.assertIn('MUSA_VISIBLE_DEVICES=6', line)
        self.assertIn('count=1', line)

    def test_multi_card_visibility_maps_ordinal_to_physical(self):
        selection = resolve_device(1, device_count=2, environ={'MUSA_VISIBLE_DEVICES': '2,5'})
        self.assertEqual((selection.ordinal, selection.physical_id), (1, 5))
        self.assertEqual(selection.visible_devices, [2, 5])

    def test_order_sensitive_visibility_warns_but_keeps_ordinal(self):
        selection = resolve_device(1, device_count=3, environ={'MUSA_VISIBLE_DEVICES': '1,2,0'})
        self.assertEqual(selection.physical_id, 2)
        self.assertIn('仍按序号解释为物理 GPU 2', selection.note)
        self.assertIn('--device 0', selection.note)

    def test_out_of_range_visibility_reports_usage_error(self):
        with self.assertRaises(DeviceSelectionError) as caught:
            resolve_device(6, device_count=1, environ={'MUSA_VISIBLE_DEVICES': '6'})
        message = str(caught.exception)
        self.assertIn('--device 6', message)
        self.assertIn("MUSA_VISIBLE_DEVICES='6'", message)
        self.assertIn('可见列表长度 1', message)
        self.assertIn('device_count()=1', message)
        self.assertIn('请设 MUSA_VISIBLE_DEVICES=6 并传 --device 0', message)

    def test_out_of_range_without_visibility_reports_count(self):
        with self.assertRaises(DeviceSelectionError) as caught:
            resolve_device(4, device_count=2, environ={})
        message = str(caught.exception)
        self.assertIn('未设置', message)
        self.assertIn('device_count()=2', message)
        self.assertIn('可用序号 0..1', message)

    def test_uuid_visibility_keeps_ordinal_and_unknown_physical(self):
        env = {'MUSA_VISIBLE_DEVICES': 'GPU-aaa,GPU-bbb'}
        selection = resolve_device(1, device_count=2, environ=env)
        self.assertEqual((selection.ordinal, selection.physical_id), (1, None))
        self.assertEqual(selection.method, 'visible-ordinal-unknown-physical')
        self.assertIn('物理 GPU 未知', format_resolution(selection.as_dict()))
        with self.assertRaises(DeviceSelectionError):
            resolve_device(2, device_count=2, environ=env)

    def test_mthreads_visibility_is_reused(self):
        selection = resolve_device(0, device_count=1, environ={'MTHREADS_VISIBLE_DEVICES': '3'})
        self.assertEqual(selection.physical_id, 3)
        self.assertEqual(selection.visible_variable, 'MTHREADS_VISIBLE_DEVICES')

    def test_mthreads_all_does_not_shadow_explicit_musa_visibility(self):
        env = {'MTHREADS_VISIBLE_DEVICES': 'all', 'MUSA_VISIBLE_DEVICES': '6'}
        selection = resolve_device(0, device_count=1, environ=env)
        self.assertEqual(selection.physical_id, 6)
        self.assertEqual(selection.visible_variable, 'MUSA_VISIBLE_DEVICES')
        plain = resolve_device(0, device_count=8, environ={'MTHREADS_VISIBLE_DEVICES': 'all'})
        self.assertEqual((plain.physical_id, plain.method), (0, 'ordinal'))

    def test_negative_and_non_integer_requests_are_usage_errors(self):
        for value in (-1, '2', True):
            with self.subTest(value=value), self.assertRaises(DeviceSelectionError):
                resolve_device(value, device_count=1, environ={})

    def test_invalid_ordinal_text_is_recognized_for_reclassification(self):
        text = 'RuntimeError: MUSA error: invalid device ordinal'
        self.assertTrue(is_invalid_ordinal(text))
        message = ordinal_error_message(text, 6)
        self.assertIn('请设 MUSA_VISIBLE_DEVICES=6 并传 --device 0', message)
        self.assertIn('不是硬件故障', message)


class CaseSelectionTests(unittest.TestCase):
    names = ['sparse_attn_bwd_csa__tilelang', 'dense_attn_fwd__triton', 'vector_add']

    def cases(self):
        return [{'name': name, 'backend': 'cpu', 'tags': ['smoke']} for name in self.names]

    def test_plain_substring_matches_without_glob(self):
        selected = filter_cases(self.cases(), 'sparse_attn_bwd_csa')
        self.assertEqual([case['name'] for case in selected], ['sparse_attn_bwd_csa__tilelang'])

    def test_glob_still_matches(self):
        selected = filter_cases(self.cases(), '*attn*')
        self.assertEqual(len(selected), 2)

    def test_zero_match_explains_pattern_names_and_alternatives(self):
        cases = self.cases()
        self.assertEqual(filter_cases(cases, 'no_such_case'), [])
        message = selection_error(self.names, 'no_such_case')
        self.assertIn("--case='no_such_case'", message)
        self.assertIn('sparse_attn_bwd_csa__tilelang', message)
        self.assertIn('--list-cases', message)
        self.assertIn('通配符', message)
        self.assertIn('子串', message)

    def test_tag_zero_match_uses_the_same_hint(self):
        self.assertEqual(filter_cases(self.cases(), tags=['missing']), [])
        message = selection_error(self.names, tags=['missing'])
        self.assertIn('--tag=missing', message)
        self.assertIn('vector_add', message)


class DeviceResolutionReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='musaopst-device-')
        self.root = Path(self.tmp.name)
        self.counter = 0

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, cases):
        self.counter += 1
        path = self.root / f'config-{self.counter}.json'
        path.write_text(json.dumps({'schema_version': 2, 'cases': cases}))
        return path

    def test_cpu_only_run_ignores_device_validation(self):
        result = run_suite(self.config([cpu_case()]), self.root / 'cpu', device=6)
        self.assertEqual(result['cases'][0]['status'], 'PASS')
        self.assertEqual(result['environment']['device_selection'], 6)
        self.assertNotIn('device_resolution', result['environment'])

    def test_resolution_is_recorded_next_to_legacy_ordinal(self):
        selection = resolve_device(0, device_count=1, environ={'MUSA_VISIBLE_DEVICES': '6'})
        out = self.root / 'resolved'
        result = run_suite(self.config([cpu_case()]), out, device=0, device_selection=selection)
        stored = json.loads((out / 'results.json').read_text())
        self.assertEqual(stored['environment']['device_selection'], 0)
        resolution = stored['environment']['device_resolution']
        self.assertEqual((resolution['requested'], resolution['ordinal'], resolution['physical_id']), (0, 0, 6))
        self.assertEqual(resolution['visible_devices'], [6])
        self.assertEqual(resolution['method'], 'visible-ordinal')
        summary = (out / 'summary.md').read_text()
        self.assertIn('--device 0 → 物理 GPU 6', summary)

    def test_zero_match_is_usage_exit_2_with_pattern_echo(self):
        out = self.root / 'selection'
        result = run_suite(self.config([cpu_case('alpha'), cpu_case('beta')]), out, pattern='no_such_case')
        self.assertEqual(result['summary']['exit_code'], 2)
        self.assertFalse(result['summary']['passed'])
        summary = (out / 'summary.md').read_text()
        self.assertIn("--case='no_such_case'", summary)
        self.assertIn('alpha', summary)
        self.assertIn('--list-cases', summary)

    def test_cli_rejects_out_of_range_device_before_any_work(self):
        from musaopst import cli
        config = self.config([gpu_case()])
        out = self.root / 'cli-device'
        with mock.patch('musaopst.device.probe_device_count', return_value=1), \
                mock.patch('sys.stderr', new_callable=__import__('io').StringIO) as stderr:
            code = cli.main(['run', str(config), '--device', '6', '--output', str(out)])
        self.assertEqual(code, 2)
        self.assertIn('请设 MUSA_VISIBLE_DEVICES=6 并传 --device 0', stderr.getvalue())
        self.assertFalse(out.exists())

    def test_list_cases_prints_rows_without_running(self):
        config = self.config([cpu_case('alpha'), cpu_case('beta')])
        process = subprocess.run(
            [sys.executable, '-m', 'musaopst', 'run', str(config), '--case', 'alpha', '--list-cases'],
            cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn('alpha\tcpu', process.stdout)
        self.assertNotIn('beta\tcpu', process.stdout)
        self.assertIn('1/2', process.stdout)

    def test_cli_zero_match_returns_2_and_writes_report(self):
        config = self.config([cpu_case('alpha'), cpu_case('beta')])
        out = self.root / 'cli-selection'
        process = subprocess.run(
            [sys.executable, '-m', 'musaopst', 'run', str(config), '--case', 'no_such_case', '--output', str(out)],
            cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
        self.assertTrue((out / 'results.json').is_file())
        self.assertIn("--case='no_such_case'", (out / 'summary.md').read_text())


if __name__ == '__main__':
    unittest.main()
