# SPDX-License-Identifier: MulanPSL-2.0
"""Offline trust-boundary regressions; fixtures never constitute GPU evidence."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from musaopst.regression import compare_baseline, merge_reports
from musaopst.reporting import render_reports
from musaopst.runner import run_suite


def passing_record(name='op'):
    comparison={'passed':True,'contract_errors':[],'outputs':[{'passed':True}]}
    return {'name':name,'backend':'cpu','mode':'CPU_PLATFORM_TEST_ONLY','required':True,
            'status':'PASS','comparison':comparison,'case_result':'case-0000/result.json',
            'post_benchmark_comparison':copy.deepcopy(comparison),
            'benchmark':{'raw_samples_ms':[1.0]*5,'policy':{'timer':'cpu'}},
            'config':{'inputs':[],'outputs':[],'attrs':{},'seed':7,'entry':'fixture:op'},
            'versions':{'fixture':'offline-only'},'input_snapshot_sha256':'fixture-only'}


def report(record=None):
    record=passing_record() if record is None else record
    return {'schema_version':2,'run_complete':True,'cases':[record],
            'planned_cases':[{'name':record['name'],'backend':'cpu','required':True}]}


class IntegrityRegressionsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_report(self,name,value):
        directory=self.root/name
        directory.mkdir()
        target=directory/'results.json'
        target.write_text(json.dumps(value))
        return target

    def test_merge_preserves_missing_planned_case_despite_raw_complete_flag(self):
        source=report()
        source['planned_cases'].append({'name':'never-executed','backend':'cpu','required':True})
        render_reports(source,self.root/'source')
        self.assertTrue(source['run_complete'])
        self.assertFalse(source['summary']['passed'])
        merged=merge_reports([self.root/'source'],self.root/'merged')
        self.assertFalse(merged['summary']['passed'])
        self.assertEqual(merged['summary']['pending'],1)
        self.assertEqual(merged['summary']['exit_code'],1)
        self.assertFalse(merged['run_complete'])
        self.assertEqual(merged['source_coverage'][0]['pending'],['never-executed'])
        self.assertEqual([c['name'] for c in merged['planned_cases']],['0:op','0:never-executed'])

    def test_merge_prefixes_plans_for_duplicate_names_across_sources(self):
        first=report();second=report()
        second['planned_cases'].append({'name':'missing','backend':'cpu','required':True})
        a=self.write_report('a',first);b=self.write_report('b',second)
        merged=merge_reports([a,b],self.root/'merged')
        self.assertEqual([c['name'] for c in merged['cases']],['0:op','1:op'])
        self.assertEqual([c['name'] for c in merged['planned_cases']],['0:op','1:op','1:missing'])
        self.assertEqual(merged['summary']['pending'],1)
        self.assertEqual(merged['summary']['fresh_gpu_passed'],0)
        self.assertTrue(all(c['provenance']=='imported' for c in merged['cases']))

    def test_merge_does_not_trust_stale_summary_and_keeps_complete_source_green(self):
        source=report()
        source['summary']={'passed':False,'pending':999,'run_complete':False}
        path=self.write_report('a',source)
        merged=merge_reports([path],self.root/'merged')
        self.assertTrue(merged['summary']['passed'])
        self.assertEqual(merged['summary']['pending'],0)

    def test_invalid_baseline_comparison_cleanup_or_post_timing_is_incomparable(self):
        current=self.write_report('current',report())
        changes=[{'comparison':{'passed':False}},
                 {'post_benchmark_comparison':{'passed':False}},
                 {'cleanup_error':'close failed'}]
        for i,update in enumerate(changes):
            with self.subTest(update=update):
                prior=report();prior['cases'][0].update(update)
                baseline=self.write_report(f'prior-{i}',prior)
                result=compare_baseline(current,baseline,self.root/f'comparison-{i}')
                self.assertEqual(result['cases'][0]['status'],'INCOMPARABLE')
                self.assertIn('baseline record is invalid',result['cases'][0]['reason'])
                self.assertEqual(result['exit_code'],2)

    def test_baseline_pending_coverage_is_incomparable_even_with_raw_complete(self):
        current=self.write_report('current',report())
        prior=report();prior['planned_cases'].append({'name':'pending','required':True})
        baseline=self.write_report('baseline',prior)
        result=compare_baseline(current,baseline,self.root/'comparison')
        self.assertEqual(result['cases'][0]['status'],'INCOMPARABLE')
        self.assertIn('incomplete',result['cases'][0]['reason'])

    def test_valid_baseline_compares_and_invalid_current_still_rejects(self):
        current=report();prior=report()
        current_path=self.write_report('current',current)
        baseline=self.write_report('baseline',prior)
        result=compare_baseline(current_path,baseline,self.root/'comparison')
        self.assertEqual(result['cases'][0]['status'],'OK')
        current['cases'][0]['comparison']['passed']=False
        current_path.write_text(json.dumps(current))
        with self.assertRaises(ValueError):
            compare_baseline(current_path,baseline,self.root/'rejected')

    def test_dynamic_device_supervision_keeps_completed_middle_case(self):
        # Only exercise parent supervision using a simulated child. No device,
        # SDK, kernel or backend is loaded; TemporaryDirectory removes fixtures.
        cases=[]
        for i in range(4):
            cases.append({'name':str(i),'backend':'torch_musa',
                'entry':'musaopst.examples.cpu_ops:add','reference':'musaopst.examples.references:add',
                'inputs':[{'shape':[7],'dtype':'float32'}]*2,'outputs':[{'shape':[7],'dtype':'float32'}],
                'timeout_s':.02,'compile_group':'supervision-unit','shape_mode':'runtime'})
        config=self.root/'config.json'
        config.write_text(json.dumps({'schema_version':2,'cases':cases}))

        class SimulatedWorker:
            def __init__(self,command,**kwargs):
                request=json.loads(Path(command[-1]).read_text())
                for child in request['requests'][:2]:
                    record=passing_record(child['case']['name'])
                    record.update(backend='torch_musa',mode='MUSA_DEVICE',
                                  message='CONTROLLED SUPERVISION FIXTURE; NO GPU EXECUTION')
                    (Path(child['workdir'])/'result.json').write_text(json.dumps(record))

            def wait(self,timeout):
                raise subprocess.TimeoutExpired('controlled-supervision-fixture',timeout)

        with patch('musaopst.runner.subprocess.Popen',SimulatedWorker), \
                patch('musaopst.runner.terminate_worker') as terminate, \
                patch('musaopst.runner.fingerprint',return_value='controlled-supervision-fixture'), \
                patch('musaopst.device_identity.probe',return_value={'verified':False,'reason':'offline fixture'}):
            result=run_suite(config,self.root/'supervision')
        terminate.assert_called_once()
        self.assertEqual([c['status'] for c in result['cases']],['PASS','PASS','TIMEOUT','SKIPPED'])
        self.assertFalse(result['summary']['passed'])
        self.assertIn('CONTROLLED SUPERVISION FIXTURE',result['cases'][1]['message'])


if __name__=='__main__':
    unittest.main()
