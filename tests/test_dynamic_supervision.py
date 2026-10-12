# SPDX-License-Identifier: MulanPSL-2.0
import json
from pathlib import Path
import tempfile
import unittest
from musaopst.runner import run_suite

class DynamicSupervision(unittest.TestCase):
    def test_completed_case_retained_when_later_binding_times_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'timed.py').write_text('import time\ndef op(x):\n if x.size>1:time.sleep(3)\n return x.copy()\ndef ref(x):return x*1\n')
            cases=[]
            for n in (1,2,3):
                cases.append({'name':str(n),'backend':'cpu','entry':'timed:op','reference':'timed:ref',
                              'inputs':[{'shape':[n],'dtype':'float32'}],'outputs':[{'shape':[n],'dtype':'float32'}],
                              'timeout_s':.5,'compile_group':'g','shape_mode':'runtime'})
            cfg=root/'c.json';cfg.write_text(json.dumps({'schema_version':2,'cases':cases}))
            result=run_suite(cfg,root/'out')
            self.assertEqual([c['status'] for c in result['cases']],['PASS','TIMEOUT','SKIPPED'])
    def test_process_lost_group_is_not_a_partial_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'lost.py').write_text('import os\ndef op(x):\n if x.size>1:os._exit(7)\n return x.copy()\ndef ref(x):return x*1\n')
            cases=[]
            for n in (1,2):
                cases.append({'name':str(n),'backend':'cpu','entry':'lost:op','reference':'lost:ref','inputs':[{'shape':[n],'dtype':'float32'}],
                              'outputs':[{'shape':[n],'dtype':'float32'}],'timeout_s':5,'compile_group':'g','shape_mode':'runtime'})
            cfg=root/'c.json';cfg.write_text(json.dumps({'schema_version':2,'cases':cases}))
            result=run_suite(cfg,root/'out');self.assertFalse(result['summary']['passed'])
            self.assertEqual(result['cases'][0]['status'],'EXECUTION_FAILED')
