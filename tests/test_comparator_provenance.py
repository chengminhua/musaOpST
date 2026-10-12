# SPDX-License-Identifier: MulanPSL-2.0
"""Explicit comparison plugins are provenance, not untracked dependencies."""
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from musaopst.artifacts import bundle
from musaopst.config import expand_config
from musaopst.runner import run_suite
from musaopst.util import sha256
from musaopst import worker


class ComparatorProvenance(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.module='comparator_'+self.root.name.replace('-','_')
        self.plugin=self.root/(self.module+'.py')
        self.plugin.write_text('def check(actual, reference, contract, policy):\n return {"passed": True}\n')
        self.case={'name':'comparison-source','backend':'cpu','entry':'musaopst.examples.cpu_ops:identity',
                   'reference':'musaopst.examples.references:identity','inputs':[{'shape':[3],'dtype':'float32'}],
                   'outputs':[{'shape':[3],'dtype':'float32'}],'compare':{'custom':self.module+':check'},'timeout_s':5}
        self.old_path=list(sys.path)

    def tearDown(self):
        sys.path[:]=self.old_path
        sys.modules.pop(self.module,None)
        self.temp.cleanup()

    def run_case(self):
        config=self.root/'config.json'
        config.write_text(json.dumps({'schema_version':2,'cases':[self.case]}))
        return run_suite(config,self.root/'result')

    def test_worker_records_explicit_comparator_source_and_bundle_contains_it(self):
        result=self.run_case()
        self.assertEqual(result['cases'][0]['status'],'PASS')
        sources=result['cases'][0]['comparator_sources']
        self.assertEqual(len(sources),1)
        self.assertEqual(sources[0]['entry'],self.module+':check')
        self.assertEqual(sources[0]['sha256'],sha256(self.plugin))
        packed=bundle(self.root/'result',self.root/'valid.tar.gz')
        with tarfile.open(packed['path']) as archive:
            self.assertIn('source/'+self.module+'.py',archive.getnames())
            manifest=json.load(archive.extractfile('bundle.json'))
        checked=[item for item in manifest['entry_source_verification'] if item['field']=='comparator_sources']
        self.assertEqual(len(checked),1)
        self.assertTrue(checked[0]['matches_recorded_execution'])

    def test_bundle_changed_comparator_is_explicitly_not_exact_historical_source(self):
        self.run_case()
        self.plugin.write_text('def check(actual, reference, contract, policy):\n return {"passed": False}\n')
        packed=bundle(self.root/'result',self.root/'changed.tar.gz')
        with tarfile.open(packed['path']) as archive:
            manifest=json.load(archive.extractfile('bundle.json'))
        checked=[item for item in manifest['entry_source_verification'] if item['field']=='comparator_sources']
        self.assertFalse(checked[0]['matches_recorded_execution'])
        self.assertTrue(any(item.get('field')=='comparator_sources' for item in packed['unresolved_sources']))

    def test_missing_comparator_remains_accuracy_failure_not_source_record_failure(self):
        self.case['compare']['custom']='missing_comparator_provenance_fixture:check'
        result=self.run_case()['cases'][0]
        self.assertEqual(result['status'],'ACCURACY_FAILED')
        self.assertEqual(result['last_stage'],'compare')
        self.assertFalse(result['comparison']['outputs'][0]['custom']['passed'])
        self.assertEqual(result['comparator_sources'][0]['sha256'],'unknown')
        self.assertIn('source_error',result['comparator_sources'][0])

    def test_source_inspection_failure_cannot_change_successful_comparison(self):
        case=expand_config({'schema_version':2,'cases':[self.case]})[0]
        workdir=self.root/'worker';workdir.mkdir()
        original=worker.entry_source
        def inspect(entry):
            if entry==self.module+':check':raise OSError('controlled source inspection failure')
            return original(entry)
        with patch('musaopst.worker.entry_source',side_effect=inspect):
            result=worker.execute({'case':case,'device':0,'config_dir':str(self.root),'workdir':str(workdir)})
        self.assertEqual(result['status'],'PASS')
        self.assertTrue(result['comparison']['passed'])
        self.assertIn('controlled source inspection failure',result['comparator_sources'][0]['source_error'])


if __name__=='__main__':unittest.main()
