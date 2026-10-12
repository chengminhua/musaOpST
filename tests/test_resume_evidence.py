# SPDX-License-Identifier: MulanPSL-2.0
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from musaopst.runner import reusable_snapshots, fingerprint, run_suite
from musaopst.util import sha256


class ResumeEvidence(unittest.TestCase):
    def test_missing_snapshot_even_when_manifest_key_deleted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=root/'case.json'
            case={'name':'add','backend':'cpu','entry':'musaopst.examples.cpu_ops:add','reference':'musaopst.examples.references:add',
                  'inputs':[{'shape':[3],'dtype':'float32'}]*2,'outputs':[{'shape':[3],'dtype':'float32'}]}
            config.write_text(json.dumps({'schema_version':2,'cases':[case]}))
            first=run_suite(config,root/'first');old=first['cases'][0]
            del old['snapshot_hashes']['actual.npz'];(root/'first/case-0000/actual.npz').unlink()
            (root/'first/results.json').write_text(json.dumps(first))
            second=run_suite(config,root/'second',resume=root/'first')
            self.assertEqual(second['summary']['reused'],0)
            self.assertEqual(second['cases'][0]['provenance'],'fresh')
            self.assertTrue(second['summary']['passed'])
    def test_optional_execution_requires_extra_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);hashes={}
            for name in ('inputs.npz','actual.npz','reference.npz'):
                (root/name).write_bytes(b'evidence');hashes[name]=sha256(root/name)
            old={'snapshot_hashes':hashes}
            with patch('musaopst.runner.validate_case_result'):
                self.assertTrue(reusable_snapshots(old,root))
                old['benchmark']={'samples':1};self.assertFalse(reusable_snapshots(old,root))
                (root/'post_benchmark.npz').write_bytes(b'evidence');hashes['post_benchmark.npz']=sha256(root/'post_benchmark.npz')
                self.assertTrue(reusable_snapshots(old,root))
                old['profile']={'gpu_kernel_count':1};self.assertFalse(reusable_snapshots(old,root))
                for name in ('profile_trace.json','profile_table.txt'):
                    (root/name).write_bytes(b'evidence');hashes[name]=sha256(root/name)
                self.assertTrue(reusable_snapshots(old,root))
                hashes['../outside']='bad';self.assertFalse(reusable_snapshots(old,root))
    def test_toolchain_change_changes_fingerprint(self):
        case={'entry':'musaopst.examples.cpu_ops:add','reference':'musaopst.examples.references:add'}
        first=fingerprint(case,'.',0,'same',{}, {'files':{'mcc':'a'}})
        second=fingerprint(case,'.',0,'same',{}, {'files':{'mcc':'b'}})
        self.assertNotEqual(first,second)

    def test_custom_comparator_source_change_prevents_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=root/'case.json';plugin=root/'customcmp.py'
            plugin.write_text('def check(*args): return {"passed": True}\n')
            case={'name':'add','backend':'cpu','entry':'musaopst.examples.cpu_ops:add','reference':'musaopst.examples.references:add',
                  'inputs':[{'shape':[3],'dtype':'float32'}]*2,'outputs':[{'shape':[3],'dtype':'float32','compare':{'custom':'customcmp:check'}}]}
            config.write_text(json.dumps({'schema_version':2,'cases':[case]}))
            first=run_suite(config,root/'first');self.assertTrue(first['summary']['passed'])
            plugin.write_text('def check(*args): return {"passed": False}\n')
            second=run_suite(config,root/'second',resume=root/'first')
            self.assertEqual(second['summary']['reused'],0)
            self.assertEqual(second['cases'][0]['status'],'ACCURACY_FAILED')
    def test_unresolved_custom_comparator_disables_reuse(self):
        case={'entry':'musaopst.examples.cpu_ops:add','reference':'musaopst.examples.references:add','compare':{'custom':'no_such_plugin:compare'}}
        self.assertIsNone(fingerprint(case,'.',0,'same'))
