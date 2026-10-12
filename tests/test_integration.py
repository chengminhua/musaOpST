# SPDX-License-Identifier: MulanPSL-2.0
import copy
import json
from pathlib import Path
import tempfile
import unittest
from musaopst.runner import run_suite
from musaopst.regression import merge_reports, compare_baseline
from musaopst.cache import cache_list, cache_prune


def base(name='op', **kw):
    c={'name':name,'backend':'cpu','entry':'musaopst.examples.cpu_ops:add',
       'reference':'musaopst.examples.references:add','inputs':[{'shape':[7],'dtype':'float32'}]*2,
       'outputs':[{'shape':[7],'dtype':'float32'}],'seed':7,'timeout_s':5}
    c.update(kw);return c


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.n=0
    def tearDown(self):self.tmp.cleanup()
    def run_cases(self,cases,**kwargs):
        self.n+=1;cfg=self.root/f'config{self.n}.json';out=self.root/f'out{self.n}'
        cfg.write_text(json.dumps({'schema_version':2,'cases':cases}));return run_suite(cfg,out,**kwargs),out,cfg
    def test_negative_match_stage_type_and_unexpected_pass(self):
        good={'status':'ACCURACY_FAILED','stage':'compare','exception':'ComparisonFailure'}
        wrong_stage=dict(good,stage='execute');wrong_type=dict(good,exception='ValueError')
        result,_,_=self.run_cases([
            base('expected',entry='musaopst.examples.cpu_ops:wrong',expect=good),
            base('stage',entry='musaopst.examples.cpu_ops:wrong',expect=wrong_stage),
            base('type',entry='musaopst.examples.cpu_ops:wrong',expect=wrong_type),
            base('unexpected',expect=good)])
        self.assertEqual([c['status'] for c in result['cases']],['EXPECTED_FAILURE','ACCURACY_FAILED','ACCURACY_FAILED','EXPECTATION_FAILED'])
        self.assertEqual(result['summary']['gpu_passed'],0);self.assertEqual(result['summary']['exit_code'],1)
    def test_all_negative_is_test_expectation_not_accuracy_pass(self):
        c=base(entry='musaopst.examples.cpu_ops:raises',expect={'status':'EXECUTION_FAILED','stage':'execute','exception':'RuntimeError'})
        result,_,_=self.run_cases([c]);self.assertEqual(result['summary']['exit_code'],0)
        self.assertEqual(result['summary']['cpu_platform_passed'],0)
        self.assertEqual(result['summary']['expected_failures'],1)
    def test_cpu_timeout_expected_and_reference_exception(self):
        result,_,_=self.run_cases([
            base('timeout',entry='musaopst.examples.cpu_ops:slow',attrs={'delay_s':3},timeout_s=.2,
                 expect={'status':'TIMEOUT','stage':'timeout','exception':'TimeoutExpired'}),
            base('ref',reference='musaopst.examples.cpu_ops:raises',expect={'status':'REFERENCE_FAILED','stage':'reference','exception':'RuntimeError'})])
        self.assertEqual([c['status'] for c in result['cases']],['EXPECTED_FAILURE']*2)
    def test_resume_hash_verifies_inputs_and_separates_provenance(self):
        c=base();first,path,_=self.run_cases([c]);second,_,_=self.run_cases([c],resume=path,rerun_failed=True)
        self.assertEqual(second['summary']['reused'],1);self.assertEqual(second['summary']['fresh_gpu_passed'],0)
        (path/'case-0000/inputs.npz').write_bytes(b'corrupt')
        third,_,_=self.run_cases([c],resume=path);self.assertEqual(third['summary']['reused'],0)
        changed,_,_=self.run_cases([base(seed=8)],resume=path);self.assertEqual(changed['summary']['reused'],0)
    def test_multiconfig_tags_and_merge_not_new_execution(self):
        first,a,ca=self.run_cases([base(tags=['smoke'])]);second,b,cb=self.run_cases([base('extra',tags=['extra'])])
        selected=run_suite([ca,cb],self.root/'selected',tags=['smoke'])
        self.assertEqual(len(selected['cases']),1)
        merged=merge_reports([a,b],self.root/'merged');self.assertEqual(merged['summary']['fresh_gpu_passed'],0)
        self.assertTrue(all(c['provenance']=='imported' for c in merged['cases']))
    def test_dynamic_same_worker_prepare_once(self):
        cases=[]
        for n in (1,17,51):
            c=base(str(n),shape_mode='runtime',compile_group='one')
            c['inputs']=[{'shape':[n],'dtype':'float32'}]*2;c['outputs']=[{'shape':[n],'dtype':'float32'}];cases.append(c)
        result,_,_=self.run_cases(cases)
        self.assertTrue(result['summary']['passed'])
        self.assertTrue(all(c['dynamic_execution']['prepare_calls']==1 for c in result['cases']))
        self.assertEqual(len({c['worker_pid'] for c in result['cases']}),1)
    def test_complex_input_tree_and_dynamic_output(self):
        (self.root/'bindings.py').write_text('import numpy as np\ndef op(*, xs, bias=None):\n return sum(xs) if bias is None else sum(xs)+bias\ndef ref(*,xs,bias=None):\n return np.sum(np.stack(xs),axis=0) if bias is None else np.sum(np.stack(xs),axis=0)+bias\ndef filt(x): return np.argwhere(x>0)\ndef ref_filter(x): return np.transpose(np.nonzero(x>0))\n')
        tensor={'shape':[3],'dtype':'float32'}
        tree=base('tree',entry='bindings:op',reference='bindings:ref',binding='named',inputs=[{'kind':'tensor_list','name':'xs','items':[tensor,tensor]},{'kind':'absent','name':'bias'}],outputs=[tensor])
        dyn=base('filter',entry='bindings:filt',reference='bindings:ref_filter',inputs=[tensor],outputs=[{'shape':'reference','dtype':'int64','rank':2,'max_elements':3}])
        result,_,_=self.run_cases([tree,dyn]);self.assertTrue(result['summary']['passed'])
    def test_baseline_rejects_unknown_gpu_identity(self):
        r,_,_=self.run_cases([base()]);c=r['cases'][0]
        c.update(backend='torch_musa',mode='MUSA_DEVICE',benchmark={'raw_samples_ms':[1]*5,'policy':{}},post_benchmark_comparison=c['comparison'])
        a=self.root/'a.json';a.write_text(json.dumps(r))
        out=compare_baseline(a,a,self.root/'basecheck');self.assertEqual(out['cases'][0]['status'],'INCOMPARABLE')
    def test_baseline_detects_regression_and_policy_mismatch(self):
        r,_,_=self.run_cases([base()]);r['cases'][0]['benchmark']={'raw_samples_ms':[1]*5,'policy':{'timer':'cpu'}};r['cases'][0]['post_benchmark_comparison']=r['cases'][0]['comparison']
        old=self.root/'old.json';old.write_text(json.dumps(r));r['cases'][0]['benchmark']['raw_samples_ms']=[2]*5
        new=self.root/'new.json';new.write_text(json.dumps(r))
        out=compare_baseline(new,old,self.root/'regression');self.assertEqual(out['exit_code'],1)
        r['cases'][0]['benchmark']['policy']['timer']='different';new.write_text(json.dumps(r))
        out=compare_baseline(new,old,self.root/'incomparable');self.assertEqual(out['exit_code'],2)
    def test_cache_never_prunes_unrecognized_folders(self):
        root=self.root/'cache';other=root/'native-unrelated';other.mkdir(parents=True)
        (other/'keep').write_text('keep');self.assertEqual(cache_prune(root)['removed'],0);self.assertTrue((other/'keep').exists())
