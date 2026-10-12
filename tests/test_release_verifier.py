# SPDX-License-Identifier: MulanPSL-2.0
"""Reject invented or incomplete evidence; CPU fixtures are not GPU evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from musaopst.backends.gpu import event_statistics
from musaopst.util import sha256
from tools.verify_release import check_constants, check_dynamic, check_profile, check_snapshots, check_timing


class VerifierTests(unittest.TestCase):
    def test_pass_requires_complete_snapshot_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('inputs.npz','actual.npz','reference.npz'):
                np.savez(root/name,value=[1.])
            case={'name':'fixture','status':'PASS','snapshot_hashes':{name:sha256(root/name) for name in ('inputs.npz','actual.npz')}}
            errors=[];check_snapshots(case,root,errors)
            self.assertTrue(any('mandatory' in e and 'reference.npz' in e for e in errors))
            case['snapshot_hashes']['reference.npz']=sha256(root/'reference.npz')
            errors=[];check_snapshots(case,root,errors);self.assertEqual(errors,[])
            case['benchmark']={'raw_samples_ms':[1.]}
            errors=[];check_snapshots(case,root,errors)
            self.assertTrue(any('post_benchmark.npz' in e for e in errors))

    def test_wall_time_cannot_claim_event_timer(self):
        config={'enabled':True,'mode':'wall_time','warmup':1,'samples':3,'iterations':5,'stream':'current'}
        timing=event_statistics([1.,2.,3.],config,allocation_included=False)
        timing['raw_batch_elapsed_ms']=[1.,2.,3.]
        timing['policy']['actual_stream']={'policy':'current'}
        timing['statistics']={key:timing[key] for key in ('median_ms','p90_ms','min_ms','max_ms')}
        timing['statistics'].update(n=3,percentile_method='numpy linear')
        case={'name':'fixture','status':'PASS','backend':'triton','benchmark':timing,'config':{'benchmark':config}}
        errors=[];check_timing(case,errors);self.assertEqual(errors,[])
        timing['policy']['timer']='musa_event'
        errors=[];check_timing(case,errors);self.assertTrue(any('timer' in e for e in errors))

    def test_cpu_only_trace_cannot_validate_gpu_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            (folder/'profile_trace.json').write_text(json.dumps({'traceEvents':[{'cat':'cpu_op','name':'fixture','dur':2.}]}))
            (folder/'profile_table.txt').write_text('offline fixture')
            case={'name':'fixture','status':'PASS','config':{'profile':{'enabled':True,'iterations':1}},
                  'profile':{'trace_path':'/remote/profile_trace.json','table_path':'/remote/profile_table.txt',
                             'gpu_kernel_count':1,'gpu_durations_us':[2.],'kernel_names':['fixture'],
                             'iterations':1,'separate_from_benchmark':True}}
            errors=[];check_profile(case,folder,errors)
            self.assertTrue(any('GPU trace events' in e for e in errors))

    def _dynamic(self, backend='triton'):
        cases=[]
        for index,size in enumerate((1,17,513)):
            launch={'compiled_artifact_ids':['a'*64],'observed_compiled_artifacts':1}
            kind='runtime_arguments; verify compiled_artifact_ids for every binding'
            if backend=='torch_musa':launch={};kind='eager_framework_dispatch'
            meta={'launch':launch,'build_count':1,'runtime_shape_kind':kind,'runtime_shape_rebindings':index}
            cases.append({'name':f'fixture{size}','backend':backend,'status':'PASS','worker_pid':123,
                          'config':{'compile_group':'group','inputs':[{'shape':[size],'dtype':'float32'}]},
                          'backend_metadata':meta,'dynamic_execution':{'group':'group','same_worker':True,'prepare_calls':1,'artifact_evidence':copy.deepcopy(meta)}})
        return cases

    def test_dynamic_re_specialization_is_rejected(self):
        cases=self._dynamic();errors=[];rows=check_dynamic(cases,errors)
        self.assertEqual(errors,[]);self.assertTrue(rows[0]['same_artifact_verified'])
        cases[-1]['backend_metadata']['launch']['compiled_artifact_ids'].append('b'*64)
        cases[-1]['dynamic_execution']['artifact_evidence']=copy.deepcopy(cases[-1]['backend_metadata'])
        errors=[];rows=check_dynamic(cases,errors)
        self.assertFalse(rows[0]['same_artifact_verified']);self.assertTrue(any('re-specialization' in e for e in errors))

    def test_eager_dynamic_never_claims_compilation(self):
        errors=[];rows=check_dynamic(self._dynamic('torch_musa'),errors)
        self.assertEqual(errors,[])
        self.assertFalse(rows[0]['same_artifact_verified'])
        self.assertIn('eager_dispatch_only',rows[0]['classification'])

    def test_constants_need_quantized_input_and_distinct_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases=[]
            for index,(dtype,value,literal) in enumerate((('float32',1.,1),('float32',2.,2),('float32',3.,3),('float16',1.099609375,1.1),('bfloat16',1.1015625,1.1))):
                folder=Path(tmp)/str(index);folder.mkdir();np.savez(folder/'inputs.npz',input_0=[1.],input_1=value)
                record={'input_index':1,'dtype':dtype,'quantized_value':value,'value_hex':value.hex(),'source':'actual_quantized_device_input','runtime_kernel_argument':False}
                record['content_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
                artifact=hashlib.sha256(str(value).encode()).hexdigest()
                launch={'compiled_constants':[record],'artifact_id':artifact,'compiled_artifact_ids':[artifact],'artifact_kind':'triton_compiled_kernel_hash','constant_mechanism':'tl.constexpr','kernel_runtime_arguments':['vector_pointer','output_pointer']}
                specs=[{'shape':[1],'dtype':dtype},{'shape':[],'dtype':dtype,'constant':True,'distribution':{'values':literal}}]
                cases.append(({'name':str(index),'backend':'triton','status':'PASS','config':{'tags':['compiled_constant'],'inputs':specs},'backend_metadata':{'launch':launch}},folder))
            errors=[];result=check_constants(cases,errors);self.assertEqual(errors,[])
            self.assertEqual(result[0]['distinct_float32_artifacts'],3)
            last=cases[-1][0]['backend_metadata']['launch']['compiled_constants'][0];last['quantized_value']=1.1
            errors=[];check_constants(cases,errors)
            self.assertTrue(any('actual quantized scalar' in e for e in errors))
            cases[1][0]['backend_metadata']['launch']['artifact_id']=cases[0][0]['backend_metadata']['launch']['artifact_id']
            errors=[];check_constants(cases,errors)
            self.assertTrue(any('three specializations' in e for e in errors))


if __name__=='__main__':unittest.main()
