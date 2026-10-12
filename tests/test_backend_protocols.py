# SPDX-License-Identifier: MulanPSL-2.0
"""Offline protocol/cache/policy tests; these never certify real GPU execution."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from musaopst.backends.build_cache import build_native_descriptor, load_cpp_artifact, _build_cpp
from musaopst.backends.gpu import GPUBackend, event_statistics
from musaopst.backends.native_protocol import encode_storage, decode_storage, NativeProtocolBackend
from musaopst.data import make_inputs, quantize_bfloat16, storage_offset
from musaopst.errors import BuildError, UnsupportedError, DeviceStateUnknownError


class CompiledConstantTests(unittest.TestCase):
    def test_constants_matrix_and_independent_quantized_reference(self):
        from musaopst.config import load_config
        from musaopst.data import reference_inputs
        from musaopst.examples.constant_reference import scale
        path=Path(__file__).resolve().parents[1]/'examples/suites/constants_matrix.json'
        cases=load_config(path)
        self.assertEqual(len(cases),19)
        self.assertEqual({case['backend'] for case in cases},{'torch_musa','musa_cpp','triton','tilelang','native'})
        for case in cases:
            values=make_inputs(case)
            reference=reference_inputs(values,case)
            result=scale(*reference)
            np.testing.assert_array_equal(result,reference[0]*float(np.asarray(reference[1])))
            if case['inputs'][1]['dtype']=='bfloat16':
                self.assertEqual(float(values[1]),1.1015625)
                self.assertNotEqual(float(values[1]),case['inputs'][1]['distribution']['values'])

    def test_compile_time_constant_cannot_be_a_runtime_value(self):
        from musaopst.examples.constant_ops import _constant_spec, unsupported
        case={'inputs':[{'shape':[17],'dtype':'float32'}, {'shape':[],'dtype':'float32'}]}
        with self.assertRaisesRegex(UnsupportedError,'constant:true'):
            _constant_spec(case)
        with self.assertRaisesRegex(UnsupportedError,'fixed runtime values'):
            unsupported()


class NativeProtocolTests(unittest.TestCase):
    def roundtrip(self, array, dtype):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'output.bin'
            storage, metadata = encode_storage(array, {'dtype':dtype})
            storage.tofile(path)
            return decode_storage(path,metadata),metadata

    def test_physical_strides_offset_preserved(self):
        base=np.arange(72,dtype=np.float32).reshape(6,12)
        array=base[1:5:2,2:11:3].T
        result,meta=self.roundtrip(array,'float32')
        np.testing.assert_array_equal(result,array)
        self.assertEqual(meta['strides'],[3,24])
        self.assertEqual(meta['storage_offset'],14)
        self.assertEqual(storage_offset(array),14)

    def test_bfloat16_exact_binary_encoding(self):
        array=quantize_bfloat16([1.1,-2.2,0.,float('nan'),float('inf')])
        result,meta=self.roundtrip(array,'bfloat16')
        np.testing.assert_array_equal(result,array)
        self.assertEqual(meta['storage_bytes'],len(array)*2)

    def test_bfloat16_scalar_preserves_32bit_decoding_storage(self):
        for number in (1.1,-2.2,0.,float('nan'),float('inf')):
            with self.subTest(number=number):
                array=quantize_bfloat16(np.asarray(number,dtype=np.float32))
                result,meta=self.roundtrip(array,'bfloat16')
                self.assertEqual(result.shape,())
                self.assertEqual(result.dtype,np.dtype('float32'))
                np.testing.assert_array_equal(result,array)
                self.assertEqual(meta['storage_bytes'],2)

    def test_all_integer_encodings_are_lossless(self):
        for dtype in ('int8','uint8','int16','uint16','int32','uint32','int64','uint64'):
            with self.subTest(dtype=dtype):
                limits=np.iinfo(dtype)
                array=np.array([limits.min,limits.max],dtype=dtype)
                result,_=self.roundtrip(array,dtype)
                np.testing.assert_array_equal(result,array)

    def test_bool_scalar_and_empty(self):
        for array in (np.array(True),np.empty((0,17),dtype=np.bool_),np.array([True,False])):
            result,_=self.roundtrip(array,'bool')
            np.testing.assert_array_equal(result,array)

    def test_negative_stride_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(UnsupportedError,'nonnegative'):
            encode_storage(np.arange(4,dtype=np.float32)[::-1],{'dtype':'float32'})

    def test_complex_native_protocol_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(UnsupportedError,'dtype'):
            encode_storage(np.ones(2,dtype=np.complex64),{'dtype':'complex64'})

    def test_output_byte_count_and_span_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'output.bin';np.arange(2,dtype=np.float32).tofile(path)
            valid={'shape':[2],'strides':[1],'dtype':'float32','storage_offset':0,'storage_bytes':8}
            for changed in ({'storage_bytes':12},{'shape':[3]},{'storage_offset':2},{'strides':[-1]}):
                with self.subTest(changed=changed),self.assertRaises(RuntimeError):
                    decode_storage(path,{**valid,**changed})

    def test_input_tree_serializes_absent_and_tensor_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=np.arange(4,dtype=np.float32)
            spec={'shape':[4],'dtype':'float32'}
            case={'backend':'native','inputs':[spec,{'kind':'absent'},{'kind':'tensor_list','items':[spec,spec]}]}
            backend=NativeProtocolBackend(case,[a,None,[a,a]],tmp)
            backend._write_inputs()
            self.assertEqual(backend.input_structure,[0,None,[1,2]])
            self.assertEqual(len(backend.input_meta),3)

    def test_native_profiler_never_accepts_cpu_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(UnsupportedError,'native profiler'):
                NativeProtocolBackend({},[],tmp).profile({})

    def test_native_missing_and_contradictory_completion_stop_device_validation(self):
        # Controlled CPU processes only; their malformed metadata is the test.
        scripts = ["raise SystemExit(1)",
                   "import pathlib,sys;pathlib.Path(sys.argv[4]).write_text('{bad')",
                   "import pathlib,sys;pathlib.Path(sys.argv[4]).write_text('{\"status\":\"PASS\"}');raise SystemExit(1)"]
        for body in scripts:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                binary=Path(tmp)/'controlled_process'
                binary.write_text('#!'+sys.executable+'\n'+body+'\n');binary.chmod(0o755)
                case={'inputs':[],'outputs':[{'shape':[1],'dtype':'float32'}],'attrs':{},'backend':'native'}
                backend=NativeProtocolBackend(case,[],tmp,{'protocol':'musaopst_native_v2','binary':str(binary)})
                backend.prepare()
                with self.assertRaises(DeviceStateUnknownError):backend.run()


class TimingPolicyTests(unittest.TestCase):
    def test_modes_have_distinct_timer_and_quantile_definitions(self):
        policy={'warmup':1,'samples':3,'iterations':5}
        batch=event_statistics([1.,2.,3.],policy,allocation_included=False)
        single=event_statistics([1.,2.,3.],{**policy,'mode':'single_event'},allocation_included=False)
        wall=event_statistics([1.,2.,3.],{**policy,'mode':'wall_time'},allocation_included=False)
        self.assertEqual(batch['policy']['iterations_per_sample'],5)
        self.assertEqual(single['policy']['iterations_per_sample'],1)
        self.assertEqual(wall['policy']['timer'],'host_wall_sync')
        self.assertFalse(single['policy']['includes_host_synchronization'])
        self.assertTrue(wall['policy']['includes_host_synchronization'])
        self.assertNotEqual(single['policy']['quantile_meaning'],batch['policy']['quantile_meaning'])

    def test_interference_is_not_claimed_as_guaranteed_cold_cache(self):
        result=event_statistics([1.],{'warmup':0,'samples':1,'iterations':1,'cache':'interference','cache_bytes':1024},allocation_included=False)
        self.assertIn('NOT guaranteed',result['policy']['cache'])
        self.assertEqual(result['policy']['cache_bytes'],1024)

    def test_runtime_rebind_rejects_undeclared_factory(self):
        backend=GPUBackend({'backend':'triton'},[],'.')
        backend.launch=lambda: []
        with self.assertRaisesRegex(UnsupportedError,'rebind hook'):
            backend.rebind({},[])

    def test_group_metadata_snapshots_do_not_alias_later_bindings(self):
        backend=GPUBackend({'backend':'triton'},[],'.')
        backend.info['launch']={'compiled_artifact_ids':['one']}
        snapshot=backend.metadata()
        backend.info['launch']['compiled_artifact_ids'].append('two')
        self.assertEqual(snapshot['launch']['compiled_artifact_ids'],['one'])

    def test_advanced_imports_never_import_gpu_packages(self):
        code='''import sys
from musaopst.examples import advanced_ops, advanced_cpp, advanced_triton, advanced_tilelang, native_protocol_ops
from musaopst.backends import native_protocol, build_cache
assert not any(p in sys.modules for p in ('torch','torch_musa','triton','tilelang'))
'''
        process=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(process.returncode,0,process.stderr)


class NativeCacheTests(unittest.TestCase):
    def test_cache_invalidates_source_header_flags_and_corrupted_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);compiler=root/'fake-mcc';source=root/'operator.mu';header=root/'constants.hpp'
            source.write_text('#include "constants.hpp"\nint main(){}\n');header.write_text('#define VALUE 1\n')
            compiler.write_text('#!'+sys.executable+'\nimport sys,pathlib\nif "--version" in sys.argv: print("offline fake compiler 1");sys.exit(0)\np=pathlib.Path(sys.argv[sys.argv.index("-o")+1]);p.write_text("#!/bin/sh\\nexit 0\\n");p.chmod(0o755)\n')
            compiler.chmod(0o755)
            with patch.dict(os.environ,{'MUSA_OPST_MCC':str(compiler),'MUSA_OPST_CACHE':str(root/'cache')}):
                spec={'source':str(source)}
                first=build_native_descriptor(spec,root/'work','mp_31')
                self.assertFalse(first['cache_hit'])
                second=build_native_descriptor(spec,root/'work','mp_31')
                self.assertTrue(second['cache_hit'])
                self.assertEqual(first['cache_key'],second['cache_key'])
                header.write_text('#define VALUE 2\n')
                third=build_native_descriptor(spec,root/'work','mp_31')
                self.assertNotEqual(first['cache_key'],third['cache_key'])
                Path(third['binary']).write_text('corrupt')
                fourth=build_native_descriptor(spec,root/'work','mp_31')
                self.assertFalse(fourth['cache_hit'])
                fifth=build_native_descriptor({**spec,'compile_flags':['-O0']},root/'work','mp_31')
                self.assertNotEqual(fourth['cache_key'],fifth['cache_key'])

    def test_prebuilt_digest_is_verified_without_compiler(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary=Path(tmp)/'operator';binary.write_text('#!/bin/sh\nexit 0\n');binary.chmod(0o755)
            with patch.dict(os.environ,{'MUSA_OPST_MCC':'compiler-that-does-not-exist'}):
                result=build_native_descriptor({'binary':str(binary)},Path(tmp)/'work')
                self.assertEqual(result['build_kind'],'explicit_prebuilt')
                with self.assertRaisesRegex(BuildError,'digest mismatch'):
                    build_native_descriptor({'binary':str(binary),'artifact_sha256':'0'*64},Path(tmp)/'work')

    def test_native_build_timeout_has_separate_error_and_keeps_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'operator.mu';source.write_text('offline fixture')
            compiler=root/'controlled_compiler'
            compiler.write_text('#!'+sys.executable+'\nimport sys,time\nif "--version" in sys.argv:print("fixture");sys.exit(0)\nprint("compile started",flush=True);time.sleep(5)\n')
            compiler.chmod(0o755)
            with patch.dict(os.environ,{'MUSA_OPST_MCC':str(compiler),'MUSA_OPST_CACHE':str(root/'cache')}):
                with self.assertRaisesRegex(TimeoutError,'owned compiler'):
                    build_native_descriptor({'source':str(source)},root/'work','mp_31',timeout_s=.1)
                self.assertIn('compile started',(root/'work/native_build.log').read_text())

    def test_prebuilt_cpp_digest_checked_before_extension_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary=Path(tmp)/'bad.so';binary.write_bytes(b'not a shared object')
            with self.assertRaisesRegex(BuildError,'digest'):
                load_cpp_artifact({'binary':str(binary),'module_name':'offline_test','artifact_sha256':'0'*64})

    def test_cpp_stale_baton_recovered_under_private_cache_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'operator.mu';source.write_text('int offline_fixture;')
            compiler=root/'fake-mcc';compiler.write_text('#!/bin/sh\necho offline-fixture\n');compiler.chmod(0o755)
            torch=types.ModuleType('torch');torch.__version__='offline';torch._C=types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)
            musa=types.ModuleType('torch_musa');musa.__version__='offline'
            utils=types.ModuleType('torch_musa.utils');extension=types.ModuleType('torch_musa.utils.musa_extension')
            calls=[]
            def fake_load(**kwargs):
                directory=Path(kwargs['build_directory']);self.assertFalse((directory/'lock').exists())
                self.assertIn('--offload-arch=mp_31',kwargs['extra_musa_cflags'])
                library=directory/(kwargs['name']+'.so');library.write_bytes(b'offline fake artifact, never loaded')
                calls.append(str(directory));return types.SimpleNamespace(__file__=str(library))
            extension.load=fake_load
            modules={'torch':torch,'torch_musa':musa,'torch_musa.utils':utils,'torch_musa.utils.musa_extension':extension}
            with patch.dict(sys.modules,modules),patch.dict(os.environ,{'MUSA_OPST_MCC':str(compiler),'MUSA_OPST_CACHE':str(root/'cache')}):
                first=_build_cpp({'source':str(source)},root/'work','mp_31')
                entry=Path(first['cache_directory']);(entry/'manifest.json').unlink();(entry/'lock').write_text('stale')
                second=_build_cpp({'source':str(source)},root/'work','mp_31')
                self.assertEqual(first['cache_key'],second['cache_key'])
                self.assertEqual(len(calls),2)
                self.assertFalse((entry/'lock').exists())

    @unittest.skipUnless(shutil.which('c++'),'C++ compiler unavailable for standalone JSON parser check')
    def test_cpp_json_parser_preserves_integer_unicode_and_rejects_duplicates(self):
        header=Path(__file__).resolve().parents[1]/'musaopst/examples/native_json.hpp'
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'parser.cpp';binary=Path(tmp)/'parser'
            source.write_text('#include "'+str(header)+'"\n#include <iostream>\nint main(){std::string s;std::getline(std::cin,s);try{std::cout<<opst::parse_json(s).dump();return 0;}catch(...){return 1;}}\n')
            compiled=subprocess.run(['c++','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(compiled.returncode,0,compiled.stderr)
            value={'integer':9223372036854775807,'unicode':'测试😀','items':[True,None,-.25]}
            run=subprocess.run([str(binary)],input=json.dumps(value),capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(json.loads(run.stdout),value)
            bad=subprocess.run([str(binary)],input='{"a":1,"a":2}',capture_output=True,text=True)
            self.assertEqual(bad.returncode,1)


if __name__=='__main__': unittest.main()
