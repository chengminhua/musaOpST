# SPDX-License-Identifier: MulanPSL-2.0
"""Offline contract/data/comparison regressions; no GPU packages imported."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from musaopst.binding import bind_inputs, flatten_tensors, map_inputs, tensor_specs, tree_description
from musaopst.compare import compare_outputs
from musaopst.config import ConfigError, V2_DTYPES, expand_config, revalidate_cases
from musaopst.data import clone_inputs, make_inputs, reference_inputs, tensor_metadata, quantize_bfloat16, source_metadata


def case(**updates):
    value = {'name': 'v2', 'backend': 'cpu', 'entry': 'musaopst.examples.cpu_ops:add',
             'reference': 'musaopst.examples.references:add',
             'inputs': [{'shape': [3], 'dtype': 'float32'}],
             'outputs': [{'shape': [3], 'dtype': 'float32'}]}
    value.update(updates)
    return value


def normalize(value, **root):
    return expand_config({'schema_version': 2, 'cases': [value], **root})[0]


def custom_check(actual, reference, contract, policy):
    actual.fill(999)
    return {'passed': False, 'message': 'controlled custom mismatch'}


def custom_invalid(*args):
    return True


class ExtendedConfigTests(unittest.TestCase):
    def test_v1_preserved_v2_marked_and_roundtrip(self):
        old = expand_config({'schema_version': 1, 'cases': [case()]})[0]
        new = normalize(case())
        self.assertNotIn('_schema_version', old)
        self.assertEqual(new['_schema_version'], 2)
        self.assertEqual(revalidate_cases([new])[0], new)

    def test_repeat_partial_overrides_keep_both_bounds(self):
        count_only = normalize(case(repeat={'count': 1}))
        self.assertEqual(count_only['repeat'], {'count': 1, 'duration_s': 0.0})
        duration_only = normalize(case(repeat={'duration_s': .01}))
        self.assertEqual(duration_only['repeat'], {'count': 0, 'duration_s': .01})
        self.assertEqual(revalidate_cases([count_only])[0]['repeat'], count_only['repeat'])

    def test_per_output_precedence_and_dtype(self):
        value = case(compare={'atol': .2}, compare_by_dtype={'int64': {'exact': True}, 'float32': {'atol': .3}},
                     outputs=[{'shape': [3], 'dtype': 'float32', 'compare': {'atol': .4}},
                              {'shape': [3], 'dtype': 'int64'}])
        resolved = normalize(value, defaults={'compare': {'atol': .1, 'rtol': .05}})
        self.assertEqual(resolved['outputs'][0]['compare']['atol'], .4)
        self.assertEqual(resolved['outputs'][0]['compare']['rtol'], .05)
        self.assertTrue(resolved['outputs'][1]['compare']['exact'])
        actual = [np.ones(3, np.float32), np.array([2**63-1, 0, 1], np.int64)]
        expected = [np.ones(3)*1.01, np.array([2**63-2, 0, 1], np.int64)]
        result = compare_outputs(actual, expected, resolved['outputs'], resolved['compare'])
        self.assertTrue(result['outputs'][0]['passed'])
        self.assertFalse(result['outputs'][1]['passed'])

    def test_symbols_and_constraints(self):
        value = case(symbols={'N': [2, 4]}, constraints=[{'left': '${N}', 'op': 'divisible', 'right': 2}],
                     inputs=[{'shape': ['${N}'], 'dtype': 'float32'}], outputs=[{'shape': ['${N}'], 'dtype': 'float32'}])
        result = expand_config({'schema_version': 2, 'cases': [value]})
        self.assertEqual([x['inputs'][0]['shape'] for x in result], [[2], [4]])
        value['symbols']['N'] = [2, 3]
        with self.assertRaisesRegex(ConfigError, 'violates constraint'):
            normalize(value)

    def test_constraint_input_and_attr_references(self):
        value = case(attrs={'axis_size': 3}, attr_types={'axis_size': 'int'},
                     constraints=[{'left': {'input': 0, 'axis': 0}, 'op': 'eq', 'right': {'attr': 'axis_size'}}])
        normalize(value)
        value['attrs']['axis_size'] = 4
        with self.assertRaises(ConfigError):
            normalize(value)

    def test_attrs_types_and_unknown_fields(self):
        for updates in ({'attrs': {'x': True}, 'attr_types': {'x': 'int'}},
                        {'attrs': {'x': [1, '2']}, 'attr_types': {'x': 'list[int]'}},
                        {'typo': 5}, {'compare': {'metrics': {'cosine_min': 2}}}):
            with self.subTest(updates=updates), self.assertRaises(ConfigError):
                normalize(case(**updates))

    def test_expect_requires_stage_exception_for_failure(self):
        normalize(case(expect={'status': 'REFERENCE_FAILED', 'stage': 'reference', 'exception': 'ValueError'}))
        for expect in ({'status': 'ENVIRONMENT_ERROR'}, {'status': 'ACCURACY_FAILED'},
                       {'status': 'BUILD_FAILED', 'stage': 'build'}):
            with self.subTest(expect=expect), self.assertRaises(ConfigError):
                normalize(case(expect=expect))

    def test_benchmark_policies_validation(self):
        resolved = normalize(case(benchmark={'mode': 'single_event', 'enabled': True, 'allow_inplace': True}))
        self.assertEqual(resolved['benchmark']['iterations'], 1)
        for policy in ({'mode': 'single_event', 'iterations': 2}, {'allow_inplace': True},
                       {'mode': 'gpu_wall'}, {'cache_bytes': 0}, {'stream': 'unknown'}):
            with self.subTest(policy=policy), self.assertRaises(ConfigError):
                normalize(case(benchmark=policy))

    def test_zero_input_and_nested_binding(self):
        self.assertEqual(make_inputs(normalize(case(inputs=[]))), [])
        specs = [{'kind': 'absent', 'name': 'bias', 'binding': 'named'},
                 {'kind': 'tensor_list', 'name': 'xs', 'items': [{'shape': [2], 'dtype': 'int32'},
                                                              {'shape': [2], 'dtype': 'float16'}]}]
        resolved = normalize(case(inputs=specs))
        values = make_inputs(resolved)
        args, kwargs = bind_inputs(values, resolved['inputs'], {'scale': 2})
        self.assertIsNone(kwargs['bias'])
        self.assertEqual(len(args[0]), 2)
        self.assertEqual(len(flatten_tensors(values)), 2)
        self.assertEqual(len(tensor_specs(specs)), 2)
        self.assertEqual(tree_description(values)[1]['items'][1]['index'], 1)
        mapped = map_inputs(values, specs, lambda value, spec: spec['dtype'])
        self.assertEqual(mapped, [None, ['int32', 'float16']])

    def test_named_binding_collisions_rejected(self):
        value = case(inputs=[{'shape': [1], 'dtype': 'float32', 'name': 'x', 'binding': 'named'}], attrs={'x': 1})
        with self.assertRaisesRegex(ConfigError, 'binding'):
            normalize(value)

    def test_negative_overlapping_strides_rejected(self):
        for strides in ([-1, 1], [1, 1], [0, 1], [4]):
            value = case(inputs=[{'shape': [2, 3], 'dtype': 'float32', 'strides': strides}])
            with self.subTest(strides=strides), self.assertRaises(ConfigError):
                normalize(value)

    def test_dynamic_output_requires_bounds(self):
        normalize(case(outputs=[{'shape': 'reference', 'dtype': 'float32', 'rank': 1, 'max_elements': 20}]))
        with self.assertRaises(ConfigError):
            normalize(case(outputs=[{'shape': 'reference', 'dtype': 'float32'}]))

    def test_complex_ulp_rejected(self):
        with self.assertRaisesRegex(ConfigError, 'ULP'):
            normalize(case(outputs=[{'shape': [3], 'dtype': 'complex64', 'compare': {'metrics': {'ulp_max': 2}}}]))


class ExtendedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def file_case(self, filename, shape=[3], dtype='float32', **source):
        return normalize(case(inputs=[{'shape': shape, 'dtype': dtype, 'source': {'path': filename, **source}}]))

    def test_npy_npz_hash_and_original_dtype(self):
        data = np.array([1, 2, 3], np.float32)
        np.save(self.directory/'x.npy', data)
        np.savez(self.directory/'x.npz', chosen=data)
        for filename, source in [('x.npy', {}), ('x.npz', {'key': 'chosen'})]:
            value = self.file_case(filename, **source)
            actual = make_inputs(value, self.directory)[0]
            np.testing.assert_array_equal(actual, data)
            self.assertEqual(source_metadata(value['inputs'][0], self.directory)['actual_sha256'],
                             hashlib.sha256((self.directory/filename).read_bytes()).hexdigest())
        value = self.file_case('x.npy', sha256='0'*64)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            make_inputs(value, self.directory)
        with self.assertRaisesRegex(ValueError, 'dtype'):
            make_inputs(self.file_case('x.npy', dtype='float16'), self.directory)
        with self.assertRaisesRegex(ValueError, 'shape'):
            make_inputs(self.file_case('x.npy', shape=[1, 3]), self.directory)

    def test_npz_missing_key_and_pickle_forbidden(self):
        np.savez(self.directory/'x.npz', actual=np.ones(3, np.float32))
        with self.assertRaises(ValueError):
            make_inputs(self.file_case('x.npz', key='missing'), self.directory)
        with self.assertRaises(ConfigError):
            self.file_case('x.npz')
        np.save(self.directory/'object.npy', np.array([{}, {}, {}], dtype=object))
        with self.assertRaises(ValueError):
            make_inputs(self.file_case('object.npy'), self.directory)

    def test_binary_endian_length_and_bf16(self):
        data = np.array([1, 2, 3], dtype='>f4')
        data.tofile(self.directory/'x.bin')
        actual = make_inputs(self.file_case('x.bin', byte_order='big'), self.directory)[0]
        np.testing.assert_array_equal(actual, data)
        with self.assertRaises(ConfigError):
            self.file_case('x.bin')
        with self.assertRaisesRegex(ValueError, 'length'):
            make_inputs(self.file_case('x.bin', shape=[4], byte_order='big'), self.directory)
        rounded = quantize_bfloat16([1.1, -2.3, 0])
        ((rounded.view(np.uint32) >> 16).astype('<u2')).tofile(self.directory/'bf.bin')
        value = self.file_case('bf.bin', dtype='bfloat16', byte_order='little')
        np.testing.assert_array_equal(make_inputs(value, self.directory)[0], rounded)

    def test_bfloat16_scalar_quantization_and_binary_roundtrip(self):
        for raw, wanted in ((0x3f808000, 0x3f800000), (0x3f818000, 0x3f820000),
                            (0xbf818000, 0xbf820000), (0x7f800000, 0x7f800000),
                            (0xff800000, 0xff800000), (0x7f800001, 0x7fc00000)):
            scalar = np.array(raw, dtype=np.uint32).view(np.float32)
            rounded = quantize_bfloat16(scalar)
            with self.subTest(raw=hex(raw)):
                self.assertEqual(rounded.shape, ())
                self.assertEqual(rounded.dtype, np.dtype('float32'))
                self.assertEqual(int(rounded.view(np.uint32)), wanted)
                bits = (rounded.view(np.uint32) >> np.uint32(16)).astype('<u2')
                bits.tofile(self.directory/'scalar.bin')
                value = make_inputs(self.file_case('scalar.bin', shape=[], dtype='bfloat16', byte_order='little'), self.directory)[0]
                self.assertEqual(value.shape, ())
                self.assertEqual(value.dtype, np.dtype('float32'))
                self.assertEqual(int(value.view(np.uint32)), wanted)

    def test_bfloat16_float_file_must_be_prequantized(self):
        np.save(self.directory/'bf.npy', np.array([1.1, 2.2, 3.3], np.float32))
        with self.assertRaisesRegex(ValueError, 'representable'):
            make_inputs(self.file_case('bf.npy', dtype='bfloat16'), self.directory)
        values = quantize_bfloat16([1.1, 2.2, 3.3])
        np.save(self.directory/'bf.npy', values)
        np.testing.assert_array_equal(make_inputs(self.file_case('bf.npy', dtype='bfloat16'), self.directory)[0], values)

    def test_more_distributions_repeatable(self):
        distributions = [{'kind': 'beta', 'alpha': 2, 'beta': 3}, {'kind': 'laplace', 'loc': 0, 'scale': 1},
                         {'kind': 'triangular', 'left': -1, 'mode': 0, 'right': 2},
                         {'kind': 'integers', 'low': -3, 'high': 4}, {'kind': 'bernoulli', 'p': .3}]
        for distribution in distributions:
            value = normalize(case(inputs=[{'shape': [20], 'dtype': 'float32', 'distribution': distribution}]))
            np.testing.assert_array_equal(make_inputs(value)[0], make_inputs(value)[0])

    def test_dtype_generation_and_complex_components(self):
        for dtype in V2_DTYPES:
            value = normalize(case(inputs=[{'shape': [3], 'dtype': dtype, 'distribution': {'kind': 'fixed', 'values': 1}}]))
            data = make_inputs(value)[0]
            self.assertEqual(data.dtype.name, 'float32' if dtype == 'bfloat16' else dtype)
        value = normalize(case(inputs=[{'shape': [3], 'dtype': 'complex64', 'distribution': {'kind': 'fixed', 'values': [1,2,3], 'imag': [4,5,6]}}]))
        np.testing.assert_array_equal(make_inputs(value)[0], np.array([1+4j, 2+5j, 3+6j], np.complex64))

    def test_integer_fixed_no_silent_overflow(self):
        for dtype, values in [('uint8', [-1]), ('int8', [128]), ('int32', [1.5])]:
            value = normalize(case(inputs=[{'shape': [1], 'dtype': dtype, 'distribution': {'kind': 'fixed', 'values': values}}]))
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                make_inputs(value)
        value = normalize(case(inputs=[{'shape': [1], 'dtype': 'uint64', 'distribution': {'kind': 'fixed', 'values': [2**64-1]}}]))
        self.assertEqual(int(make_inputs(value)[0][0]), 2**64-1)

    def test_uint64_mixed_fixed_values_do_not_round_through_float(self):
        values = [0, 2**64-1, 2**64-2]
        config = normalize(case(inputs=[{'shape':[3], 'dtype':'uint64',
                                        'distribution':{'kind':'fixed','values':values}}]))
        self.assertEqual(make_inputs(config)[0].tolist(), values)
        config['inputs'][0]['distribution']['values'] = [0, 2**64, 1]
        with self.assertRaises(ValueError):
            make_inputs(config)

    def test_explicit_stride_offset_clone_and_reference(self):
        value = normalize(case(inputs=[{'shape': [2,3], 'dtype': 'float16', 'strides': [1,5], 'storage_offset': 7}]))
        actual = make_inputs(value)[0]
        for item in (actual, clone_inputs([actual])[0], reference_inputs([actual], 'float32')[0]):
            self.assertEqual(tensor_metadata(item)['strides'], [1,5])
            self.assertEqual(tensor_metadata(item)['storage_offset'], 7)
        self.assertFalse(np.shares_memory(actual, clone_inputs([actual])[0]))

    def test_transpose_and_slice_keep_views(self):
        value = normalize(case(inputs=[{'shape': [2,2], 'base_shape': [3,4], 'dtype': 'float32',
                                       'transforms': [{'kind': 'transpose', 'axes': [1,0]},
                                                      {'kind': 'slice', 'start': [1,0], 'stop': [4,2], 'step': [2,1]}]}]))
        actual = make_inputs(value)[0]
        self.assertEqual(actual.shape, (2,2))
        self.assertEqual(tensor_metadata(actual)['strides'], [2,4])
        self.assertEqual(tensor_metadata(actual)['storage_offset'], 1)
        with self.assertRaises(ConfigError):
            normalize(case(inputs=[{'shape':[2], 'base_shape':[4], 'dtype':'float32', 'transforms':[{'kind':'slice','step':[-1]}]}]))

    def test_reference_precision_tree_preserves_integer_exactness(self):
        values = [None, [np.array([1.1], np.float16), np.array([2**63-1], np.int64), np.array([1+2j], np.complex64)]]
        for policy, real, complex_dtype in [('float64','float64','complex128'), ('float32','float32','complex64'), ('preserve','float16','complex64')]:
            refs = reference_inputs(values, policy)
            self.assertIsNone(refs[0])
            self.assertEqual(refs[1][0].dtype.name, real)
            self.assertEqual(refs[1][1].dtype.name, 'int64')
            self.assertEqual(int(refs[1][1][0]), 2**63-1)
            self.assertEqual(refs[1][2].dtype.name, complex_dtype)


class ExtendedCompareTests(unittest.TestCase):
    def check(self, actual, expected, policy=None, contract=None, **kwargs):
        contract = contract or {'shape': list(actual.shape), 'dtype': actual.dtype.name}
        return compare_outputs([actual], [expected], [contract], policy or {}, **kwargs)

    def test_bool_and_uint64_exact(self):
        for actual, expected in [(np.array([True, False]), np.array([True, True])),
                                 (np.array([2**64-1], np.uint64), np.array([2**64-2], np.uint64))]:
            result = self.check(actual, expected, {'exact': True})
            self.assertFalse(result['passed'])
            self.assertEqual(result['outputs'][0]['max_absolute_error'], 1)

    def test_complex_checks_both_components_and_nan_policy(self):
        actual = np.array([1+2j], np.complex64)
        result = self.check(actual, np.array([1+3j]), {'atol': 0, 'rtol': 0})
        self.assertFalse(result['passed'])
        self.assertTrue(result['outputs'][0]['failures'][0]['components']['real']['passed'])
        self.assertFalse(result['outputs'][0]['failures'][0]['components']['imag']['passed'])
        actual = np.array([complex(float('nan'),2)], np.complex64)
        self.assertTrue(self.check(actual, actual.copy(), {'nan':'equal'})['passed'])
        self.assertFalse(self.check(actual, np.array([complex(float('nan'),3)]), {'nan':'equal'})['passed'])
        json.dumps(result, allow_nan=False)

    def test_ulp_threshold_and_signed_zero(self):
        actual = np.array([1], np.float32)
        reference = np.nextafter(actual, np.float32(2))
        passed = self.check(actual, reference, {'metrics': {'ulp_max':1}})
        failed = self.check(actual, reference, {'metrics': {'ulp_max':0}})
        self.assertTrue(passed['passed'])
        self.assertEqual(passed['outputs'][0]['metrics']['max_ulp'], 1)
        self.assertFalse(failed['passed'])
        self.assertTrue(self.check(np.array([-0.],np.float32), np.array([0.]), {'metrics':{'ulp_max':0}})['passed'])

    def test_rmse_cosine_only_gate_when_requested(self):
        actual = np.array([1,2], np.float32)
        reference = np.array([1,2.1])
        self.assertTrue(self.check(actual,reference, {'atol':.2})['passed'])
        self.assertFalse(self.check(actual,reference, {'atol':.2,'metrics':{'rmse_max':.01}})['passed'])
        self.assertFalse(self.check(actual,-actual, {'atol':5,'metrics':{'cosine_min':0}})['passed'])
        self.assertTrue(self.check(np.zeros(3,np.float32), np.zeros(3), {'metrics':{'cosine_min':1}})['passed'])

    def test_empty_and_no_finite_metric_population(self):
        self.assertTrue(self.check(np.empty(0,np.float32), np.empty(0), {'metrics':{'rmse_max':0}})['passed'])
        self.assertFalse(self.check(np.array([np.nan],np.float32),np.array([np.nan]), {'nan':'equal','metrics':{'rmse_max':0}})['passed'])

    def test_dynamic_contract_uses_independent_reference(self):
        contract = {'shape':'reference','rank':1,'max_elements':4,'dtype':'float32'}
        self.assertTrue(self.check(np.ones(2,np.float32),np.ones(2),contract=contract)['passed'])
        self.assertFalse(self.check(np.ones(3,np.float32),np.ones(2),contract=contract)['passed'])
        self.assertFalse(self.check(np.ones(5,np.float32),np.ones(5),contract=contract)['passed'])
        self.assertFalse(self.check(np.ones((1,2),np.float32),np.ones((1,2)),contract=contract)['passed'])

    def test_layout_and_alias_require_evidence(self):
        actual = np.arange(3,dtype=np.float32)
        contract={'shape':[3],'dtype':'float32','layout':{'strides':[2]},'alias_input':0}
        self.assertFalse(self.check(actual,actual,contract=contract)['passed'])
        meta={'shape':[3],'dtype':'float32','strides':[2],'alias_inputs':[0]}
        self.assertTrue(self.check(actual,actual,contract=contract,actual_metadata=[meta])['passed'])

    def test_custom_return_contract_and_isolation(self):
        actual = np.ones(3,np.float32)
        result=self.check(actual,actual.copy(),{'custom':__name__+':custom_check'})
        self.assertFalse(result['passed'])
        np.testing.assert_array_equal(actual,np.ones(3))
        invalid=self.check(actual,actual.copy(),{'custom':__name__+':custom_invalid'})
        self.assertIn('boolean passed',invalid['outputs'][0]['custom']['error'])


if __name__ == '__main__':
    unittest.main()
