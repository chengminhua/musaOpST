# SPDX-License-Identifier: MulanPSL-2.0
"""Exact comparison must not round large integer operands through NumPy floats."""
import unittest
import numpy as np
from musaopst.compare import compare_outputs


class IntegerReferenceIntegrity(unittest.TestCase):
    def compare(self,actual,reference):
        return compare_outputs([actual],[reference],
            [{'shape':list(actual.shape),'dtype':actual.dtype.name}],
            {'exact':True,'atol':0,'rtol':0})

    def test_large_integer_output_and_float_reference_do_not_false_pass(self):
        values=[('uint64',2**63+1,2**63),('uint64',2**64-1,2**64),
                ('int64',2**53+1,2**53),('int64',-(2**53+1),-(2**53)),
                ('int64',2**63-1,2**63)]
        for dtype,actual,reference in values:
            with self.subTest(dtype=dtype,actual=actual):
                result=self.compare(np.array([actual],dtype=dtype),np.array([reference],dtype=np.float64))
                self.assertFalse(result['passed'])
                self.assertEqual(result['outputs'][0]['mismatch_count'],1)
                self.assertEqual(result['outputs'][0]['failures'][0]['reason'],'exact')

    def test_float_or_complex_output_cannot_hide_integer_reference_difference(self):
        for dtype in ('float64','complex128'):
            for integer_dtype,value in (('uint64',2**63+1),('int64',2**53+1)):
                with self.subTest(dtype=dtype,integer_dtype=integer_dtype):
                    result=self.compare(np.array([value-1],dtype=dtype),np.array([value],dtype=integer_dtype))
                    self.assertFalse(result['passed'])
                    self.assertEqual(result['outputs'][0]['mismatch_count'],1)

    def test_exactly_equal_mixed_scalar_and_vector_values_still_pass(self):
        pairs=[(np.array(2**63,dtype=np.uint64),np.array(2**63,dtype=np.float64)),
               (np.array([-2**63,0,2**53],dtype=np.int64),np.array([-2**63,0,2**53],dtype=np.float64)),
               (np.array([2**63],dtype=np.complex128),np.array([2**63],dtype=np.uint64)),
               (np.array([False,True],dtype=np.bool_),np.array([0.,1.],dtype=np.float64))]
        for actual,reference in pairs:
            with self.subTest(actual=actual,reference=reference):
                self.assertTrue(self.compare(actual,reference)['passed'])


if __name__=='__main__':
    unittest.main()
