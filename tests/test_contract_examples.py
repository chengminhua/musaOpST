# SPDX-License-Identifier: MulanPSL-2.0
"""Offline construction checks only: these do not certify any GPU backend."""
import importlib
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from musaopst.binding import bind_inputs, flatten_tensors
from musaopst.config import load_config
from musaopst.data import make_inputs, reference_inputs
from musaopst.util import as_outputs, load_entry


class ContractExampleTests(unittest.TestCase):
    def test_every_contract_example_constructs_independent_reference(self):
        path=Path(__file__).resolve().parents[1]/'examples/suites/contracts_matrix.json'
        cases=load_config(path)
        self.assertEqual({c['backend'] for c in cases},{'torch_musa','musa_cpp','triton','tilelang','native'})
        for case in cases:
            with self.subTest(case=case['name']):
                inputs=make_inputs(case,path.parent)
                refs=reference_inputs(inputs,case.get('reference_policy','float64'))
                self.assertTrue(all(not np.shares_memory(a,b) for a,b in zip(flatten_tensors(inputs),flatten_tensors(refs))))
                args,kwargs=bind_inputs(refs,case['inputs'],case['attrs'],binding=case.get('binding','positional'))
                outputs=as_outputs(load_entry(case['reference'])(*args,**kwargs))
                self.assertEqual(len(outputs),len(case['outputs']))
                for output,spec in zip(outputs,case['outputs']):
                    if spec['shape']=='reference':
                        self.assertEqual(output.ndim,spec['rank'])
                        self.assertLessEqual(output.size,spec['max_elements'])
                    else:self.assertEqual(list(output.shape),spec['shape'])

    def test_contract_callable_module_import_is_sdk_free(self):
        import builtins
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in {'torch','torch_musa','triton','tilelang'}:
                raise AssertionError('GPU dependency imported during example discovery')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded):
            module=importlib.import_module('musaopst.examples.contract_ops')
            importlib.reload(module)
            self.assertTrue(callable(module.max_with_index))


if __name__=='__main__':unittest.main()
