# SPDX-License-Identifier: MulanPSL-2.0
"""CPU-only validation of the legacy native protocol's supported policies."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from musaopst.backends.native import NativeBackend
from musaopst.errors import UnsupportedError


class LegacyNativePolicyTests(unittest.TestCase):
    def make_backend(self, folder, constant=False):
        case = {'backend': 'native', 'entry': 'fixture:entry', 'attrs': {},
                'inputs': [{'shape': [3], 'dtype': 'float32', 'layout': 'contiguous'},
                           {'shape': [3], 'dtype': 'float32', 'layout': 'contiguous', 'constant': constant}]}
        return NativeBackend(case, [np.zeros(3, dtype=np.float32) for _ in range(2)], folder)

    def test_v2_timing_policies_rejected_before_native_execution(self):
        base = {'warmup': 1, 'samples': 2, 'iterations': 3, 'mode': 'batch_event',
                'stream': 'current', 'cache': 'warm', 'allow_inplace': False}
        with tempfile.TemporaryDirectory() as folder:
            backend = self.make_backend(folder)
            with patch.object(backend, '_invoke', side_effect=AssertionError('native execution must not be reached')):
                for override in ({'mode': 'single_event'}, {'mode': 'wall_time'},
                                 {'stream': 'independent'}, {'cache': 'interference', 'cache_bytes': 4096},
                                 {'allow_inplace': True}):
                    with self.subTest(override=override), self.assertRaises(UnsupportedError):
                        backend.benchmark({**base, **override})

    def test_legacy_batch_event_keeps_actual_timing_scope(self):
        policy = {'warmup': 1, 'samples': 2, 'iterations': 3}
        with tempfile.TemporaryDirectory() as folder:
            backend = self.make_backend(folder)
            with patch.object(backend, '_invoke', return_value={'raw_samples_ms': [1., 2.], 'raw_batch_elapsed_ms': [3., 6.]}) as invoke:
                result = backend.benchmark(policy)
                invoke.assert_called_once_with(policy)
            self.assertEqual(result['policy']['timer'], 'musa_event')
            self.assertEqual(result['policy']['mode'], 'batch_event')
            self.assertEqual(result['policy']['stream'], 'native default stream (0)')
            self.assertEqual(result['policy']['iterations_per_sample'], 3)
            self.assertIn('NOT individual-call', result['policy']['quantile_meaning'])

    def test_compile_time_constant_rejected_before_legacy_build(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = self.make_backend(folder, constant=True)
            descriptor = {'protocol': 'vector_add_f32_v1', 'source': str(Path(folder) / 'never_compile.mu')}
            with patch('musaopst.backends.native.load_entry', return_value=lambda *args: descriptor), \
                 patch('musaopst.backends.native.subprocess.run', side_effect=AssertionError('compiler must not be reached')):
                with self.assertRaises(UnsupportedError):
                    backend.prepare()


if __name__ == '__main__':
    unittest.main()
