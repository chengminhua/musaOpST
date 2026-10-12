# SPDX-License-Identifier: MulanPSL-2.0
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from musaopst.backends.native import NativeBackend
from musaopst.errors import BuildError

class LegacyNativeDescriptor(unittest.TestCase):
    def test_input_specs_do_not_overwrite_explicit_v1_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            descriptor={'protocol':'vector_add_f32_v1','source':str(Path(folder)/'missing.mu')}
            case={'entry':'plugin:entry','inputs':[{'dtype':'float32','layout':'contiguous'}]*2,'attrs':{}}
            backend=NativeBackend(case,[np.zeros(3,dtype=np.float32)]*2,folder)
            with patch('musaopst.backends.native.load_entry',return_value=lambda *args:descriptor):
                with self.assertRaisesRegex(BuildError,'Native source does not exist'):
                    backend.prepare()
