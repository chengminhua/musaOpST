# SPDX-License-Identifier: MulanPSL-2.0
"""Acceptance driver's expected failures must not hide device loss."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).resolve().parents[1]/'tools/run_gpu_acceptance.py'
spec=importlib.util.spec_from_file_location('gpu_acceptance_driver',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AcceptanceDriverTests(unittest.TestCase):
    def test_expected_exit_cannot_hide_fatal_or_wrong_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'negative').mkdir()
            p=out/'negative/results.json'
            for case in [{'status':'ENVIRONMENT_ERROR'}, {'status':'ACCURACY_FAILED','fatal_device_error':True}]:
                p.write_text(json.dumps({'run_complete':True,'cases':[case]}))
                self.assertFalse(module.validate_outcome('negative',1,1,out))

    def test_safe_negative_requires_completion_and_no_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'negative').mkdir()
            p=out/'negative/results.json'
            result={'run_complete':True,'cases':[{'status':'ACCURACY_FAILED'}]}
            p.write_text(json.dumps(result))
            self.assertTrue(module.validate_outcome('negative',1,1,out))
            result['cases'][0]['benchmark']={}
            p.write_text(json.dumps(result))
            self.assertFalse(module.validate_outcome('negative',1,1,out))
