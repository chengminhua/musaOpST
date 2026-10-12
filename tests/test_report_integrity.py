# SPDX-License-Identifier: MulanPSL-2.0
"""Reported success requires complete matching evidence, even when read offline."""
import copy
import unittest
from musaopst.reporting import validate_case_result, summarize
from musaopst.expectations import apply_expectation


def passing():
    return {'name':'example','backend':'cpu','mode':'CPU_PLATFORM_TEST_ONLY','required':True,'status':'PASS',
            'comparison':{'passed':True,'contract_errors':[],'outputs':[{'passed':True}]}}


def expected():
    result=passing()
    result.update(status='ACCURACY_FAILED',last_stage='compare',exception_type='ComparisonFailure',message='bad values')
    result['comparison']['passed']=False
    return apply_expectation(result,{'backend':'cpu','expect':{'status':'ACCURACY_FAILED','stage':'compare','exception':'ComparisonFailure'}})


class ReportIntegrityTests(unittest.TestCase):
    def test_expected_failure_is_valid_but_never_accuracy_pass(self):
        record=expected()
        validate_case_result(record)
        summary=summarize([record])
        self.assertTrue(summary['passed'])
        self.assertEqual(summary['expected_failures'],1)
        self.assertEqual(summary['gpu_passed'],0)
        self.assertEqual(summary['cpu_platform_passed'],0)

    def test_expected_failure_revalidates_status_stage_and_exception(self):
        modifications=[{'observed_status':'made-up'}, {'last_stage':'build'}, {'exception_type':'OtherError'},
                       {'expectation':{}}, {'expectation_evaluated':False}, {'expectation_met':False},
                       {'mode':'MUSA_DEVICE'}, {'fatal_device_error':True}, {'cleanup_error':'failed'},
                       {'expectation':{'status':'ACCURACY_FAILED','stage':'compare','exception':'ComparisonFailure','message_contains':'absent-token'}}]
        for changes in modifications:
            record=expected();record.update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                validate_case_result(record)

    def test_device_timeout_cannot_be_forged_as_expected(self):
        record=expected()
        record.update(backend='torch_musa',mode='MUSA_DEVICE',observed_status='TIMEOUT',last_stage='timeout',exception_type='TimeoutExpired',
                      expectation={'status':'TIMEOUT','stage':'timeout','exception':'TimeoutExpired'})
        with self.assertRaises(ValueError):validate_case_result(record)

    def test_pass_rejects_fatal_cleanup_repeat_and_post_timing_failure(self):
        changes=[{'fatal_device_error':True},{'cleanup_error':'close error'},{'repeat':{'anomalies':[{'iteration':1}]}},
                 {'repeat':{'anomalies':None}}, {'post_benchmark_comparison':{'passed':False}},
                 {'benchmark':{'raw_samples_ms':[1,2,3]}}, {'expectation_evaluated':True,'expectation_met':False}]
        for update in changes:
            record=passing();record.update(update)
            with self.subTest(update=update),self.assertRaises(ValueError):validate_case_result(record)

    def test_valid_timing_requires_comparison_but_old_plain_pass_reads(self):
        record=passing();validate_case_result(record)
        record.update(benchmark={'raw_samples_ms':[1,2,3]},post_benchmark_comparison=copy.deepcopy(record['comparison']),repeat={'anomalies':[]})
        validate_case_result(record)


if __name__=='__main__':unittest.main()
