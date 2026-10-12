# SPDX-License-Identifier: MulanPSL-2.0
"""Test expectations never change the observed numerical/device outcome."""


def apply_expectation(result, case):
    expectation = case.get('expect')
    if not expectation or result.get('expectation_evaluated'):
        return result
    if isinstance(expectation, str):
        expectation = {'status': expectation}
    observed = result['status']
    wanted = expectation.get('status', 'PASS')
    result.update(observed_status=observed, expectation=expectation, expectation_evaluated=True)
    unsafe = (result.get('fatal_device_error') or result.get('cleanup_error')
              or observed in {'ENVIRONMENT_ERROR', 'SKIPPED'}
              or (observed == 'TIMEOUT' and case['backend'] != 'cpu'))
    matched = not unsafe and observed == wanted
    if wanted != 'PASS':
        matched = matched and result.get('last_stage') == expectation.get('stage')
        if expectation.get('exception'):
            matched = matched and result.get('exception_type') == expectation['exception']
        if expectation.get('message_contains'):
            matched = matched and expectation['message_contains'] in result.get('message', '')
    result['expectation_met'] = bool(matched)
    if matched and wanted != 'PASS':
        result['status'] = 'EXPECTED_FAILURE'
        result['message'] = '负向测试符合预期；不计为算子精度通过。' + result.get('message', '')
    elif not matched and observed == 'PASS':
        result.update(status='EXPECTATION_FAILED', message='预期失败，但用例意外成功')
    return result
