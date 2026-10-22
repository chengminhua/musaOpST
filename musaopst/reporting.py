# SPDX-License-Identifier: MulanPSL-2.0
"""JSON / Markdown / JUnit share one fail-closed coverage policy."""
from collections import Counter
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from .util import write_json

UNEXECUTED = {'SKIPPED', 'UNSUPPORTED'}
STATUSES = {'CONFIG_ERROR', 'BUILD_FAILED', 'REFERENCE_FAILED', 'EXECUTION_FAILED',
            'ACCURACY_FAILED', 'TIMEOUT', 'ENVIRONMENT_ERROR', 'UNSUPPORTED', 'SKIPPED', 'PASS', 'EXPECTED_FAILURE', 'EXPECTATION_FAILED'}
GPU_BACKENDS = {'torch_musa', 'musa_cpp', 'triton', 'tilelang', 'native'}


def validate_case_result(case):
    """Reject malformed records and PASS claims lacking comparison evidence."""
    if not isinstance(case, dict):
        raise ValueError('结果中的用例必须是对象')
    if not isinstance(case.get('name'), str) or not case['name'].strip():
        raise ValueError('结果中的用例缺少有效 name')
    if not isinstance(case.get('status'), str) or case['status'] not in STATUSES:
        raise ValueError(f"未知用例结果状态：{case.get('status')!r}")
    if type(case.get('required', True)) is not bool:
        raise ValueError('结果中的 required 必须是布尔值')
    elapsed = case.get('elapsed_s', 0)
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed < 0:
        raise ValueError('结果中的 elapsed_s 必须是有限非负数')
    try:
        if not math.isfinite(elapsed):
            raise ValueError('结果中的 elapsed_s 必须是有限非负数')
    except OverflowError as exc:
        raise ValueError('结果中的 elapsed_s 过大') from exc
    for field in ('backend', 'mode', 'message', 'log', 'case_result', 'traceback'):
        if field in case and not isinstance(case[field], str):
            raise ValueError(f'结果中的 {field} 必须是字符串')
    for field in ('comparison', 'benchmark', 'repeat'):
        if field in case and not isinstance(case[field], dict):
            raise ValueError(f'结果中的 {field} 必须是对象')
    if case['status'] in {'PASS', 'EXPECTED_FAILURE'}:
        backend = case.get('backend')
        if not isinstance(backend, str) or backend not in {'cpu', *GPU_BACKENDS}:
            raise ValueError('通过结果缺少有效执行后端')
        mode = 'CPU_PLATFORM_TEST_ONLY' if backend == 'cpu' else 'MUSA_DEVICE'
        if case.get('mode') != mode:
            raise ValueError('通过结果的执行模式与后端不一致')
        if case.get('fatal_device_error') or case.get('cleanup_error'):
            raise ValueError('设备或清理故障不能记为通过')
    if case['status'] == 'EXPECTED_FAILURE':
        expectation = case.get('expectation')
        observed = case.get('observed_status')
        allowed = {'CONFIG_ERROR', 'BUILD_FAILED', 'REFERENCE_FAILED', 'EXECUTION_FAILED',
                   'ACCURACY_FAILED', 'TIMEOUT', 'UNSUPPORTED'}
        if (case.get('expectation_met') is not True or case.get('expectation_evaluated') is not True
                or not isinstance(expectation, dict) or not isinstance(observed, str) or observed not in allowed
                or expectation.get('status') != observed):
            raise ValueError('预期失败结果缺少真实状态和明确匹配证据')
        if observed == 'TIMEOUT' and case.get('backend') != 'cpu':
            raise ValueError('设备超时不能作为预期失败通过')
        if (not isinstance(expectation.get('stage'), str) or not expectation['stage']
                or expectation['stage'] != case.get('last_stage')
                or not isinstance(expectation.get('exception'), str) or not expectation['exception']
                or expectation['exception'] != case.get('exception_type')):
            raise ValueError('预期失败的阶段或异常类型与实际结果不匹配')
        message = expectation.get('message_contains')
        if message is not None and (not isinstance(message, str) or message not in case.get('message', '')):
            raise ValueError('预期失败消息与实际结果不匹配')
    if case['status'] == 'PASS':
        def passed_comparison(comparison):
            if not isinstance(comparison, dict):
                return False
            outputs = comparison.get('outputs')
            return (comparison.get('passed') is True and comparison.get('contract_errors') == []
                    and isinstance(outputs, list) and bool(outputs)
                    and all(isinstance(output, dict) and output.get('passed') is True for output in outputs))
        if not passed_comparison(case.get('comparison')):
            raise ValueError('PASS 用例缺少已执行且通过的输出契约和独立参考比较')
        if 'post_benchmark_comparison' in case and not passed_comparison(case['post_benchmark_comparison']):
            raise ValueError('PASS 用例的计时后比较失败或证据不完整')
        if case.get('benchmark') and 'post_benchmark_comparison' not in case:
            raise ValueError('PASS 性能结果缺少计时后正确性比较')
        if 'repeat' in case:
            anomalies = case['repeat'].get('anomalies', [])
            if not isinstance(anomalies, list) or anomalies:
                raise ValueError('发现重复测试异常，不能记为 PASS')
        if case.get('expectation_evaluated'):
            expectation = case.get('expectation')
            if (case.get('expectation_met') is not True or not isinstance(expectation, dict)
                    or expectation.get('status') != 'PASS' or case.get('observed_status') != 'PASS'):
                raise ValueError('测试期望未满足或匹配证据不完整，不能记为 PASS')



def summarize(cases):
    if not isinstance(cases, list):
        raise ValueError('结果中的 cases 必须是列表')
    for case in cases:
        validate_case_result(case)
    counts = dict(Counter(c['status'] for c in cases))
    required = [c for c in cases if c.get('required', True)]
    required_missing = sum(c['status'] not in {'PASS','EXPECTED_FAILURE'} for c in required)
    failures = sum(c['status'] not in {'PASS', 'EXPECTED_FAILURE', *UNEXECUTED} for c in cases)
    passed = bool(counts.get('PASS') or counts.get('EXPECTED_FAILURE')) and required_missing == 0 and failures == 0
    return {'passed': passed, 'exit_code': 0 if passed else 1, 'total': len(cases),
            'expected_failures': counts.get('EXPECTED_FAILURE',0),
            'fresh_gpu_passed': sum(c['status']=='PASS' and c.get('backend') in GPU_BACKENDS and c.get('provenance','fresh')=='fresh' for c in cases),
            'reused': sum(c.get('provenance')=='reused' for c in cases),
            'counts': counts, 'required_total': len(required), 'required_nonpass': required_missing,
            'executed': sum(c['status'] not in UNEXECUTED for c in cases),
            'optional_unexecuted': sum(not c.get('required', True) and c['status'] in UNEXECUTED for c in cases),
            'gpu_passed': sum(c['status'] == 'PASS' and c.get('backend') in GPU_BACKENDS for c in cases),
            'cpu_platform_passed': sum(c['status'] == 'PASS' and c.get('backend') == 'cpu' for c in cases)}


def _escape(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def render_reports(result, output_dir):
    if not isinstance(result, dict) or type(result.get('schema_version')) is not int or result['schema_version'] not in (1,2):
        raise ValueError('不是 schema_version=1/2 的 musaopst 结果')
    if not isinstance(result.get('cases'), list):
        raise ValueError('结果中的 cases 必须是列表')
    # Validate before writing so an invalid report cannot overwrite old artifacts.
    summary = summarize(result['cases'])
    # A selector or usage mistake is not a device verdict: exit 2 keeps CI able
    # to distinguish "bad argument" from "required case failed".
    if result.get('usage_error'):
        summary['exit_code'] = 2
    if 'run_complete' in result and type(result['run_complete']) is not bool:
        raise ValueError('run_complete 必须是布尔值')
    planned = result.get('planned_cases', [])
    if not isinstance(planned, list):
        raise ValueError('planned_cases 必须是列表')
    completed_names = {c['name'] for c in result['cases']}
    pending = [c for c in planned if c['name'] not in completed_names]
    summary['pending'] = len(pending)
    summary['run_complete'] = result.get('run_complete', True) and not pending
    if not summary['run_complete']:
        summary['passed'] = False
        summary['exit_code'] = 1
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    result['summary'] = summary
    write_json(dest / 'results.json', result)
    summary = result['summary']
    lines = ['# musaopst 测试报告', '',
             f"整体测试期望：**{'满足' if summary['passed'] else '未满足'}**；退出码 `{summary['exit_code']}`。", '']
    resolution = (result.get('environment') or {}).get('device_resolution') if isinstance(result.get('environment'), dict) else None
    if isinstance(resolution, dict):
        from .device import format_resolution
        lines += [f"设备：{format_resolution(resolution)}", '']
        if resolution.get('note'):
            lines += [f"设备说明：{_escape(resolution['note'])}", '']
    selection = result.get('selection') if isinstance(result.get('selection'), dict) else None
    if selection and (selection.get('pattern') or selection.get('tags')):
        wanted = []
        if selection.get('pattern'):
            wanted.append(f"--case={selection['pattern']!r}")
        if selection.get('tags'):
            wanted.append('--tag=' + ','.join(selection['tags']))
        lines += [f"筛选：{' '.join(wanted)}；命中 {selection.get('matched', '?')}/{selection.get('available_count', '?')} 个用例。", '']
    if selection and selection.get('available'):
        names = selection['available']
        lines += [f"可用用例（前 {len(names)} 个）：{'、'.join(names)}。筛选支持 --case 通配符/子串与 --tag；使用 --list-cases 查看全部用例。", '']
    lines += [
             f"真实 GPU 通过 {summary['gpu_passed']} 项；CPU 平台测试通过 {summary['cpu_platform_passed']} 项。",
             f"必需项未通过 {summary['required_nonpass']} 项；可选项未执行 {summary['optional_unexecuted']} 项；待执行 {summary['pending']} 项。", '',
             f"调度完成：{summary['run_complete']}。未完成的运行不能整体通过。", '',
             f"本次新 GPU 精度通过 {summary['fresh_gpu_passed']} 项；复用记录 {summary['reused']} 项；预期失败 {summary['expected_failures']} 项（不计精度通过）。", '',
             'CPU 结果不能证明 GPU 正确；有限重复测试不代表长期稳定性认证。', '',
             '| 用例 | 后端 | 必需 | 状态 | 耗时秒 | 说明 |', '|---|---|---|---|---:|---|']
    for c in result['cases']:
        lines.append(f"| {_escape(c['name'])} | {_escape(c.get('backend','unknown'))} | {c.get('required',True)} | {c['status']} | {c.get('elapsed_s',0):.3f} | {_escape(c.get('message',''))} |")
    lines += ['', '## 复现', '', '配置和源码必须与记录的摘要匹配；设备编号由运行者选择。', '', '```bash', result.get('reproduce', 'unknown'), '```', '', '## 明细', '']
    for c in result['cases']:
        lines += [f"### {_escape(c['name'])}", '', f"状态：`{c['status']}`；模式：`{c.get('mode','unknown')}`。", '']
        if c.get('comparison'):
            lines += ['```json', __import__('json').dumps(c['comparison'], ensure_ascii=False, indent=2), '```', '']
        if c.get('benchmark'):
            b = c['benchmark']
            lines += [f"计时政策：`{b.get('policy',{})}`。", '', f"样本统计（毫秒）：`{b.get('statistics',{})}`。", '', 'P90 是所声明样本单位的分位数；批均值的 P90 不是单次调用尾延迟。', '']
        if c.get('repeat'):
            lines += [f"有限重复测试：`{c['repeat']}`。", '']
        if c.get('log'):
            lines += [f"日志：[{c['log']}]({c['log']})；详情：[{c.get('case_result','result.json')}]({c.get('case_result','result.json')})", '']
        if c.get('traceback'):
            lines += ['```text', c['traceback'], '```', '']
    (dest / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    suite = ET.Element('testsuite', name='musaopst', tests=str(len(result['cases'])),
                       failures=str(sum(c['status'] not in {'PASS','EXPECTED_FAILURE'} and (c.get('required',True) or c['status'] not in UNEXECUTED) for c in result['cases'])),
                       skipped=str(sum(c['status'] in UNEXECUTED and not c.get('required',True) for c in result['cases'])),
                       errors='0', time=str(sum(c.get('elapsed_s',0) for c in result['cases'])))
    props = ET.SubElement(suite, 'properties')
    for key, val in summary.items():
        ET.SubElement(props, 'property', name=str(key), value=str(val))
    for c in result['cases']:
        node = ET.SubElement(suite, 'testcase', name=c['name'], classname=c.get('backend','config'), time=str(c.get('elapsed_s',0)))
        if c['status'] in UNEXECUTED and not c.get('required',True):
            ET.SubElement(node, 'skipped', message=c.get('message',c['status']))
        elif c['status'] not in {'PASS','EXPECTED_FAILURE'}:
            ET.SubElement(node, 'failure', type=c['status'], message=c.get('message',c['status'])).text = c.get('traceback',str(c.get('comparison','')))
        ET.SubElement(node, 'system-out').text = f"observed_status={c.get('observed_status',c['status'])} provenance={c.get('provenance','fresh')} mode={c.get('mode','unknown')} log={c.get('log','unknown')}"
    # An optional all-skipped suite still fails CI, rather than becoming green.
    if not summary['passed'] and int(suite.get('failures')) == 0:
        node = ET.SubElement(suite, 'testcase', name='required_coverage', classname='coverage', time='0')
        ET.SubElement(node, 'failure', type='NO_PASS', message='No executed passing cases or run not complete')
        suite.set('tests', str(len(result['cases'])+1))
        suite.set('failures', '1')
    ET.indent(suite)
    ET.ElementTree(suite).write(dest / 'junit.xml', encoding='utf-8', xml_declaration=True)
    return summary['exit_code']
