#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Run real offline unittests and preserve text, JSON and JUnit evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def startTest(self, test):
        self.started = time.monotonic()
        self.current = {'name': test.id(), 'status': 'PASS', 'details': []}
        super().startTest(test)

    def addFailure(self, test, err):
        self.current['status'] = 'FAIL'
        self.current['details'].append(self._exc_info_to_string(err, test))
        super().addFailure(test, err)

    def addError(self, test, err):
        self.current['status'] = 'ERROR'
        self.current['details'].append(self._exc_info_to_string(err, test))
        super().addError(test, err)

    def addSkip(self, test, reason):
        self.current.update(status='SKIP', details=[reason])
        super().addSkip(test, reason)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self.current['status'] = 'FAIL' if issubclass(err[0], test.failureException) else 'ERROR'
            self.current['details'].append(str(subtest) + '\n' + self._exc_info_to_string(err, test))
        super().addSubTest(test, subtest, err)

    def addExpectedFailure(self, test, err):
        self.current.update(status='SKIP', details=['expected failure: ' + self._exc_info_to_string(err, test)])
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        self.current.update(status='FAIL', details=['unexpected success'])
        super().addUnexpectedSuccess(test)

    def stopTest(self, test):
        self.current['elapsed_s'] = time.monotonic() - self.started
        self.records.append(self.current)
        super().stopTest(test)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    if options.output.exists() and any(options.output.iterdir()):
        parser.error('output directory is nonempty; refusing to overwrite')
    options.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    timestamp = datetime.now(timezone.utc).isoformat()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
    with (options.output / 'unittest.log').open('w', encoding='utf-8') as log:
        runner = unittest.TextTestRunner(stream=Tee(sys.stderr, log), verbosity=2, resultclass=RecordedResult)
        result = runner.run(suite)
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for parent in ('musaopst', 'tests', 'tools') for path in sorted((ROOT / parent).rglob('*.py'))}
    report = {'schema_version': 1, 'mode': 'CPU_PLATFORM_UNIT_TESTS_ONLY', 'started_utc': timestamp,
              'python': sys.version, 'platform': platform.platform(), 'tests_run': result.testsRun,
              'passed': result.wasSuccessful() and result.testsRun > 0,
              'elapsed_s': time.monotonic() - start, 'cases': result.records, 'source_sha256': hashes,
              'reproduce': [sys.executable, str(Path(__file__).resolve()), '--output', str(options.output) + '-rerun']}
    (options.output / 'unittest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    root = ET.Element('testsuite', name='musaopst-offline-unit-tests', tests=str(result.testsRun),
                      failures=str(sum(c['status'] == 'FAIL' for c in result.records)),
                      errors=str(sum(c['status'] == 'ERROR' for c in result.records)),
                      skipped=str(sum(c['status'] == 'SKIP' for c in result.records)), time=str(report['elapsed_s']))
    properties = ET.SubElement(root, 'properties')
    ET.SubElement(properties, 'property', name='mode', value=report['mode'])
    for record in result.records:
        module, _, name = record['name'].rpartition('.')
        node = ET.SubElement(root, 'testcase', classname=module, name=name, time=str(record['elapsed_s']))
        tag = {'FAIL': 'failure', 'ERROR': 'error', 'SKIP': 'skipped'}.get(record['status'])
        if tag:
            ET.SubElement(node, tag).text = '\n'.join(record['details'])
    ET.indent(root)
    ET.ElementTree(root).write(options.output / 'junit.xml', encoding='utf-8', xml_declaration=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
