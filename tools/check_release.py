#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Audit the publishable source tree; never upload or rewrite project files."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT_FILES = {'README.md', 'LICENSE', 'NOTICE', 'pyproject.toml', 'MANIFEST.in',
              '.gitignore', 'AGENTS.md', 'CONTRIBUTING.md', 'CHANGELOG.md'}
ROOT_DIRS = {'musaopst', 'examples', 'docs', 'skills', 'tests', 'tools'}
LOCAL_DIRS = {'.git', '.venv', 'venv', '.cache', '.pytest_cache', '.musaopst-cache',
              '.deps', '.tmp', 'build', 'dist', 'runs', 'workspace', 'bundles'}
CODE_SUFFIXES = {'.py', '.sh', '.cpp', '.mu', '.h', '.hpp'}
TEXT_SUFFIXES = CODE_SUFFIXES | {'.md', '.json', '.toml', '.yaml', '.yml', '.txt', '.in'}
PRIVATE_ADDRESS = re.compile(r'(?<![\w.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?![\w.])')
LOCAL_HOME = re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/')
SECRET = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b')


def audit(root):
    root = Path(root).resolve()
    errors, excluded, files = [], [], {}
    def error(path, reason):
        errors.append({'path': str(path), 'reason': reason})
    for name in sorted(ROOT_FILES | ROOT_DIRS):
        if not (root / name).exists():
            error(name, 'required source item missing')
    for item in sorted(root.iterdir()):
        if item.name in ROOT_FILES or item.name in ROOT_DIRS:
            pass
        elif item.name in LOCAL_DIRS or item.name.startswith('.venv-') or item.name.endswith('.egg-info'):
            excluded.append(item.name)
            continue
        else:
            error(item.name, 'unexpected root item; review before distribution')
            continue
        if item.is_symlink():
            error(item.name, 'symlink is not a portable release source')
            continue
        candidates = item.rglob('*') if item.is_dir() else [item]
        for p in sorted(candidates):
            rel = p.relative_to(root)
            if '__pycache__' in rel.parts or p.suffix in {'.pyc', '.pyo'}:
                excluded.append(str(rel))
                continue
            if p.is_symlink():
                error(rel, 'symlink is not a portable release source')
                continue
            if not p.is_file():
                continue
            if p.name.startswith('.env') or p.suffix in {'.pem', '.key', '.so', '.whl', '.log', '.tar', '.gz'}:
                error(rel, 'credential-like or generated file inside source tree')
            data = p.read_bytes()
            files[str(rel)] = hashlib.sha256(data).hexdigest()
            if p.suffix not in TEXT_SUFFIXES and p.name not in ROOT_FILES:
                if p.suffix not in {'.npy', '.npz', '.bin'} or rel.parts[0] != 'examples':
                    error(rel, 'unclassified source file')
                continue
            try:
                text = data.decode('utf-8')
            except UnicodeDecodeError:
                error(rel, 'text file is not UTF-8')
                continue
            if PRIVATE_ADDRESS.search(text) or LOCAL_HOME.search(text) or SECRET.search(text):
                error(rel, 'private environment detail or credential pattern; value withheld')
            if p.suffix in CODE_SUFFIXES and 'SPDX-License-Identifier: MulanPSL-2.0' not in text[:300]:
                error(rel, 'first-party source license identifier missing')
            if p.suffix == '.md':
                for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
                    path = target.split('#')[0]
                    if path and not re.match(r'[a-z]+://', path) and not (p.parent / path).exists():
                        error(rel, 'missing local Markdown target: ' + path)
    project = (root / 'pyproject.toml').read_text() if (root / 'pyproject.toml').is_file() else ''
    for marker in ('name = "musaopst"', 'license = "MulanPSL-2.0"',
                   'license-files = ["LICENSE", "NOTICE"]', 'musaopst = "musaopst.cli:main"'):
        if marker not in project:
            error('pyproject.toml', 'release metadata missing: ' + marker)
    version_match = re.search(r'^version = "([^"]+)"', project, re.M)
    version = version_match.group(1) if version_match else None
    init = root / 'musaopst/__init__.py'
    if init.is_file():
        module_version = None
        for node in ast.parse(init.read_text()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__version__' for t in node.targets):
                module_version = ast.literal_eval(node.value)
        if not version or version != module_version:
            error('musaopst/__init__.py', 'version differs from distribution metadata')
    return {'passed': not errors, 'version': version, 'checked_files': len(files),
            'files_sha256': files, 'excluded_local_items': excluded, 'errors': errors,
            'scope': 'Source checks only; package installation, tests, and GPU validation are separate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, help='New JSON evidence path, preferably outside the repository')
    args = parser.parse_args()
    result = audit(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files_sha256'}, ensure_ascii=False, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
