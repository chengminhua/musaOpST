# SPDX-License-Identifier: MulanPSL-2.0
"""Read-only physical identity for conservative report reuse (no SDK import)."""
import os
import re
import shutil
import subprocess


VISIBILITY_VARIABLES = ('MTHREADS_VISIBLE_DEVICES', 'MUSA_VISIBLE_DEVICES')


def visibility_spec(environ=None):
    """Single interpretation of container visibility for identity and --device.

    `indices` is a physical-id list only for a pure numeric visibility value;
    UUID lists stay as raw text because the host-side index is unknown.
    """
    env = os.environ if environ is None else environ
    for name in VISIBILITY_VARIABLES:
        raw = env.get(name)
        if raw is None:
            continue
        text = str(raw).strip()
        # Images commonly ship MTHREADS_VISIBLE_DEVICES=all as "no restriction";
        # it must not shadow an explicit MUSA_VISIBLE_DEVICES value.
        if not text or text.lower() == 'all':
            continue
        parts = [part.strip() for part in text.split(',')]
        numeric = bool(parts) and all(part.isdigit() for part in parts)
        return {'variable': name, 'raw': text,
                'indices': [int(part) for part in parts] if numeric else None,
                'numeric': numeric}
    return {'variable': None, 'raw': None, 'indices': None, 'numeric': False}


def probe(device=0):
    binary=shutil.which('mthreads-gmi')
    if not binary:return {'verified':False,'reason':'mthreads-gmi unavailable'}
    # Container visibility may renumber devices. Query all and accept an
    # unambiguous single device, or map through an explicit numeric visibility.
    try:
        process=subprocess.run([binary,'-q'],capture_output=True,text=True,timeout=10)
        if process.returncode:return {'verified':False,'reason':'mthreads-gmi -q failed'}
        text=process.stdout
        records=[]
        for block in re.split(r'\n(?=GPU\d+\s)',text)[1:]:
            index=re.match(r'GPU(\d+)',block)
            uuid=re.search(r'GPU UUID\s*:\s*(\S+)',block)
            if index and uuid:records.append((int(index[1]),uuid[1]))
        selected=None
        if len(records)==1 and device==0:selected=records[0]
        else:
            visibility=visibility_spec()
            if visibility['numeric']:
                indices=visibility['indices']
                if device<len(indices):selected=next((r for r in records if r[0]==indices[device]),None)
            elif not visibility['raw']:selected=next((r for r in records if r[0]==device),None)
        if not selected:return {'verified':False,'reason':'ambiguous device visibility'}
        driver=re.search(r'Driver Version\s*:\s*(\S+)',text)
        return {'verified':True,'uuid':selected[1],'driver':driver[1] if driver else 'unknown',
                'method':'mthreads-gmi -q with explicit visibility mapping'}
    except (OSError,subprocess.TimeoutExpired) as exc:return {'verified':False,'reason':str(exc)}


def toolchain_probe():
    """Content identity for compiler/runtime/header changes, without loading MUSA."""
    import hashlib
    from pathlib import Path
    roots=[]
    for value in (os.environ.get('MUSA_HOME'),os.environ.get('MUSA_PATH')):
        if value:roots.append(Path(value))
    compiler=shutil.which('mcc')
    if compiler:roots.append(Path(compiler).resolve().parent.parent)
    roots.append(Path('/usr/local/musa'))
    root=next((p for p in roots if (p/'include/musa_runtime.h').is_file()),None)
    if root is None:return {'verified':False,'reason':'SDK root and musa_runtime.h not found'}
    paths=[root/'bin/mcc',root/'include/musa_runtime.h',root/'include/musa_runtime_api.h']
    runtime=next((p for folder in ('lib','lib64') for p in sorted((root/folder).glob('libmusart.so*')) if p.is_file()),None)
    if runtime:paths.append(runtime)
    hashes={}
    try:
        for path in paths:
            if path.is_file():
                digest=hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
                hashes[str(path.relative_to(root))]=digest.hexdigest()
    except OSError as exc:return {'verified':False,'reason':str(exc)}
    complete=(root/'bin/mcc').is_file() and runtime is not None
    return {'verified':complete,'files':hashes,'scope':'MCC binary, public runtime headers, libmusart; package versions and declared sources fingerprinted separately',
            'reason':'verified declared toolchain components' if complete else 'compiler or runtime library unavailable'}
