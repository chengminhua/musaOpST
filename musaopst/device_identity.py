# SPDX-License-Identifier: MulanPSL-2.0
"""Read-only physical identity for conservative report reuse (no SDK import)."""
import os
import re
import shutil
import subprocess


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
            visibility=os.environ.get('MTHREADS_VISIBLE_DEVICES') or os.environ.get('MUSA_VISIBLE_DEVICES')
            if visibility and all(x.isdigit() for x in visibility.split(',')):
                indices=[int(x) for x in visibility.split(',')]
                if device<len(indices):selected=next((r for r in records if r[0]==indices[device]),None)
            elif not visibility:selected=next((r for r in records if r[0]==device),None)
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
