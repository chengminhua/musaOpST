# SPDX-License-Identifier: MulanPSL-2.0
"""Manage only complete native and C++ cache entries owned by musaopst."""
import json
import os
from pathlib import Path
import shutil
from .util import sha256


def cache_list(root=None):
    root=Path(root or os.environ.get('MUSA_OPST_CACHE',Path.cwd()/'.musaopst-cache')).resolve()
    entries=[]
    if root.is_dir():
        for path in sorted([*root.glob('native-*'),*root.glob('cpp-*')]):
            if path.is_symlink() or not path.is_dir():continue
            manifest=path/'manifest.json'
            try:
                data=json.loads(manifest.read_text());binary=Path(data.get('binary',path/'operator'))
                if data.get('build_kind') not in {'native_mcc','musa_cpp_extension'} or binary.resolve().parent!=path.resolve():continue
                entries.append({'path':str(path),'key':data.get('cache_key'),
                                'valid':binary.is_file() and not binary.is_symlink() and sha256(binary)==data.get('artifact_sha256')})
            except (OSError,ValueError):continue
    return {'root':str(root),'entries':entries,'scope':'native and musaopst C++ complete entries; other JIT caches are framework-managed'}


def cache_prune(root=None):
    result=cache_list(root)
    import fcntl
    for entry in result['entries']:
        key=entry['key']
        if not isinstance(key,str) or not key.isalnum():continue
        with (Path(result['root'])/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            shutil.rmtree(entry['path'])
    result['removed']=len(result['entries']);return result
