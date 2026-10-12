# SPDX-License-Identifier: MulanPSL-2.0
"""Read-only capability probe, always in a newly executed interpreter."""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from .util import write_json


def probe(device=0):
    data = {'schema_version':1,'python':sys.version,'platform':platform.platform(),
            'host':platform.node(),'device_requested':device,'gpu_available':False,'packages':{},
            'capabilities':{},'errors':[],'note':'doctor 仅检查能力；不代表算子精度验收通过'}
    for package in ['numpy','torch','torch_musa','triton','tilelang','tilelang-musa','apache-tvm-ffi','z3-solver']:
        try:
            data['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            data['packages'][package] = 'unknown'
    for binary in ['mcc','mthreads-gmi','ninja','c++']:
        path = shutil.which(binary)
        data['capabilities'][binary] = path or 'unavailable'
        if path and binary in ['mcc','mthreads-gmi']:
            try:
                proc = subprocess.run([path] + (['--version'] if binary=='mcc' else []),capture_output=True,text=True,timeout=10)
                data[binary+'_output'] = proc.stdout+proc.stderr
            except Exception as exc:
                data[binary+'_output'] = repr(exc)
    try:
        import torch
        import torch_musa
        data['gpu_available'] = bool(torch.musa.is_available())
        data['device_count'] = torch.musa.device_count()
        if not data['gpu_available'] or device >= data['device_count']:
            raise RuntimeError('请求的 MUSA 设备不可用；不回退 CPU')
        torch.musa.set_device(device)
        prop = torch.musa.get_device_properties(device)
        data['device'] = {k:getattr(prop,k,'unknown') for k in ['name','major','minor','total_memory','multi_processor_count','warp_size']}
        data['capabilities']['torch_musa'] = True
        data['capabilities']['event_api'] = hasattr(torch.musa,'Event')
        data['capabilities']['memory_allocated'] = hasattr(torch.musa,'memory_allocated')
        data['capabilities']['memory_reserved'] = hasattr(torch.musa,'memory_reserved')
        data['capabilities']['musa_cpp_extension'] = importlib.util.find_spec('torch_musa.utils.musa_extension') is not None
        a,b=torch.musa.Event(enable_timing=True),torch.musa.Event(enable_timing=True)
        a.record(); b.record(); b.synchronize()
        data['event_probe_ms'] = a.elapsed_time(b)
        for name in ['triton','tilelang']:
            try:
                mod = __import__(name)
                data['capabilities'][name+'_import'] = True
                if name=='triton':
                    data['triton_target'] = str(mod.runtime.driver.active.get_current_target())
            except Exception as exc:
                data['capabilities'][name+'_import'] = False
                data['errors'].append(f'{name}: {exc!r}')
    except Exception as exc:
        data['errors'].append(repr(exc))
    data['visibility'] = {k:os.environ.get(k,'unknown') for k in ['MUSA_VISIBLE_DEVICES','MTHREADS_VISIBLE_DEVICES']}
    return data


if __name__=='__main__':
    write_json(sys.argv[1],probe(int(sys.argv[2])))
