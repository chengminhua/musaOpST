#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Bounded, read-only capability checks plus one local object compilation."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--arch');parser.add_argument('--host-target',default='aarch64-linux-gnu');args=parser.parse_args()
    root=args.output
    if root.exists() and any(root.iterdir()):raise ValueError('Refusing to overwrite evidence')
    root.mkdir(parents=True,exist_ok=True)
    result={'platform':platform.platform(),'python':sys.version,'packages':{},'tools':{}}
    for name in ('tensorflow','onnx'):
        try:spec=importlib.util.find_spec(name);result['packages'][name]={'found':bool(spec),'origin':spec.origin if spec else None}
        except Exception as exc:result['packages'][name]={'found':False,'error':repr(exc)}
    for name in ('mcc','msys','mcu','musa-sim','musa_simulator','musa-sanitizer','aarch64-linux-gnu-g++'):
        path=shutil.which(name)
        if not path:
            matches=[p for p in Path('/usr/local/musa').glob('**/'+name) if p.is_file()]
            path=str(matches[0]) if matches else None
        row={'path':path or 'unavailable'}
        if path:
            try:
                cp=subprocess.run([path,'--help'],capture_output=True,text=True,timeout=15)
                (root/(name+'.help.txt')).write_text(cp.stdout+cp.stderr);row.update(returncode=cp.returncode,help=name+'.help.txt')
            except Exception as exc:row['error']=repr(exc)
        result['tools'][name]=row
    mcc=result['tools']['mcc']['path']
    if mcc!='unavailable' and args.arch:
        source=root/'host_cross_probe.mu';source.write_text('#include <musa_runtime.h>\n__global__ void k(float* x) { if(threadIdx.x==0)x[0]=1; }\nint main(){return 0;}\n')
        command=[mcc,'--target='+args.host_target,'--offload-arch='+args.arch,'-c',str(source),'-o',str(root/'host_cross_probe.o')]
        try:
            cp=subprocess.run(command,capture_output=True,text=True,timeout=30)
            (root/'host_cross_compile.log').write_text(cp.stdout+cp.stderr)
            result['host_cross_compile']={'command':command,'returncode':cp.returncode,'status':'OBJECT_BUILT_NOT_EXECUTED' if cp.returncode==0 else 'BLOCKED'}
        except Exception as exc:result['host_cross_compile']={'status':'BLOCKED','error':repr(exc)}
    else:result['host_cross_compile']={'status':'BLOCKED','reason':'mcc unavailable or explicit --arch missing'}
    result['simulator']={'status':'UNAVAILABLE' if all(result['tools'][x]['path']=='unavailable' for x in ('musa-sim','musa_simulator')) else 'NEEDS_DOCUMENTED_INTERFACE','reason':'No simulator API is guessed from hardware scientific simulation libraries'}
    (root/'probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
