# SPDX-License-Identifier: MulanPSL-2.0
"""Content-addressed native builds; no device initialization is required."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import platform

from musaopst.errors import BuildError, EnvironmentError


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sources(spec):
    sources = [Path(p).resolve() for p in spec.get("sources", [spec.get("source")]) if p]
    if not sources:
        raise BuildError("Native build requires source or sources")
    dependencies = set(sources) | {Path(p).resolve() for p in spec.get("headers", [])}
    queue = list(dependencies)
    while queue:
        path = queue.pop()
        if not path.is_file():
            raise BuildError(f"Missing native source/dependency: {path}")
        # Follow local quoted includes. SDK/system headers are covered by the
        # SDK fingerprint; extra include roots must be declared in headers.
        for name in re.findall(r'^\s*#\s*include\s*"([^"]+)"', path.read_text(errors="replace"), re.M):
            candidate = (path.parent / name).resolve()
            if candidate.is_file() and candidate not in dependencies:
                dependencies.add(candidate)
                queue.append(candidate)
    return sources, sorted(dependencies)


def build_native_descriptor(spec, workdir, arch=None, *, timeout_s=300, isolated=True):
    """Build explicit descriptor or verify a supplied executable.

    The caller must provide a target architecture for SDK-only/offline builds.
    Passing --offload-arch chooses GPU ISA; this is NOT host cross-compilation.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if spec.get("binary"):
        binary = Path(spec["binary"]).resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise BuildError(f"Native binary is not executable: {binary}")
        actual_sha = _digest(binary)
        if spec.get("artifact_sha256") and spec["artifact_sha256"] != actual_sha:
            raise BuildError("Prebuilt native artifact digest mismatch")
        return {"binary": str(binary), "artifact_sha256": actual_sha,
                "build_kind": "explicit_prebuilt", "cache_hit": False,
                "arch": spec.get("arch", arch or "unknown"), "host_compatibility": "checked by execution"}
    home = Path(os.environ.get("MUSA_HOME", "/usr/local/musa"))
    compiler = shutil.which(os.environ.get("MUSA_OPST_MCC", "mcc"))
    if not compiler and (home / "bin/mcc").is_file():
        compiler = str(home / "bin/mcc")
    if not compiler:
        raise EnvironmentError("mcc unavailable; set PATH/MUSA_HOME or MUSA_OPST_MCC")
    arch = spec.get("arch") or arch or os.environ.get("MUSA_OPST_ARCH")
    if not arch or not re.fullmatch(r"mp_[0-9]+", arch):
        raise EnvironmentError("Native build needs an explicit mp_<digits> arch; no GPU architecture is guessed")
    sources, dependencies = _sources(spec)
    flags = spec.get("compile_flags", ["-O2", "-std=c++17"])
    if not isinstance(flags, list) or any(not isinstance(f, str) for f in flags):
        raise BuildError("compile_flags must be an argv list of strings")
    version = subprocess.run([compiler, "--version"], text=True, capture_output=True, check=False, timeout=10)
    if version.returncode:
        raise BuildError("mcc --version failed: " + version.stderr)
    sdk_headers = [home / "include/musa_runtime.h", home / "include/musa_runtime_api.h"]
    fingerprint = {"format": 1, "sources": {str(p): _digest(p) for p in dependencies},
                   "compiler": str(Path(compiler).resolve()), "compiler_sha256": _digest(compiler),
                   "compiler_version": version.stdout.strip() or version.stderr.strip(),
                   "sdk_headers": {str(p): _digest(p) for p in sdk_headers if p.is_file()},
                   "musa_home": str(home.resolve()), "arch": arch, "flags": flags,
                   "link_flags": spec.get("link_flags", []), "host_machine": platform.machine(),
                   "host_libc": platform.libc_ver(),
                   "search_environment": {k:os.environ.get(k, "") for k in ("CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH")}}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    root = Path(os.environ.get("MUSA_OPST_CACHE", str(Path.cwd() / ".musaopst-cache"))).resolve()
    cache = root / ("native-" + key)
    root.mkdir(parents=True, exist_ok=True)
    # Advisory lock prevents concurrent writers even if a user starts two serial
    # suites. All artifacts are private to this cache; no external path is removed.
    import fcntl
    with (root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = cache / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
                binary = cache / "operator"
                if binary.is_file() and _digest(binary) == manifest["artifact_sha256"]:
                    return dict(manifest, cache_hit=True)
            except (ValueError, KeyError, OSError):
                pass
        with tempfile.TemporaryDirectory(prefix="build-", dir=root) as temp:
            binary = Path(temp) / "operator"
            command = [compiler, *flags, f"--offload-arch={arch}", *map(str, sources),
                       "-L" + str(home / "lib"), "-Wl,-rpath," + str(home / "lib"),
                       "-lmusart", *spec.get("link_flags", []), "-o", str(binary)]
            import signal
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, start_new_session=isolated)
            try:
                # In the device worker, stay in its group and let its parent
                # enforce the complete case timeout, including compiler children.
                stdout, stderr = process.communicate(timeout=timeout_s if isolated else None)
            except subprocess.TimeoutExpired as exc:
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: stdout, stderr = process.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    stdout, stderr = process.communicate()
                (workdir / "native_build.log").write_text(stdout + stderr)
                raise TimeoutError("Native build timed out; only its owned compiler process group was terminated") from exc
            log = stdout + stderr
            (workdir / "native_build.log").write_text(log)
            if process.returncode or not binary.is_file():
                raise BuildError(f"Native build failed ({process.returncode}): {log[-4000:]}")
            cache.mkdir(parents=True, exist_ok=True)
            os.replace(binary, cache / "operator")
            manifest = {"binary": str(cache / "operator"), "artifact_sha256": _digest(cache / "operator"),
                        "build_kind": "native_mcc", "cache_hit": False, "cache_key": key,
                        "cache_directory": str(cache), "fingerprint": fingerprint, "arch": arch,
                        "build_command": command, "host_cross_compilation": False}
            temporary_manifest = cache / "manifest.tmp"
            temporary_manifest.write_text(json.dumps(manifest, indent=2))
            os.replace(temporary_manifest, manifest_path)
            (cache / "build.log").write_text(log)
            return manifest


def load_cpp_artifact(manifest):
    """Import an explicitly named Python extension after validating its bytes."""
    import importlib.util
    import sys
    binary = Path(manifest['binary']).resolve()
    if not binary.is_file() or _digest(binary) != manifest['artifact_sha256']:
        raise BuildError('Prebuilt C++ extension is missing or has a different digest')
    name = manifest['module_name']
    if name in sys.modules:
        loaded = sys.modules[name]
        if Path(loaded.__file__).resolve() != binary:
            raise BuildError('Extension module name is already loaded from another path')
        return loaded
    module_spec = importlib.util.spec_from_file_location(name, str(binary))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _build_cpp(spec, workdir, arch):
    """Called only by an isolated builder or an already isolated GPU worker."""
    import platform
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    try:
        import torch
        import torch_musa
    except Exception as exc:
        raise EnvironmentError('C++ extension build requires compatible torch and torch_musa: ' + str(exc)) from exc
    if spec.get('binary'):
        binary = Path(spec['binary']).resolve()
        if not binary.is_file() or not spec.get('artifact_sha256') or _digest(binary) != spec['artifact_sha256']:
            raise BuildError('Prebuilt C++ extension digest mismatch')
        if not spec.get('module_name'):
            raise BuildError('Prebuilt C++ extension needs its compiled module_name')
        if arch and spec.get('arch') and spec['arch'] != arch:
            raise EnvironmentError('Prebuilt C++ target architecture differs from the selected device')
        previous = spec.get('fingerprint', {})
        expected = {'torch':str(torch.__version__), 'torch_musa':getattr(torch_musa,'__version__','unknown'),
                    'machine':platform.machine(), 'cxx11_abi':bool(torch._C._GLIBCXX_USE_CXX11_ABI)}
        for key, current in expected.items():
            if key in previous and previous[key] != current:
                raise EnvironmentError(f'Prebuilt C++ {key} differs from the current environment')
        return {'binary':str(binary),'artifact_sha256':spec['artifact_sha256'],
                'module_name':spec['module_name'],'build_kind':'explicit_prebuilt_cpp',
                'arch':spec.get('arch',arch or 'unknown'),'cache_hit':False}
    arch = spec.get('arch') or arch or os.environ.get('MUSA_OPST_ARCH')
    if not arch or not re.fullmatch(r'mp_[0-9]+',arch):
        raise EnvironmentError('C++ build needs an explicit mp_<digits> target architecture')
    os.environ['TORCH_MUSA_ARCH_LIST'] = arch.removeprefix('mp_')
    # Respect an explicit user budget; otherwise one compiler job is sufficient
    # for examples and does not take all CPUs of a shared host.
    os.environ.setdefault('MAX_JOBS','1')
    from torch_musa.utils.musa_extension import load
    sources, dependencies = _sources(spec)
    home = Path(os.environ.get('MUSA_HOME','/usr/local/musa'))
    compiler = shutil.which(os.environ.get('MUSA_OPST_MCC','mcc')) or str(home/'bin/mcc')
    if not Path(compiler).is_file():
        raise EnvironmentError('mcc is unavailable for C++ extension build')
    # musa_extension honors PYTORCH_MCC, whereas the public platform override is
    # MUSA_OPST_MCC. Bind the actual build command to the compiler we fingerprint.
    os.environ['PYTORCH_MCC'] = str(Path(compiler).resolve())
    version = subprocess.run([compiler,'--version'],capture_output=True,text=True,check=False)
    cxx = shutil.which(os.environ.get('CXX', 'c++'))
    cxx_version = subprocess.run([cxx,'--version'],capture_output=True,text=True,check=False).stdout if cxx else 'unknown'
    sdk_headers = [home/'include/musa_runtime.h', home/'include/musa_runtime_api.h', home/'include/driver_types.h']
    musa_flags = list(spec.get('extra_musa_cflags',['-O2','-std=c++17']))
    explicit_arch = '--offload-arch=' + arch
    for flag in musa_flags:
        if flag.startswith('--offload-arch=') and flag != explicit_arch:
            raise BuildError('C++ extra_musa_cflags conflicts with the declared target architecture')
    if explicit_arch not in musa_flags:
        musa_flags.append(explicit_arch)
    fingerprint = {'sources':{str(path):_digest(path) for path in dependencies},
                   'arch':arch,'compiler':str(Path(compiler).resolve()),'compiler_sha256':_digest(compiler),
                   'compiler_version':version.stdout.strip(),'torch':str(torch.__version__),
                   'torch_musa':getattr(torch_musa,'__version__','unknown'),'python':sys.version,
                   'machine':platform.machine(),'cxx11_abi':bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                   'cxx_compiler':str(Path(cxx).resolve()) if cxx else 'unknown',
                   'cxx_environment':os.environ.get('CXX','c++'),
                   'cxx_compiler_sha256':_digest(cxx) if cxx else 'unknown','cxx_version':cxx_version,
                   'sdk_headers':{str(path):_digest(path) for path in sdk_headers if path.is_file()},
                   'host_libc':platform.libc_ver(),
                   'search_environment':{k:os.environ.get(k,'') for k in ('CPATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH')},
                   'cflags':spec.get('extra_cflags',['-O2']),
                   'musa_cflags':musa_flags,
                   'ldflags':spec.get('extra_ldflags',[])}
    key = hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    name = 'opst_cpp_' + key[:20]
    root = Path(os.environ.get('MUSA_OPST_CACHE',str(Path.cwd()/'.musaopst-cache'))).resolve()
    cache = root / ('cpp-' + key)
    cache.mkdir(parents=True,exist_ok=True)
    workdir = Path(workdir)
    workdir.mkdir(parents=True,exist_ok=True)
    manifest_path = cache/'manifest.json'
    import fcntl
    with (root/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if manifest_path.is_file():
            try:
                old = json.loads(manifest_path.read_text())
                old_binary = Path(old['binary']).resolve()
                if old_binary.parent == cache and old_binary.is_file() and _digest(old_binary)==old['artifact_sha256']:
                    return dict(old,cache_hit=True)
                if old_binary.parent == cache and old_binary.is_file():
                    old_binary.unlink()  # Discard only this cache's corrupt generated library.
            except (ValueError,KeyError,OSError):
                pass
        # torch's FileBaton is a marker file, not an OS lock; killing our isolated
        # builder on timeout can leave it behind. We hold the outer process flock
        # here, so no live musaopst builder can own this private cache entry.
        # Remove only this entry's marker before allowing ninja to recover.
        torch_lock = cache/'lock'
        if torch_lock.exists():
            torch_lock.unlink()
        with (workdir/'cpp_build.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
            try:
                module = load(name=name,sources=list(map(str,sources)),build_directory=str(cache),
                              extra_cflags=fingerprint['cflags'],extra_musa_cflags=fingerprint['musa_cflags'],
                              extra_ldflags=fingerprint['ldflags'],with_musa=True,verbose=True)
            except Exception as exc:
                raise BuildError('C++ extension build failed: '+str(exc)) from exc
        binary = Path(module.__file__).resolve()
        manifest = {'binary':str(binary),'artifact_sha256':_digest(binary),'module_name':name,
                    'build_kind':'musa_cpp_extension','arch':arch,'cache_hit':False,'cache_key':key,
                    'cache_directory':str(cache),'fingerprint':fingerprint,'device_execution':False,
                    'runtime_device_probe_possible':True,'host_cross_compilation':False}
        temporary = cache/'manifest.tmp'
        temporary.write_text(json.dumps(manifest,indent=2))
        os.replace(temporary,manifest_path)
        return manifest


def build_cpp_descriptor(spec, workdir, arch=None, *, isolated=True, timeout_s=300):
    """Build a declared MUSA extension without launching any operator kernel."""
    if not isolated:
        return _build_cpp(spec,workdir,arch)
    import signal
    import sys
    workdir=Path(workdir).resolve()
    workdir.mkdir(parents=True,exist_ok=True)
    request=workdir/'cpp_build_request.json'
    result=workdir/'cpp_build_result.json'
    request.write_text(json.dumps({'spec':spec,'workdir':str(workdir),'arch':arch}))
    with (workdir/'cpp_builder.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-m','musaopst.backends.build_cache',str(request),str(result)],
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
            raise TimeoutError('Isolated C++ build timed out') from exc
    if not result.is_file():
        raise BuildError(f'C++ builder exited {code} without a structured result')
    value=json.loads(result.read_text())
    if code or value.get('status')!='BUILT':
        error=EnvironmentError if value.get('error_type')=='EnvironmentError' else BuildError
        raise error(value.get('message','C++ build failed'))
    return value['build']


if __name__=='__main__':
    import sys
    import traceback
    args=json.loads(Path(sys.argv[1]).read_text())
    try:
        build=_build_cpp(args['spec'],args['workdir'],args.get('arch'))
        Path(sys.argv[2]).write_text(json.dumps({'status':'BUILT','build':build},indent=2))
    except Exception as exc:
        Path(sys.argv[2]).write_text(json.dumps({'status':'BUILD_FAILED','error_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()},indent=2))
        raise SystemExit(1)
