# SPDX-License-Identifier: MulanPSL-2.0
"""SDK-free artifact identity, verified snapshots and portable source bundles."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile

from .util import sha256, write_json


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def case_fingerprint(case, *, source=None, inputs=None, environment=None, entries=None):
    """Hash explicit execution identities; callers decide when evidence is complete."""
    normalized = {k: v for k, v in case.items() if k not in {"parameters_selected", "_config_dir"}}
    return content_digest({"case": normalized, "source": source, "inputs": inputs,
                           "environment": environment, "entries": entries})


def resume_identity(case, *, source, environment, entries, inputs=None):
    """Unknown tool, plugin or environment identities must never become cache hits."""
    reasons = []
    if not isinstance(source, dict) or source.get("content_sha256") in (None, "unknown", ""):
        reasons.append("tool content digest unavailable")
    if not isinstance(entries, (list, tuple)) or len(entries) < 2:
        reasons.append("implementation/reference identities unavailable")
    elif any(not isinstance(x, dict) or x.get("sha256") in (None, "unknown", "") for x in entries):
        reasons.append("implementation/reference source digest unavailable")
    def unknown(value):
        if isinstance(value, dict):
            return any(unknown(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(unknown(v) for v in value)
        return value is None or value == "unknown"
    if not environment or unknown(environment):
        reasons.append("execution environment identity unavailable")
    return {"fingerprint": case_fingerprint(case, source=source, inputs=inputs,
                                           environment=environment, entries=entries),
            "reusable": not reasons, "reasons": reasons}


def file_manifest(root, paths=None):
    root = Path(root).resolve()
    result = {}
    for path in sorted(paths if paths is not None else root.rglob("*")):
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            continue
        path.resolve().relative_to(root)
        result[str(path.relative_to(root))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return result


def verify_manifest(root, manifest):
    root = Path(root).resolve()
    errors = []
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(f"path escapes artifact root: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing artifact: {relative}")
        elif path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            errors.append(f"artifact digest mismatch: {relative}")
    return {"passed": not errors, "errors": errors, "checked": len(manifest)}


def verified_reuse(record, expected_fingerprint, root):
    if record.get("status") != "PASS" or record.get("comparison", {}).get("passed") is not True:
        return {"reusable": False, "reason": "only independently compared PASS records may be reused"}
    identity = record.get("resume_identity", {})
    if not identity.get("reusable") or identity.get("fingerprint") != expected_fingerprint:
        return {"reusable": False, "reason": "execution identity differs or is incomplete"}
    manifest = record.get("artifact_manifest")
    if not manifest:
        return {"reusable": False, "reason": "artifact integrity manifest is absent"}
    checked = verify_manifest(root, manifest)
    return {"reusable": checked["passed"], "reason": "; ".join(checked["errors"]) or "verified prior evidence"}


def result_path(path):
    path = Path(path).resolve()
    return path / "results.json" if path.is_dir() else path


def case_directory(results_path, record, index):
    root = result_path(results_path).parent
    relative = record.get("case_result", f"case-{index:04d}/result.json")
    directory = (root / relative).resolve().parent
    directory.relative_to(root)
    return directory


def _credential_path(path):
    name = Path(path).name.lower()
    return (name in {".netrc", ".npmrc", ".pypirc", ".git-credentials", "credentials", "credentials.json",
                     "token.json", "tokens.json", "secrets.json", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}
            or name == ".env" or name.startswith(".env.")
            or Path(name).suffix in {".pem", ".key", ".p12", ".pfx", ".keystore"}
            or any(part in {".ssh", ".aws", ".kube"} for part in Path(path).parts))


def bundle(results_path, output_file, *, max_bytes=512 * 1024 * 1024):
    """Archive evidence plus known source files; never claim to package GPU SDKs."""
    results_path = result_path(results_path)
    root = results_path.parent
    result = json.loads(results_path.read_text(encoding="utf-8"))
    output = Path(output_file).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    members = {}
    total = 0

    def add(path, name):
        nonlocal total
        path = Path(path)
        if _credential_path(path):
            raise ValueError(f"refusing common credential/key file in bundle: {path.name}")
        if not path.is_file() or path.is_symlink():
            return
        if name in members:
            if sha256(members[name]) != sha256(path):
                raise ValueError(f"conflicting bundled source path: {name}")
            return
        total += path.stat().st_size
        if total > max_bytes:
            raise ValueError(f"bundle exceeds max_bytes={max_bytes}; no archive written")
        members[name] = path

    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part.startswith(".build-cache") or part in {"__pycache__", ".git", ".venv", ".deps"}
               for part in rel.parts) or path.resolve() == output:
            continue
        add(path, "evidence/" + str(rel))
    package_root = Path(__file__).resolve().parent
    for path in sorted(package_root.rglob("*")):
        if "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".mu", ".cpp", ".h", ".hpp"}:
            add(path, "source/musaopst/" + str(path.relative_to(package_root)))
    unresolved = []
    entry_source_checks = []
    module_roots = {package_root.parent}
    for case in result.get("cases", []):
        entries = [(field, case.get(field, {})) for field in ("implementation_source", "reference_source")]
        entries += [("comparator_sources", entry) for entry in case.get("comparator_sources", [])]
        config = case.get("config", {})
        policies = [config.get("compare", {}), *[item.get("compare", {}) for item in config.get("outputs", [])]]
        custom_entries = {policy["custom"] for policy in policies if policy.get("custom")}
        recorded_custom = {entry.get("entry") for entry in case.get("comparator_sources", [])}
        for missing in sorted(custom_entries - recorded_custom):
            unresolved.append({"case": case.get("name"), "field": "comparator_sources", "entry": missing,
                               "reason": "declared custom comparator source identity was not recorded"})
        for field, entry in entries:
            filename, symbol = entry.get("file"), entry.get("entry", "")
            if filename in (None, "unknown") or not Path(filename).is_file():
                unresolved.append({"case": case.get("name"), "field": field, "reason": "source unavailable"})
                continue
            recorded_digest = entry.get("sha256")
            actual_digest = sha256(filename)
            matches = recorded_digest not in (None, "unknown", "") and recorded_digest == actual_digest
            entry_source_checks.append({"case": case.get("name"), "field": field, "entry": symbol,
                                        "recorded_sha256": recorded_digest or "unknown", "bundled_sha256": actual_digest,
                                        "matches_recorded_execution": matches})
            if not matches:
                unresolved.append({"case": case.get("name"), "field": field, "entry": symbol,
                                   "reason": "entry source differs from recorded execution or recorded digest is unavailable"})
            module = symbol.partition(":")[0]
            if module.startswith("musaopst."):
                continue
            if not module or not all(part.isidentifier() for part in module.split(".")):
                unresolved.append({"case": case.get("name"), "field": field, "reason": "module identity unavailable"})
                continue
            source = Path(filename)
            module_path = Path(*module.split("."))
            package_entry = source.name == "__init__.py"
            module_roots.add(source.resolve().parents[len(module.split(".")) if package_entry else len(module.split(".")) - 1])
            target = module_path / "__init__.py" if package_entry else module_path.with_suffix(".py")
            add(source, "source/" + str(target))
            # Explicit literal neighboring resources and local quoted includes
            # are source dependencies, unlike arbitrary files in the directory.
            resources = [(source.parent / name, target.parent / name)
                         for name in re.findall(r'\.with_name\([\'"]([^\'"]+)[\'"]\)', source.read_text(errors="replace"))
                         if Path(name).name == name and Path(name).suffix in {".mu", ".cpp", ".h", ".hpp"}]
            visited_resources = set()
            while resources:
                resource, archive_path = resources.pop()
                if resource in visited_resources or not resource.is_file():
                    continue
                visited_resources.add(resource)
                add(resource, "source/" + str(archive_path))
                for name in re.findall(r'^\s*#\s*include\s*"([^"]+)"', resource.read_text(errors="replace"), re.M):
                    relative = Path(name)
                    if relative.is_absolute() or ".." in relative.parts:
                        unresolved.append({"source": str(resource), "reason": "external include requires explicit source packaging", "include": name})
                    else:
                        resources.append((resource.parent / relative, archive_path.parent / relative))
            # Copy only known package markers, never the whole external directory.
            ancestor = source.parent.parent if package_entry else source.parent
            for depth in range(len(module.split(".")) - 1, 0, -1):
                add(ancestor / "__init__.py", "source/" + str(Path(*module.split(".")[:depth]) / "__init__.py"))
                ancestor = ancestor.parent
    declared_sources = []
    for case in result.get("cases", []):
        config = case.get("config", {})
        config_dir = config.get("_config_dir")
        if not config_dir and result.get("config_file") not in (None, "unknown"):
            config_dir = str(Path(result["config_file"]).resolve().parent)
        for value in config.get("sources", []):
            raw = Path(value)
            if not raw.is_absolute() and not config_dir:
                raise ValueError(f"cannot resolve declared source without original config directory: {value}")
            selected = raw if raw.is_absolute() else Path(config_dir) / raw
            if selected.is_symlink() or not selected.is_file():
                raise ValueError(f"declared source must be a regular file, not directory/symlink: {selected}")
            path = selected.resolve()
            if _credential_path(selected) or _credential_path(path):
                raise ValueError(f"refusing common credential/key file declared as source: {selected.name}")
            # Module roots determine Python import paths even when the config
            # lives deeper inside a package. Config-relative layout is fallback.
            roots = sorted(module_roots, key=lambda item: len(item.parts), reverse=True)
            if config_dir and Path(config_dir).resolve() not in roots:
                roots.append(Path(config_dir).resolve())
            relative = None
            for candidate in roots:
                try:
                    relative = path.relative_to(candidate)
                    break
                except ValueError:
                    continue
            relocation = relative is None
            if relocation:
                relative = Path("declared-external") / hashlib.sha256(str(path.parent).encode()).hexdigest()[:12] / path.name
                unresolved.append({"source": str(path), "reason": "explicit external source preserved; absolute import/resource paths require relocation"})
            archive_path = "source/" + str(relative)
            add(path, archive_path)
            declared_sources.append({"case": case.get("name"), "original": str(path), "archive": archive_path,
                                     "sha256": sha256(path), "relocation_required": relocation})
    source_checks = []
    for relative, expected in result.get("source", {}).get("files", {}).items():
        path = members.get("source/musaopst/" + relative)
        if path:
            actual = sha256(path)
            source_checks.append({"path": relative, "recorded_sha256": expected, "bundled_sha256": actual,
                                  "matches_recorded_execution": actual == expected})
    if any(not item["matches_recorded_execution"] for item in source_checks):
        unresolved.append({"reason": "bundled tool source differs from recorded execution; historical evidence is intact but exact code replay is not established"})
    manifest = {"bundle_version": 1, "kind": "source-and-evidence",
                "gpu_validation": "copied historical evidence, not a new execution",
                "environment": result.get("environment", {}), "source": result.get("source", {}),
                "files": {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                          for name, path in members.items()}, "unresolved_sources": unresolved,
                "declared_sources": declared_sources,
                "tool_source_verification": source_checks,
                "entry_source_verification": entry_source_checks,
                "requires": ["compatible Python and declared dependencies", "GPU driver/SDK and backend packages for device replay"],
                "limitations": ["external plugin transitive imports and SDKs are not automatically vendored",
                                "recreate replay config after extraction; absolute paths in historical reports are provenance only"],
                "replay_command": "PYTHONPATH=source python -m musaopst replay evidence/results.json --output replay"}
    with tempfile.TemporaryDirectory(prefix="musaopst-bundle-") as temporary:
        manifest_path = Path(temporary) / "bundle.json"
        write_json(manifest_path, manifest)
        temporary_archive = output.with_name(output.name + ".tmp")
        try:
            with tarfile.open(temporary_archive, "w:gz") as archive:
                archive.add(manifest_path, arcname="bundle.json", recursive=False)
                for name, path in sorted(members.items()):
                    archive.add(path, arcname=name, recursive=False)
            os.replace(temporary_archive, output)
        finally:
            temporary_archive.unlink(missing_ok=True)
    return {"path": str(output), "sha256": sha256(output), "files": len(members),
            "bytes": output.stat().st_size, "unresolved_sources": unresolved,
            "portable_environment": False}
