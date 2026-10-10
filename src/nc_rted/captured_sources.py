"""Verified relocation of an already-admitted source closure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class CapturedSourceError(ValueError):
    pass


def load_captured_sources(path: str | Path, admission: dict) -> dict[str, Path]:
    """Return original-name to immutable-copy paths after checking every byte."""
    descriptor=Path(path).resolve()
    try:
        document=json.loads(descriptor.read_text())
        mapping=document["files"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CapturedSourceError("captured source map is invalid") from error
    expected=admission.get("source_files")
    if document.get("schema") != "nc_rted_captured_source_map/v1" or not isinstance(mapping,dict) or not isinstance(expected,dict) or set(mapping) != set(expected):
        raise CapturedSourceError("captured source map differs from formal admission")
    root=descriptor.parent.resolve(); resolved={}
    for original, relative in mapping.items():
        if not isinstance(original,str) or not isinstance(relative,str):
            raise CapturedSourceError("captured source map entry is invalid")
        candidate=(root/relative).resolve()
        if Path(relative).is_absolute() or ".." in Path(relative).parts or root not in candidate.parents or not candidate.is_file() or candidate.is_symlink():
            raise CapturedSourceError("captured source map escapes immutable input root")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected[original]:
            raise CapturedSourceError("captured source differs from formal admission")
        resolved[original]=candidate
    return resolved


def load_captured_runtime(path: str | Path, runtime: dict) -> dict[str, str]:
    """Verify and return physical-path substitutions for bound runtime source files."""
    descriptor=Path(path).resolve(); root=descriptor.parent.resolve()
    try:
        document=json.loads(descriptor.read_text()); mapping=document["runtime"]
        inherited=runtime["inherited"]; stage2=runtime["stage2_cache"]
        required={"inherited_external_root":"inherited","inherited_source_manifest":"inherited-source-manifest.json","stage2_module":"stage2-module.py"}
        if document.get("schema") != "nc_rted_captured_source_map/v1" or mapping != required: raise ValueError
        manifest=root/mapping["inherited_source_manifest"]
        files=json.loads(manifest.read_text())["files"]
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != inherited["source_manifest_sha256"] or not isinstance(files,dict): raise ValueError
        external=root/mapping["inherited_external_root"]
        for relative, digest in files.items():
            candidate=(external/relative).resolve()
            if (not isinstance(relative,str) or not isinstance(digest,str) or Path(relative).is_absolute() or ".." in Path(relative).parts or
                    external not in candidate.parents or not candidate.is_file() or candidate.is_symlink() or hashlib.sha256(candidate.read_bytes()).hexdigest() != digest): raise ValueError
        stage=root/mapping["stage2_module"]
        if not stage.is_file() or stage.is_symlink() or hashlib.sha256(stage.read_bytes()).hexdigest() != stage2["module_sha256"]: raise ValueError
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CapturedSourceError("captured runtime source map is invalid") from error
    return {"external_root":str(external), "source_manifest":str(manifest), "stage2_module":str(stage)}


def validate_admitted_sources(files: dict, code_sha256: str, required_paths: set[Path], captured: dict[str, Path] | None = None) -> None:
    """Validate canonical admission identities against original or captured bytes."""
    digest=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    if digest != code_sha256:
        raise CapturedSourceError("source manifest differs from run identity")
    paths={name: Path(name) if captured is None else captured.get(name) for name in files}
    if any(path is None for path in paths.values()):
        raise CapturedSourceError("captured source map omits admitted source")
    resolved={path.resolve() for path in paths.values()}
    if not {path.resolve() for path in required_paths}.issubset(resolved):
        raise CapturedSourceError("formal admission omits NC-RTED source")
    for name, expected in files.items():
        path=paths[name]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise CapturedSourceError("accepted implementation changed")
