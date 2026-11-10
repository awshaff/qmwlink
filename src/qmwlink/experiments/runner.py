"""Experiment runner: validation, results cache, run metadata. (FR-EXP-1..4, TSD §12)

Outputs go to ``<out_root>/<experiment>/<config_hash>/``. A run is a cache hit when that directory
holds a ``meta.json`` marked complete whose external-input hash is unchanged; ``force`` recomputes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from qmwlink.experiments import registry

logger = logging.getLogger(__name__)

PACKAGES = ("qmwlink", "numpy", "scipy", "mpmath", "matplotlib", "pydantic", "pyyaml")


class ConfigError(ValueError):
    """Invalid experiment config; the message names the file and the offending fields."""


@dataclass(frozen=True)
class RunResult:
    experiment: str
    out_dir: Path
    cache_hit: bool
    wall_time_s: float


def package_versions() -> dict[str, str]:
    """Installed versions of the pinned runtime packages (FR-EXP-3)."""
    out: dict[str, str] = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:  # pragma: no cover - only for a broken install
            out[name] = "unknown"
    return out


def _external_files(data: Any) -> list[str]:
    """Paths of external input files referenced by a config (``literature*`` keys)."""
    found: list[str] = []
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, str) and str(key).startswith("literature") and val.endswith(".yaml"):
                found.append(val)
            else:
                found += _external_files(val)
    elif isinstance(data, list):
        for item in data:
            found += _external_files(item)
    return sorted(set(found))


def inputs_hash(data: Any) -> str:
    """Hash of the external input files a config points to (missing file hashes as absent)."""
    h = hashlib.sha256()
    for name in _external_files(data):
        p = Path(name)
        h.update(name.encode())
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
    return h.hexdigest()[:12]


def load_and_validate(
    config_path: str | Path,
) -> tuple[registry.ExperimentSpec, dict[str, Any], str]:
    """Parse the YAML, pick the experiment from its ``experiment`` key, validate the schema."""
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML: {exc}") from exc
    if not isinstance(data, dict) or "experiment" not in data:
        raise ConfigError(f"{path}: top-level key 'experiment' is required")
    try:
        spec = registry.get(str(data["experiment"]))
    except KeyError as exc:
        raise ConfigError(f"{path}: {exc.args[0]}") from exc
    try:
        spec.model.model_validate(data)
    except ValidationError as exc:
        lines = [f"  {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
                 for e in exc.errors()]  # fmt: skip
        raise ConfigError(f"{path}: invalid {spec.name} config:\n" + "\n".join(lines)) from exc
    return spec, data, hashlib.sha256(text.encode()).hexdigest()[:12]


def _read_meta(path: Path) -> dict[str, Any]:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def run_config(
    config_path: str | Path, out_root: str | Path = "results", *, force: bool = False
) -> RunResult:
    """Run one experiment config with the cache. (FR-EXP-2,3,4)"""
    spec, data, cfg_hash = load_and_validate(config_path)
    out = Path(out_root) / spec.name / cfg_hash
    meta_path = out / "meta.json"
    ext = inputs_hash(data)
    old = _read_meta(meta_path)
    if not force and old.get("complete") is True and old.get("inputs_hash") == ext:
        logger.info("%s: cache hit at %s", spec.name, out)
        return RunResult(spec.name, out, True, float(old.get("wall_time_s", 0.0)))
    t0 = time.perf_counter()
    result_dir = Path(spec.run(config_path, out_root))
    wall = time.perf_counter() - t0
    meta = _read_meta(result_dir / "meta.json")
    meta.update(
        {
            "complete": True,
            "wall_time_s": round(wall, 3),
            "seed": data.get("seed"),
            "config_file": str(config_path),
            "inputs_hash": ext,
            "external_inputs": _external_files(data),
            "packages": package_versions(),
        }
    )
    (result_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    logger.info("%s: computed in %.1f s -> %s", spec.name, wall, result_dir)
    return RunResult(spec.name, result_dir, False, wall)
