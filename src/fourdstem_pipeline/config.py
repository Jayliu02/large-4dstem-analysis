"""Workflow configuration loading with schema validation.

``load_workflow_config()`` validates the shape of every top-level section so
that YAML typos and missing required blocks are surfaced immediately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema definition -- each top-level key maps to a validator.
# ---------------------------------------------------------------------------

_SCHEMA: dict[str, dict[str, Any]] = {
    "pipeline": {
        "allowed": {"stages", "output_dir"},
        "types": {
            "stages": (list, tuple),
            "output_dir": (str, type(None)),
        },
        "defaults": {"stages": ["stage1", "stage2a", "stage2b"]},
    },
    "project": {
        "allowed": {"name", "output_dir"},
        "types": {"name": str, "output_dir": (str,)},
        "defaults": {"name": "large_4dstem_analysis", "output_dir": "outputs"},
    },
    "data": {
        "allowed": {
            "path",
            "directory",
            "pattern",
            "index",
            "backend",
            "lazy",
            "cache",
            "chunks",
            "scan_shape",
            "detector_shape",
            "dtype",
            "mib_header_bytes",
        },
        "types": {
            "path": (str, type(None)),
            "directory": (str, type(None)),
            "pattern": (str,),
            "index": (int,),
            "backend": (str, type(None)),
            "lazy": (bool,),
            "cache": (str, type(None)),
            "scan_shape": (list, tuple, type(None)),
            "detector_shape": (list, tuple, type(None)),
            "dtype": (str, type(None)),
            "mib_header_bytes": (int, type(None)),
        },
    },
    "preprocess": {
        "allowed": {"q_crop", "q_bin", "r_bin"},
        "types": {
            "q_crop": (list, tuple, type(None)),
            "q_bin": (int, float),
            "r_bin": (int, float),
        },
    },
    "geometry": {
        "allowed": {"center", "radial_bins"},
        "types": {"center": (list, tuple, type(None)), "radial_bins": (int, float)},
    },
    "virtual_images": {
        "allowed": {"masks"},
    },
    "phase_screening": {
        "allowed": {
            "method",
            "candidate_phases",
            "n_components",
            "n_clusters",
        },
        "types": {
            "method": (str,),
            "candidate_phases": (list, type(None)),
            "n_components": (int, float),
            "n_clusters": (int, float),
        },
        "defaults": {"method": "pca_nmf_cluster", "n_components": 3},
    },
    "orientation": {
        "allowed": {
            "enabled",
            "phase_candidates",
            "preview_binning",
            "roi",
            "confidence_threshold",
        },
        "types": {
            "enabled": (bool,),
            "phase_candidates": (list, type(None)),
            "preview_binning": (list, tuple),
            "roi": (list, tuple, type(None)),
            "confidence_threshold": (int, float),
        },
        "defaults": {"enabled": True, "preview_binning": [2, 2], "confidence_threshold": 0.05},
    },
    "roi_bragg": {
        "allowed": {
            "enabled",
            "roi",
            "thin_r",
            "bin_q",
            "mem",
            "corr_power",
            "sigma_cc",
            "edge_boundary",
            "min_relative_intensity",
            "min_peak_spacing",
            "subpixel",
            "max_num_peaks",
            "cuda",
        },
        "types": {
            "enabled": (bool,),
            "roi": (list, tuple),
            "thin_r": (int, float),
            "bin_q": (int, float),
            "mem": (str,),
            "corr_power": (int, float),
            "sigma_cc": (int, float),
            "edge_boundary": (int, float),
            "min_relative_intensity": (int, float),
            "min_peak_spacing": (int, float),
            "subpixel": (str,),
            "max_num_peaks": (int, float),
            "cuda": (bool,),
        },
        "defaults": {"enabled": False},
    },
    "sample_mask": {
        "allowed": {
            "enabled",
            "source",
            "method",
            "percentile",
            "fill_holes",
            "min_size",
            "background_label",
        },
        "types": {
            "enabled": (bool,),
            "source": (str,),
            "method": (str,),
            "percentile": (int, float),
            "fill_holes": (bool,),
            "min_size": (int, float),
            "background_label": (int, float),
        },
        "defaults": {
            "enabled": True,
            "source": "adf",
            "method": "percentile",
            "percentile": 15,
            "fill_holes": True,
            "min_size": 100,
            "background_label": -1,
        },
    },
    "stage2a": {
        "allowed": {
            "stage1_dir",
            "data_path",
            "output_dir",
            "roi_source",
            "max_rois",
            "thin_r",
            "bin_q",
            "mem",
            "scan_shape",
            "corr_power",
            "sigma_cc",
            "edge_boundary",
            "min_relative_intensity",
            "min_peak_spacing",
            "subpixel",
            "max_num_peaks",
            "cuda",
            "central_exclusion_radius",
            "save_roi_data",
        },
    },
    "stage2b": {
        "allowed": {
            "stage2_dir",
            "output_dir",
            "template_generation",
            "matching",
            "candidate_cifs",
        },
    },
    "stage2c": {
        "allowed": {
            "stage2_dir",
            "stage2b_dir",
            "output_dir",
            "enabled",
            "backend",
            "dry_run",
            "input",
            "results_npz",
            "candidates",
            "candidate_phases",
            "calibration",
            "polar",
            "orientation",
            "qc",
            "runtime",
        },
    },
    "consensus": {
        "allowed": {
            "output_dir",
            "stage2b_manifest",
            "stage2b_results",
            "stage2c_manifest",
            "stage2c_results",
        },
    },
}

_KNOWN_TOP_KEYS = set(_SCHEMA)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_workflow_config(path: str | Path = "configs/pipeline.yaml") -> dict[str, Any]:
    """Load and validate a YAML workflow configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    if not isinstance(raw, dict):
        raise ValueError(f"Workflow config must be a YAML mapping. Got {type(raw).__name__} from {config_path}.")

    _validate(raw, source=str(config_path))

    # Merge defaults for sections present in the config.
    for section, spec in _SCHEMA.items():
        if section in raw and "defaults" in spec:
            for key, value in spec["defaults"].items():
                raw[section].setdefault(key, value)

    return raw


def validate_workflow_config(cfg: dict[str, Any]) -> None:
    """Validate a config dict (e.g. one constructed in tests)."""
    _validate(cfg, source="<dict>")


# ---------------------------------------------------------------------------
# Data-path resolution (extracted from workflow.py)
# ---------------------------------------------------------------------------


def resolve_data_config(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``data.path`` from ``data.directory`` / ``pattern`` / ``index``.

    Returns a copy of *data_cfg* with ``path`` set.  Falls back to
    ``synthetic://demo`` when neither ``path`` nor ``directory`` is given.
    """
    resolved = dict(data_cfg)
    if resolved.get("path"):
        return resolved

    directory = resolved.get("directory")
    if not directory:
        resolved["path"] = "synthetic://demo"
        log.info("No data.path or data.directory specified — using synthetic://demo")
        return resolved

    pattern = resolved.get("pattern", "*.mib")
    candidates = sorted(Path(directory).glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No data files matched {Path(directory) / pattern}.")

    index = int(resolved.get("index", 0))
    if index < 0 or index >= len(candidates):
        raise IndexError(f"data.index {index} is out of range for {len(candidates)} matched files.")
    resolved["path"] = str(candidates[index])
    resolved["matched_files"] = [str(p) for p in candidates]
    log.info("Resolved data.path=%s from %d matched file(s)", resolved["path"], len(candidates))
    return resolved


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------


def _validate(raw: dict[str, Any], *, source: str) -> None:
    _check_unknown_top_level_keys(raw, source)
    for section, spec in _SCHEMA.items():
        if section not in raw:
            log.debug("Section %r not present in config, skipping validation.", section)
            continue
        value = raw[section]
        if not isinstance(value, dict):
            raise TypeError(f"[{source}] {section!r} must be a mapping, got {type(value).__name__}.")
        _check_unknown_keys(section, value, spec, source)
        _check_types(section, value, spec, source)


def _check_unknown_top_level_keys(raw: dict[str, Any], source: str) -> None:
    unknown = set(raw) - _KNOWN_TOP_KEYS
    if unknown:
        log.warning(
            "[%s] Unknown top-level config section(s): %s. These will be ignored.",
            source,
            ", ".join(sorted(unknown)),
        )


def _check_unknown_keys(section: str, value: dict[str, Any], spec: dict[str, Any], source: str) -> None:
    unknown = set(value) - spec.get("allowed", set())
    if unknown:
        log.warning(
            "[%s] %r section: unknown key(s): %s. These will be ignored.",
            source,
            section,
            ", ".join(sorted(unknown)),
        )


def _check_types(section: str, value: dict[str, Any], spec: dict[str, Any], source: str) -> None:
    type_map = spec.get("types", {})
    for key, val in value.items():
        expected = type_map.get(key)
        if expected is None:
            continue
        if isinstance(val, expected):
            continue
        type_names = " | ".join(t.__name__ for t in expected)
        log.warning(
            "[%s] %s.%s: expected %s, got %s (%r).",
            source,
            section,
            key,
            type_names,
            type(val).__name__,
            val,
        )
