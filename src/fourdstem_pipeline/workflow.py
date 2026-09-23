"""Single non-visual 4D-STEM analysis workflow.

Orchestrates data loading, preprocessing, virtual imaging, radial
fingerprinting, unsupervised phase screening, orientation preview, and
optional ROI Bragg detection.  Each stage is logged with timing, and
individual stage failures are captured rather than aborting the entire run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_workflow_config, resolve_data_config
from .contracts import DataContract
from .dataset import DatasetHandle
from .diagnostics import run_stage1_diagnostics
from .export import save_annotated_label_png, save_label_png, save_png, save_profile_png, save_report, save_summary
from .fingerprints import FingerprintResult, compute_radial_fingerprints
from .loaders import load_dataset
from .logging import configure_pipeline_logging, get_logger, log_stage_end, log_stage_start
from .masks import build_annular_masks
from .orientation import OrientationResult, run_orientation_preview
from .phase import PhaseScreeningResult, screen_phases
from .preprocess import apply_preprocess
from .provenance import collect_provenance, save_provenance
from .qc import QCResult, run_qc_checks, save_qc_summary
from .sample_mask import apply_mask_to_labels, clean_mask, make_sample_mask, save_sample_mask_outputs
from .virtual import VirtualImageResult, compute_virtual_images

log = get_logger(__name__)


@dataclass(slots=True)
class WorkflowResult:
    output_dir: Path
    dataset: DatasetHandle | None
    virtual_images: VirtualImageResult | None = None
    fingerprints: FingerprintResult | None = None
    phase_screening: PhaseScreeningResult | None = None
    orientation: OrientationResult | None = None
    roi_bragg: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    summary_path: Path | None = None
    report_path: Path | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)


def run_workflow(
    config: str | Path | dict[str, Any] = "configs/pipeline.yaml",
    *,
    log_level: str = "INFO",
) -> WorkflowResult:
    """Run the single non-visual 4D-STEM analysis workflow.

    Parameters
    ----------
    config:
        Path to a YAML config file, or an already-parsed dict.
    log_level:
        One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``.  Also
        controllable via the ``FOURDSTEM_LOG_LEVEL`` environment variable.
    """
    configure_pipeline_logging(level=log_level)
    t0 = time.perf_counter()
    start_time = datetime.now(timezone.utc)

    cfg = load_workflow_config(config) if isinstance(config, (str, Path)) else dict(config)
    _validate_dict_config(cfg)

    project_cfg = cfg.get("project", {})
    output_dir = Path(project_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("4D-STEM analysis workflow starting")
    log.info("  project: %s", project_cfg.get("name", "unnamed"))
    log.info("  output:  %s", output_dir)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Stage 0: Load & preprocess
    # ------------------------------------------------------------------
    data_cfg = resolve_data_config(cfg.get("data", {}))
    block_shape = _navigation_block_shape(cfg, data_cfg)

    errors: list[dict[str, Any]] = []

    dataset = _run_stage(
        "load",
        lambda: load_dataset(
            data_cfg.get("path", "synthetic://demo"),
            lazy=bool(data_cfg.get("lazy", True)),
            cache=data_cfg.get("cache"),
            chunks=data_cfg.get("chunks"),
            backend=data_cfg.get("backend"),
            scan_shape=data_cfg.get("scan_shape"),
            detector_shape=data_cfg.get("detector_shape"),
            dtype=data_cfg.get("dtype"),
            mib_header_bytes=data_cfg.get("mib_header_bytes"),
        ),
        errors=errors,
    )

    if dataset is None:
        if not any(err.get("stage") == "load" for err in errors):
            errors.append({"stage": "load", "error": "Data loading failed.", "elapsed_s": 0})
        return _make_error_result(output_dir, dataset, errors)

    log.info("  data backend:  %s", dataset.source_backend)
    log.info("  navigation:    %s", dataset.navigation_shape)
    log.info("  signal:        %s", dataset.signal_shape)
    log.info("  block shape:   %s", block_shape)

    dataset = _run_stage(
        "preprocess",
        lambda: apply_preprocess(dataset, **cfg.get("preprocess", {})),
        errors=errors,
    )

    if dataset is None:
        log.error("Preprocessing failed — cannot continue. Aborting workflow.")
        if not any(err.get("stage") == "preprocess" for err in errors):
            errors.append({"stage": "preprocess", "error": "Preprocessing failed.", "elapsed_s": 0})
        return _make_error_result(output_dir, dataset, errors)

    # ------------------------------------------------------------------
    # Set up output directories
    # ------------------------------------------------------------------
    geometry = cfg.get("geometry", {})
    masks = build_annular_masks(
        dataset.signal_shape,
        cfg.get("virtual_images", {}).get("masks", {}),
        center=geometry.get("center"),
    )

    preprocess_dir = output_dir / "00_preprocess"
    virtual_dir = output_dir / "virtual"
    fingerprints_dir = output_dir / "fingerprints"
    classes_dir = output_dir / "fingerprint_classes"
    orientation_dir = output_dir / "orientation"
    png_dir = output_dir / "png"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: Virtual images
    # ------------------------------------------------------------------
    virtual = _run_stage(
        "virtual images",
        lambda: compute_virtual_images(dataset, masks, output_dir=virtual_dir, block_shape=block_shape),
        errors=errors,
    )

    # ------------------------------------------------------------------
    # Sample mask (from virtual ADF / HAADF)
    # ------------------------------------------------------------------
    sample_mask_cfg = cfg.get("sample_mask", {})
    sample_mask: np.ndarray | None = None
    if bool(sample_mask_cfg.get("enabled", False)) and virtual is not None:
        source_name = str(sample_mask_cfg.get("source", "adf"))
        source_image = virtual.images.get(source_name)
        if source_image is not None:
            sample_mask = clean_mask(
                make_sample_mask(
                    source_image,
                    percentile=float(sample_mask_cfg.get("percentile", 15)),
                ),
                min_size=int(sample_mask_cfg.get("min_size", 100)),
                fill_holes=bool(sample_mask_cfg.get("fill_holes", True)),
            )
            mask_outputs = save_sample_mask_outputs(
                preprocess_dir, png_dir, sample_mask, source_image,
            )
            sample_frac = float(sample_mask.mean())
            log.info(
                "Sample mask from %r: %.1f%% sample, %.1f%% background",
                source_name,
                100 * sample_frac,
                100 * (1 - sample_frac),
            )
        else:
            log.warning(
                "Sample mask enabled but source %r not found in virtual images. "
                "Available: %s. Skipping mask.",
                source_name,
                ", ".join(sorted(virtual.images.keys())),
            )
    elif bool(sample_mask_cfg.get("enabled", False)):
        log.warning("Sample mask enabled but virtual images failed — skipping mask.")

    # ------------------------------------------------------------------
    # Stage 2: Radial fingerprints
    # ------------------------------------------------------------------
    fingerprints = _run_stage(
        "radial fingerprints",
        lambda: compute_radial_fingerprints(
            dataset,
            geometry,
            int(geometry.get("radial_bins", 48)),
            output_dir=fingerprints_dir,
            block_shape=block_shape,
        ),
        errors=errors,
    )

    # Apply sample mask to fingerprint profiles before clustering so that
    # vacuum / background positions do not influence PCA / NMF / KMeans.
    if sample_mask is not None and fingerprints is not None:
        fingerprints.profiles[~sample_mask] = 0

    # ------------------------------------------------------------------
    # Stage 3: Phase screening
    # ------------------------------------------------------------------
    phase_cfg = cfg.get("phase_screening", {})
    phase = _run_stage(
        "phase screening",
        lambda: screen_phases(
            fingerprints,
            method=phase_cfg.get("method", "pca_nmf_cluster"),
            candidate_phases=phase_cfg.get("candidate_phases"),
            n_components=int(phase_cfg.get("n_components", 3)),
            n_clusters=int(phase_cfg.get("n_clusters", phase_cfg.get("n_components", 3))),
            output_dir=classes_dir,
        ),
        errors=errors,
    )

    # Set labels outside the sample mask to background_label so that
    # downstream diagnostics, ROI candidates, and reports clearly
    # distinguish sample from vacuum / excluded regions.
    if sample_mask is not None and phase is not None:
        background_label = int(sample_mask_cfg.get("background_label", -1))
        apply_mask_to_labels(phase.labels, sample_mask, background_label)

    # ------------------------------------------------------------------
    # Stage 4: Orientation preview
    # ------------------------------------------------------------------
    orientation_cfg = cfg.get("orientation", {})
    orientation_roi_invalid = False
    roi_raw = orientation_cfg.get("roi")

    if bool(orientation_cfg.get("enabled", True)) and roi_raw is not None:
        y0_raw, y1_raw, x0_raw, x1_raw = [int(v) for v in roi_raw]
        # Pre-validate: reject ROI where end <= start before clamping
        if y1_raw <= y0_raw or x1_raw <= x0_raw:
            orientation_roi_invalid = True
            log.warning(
                "Orientation ROI %s has zero area (y1=%d <= y0=%d or x1=%d <= x0=%d) "
                "-- skipping orientation preview.",
                roi_raw, y1_raw, y0_raw, x1_raw, x0_raw,
            )

    if not bool(orientation_cfg.get("enabled", True)) or orientation_roi_invalid:
        orientation = None
    else:
        orientation = _run_stage(
            "orientation preview",
            lambda: run_orientation_preview(
                dataset,
                phase_candidates=orientation_cfg.get("phase_candidates"),
                binning=orientation_cfg.get("preview_binning", (2, 2)),
                roi=roi_raw,
                confidence_threshold=float(orientation_cfg.get("confidence_threshold", 0.05)),
                output_dir=orientation_dir,
                block_shape=block_shape,
            ),
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Stage 5: ROI Bragg (optional)
    # ------------------------------------------------------------------
    roi_bragg = None
    roi_cfg = cfg.get("roi_bragg", {})
    if bool(roi_cfg.get("enabled", False)):
        roi_bragg = _run_stage(
            "ROI Bragg detection",
            lambda: _run_roi_bragg(data_cfg, roi_cfg, output_dir / "roi_bragg"),
            errors=errors,
        )

    # ------------------------------------------------------------------
    # PNG exports
    # ------------------------------------------------------------------
    png_outputs: dict[str, Path] = {}
    if virtual is not None and fingerprints is not None and phase is not None:
        png_outputs = _save_png_outputs(png_dir, virtual, fingerprints, phase, orientation)
    else:
        log.warning("Skipping PNG exports because one or more upstream stages failed.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    diagnostics: dict[str, Any] = {}
    if virtual is not None and fingerprints is not None and phase is not None:
        diagnostics = _run_stage(
            "stage-1 diagnostics",
            lambda: run_stage1_diagnostics(
                dataset,
                fingerprints,
                phase,
                virtual,
                orientation,
                output_dir=output_dir,
                png_dir=png_dir,
                block_shape=block_shape,
                confidence_threshold=float(orientation_cfg.get("confidence_threshold", 0.05)),
                radial_center=geometry.get("center"),
            ),
            errors=errors,
        ) or {}

        for key, filename in {
            "cluster_mean_radial_profiles": "cluster_mean_radial_profiles.png",
            "cluster_virtual_image_statistics": "cluster_virtual_image_statistics.png",
            "roi_candidates_overlay": "roi_candidates_overlay.png",
            "mean_dp_with_center_marker": "mean_dp_with_center_marker.png",
            "orientation_score_masked": "orientation_score_masked.png",
            "k_sweep_metrics": "k_sweep_metrics.png",
            "cluster_cleaned_labels": "cluster_cleaned_labels.png",
            "cluster_vs_orientation_heatmap": "cluster_vs_orientation_heatmap.png",
            "ring_2_over_ring_1": "ring_2_over_ring_1.png",
            "ring_3_over_ring_1": "ring_3_over_ring_1.png",
            "ring_3_over_ring_2": "ring_3_over_ring_2.png",
        }.items():
            path = png_dir / filename
            if path.exists():
                png_outputs[key] = path
    else:
        log.warning("Skipping diagnostics because virtual, fingerprints, or phase screening failed.")

    # Sample mask PNGs (may exist even when diagnostics are skipped)
    for key, filename in {
        "sample_mask": "sample_mask.png",
        "sample_mask_overlay_adf": "sample_mask_overlay_adf.png",
    }.items():
        path = png_dir / filename
        if path.exists():
            png_outputs[key] = path

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    end_time = datetime.now(timezone.utc)
    input_path = data_cfg.get("path") if data_cfg.get("path") and data_cfg.get("path") not in ("synthetic://demo",) else None
    config_for_prov = config if isinstance(config, (str, Path)) else cfg
    random_seed = cfg.get("random_seed") if isinstance(cfg.get("random_seed"), int) else None
    provenance = collect_provenance(
        config=config_for_prov,
        input_path=input_path,
        run_name=str(project_cfg.get("name", "unnamed")),
        start_time=start_time,
        end_time=end_time,
        random_seed=random_seed,
    )
    save_provenance(output_dir, provenance)

    # ------------------------------------------------------------------
    # QC
    # ------------------------------------------------------------------
    qc_result = run_qc_checks(
        dataset=dataset,
        virtual=virtual,
        fingerprints=fingerprints,
        phase=phase,
        orientation=orientation,
        diagnostics=diagnostics,
        errors=errors if errors else None,
        confidence_threshold=float(orientation_cfg.get("confidence_threshold", 0.05)),
        sample_mask=sample_mask,
        orientation_roi_invalid=orientation_roi_invalid,
        orientation_enabled=bool(orientation_cfg.get("enabled", True)),
        radial_center=geometry.get("center"),
    )
    save_qc_summary(output_dir, qc_result)

    # ------------------------------------------------------------------
    # Summary & report
    # ------------------------------------------------------------------
    data_contract = DataContract(
        array_shape=dataset.shape if dataset else None,
        nav_shape=dataset.navigation_shape if dataset else None,
        sig_shape=dataset.signal_shape if dataset else None,
    )

    summary = {
        "project": project_cfg,
        "data_config": data_cfg,
        "preprocess": cfg.get("preprocess", {}),
        "dataset": dataset.describe(),
        "data_contract": data_contract.to_dict(),
        "provenance": provenance,
        "qc": qc_result.to_dict(),
        "dependencies": {
            "source_backend": dataset.metadata.get("source_backend", "unknown") if dataset else "none",
            "pyxem_available": dataset.metadata.get("pyxem_available", False) if dataset else False,
            "pyxem_signal_type": dataset.metadata.get("pyxem_signal_type") if dataset else None,
            "py4DSTEM_used": roi_bragg is not None,
        },
        "outputs": {
            "virtual": str(virtual_dir),
            "fingerprints": str(fingerprints_dir),
            "fingerprint_classes": str(classes_dir),
            "orientation": str(orientation_dir),
            "cluster_diagnostics": diagnostics.get("cluster_diagnostics"),
            "roi_candidates": diagnostics.get("roi_candidates"),
            "roi_bragg": roi_bragg,
            "png": {name: str(path) for name, path in png_outputs.items()},
            "diagnostics": diagnostics,
        },
        "shapes": {
            "virtual_images": {name: image.shape for name, image in virtual.images.items()} if virtual else {},
            "radial_fingerprints": fingerprints.profiles.shape if fingerprints else None,
            "fingerprint_class_labels": phase.labels.shape if phase else None,
            "orientation_index": orientation.orientation_index.shape if orientation else None,
        },
        "errors": errors if errors else None,
        "sample_mask": {
            "enabled": bool(sample_mask_cfg.get("enabled", False)),
            "source": str(sample_mask_cfg.get("source", "adf")),
            "method": str(sample_mask_cfg.get("method", "percentile")),
            "percentile": float(sample_mask_cfg.get("percentile", 15)),
            "fill_holes": bool(sample_mask_cfg.get("fill_holes", True)),
            "min_size": int(sample_mask_cfg.get("min_size", 100)),
            "background_label": int(sample_mask_cfg.get("background_label", -1)),
            "generated": sample_mask is not None,
            "sample_pixels": int(sample_mask.sum()) if sample_mask is not None else None,
            "background_pixels": int((~sample_mask).sum()) if sample_mask is not None else None,
            "sample_fraction": round(float(sample_mask.mean()), 6) if sample_mask is not None else None,
        },
    }

    import numpy as _np

    # ------------------------------------------------------------------
    # Stage 1 → Stage 2 file interface
    # ------------------------------------------------------------------
    _save_stage1_outputs(
        output_dir=output_dir,
        virtual_dir=virtual_dir,
        fingerprints_dir=fingerprints_dir,
        classes_dir=classes_dir,
        orientation_dir=orientation_dir,
        dataset=dataset,
        virtual=virtual,
        fingerprints=fingerprints,
        phase=phase,
        orientation=orientation,
        diagnostics=diagnostics,
        data_contract=data_contract,
        cfg=cfg,
        qc_result=qc_result,
        roi_bragg=roi_bragg,
    )
    summary["outputs"]["roi_candidates"] = diagnostics.get("roi_candidates")
    summary["outputs"]["diagnostics"] = diagnostics

    try:
        default_nav = dataset.navigation_shape if dataset else (1, 1)
    except (ValueError, AttributeError):
        default_nav = (1, 1)
    report_path = save_report(
        output_dir, summary,
        phase.labels if phase else _np.zeros(default_nav, dtype=_np.int16),
    )
    summary["outputs"]["report"] = str(report_path)
    html_report_path = report_path.with_suffix(".html")
    if html_report_path.exists():
        summary["outputs"]["report_html"] = str(html_report_path)
    summary_path = save_summary(output_dir, summary)

    elapsed = time.perf_counter() - t0
    if errors:
        log.warning("=" * 60)
        log.warning("Workflow finished with %d error(s) in %.1f s", len(errors), elapsed)
        for err in errors:
            log.warning("  ✗  %s: %s", err["stage"], err["error"])
    else:
        log.info("=" * 60)
        log.info("Workflow finished successfully in %.1f s", elapsed)
    log.info("  summary: %s", summary_path)
    log.info("  report:  %s", report_path)
    log.info("=" * 60)

    return WorkflowResult(
        output_dir=output_dir,
        dataset=dataset,
        virtual_images=virtual,
        fingerprints=fingerprints,
        phase_screening=phase,
        orientation=orientation,
        roi_bragg=roi_bragg,
        diagnostics=diagnostics,
        summary_path=summary_path,
        report_path=report_path,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_stage1_outputs(
    *,
    output_dir: Path,
    virtual_dir: Path,
    fingerprints_dir: Path,
    classes_dir: Path,
    orientation_dir: Path,
    dataset: DatasetHandle | None,
    virtual: VirtualImageResult | None,
    fingerprints: FingerprintResult | None,
    phase: PhaseScreeningResult | None,
    orientation: OrientationResult | None,
    diagnostics: dict[str, Any],
    data_contract: DataContract,
    cfg: dict[str, Any],
    qc_result: QCResult,
    roi_bragg: dict[str, Any] | None = None,
) -> None:
    """Write the stable Stage-1 → Stage-2 file interface.

    Creates ``stage1_summary.json`` (the canonical manifest for Stage 2),
    ``data_contract.json``, ``preprocess_info.json``, bundles virtual images
    into ``virtual/virtual_images.npz``, and saves ``radial_axis.npy``
    alongside the fingerprint profiles.
    """
    import json as _json

    project_cfg = cfg.get("project", {})
    preprocess_cfg = cfg.get("preprocess", {})

    # --- data_contract.json --------------------------------------------------
    (output_dir / "data_contract.json").write_text(
        _json.dumps(data_contract.to_dict(), indent=2), encoding="utf-8",
    )

    # --- preprocess_info.json ------------------------------------------------
    preprocess_info = {
        "q_crop": preprocess_cfg.get("q_crop"),
        "q_bin": int(preprocess_cfg.get("q_bin", 1)),
        "r_bin": int(preprocess_cfg.get("r_bin", 1)),
        "original_shape": list(dataset.shape) if dataset else None,
        "preprocessed_shape": list(dataset.shape) if dataset else None,
        "nav_shape": list(dataset.navigation_shape) if dataset else None,
        "sig_shape": list(dataset.signal_shape) if dataset else None,
    }
    (output_dir / "preprocess_info.json").write_text(
        _json.dumps(preprocess_info, indent=2), encoding="utf-8",
    )

    # --- virtual/virtual_images.npz ------------------------------------------
    if virtual is not None and virtual.images:
        import numpy as _np
        _np.savez_compressed(
            virtual_dir / "virtual_images.npz",
            **{name: _np.asarray(img) for name, img in virtual.images.items()},
        )

    # --- fingerprints/radial_axis.npy ----------------------------------------
    if fingerprints is not None and fingerprints.radii is not None:
        import numpy as _np
        _np.save(fingerprints_dir / "radial_axis.npy", fingerprints.radii)

    # --- stage1_summary.json -------------------------------------------------
    _ensure_roi_candidates_file(output_dir, diagnostics)

    labels_path = classes_dir / "fingerprint_class_labels.npy"
    roi_candidates_path = output_dir / "roi_candidates" / "roi_candidates.yaml"
    stage1_summary = {
        "run_name": str(project_cfg.get("name", "unnamed")),
        "preprocessed_shape": list(dataset.shape) if dataset else None,
        "nav_shape": list(dataset.navigation_shape) if dataset else None,
        "sig_shape": list(dataset.signal_shape) if dataset else None,
        "q_crop": preprocess_cfg.get("q_crop"),
        "q_bin": int(preprocess_cfg.get("q_bin", 1)),
        "r_bin": int(preprocess_cfg.get("r_bin", 1)),
        "labels_path": _relative_path_if_exists(labels_path, output_dir),
        "roi_candidates_path": _relative_path_if_exists(roi_candidates_path, output_dir),
        "qc_status": qc_result.stage1_status,
        "virtual_images_path": _relative_path_if_exists(virtual_dir / "virtual_images.npz", output_dir),
        "fingerprints_path": _relative_path_if_exists(fingerprints_dir / "radial_fingerprints.npy", output_dir),
        "radial_axis_path": _relative_path_if_exists(fingerprints_dir / "radial_axis.npy", output_dir),
        "orientation_index_path": _relative_path_if_exists(orientation_dir / "orientation_index.npy", output_dir),
        "orientation_score_path": _relative_path_if_exists(orientation_dir / "orientation_score.npy", output_dir),
        "data_contract_path": _relative_path_if_exists(output_dir / "data_contract.json", output_dir),
        "preprocess_info_path": _relative_path_if_exists(output_dir / "preprocess_info.json", output_dir),
        "provenance_path": _relative_path_if_exists(output_dir / "provenance.json", output_dir),
        "qc_summary_path": _relative_path_if_exists(output_dir / "qc_summary.json", output_dir),
        "cluster_summary_path": _relative_path_if_exists(classes_dir / "cluster_summary.csv", output_dir),
        "cluster_mean_radial_profiles_path": _relative_path_if_exists(
            classes_dir / "cluster_mean_radial_profiles.npy", output_dir,
        ),
        "dependencies": {
            "source_backend": dataset.metadata.get("source_backend", "unknown") if dataset else "none",
            "pyxem_available": dataset.metadata.get("pyxem_available", False) if dataset else False,
            "pyxem_signal_type": dataset.metadata.get("pyxem_signal_type") if dataset else None,
            "py4DSTEM_used": roi_bragg is not None,
        },
    }
    (output_dir / "stage1_summary.json").write_text(
        _json.dumps(stage1_summary, indent=2, default=str), encoding="utf-8",
    )
    log.info("Stage-1 → Stage-2 interface written to %s", output_dir)


def _relative_path_if_exists(path: Path, base: Path) -> str | None:
    """Return a POSIX relative path only when *path* exists."""
    if not path.exists():
        return None
    return path.relative_to(base).as_posix()


def _ensure_roi_candidates_file(output_dir: Path, diagnostics: dict[str, Any]) -> None:
    """Create a valid empty ROI-candidates file pair when diagnostics skipped it."""
    roi_dir = output_dir / "roi_candidates"
    yaml_path = roi_dir / "roi_candidates.yaml"
    csv_path = roi_dir / "roi_candidates.csv"
    if yaml_path.exists() and csv_path.exists():
        return

    roi_dir.mkdir(parents=True, exist_ok=True)
    if not yaml_path.exists():
        yaml_path.write_text("rois: []\n", encoding="utf-8")
    if not csv_path.exists():
        csv_path.write_text("name,bbox,center,size,reason,recommended_stage2,cluster\n", encoding="utf-8")

    roi_outputs = diagnostics.setdefault("roi_outputs", {})
    if isinstance(roi_outputs, dict):
        roi_outputs.setdefault("yaml", str(yaml_path))
        roi_outputs.setdefault("csv", str(csv_path))
    diagnostics.setdefault("roi_candidates", str(roi_dir))


def _make_error_result(
    output_dir: Path,
    dataset: DatasetHandle | None,
    errors: list[dict[str, Any]],
) -> WorkflowResult:
    """Return a result with everything set to None when the workflow cannot proceed."""
    log.error("Workflow aborted due to unrecoverable error(s).")
    qc_result = run_qc_checks(
        dataset=dataset,
        virtual=None,
        fingerprints=None,
        phase=None,
        orientation=None,
        diagnostics={},
        errors=errors,
    )
    save_qc_summary(output_dir, qc_result)
    summary = {
        "dataset": dataset.describe() if dataset else None,
        "qc": qc_result.to_dict(),
        "outputs": {
            "qc_summary": str(output_dir / "qc_summary.json"),
        },
        "errors": errors,
    }
    summary_path = save_summary(output_dir, summary)
    return WorkflowResult(
        output_dir=output_dir,
        dataset=dataset,
        summary_path=summary_path,
        report_path=None,
        errors=errors,
    )


def _run_stage(
    label: str,
    fn: Any,
    *,
    errors: list[dict[str, Any]] | None = None,
) -> Any:
    """Run a pipeline stage with timing and optional error capture."""
    log_stage_start(log, label)
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        log.error("✗  %s FAILED after %.1f s: %s", label, elapsed, exc)
        if errors is not None:
            errors.append({"stage": label, "error": str(exc), "elapsed_s": elapsed})
        return None
    elapsed = time.perf_counter() - t0
    log_stage_end(log, label, elapsed)
    return result


def _navigation_block_shape(cfg: dict[str, Any], data_cfg: dict[str, Any]) -> tuple[int, int]:
    block_shape = cfg.get("block_shape")
    if block_shape is None:
        chunks = data_cfg.get("chunks", {})
        if isinstance(chunks, dict):
            block_shape = chunks.get("navigation", (8, 8))
        else:
            block_shape = chunks[:2] if chunks else (8, 8)
    by, bx = [max(1, int(v)) for v in block_shape]
    # Guard against pathologically small blocks that turn every dask
    # .compute() call into pure scheduler overhead (thousands of 4×4
    # chunks on a 512×512 scan → pipeline appears hung).
    MIN_BLOCK = 16
    if by < MIN_BLOCK or bx < MIN_BLOCK:
        log.warning(
            "navigation block shape (%d, %d) is very small — each block "
            "triggers a dask .compute() call with full scheduler overhead.  "
            "Consider setting chunks.navigation to at least [%d, %d] in your "
            "config to avoid apparent hangs on large datasets.",
            by, bx, max(by, MIN_BLOCK), max(bx, MIN_BLOCK),
        )
    return by, bx


def _validate_dict_config(cfg: dict[str, Any]) -> None:
    """Run schema validation on an inline config dict (not from file)."""
    from .config import validate_workflow_config

    validate_workflow_config(cfg)


def _save_png_outputs(
    output_dir: Path,
    virtual: VirtualImageResult,
    fingerprints: FingerprintResult,
    phase: PhaseScreeningResult,
    orientation: OrientationResult | None,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in virtual.images.items():
        paths[f"virtual_{name}"] = save_png(output_dir / f"virtual_{name}.png", image)
    paths["com_x"] = save_png(output_dir / "com_x.png", virtual.com_x)
    paths["com_y"] = save_png(output_dir / "com_y.png", virtual.com_y)
    paths["max_diffraction"] = save_png(output_dir / "max_diffraction.png", virtual.max_diffraction)
    paths["mean_radial_profile"] = save_profile_png(
        output_dir / "mean_radial_profile.png",
        fingerprints.radii,
        fingerprints.profiles,
    )
    paths["fingerprint_class_labels_annotated"] = save_annotated_label_png(
        output_dir / "fingerprint_class_labels_annotated.png",
        phase.labels,
        title="Fingerprint-class map from radial profiles",
    )
    paths["fingerprint_class_low_confidence"] = save_png(output_dir / "fingerprint_class_low_confidence_mask.png", phase.low_confidence_mask)
    for name, score in phase.candidate_scores.items():
        paths[f"candidate_score_{name}"] = save_png(output_dir / f"candidate_score_{name}.png", score)
    if orientation is not None:
        paths["orientation_index"] = save_label_png(output_dir / "orientation_index.png", orientation.orientation_index)
        paths["orientation_score"] = save_png(output_dir / "orientation_score.png", orientation.score)
        paths["orientation_low_confidence"] = save_png(output_dir / "orientation_low_confidence_mask.png", orientation.low_confidence_mask)
    return paths


def _run_roi_bragg(data_cfg: dict[str, Any], roi_cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    try:
        import py4DSTEM
    except ImportError as exc:
        raise ImportError("roi_bragg.enabled requires py4DSTEM. Install optional diffraction dependencies first.") from exc

    import numpy as np

    path = data_cfg.get("path")
    if not path:
        raise ValueError("roi_bragg requires data.path to point to a real MIB file.")
    scan_shape = tuple(data_cfg.get("scan_shape", (512, 512)))
    cube = py4DSTEM.import_file(str(path), mem=roi_cfg.get("mem", "MEMMAP"), scan=scan_shape)

    y0, y1, x0, x1 = [int(v) for v in roi_cfg.get("roi", (0, scan_shape[0], 0, scan_shape[1]))]
    thin_r = max(1, int(roi_cfg.get("thin_r", 2)))
    bin_q = max(1, int(roi_cfg.get("bin_q", 2)))
    roi_data = np.asarray(cube.data[y0:y1:thin_r, x0:x1:thin_r, :, :], dtype=np.float32)
    dc_roi = py4DSTEM.DataCube(roi_data, name="roi", calibration=cube.calibration)
    if bin_q > 1:
        dc_roi = dc_roi.bin_Q(bin_q, dtype=np.float32)

    template = np.asarray(dc_roi.data.mean(axis=(0, 1), dtype=np.float32), dtype=np.float32)
    bragg = dc_roi.find_Bragg_disks(
        template=template,
        corrPower=float(roi_cfg.get("corr_power", 1.0)),
        sigma_cc=float(roi_cfg.get("sigma_cc", 1)),
        edgeBoundary=int(roi_cfg.get("edge_boundary", 10)),
        minRelativeIntensity=float(roi_cfg.get("min_relative_intensity", 0.05)),
        minPeakSpacing=int(roi_cfg.get("min_peak_spacing", 4)),
        subpixel=roi_cfg.get("subpixel", "poly"),
        maxNumPeaks=int(roi_cfg.get("max_num_peaks", 70)),
        CUDA=bool(roi_cfg.get("cuda", False)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    histogram = bragg.histogram(mode="cal")
    np.save(output_dir / "roi_bragg_vector_map.npy", np.asarray(histogram.data, dtype=np.float32))
    return {
        "output_dir": str(output_dir),
        "roi": [y0, y1, x0, x1],
        "thin_r": thin_r,
        "bin_q": bin_q,
        "bragg_vector_map": str(output_dir / "roi_bragg_vector_map.npy"),
    }
