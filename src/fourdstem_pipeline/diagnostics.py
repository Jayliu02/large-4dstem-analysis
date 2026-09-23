"""Stage-1 diagnostics orchestrator.

Delegates to sub-modules:
- ``diagnostics_cluster`` -- cluster mean DPs, radial profiles, virtual stats,
  normalisation comparison, K-sweep.
- ``diagnostics_spatial`` -- beam centre, connected components, orientation
  reliability, ROI candidates, overlays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dataset import DatasetHandle
from .diagnostics_cluster import (
    cluster_orientation_table,
    cluster_mean_diffraction,
    cluster_radial_profiles,
    cluster_virtual_statistics,
    interleave_mean_std,
    k_sweep,
    normalisation_comparison,
    ring_ratio_maps,
    stats_bar_values,
    write_cluster_summary,
)
from .diagnostics_spatial import (
    beam_diagnostics,
    connected_component_diagnostics,
    orientation_reliability,
    roi_candidates,
)
from .export import save_bar_png, save_lines_png, save_png
from .fingerprints import FingerprintResult
from .logging import get_logger
from .orientation import OrientationResult
from .phase import PhaseScreeningResult
from .virtual import VirtualImageResult

log = get_logger(__name__)


def run_stage1_diagnostics(
    dataset: DatasetHandle,
    fingerprints: FingerprintResult,
    phase: PhaseScreeningResult,
    virtual: VirtualImageResult,
    orientation: OrientationResult | None,
    *,
    output_dir: str | Path,
    png_dir: str | Path,
    block_shape: tuple[int, int],
    confidence_threshold: float,
    radial_center: tuple[float, float] | list[float] | None = None,
) -> dict[str, Any]:
    """Run the full suite of stage-1 diagnostics and return output paths.

    Each diagnostic substep is isolated in its own try/except block so that
    a failure in a late substep (e.g. orientation reliability or ROI
    candidates) does not discard outputs already written by earlier
    substeps.  Substeps that depend on *orientation* are skipped when it
    is ``None``.
    """
    output_dir = Path(output_dir)
    png_dir = Path(png_dir)
    cluster_dir = output_dir / "05_cluster_diagnostics"
    roi_dir = output_dir / "roi_candidates"
    class_dir = output_dir / "fingerprint_classes"
    orientation_dir = output_dir / "orientation"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)
    class_dir.mkdir(parents=True, exist_ok=True)
    orientation_dir.mkdir(parents=True, exist_ok=True)

    labels = phase.labels
    cluster_ids = sorted(int(v) for v in np.unique(labels) if v >= 0)

    result: dict[str, Any] = {}
    diag_errors: list[str] = []

    # Group 1 -- Cluster mean diffraction patterns (no orientation dep)
    try:
        cluster_mean_dps = cluster_mean_diffraction(dataset, labels, cluster_ids, block_shape)
        np.save(cluster_dir / "cluster_mean_dps.npy", cluster_mean_dps)
        for idx, cluster_id in enumerate(cluster_ids):
            save_png(png_dir / f"cluster_mean_dp_log_{cluster_id}.png", np.log1p(cluster_mean_dps[idx]))
    except Exception as exc:
        log.error("Cluster mean diffraction patterns failed: %s", exc)
        diag_errors.append(f"cluster_mean_diffraction: {exc}")

    # Group 2 -- Cluster radial profiles (no orientation dep)
    try:
        radial_mean, radial_std = cluster_radial_profiles(fingerprints.profiles, labels, cluster_ids)
        np.save(cluster_dir / "cluster_mean_radial_profiles.npy", radial_mean)
        np.save(class_dir / "cluster_mean_radial_profiles.npy", radial_mean)
        np.save(cluster_dir / "cluster_radial_profile_std.npy", radial_std)
        save_lines_png(png_dir / "cluster_mean_radial_profiles.png", fingerprints.radii, radial_mean)
        save_lines_png(png_dir / "cluster_radial_profile_mean_std.png", fingerprints.radii,
                       interleave_mean_std(radial_mean, radial_std))
    except Exception as exc:
        log.error("Cluster radial profiles failed: %s", exc)
        diag_errors.append(f"cluster_radial_profiles: {exc}")

    # Group 3 -- Cluster virtual-image statistics (no orientation dep)
    try:
        stats_rows = cluster_virtual_statistics(labels, virtual, cluster_ids)
        write_cluster_summary(cluster_dir, stats_rows)
        write_cluster_summary(class_dir, stats_rows)
        save_bar_png(png_dir / "cluster_virtual_image_statistics.png", stats_bar_values(stats_rows))
    except Exception as exc:
        log.error("Cluster virtual statistics failed: %s", exc)
        diag_errors.append(f"cluster_virtual_statistics: {exc}")

    # Group 4 -- Ring ratio maps (no orientation dep) + cluster-vs-orientation
    ring_ratio_outputs: dict[str, Any] = {}
    try:
        ring_ratio_outputs = ring_ratio_maps(virtual, cluster_dir, png_dir)
    except Exception as exc:
        log.error("Ring ratio maps failed: %s", exc)
        diag_errors.append(f"ring_ratio_maps: {exc}")

    cluster_orientation_outputs: dict[str, Any] = {}
    if orientation is not None:
        try:
            cluster_orientation_outputs = cluster_orientation_table(
                labels, orientation, cluster_ids, cluster_dir, png_dir)
        except Exception as exc:
            log.error("Cluster vs orientation table failed: %s", exc)
            diag_errors.append(f"cluster_orientation_table: {exc}")

    # Group 5 -- Normalisation comparison & K-sweep (no orientation dep)
    norm_outputs: dict[str, Any] = {}
    try:
        norm_outputs = normalisation_comparison(fingerprints.profiles, len(cluster_ids), class_dir, png_dir)
    except Exception as exc:
        log.error("Normalisation comparison failed: %s", exc)
        diag_errors.append(f"normalisation_comparison: {exc}")

    k_sweep_outputs: dict[str, Any] = {}
    try:
        k_sweep_outputs = k_sweep(fingerprints.profiles, [2, 3, 4, 5, 6, 8], class_dir, png_dir)
    except Exception as exc:
        log.error("K-sweep failed: %s", exc)
        diag_errors.append(f"k_sweep: {exc}")

    # Group 6 -- Beam diagnostics (no orientation dep)
    beam_outputs: dict[str, Any] = {}
    try:
        beam_outputs = beam_diagnostics(virtual, png_dir, output_dir / "00_preprocess", radial_center=radial_center)
    except Exception as exc:
        log.error("Beam diagnostics failed: %s", exc)
        diag_errors.append(f"beam_diagnostics: {exc}")

    # Group 7 -- Connected components (no orientation dep)
    component_outputs: dict[str, Any] = {}
    try:
        component_outputs = connected_component_diagnostics(
            labels, virtual.images, cluster_ids, cluster_dir, png_dir, class_dir)
    except Exception as exc:
        log.error("Connected component diagnostics failed: %s", exc)
        diag_errors.append(f"connected_component_diagnostics: {exc}")

    # Group 8 -- Orientation reliability (orientation-dependent)
    orientation_outputs: dict[str, Any] = {}
    if orientation is not None:
        try:
            orientation_outputs = orientation_reliability(
                orientation, confidence_threshold, png_dir, orientation_dir)
        except Exception as exc:
            log.error("Orientation reliability failed: %s", exc)
            diag_errors.append(f"orientation_reliability: {exc}")

    # Group 9 -- ROI candidates (orientation-dependent)
    roi_outputs: dict[str, Any] = {}
    if orientation is not None:
        try:
            roi_outputs = roi_candidates(
                labels, virtual.images, orientation, cluster_ids, roi_dir, png_dir)
        except Exception as exc:
            log.error("ROI candidates failed: %s", exc)
            diag_errors.append(f"roi_candidates: {exc}")

    result = {
        "cluster_diagnostics": str(cluster_dir),
        "roi_candidates": str(roi_dir),
        "cluster_summary_csv": str(cluster_dir / "cluster_summary.csv"),
        "cluster_summary_md": str(cluster_dir / "cluster_summary.md"),
        "ring_ratio_maps": ring_ratio_outputs,
        "cluster_vs_orientation": cluster_orientation_outputs,
        "normalization_comparison": norm_outputs,
        "k_sweep": k_sweep_outputs,
        "beam": beam_outputs,
        "connected_components": component_outputs,
        "orientation_reliability": orientation_outputs,
        "roi_outputs": roi_outputs,
    }
    if diag_errors:
        result["_errors"] = diag_errors
        log.warning("Stage-1 diagnostics completed with %d partial failure(s).", len(diag_errors))
    return result
