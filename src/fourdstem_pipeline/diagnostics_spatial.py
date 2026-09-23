"""Spatial diagnostics: beam centre estimation, connected-component
analysis, orientation reliability, ROI candidate generation, and overlay
visualisations.
"""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .export import save_bar_png, save_label_png, save_png
from .orientation import OrientationResult
from .virtual import VirtualImageResult


# ---------------------------------------------------------------------------
# Beam diagnostics
# ---------------------------------------------------------------------------


def beam_diagnostics(
    virtual: VirtualImageResult,
    png_dir: Path,
    preprocess_dir: Path,
    *,
    radial_center: tuple[float, float] | list[float] | None = None,
) -> dict[str, str]:
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    mean_dp = virtual.mean_diffraction
    max_dp = virtual.max_diffraction
    yy, xx = np.indices(mean_dp.shape)
    total = max(float(mean_dp.sum()), 1e-12)
    cy = float((mean_dp * yy).sum() / total)
    cx = float((mean_dp * xx).sum() / total)
    if radial_center is None:
        radial_center = ((mean_dp.shape[0] - 1) / 2, (mean_dp.shape[1] - 1) / 2)
    offset = float(np.hypot(cy - radial_center[0], cx - radial_center[1]))
    text = (
        f"estimated_center_yx: [{cy:.3f}, {cx:.3f}]\n"
        f"radial_center_yx: [{radial_center[0]:.3f}, {radial_center[1]:.3f}]\n"
        f"offset_pixels: {offset:.3f}\n"
    )
    if offset > 3:
        text += (
            "WARNING: Estimated beam center is offset from radial integration center. "
            "Radial fingerprints and fingerprint-class labels may be biased.\n"
        )
    (preprocess_dir / "beam_center_estimate.txt").write_text(text, encoding="utf-8")
    save_png(png_dir / "mean_dp_log.png", np.log1p(mean_dp))
    save_png(png_dir / "max_dp_log.png", np.log1p(max_dp))
    save_png(png_dir / "mean_dp_with_center_marker.png", _center_overlay(mean_dp, cy, cx))
    save_png(png_dir / "mean_dp_with_radial_bins.png", _radial_overlay(mean_dp, radial_center))
    save_png(png_dir / "radial_mask_overlay.png", _radial_overlay(mean_dp, radial_center))
    central_mask = ((yy - radial_center[0]) ** 2 + (xx - radial_center[1]) ** 2) <= 7**2
    save_png(png_dir / "central_disk_mask.png", central_mask)
    if virtual.saturation_fraction is not None:
        save_png(png_dir / "saturation_fraction_map.png", virtual.saturation_fraction)
    return {"beam_center": str(preprocess_dir / "beam_center_estimate.txt")}


# ---------------------------------------------------------------------------
# Connected-component diagnostics
# ---------------------------------------------------------------------------


def connected_component_diagnostics(
    labels: np.ndarray,
    images: dict[str, np.ndarray],
    cluster_ids: list[int],
    output_dir: Path,
    png_dir: Path,
    class_dir: Path | None = None,
) -> dict[str, str]:
    rows = []
    cleaned = labels.copy()
    for cluster_id in cluster_ids:
        comps = connected_components(labels == cluster_id)
        sizes = np.asarray([len(comp) for comp in comps], dtype=np.int64)
        small = int(np.sum(sizes < 16)) if sizes.size else 0
        largest = int(sizes.max()) if sizes.size else 0
        rows.append(
            {
                "cluster": cluster_id,
                "component_count": len(comps),
                "largest_component_size": largest,
                "largest_component_fraction_within_class": largest / max(int(np.sum(labels == cluster_id)), 1),
                "median_component_size": float(np.median(sizes)) if sizes.size else 0.0,
                "small_component_count": small,
            }
        )
        for comp, size in zip(comps, sizes):
            if size < 16:
                for y, x in comp:
                    cleaned[y, x] = _majority_neighbor(labels, y, x)
    keys = [
        "cluster",
        "component_count",
        "largest_component_size",
        "largest_component_fraction_within_class",
        "median_component_size",
        "small_component_count",
    ]
    with (output_dir / "cluster_connected_components.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    np.save(output_dir / "cluster_cleaned_labels.npy", cleaned.astype(np.int16))
    cleaned_png = save_label_png(png_dir / "cluster_cleaned_labels.png", cleaned)
    class_outputs: dict[str, str] = {}
    if class_dir is not None:
        class_dir.mkdir(parents=True, exist_ok=True)
        class_npy = class_dir / "fingerprint_class_labels_cleaned.npy"
        np.save(class_npy, cleaned.astype(np.int16))
        class_outputs = {
            "fingerprint_class_labels_cleaned_npy": str(class_npy),
            "fingerprint_class_labels_cleaned_png": str(cleaned_png),
        }
    hist_values = np.asarray(
        [[row["largest_component_size"], row["component_count"]] for row in rows],
        dtype=np.float32,
    )
    if hist_values.size == 0:
        hist_values = np.zeros((1, 2), dtype=np.float32)
    save_bar_png(png_dir / "cluster_area_histogram.png", hist_values)
    base = images.get("adf") if "adf" in images else next(iter(images.values()))
    save_png(png_dir / "cluster_boundary_overlay_on_adf.png", _boundary_overlay(base, labels))
    return {
        "connected_components_csv": str(output_dir / "cluster_connected_components.csv"),
        "cleaned_labels_npy": str(output_dir / "cluster_cleaned_labels.npy"),
        "cleaned_labels_png": str(cleaned_png),
        **class_outputs,
    }


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """BFS-based 4-connected component labelling on a boolean mask."""
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        seen[y, x] = True
        comp: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            comp.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        components.append(comp)
    return components


# ---------------------------------------------------------------------------
# Orientation reliability
# ---------------------------------------------------------------------------


def orientation_reliability(
    orientation: OrientationResult,
    threshold: float,
    png_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    unreliable = orientation.score < threshold
    masked_score = orientation.score.copy()
    masked_index = orientation.orientation_index.copy()
    masked_score[unreliable] = 0
    masked_index[unreliable] = -1
    np.save(output_dir / "orientation_score_masked.npy", masked_score)
    np.save(output_dir / "orientation_index_masked.npy", masked_index)
    save_png(png_dir / "orientation_score_masked.png", masked_score)
    save_label_png(png_dir / "orientation_index_masked.png", np.maximum(masked_index, 0))
    hist, _ = np.histogram(orientation.score, bins=32)
    save_bar_png(png_dir / "orientation_score_histogram.png", hist)
    return {"masked_score": str(output_dir / "orientation_score_masked.npy")}


# ---------------------------------------------------------------------------
# ROI candidate generation
# ---------------------------------------------------------------------------


def roi_candidates(
    labels: np.ndarray,
    images: dict[str, np.ndarray],
    orientation: OrientationResult,
    cluster_ids: list[int],
    output_dir: Path,
    png_dir: Path,
) -> dict[str, str]:
    rois: list[dict[str, Any]] = []
    for cluster_id in cluster_ids:
        comps = connected_components(labels == cluster_id)
        if not comps:
            continue
        largest = max(comps, key=len)
        rois.append(
            _roi_from_component(
                f"cluster{cluster_id}_core_01", largest, labels.shape,
                cluster=cluster_id, reason="largest connected component core", size=64,
            )
        )
        if np.sum(labels == cluster_id) / labels.size < 0.15:
            rois.append(
                _roi_from_component(
                    f"cluster{cluster_id}_minority_01", largest, labels.shape,
                    cluster=cluster_id, reason="minority fingerprint class", size=32,
                )
            )
    boundary = np.argwhere(_boundary_mask(labels))
    if boundary.size:
        center = boundary[len(boundary) // 2]
        rois.append(
            _roi_from_center("boundary_classes_01", int(center[0]), int(center[1]), labels.shape, 64, "class boundary", None)
        )
    rois.extend(_orientation_rois(orientation, labels.shape))
    rois.extend(_intensity_rois(images, labels.shape))

    yaml_path = output_dir / "roi_candidates.yaml"
    csv_path = output_dir / "roi_candidates.csv"
    yaml_path.write_text(yaml.safe_dump({"rois": rois}, sort_keys=False), encoding="utf-8")
    keys = sorted({key for roi in rois for key in roi}) or [
        "name",
        "bbox",
        "center",
        "size",
        "reason",
        "recommended_stage2",
        "cluster",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{key: roi.get(key, "") for key in keys} for roi in rois])
    base = images.get("adf") if "adf" in images else next(iter(images.values()))
    save_png(png_dir / "roi_candidates_overlay.png", _roi_overlay(base, rois))
    return {"yaml": str(yaml_path), "csv": str(csv_path)}


def _roi_from_component(
    name: str,
    component: list[tuple[int, int]],
    shape: tuple[int, int],
    *,
    cluster: int,
    reason: str,
    size: int,
) -> dict[str, Any]:
    coords = np.asarray(component)
    y = int(np.median(coords[:, 0]))
    x = int(np.median(coords[:, 1]))
    return _roi_from_center(name, y, x, shape, size, reason, cluster)


def _roi_from_center(
    name: str,
    y: int,
    x: int,
    shape: tuple[int, int],
    size: int,
    reason: str,
    cluster: int | None,
) -> dict[str, Any]:
    """Create an ROI dict following the unified data contract.

    Parameters
    ----------
    y, x:
        Centre position in ``(y, x)`` (row, column) order per the
        ``center_order: y_x`` convention.
    """
    half = size // 2
    y0 = max(0, min(y - half, shape[0] - size))
    x0 = max(0, min(x - half, shape[1] - size))
    y1 = min(shape[0], y0 + size)
    x1 = min(shape[1], x0 + size)
    roi: dict[str, Any] = {
        "name": name,
        "center": [int((y0 + y1) / 2), int((x0 + x1) / 2)],
        "bbox": [int(y0), int(y1), int(x0), int(x1)],
        "size": [int(y1 - y0), int(x1 - x0)],
        "reason": reason,
        "recommended_stage2": "py4dstem_bragg_indexing",
    }
    if cluster is not None:
        roi["cluster"] = int(cluster)
    return roi


def _orientation_rois(
    orientation: OrientationResult,
    label_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    scale_y = label_shape[0] / orientation.score.shape[0]
    scale_x = label_shape[1] / orientation.score.shape[1]
    high = np.unravel_index(int(np.argmax(orientation.score)), orientation.score.shape)
    low = np.unravel_index(int(np.argmin(orientation.score)), orientation.score.shape)
    return [
        _roi_from_center(
            "high_orientation_score_01",
            int((high[0] + 0.5) * scale_y), int((high[1] + 0.5) * scale_x),
            label_shape, 64, "high orientation score", None,
        ),
        _roi_from_center(
            "low_orientation_score_01",
            int((low[0] + 0.5) * scale_y), int((low[1] + 0.5) * scale_x),
            label_shape, 64, "low orientation score", None,
        ),
    ]


def _intensity_rois(
    images: dict[str, np.ndarray],
    shape: tuple[int, int],
) -> list[dict[str, Any]]:
    rois = []
    for name in ("adf", "haadf"):
        if name in images:
            y, x = np.unravel_index(int(np.argmax(images[name])), images[name].shape)
            rois.append(
                _roi_from_center(
                    f"{name}_high_intensity_01", int(y), int(x),
                    shape, 64, f"{name.upper()} intensity anomaly", None,
                )
            )
    return rois


def _majority_neighbor(labels: np.ndarray, y: int, x: int) -> int:
    y0, y1 = max(0, y - 1), min(labels.shape[0], y + 2)
    x0, x1 = max(0, x - 1), min(labels.shape[1], x + 2)
    values, counts = np.unique(labels[y0:y1, x0:x1], return_counts=True)
    return int(values[np.argmax(counts)])


def _boundary_mask(labels: np.ndarray) -> np.ndarray:
    mask = np.zeros(labels.shape, dtype=bool)
    mask[:-1, :] |= labels[:-1, :] != labels[1:, :]
    mask[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return mask


# ---------------------------------------------------------------------------
# Visualisation overlays
# ---------------------------------------------------------------------------


def _scale_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (1, 99))
    if hi <= lo:
        hi = float(finite.max())
        lo = float(finite.min())
    scaled = (arr - lo) / max(hi - lo, 1e-12)
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def _boundary_overlay(base: np.ndarray, labels: np.ndarray) -> np.ndarray:
    gray = _scale_gray(base)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    rgb[_boundary_mask(labels)] = [235, 40, 40]
    return rgb


def _roi_overlay(base: np.ndarray, rois: list[dict[str, Any]]) -> np.ndarray:
    gray = _scale_gray(base)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    for roi in rois:
        y0, y1, x0, x1 = [int(v) for v in roi["bbox"]]
        rgb[y0:y1, x0] = [255, 215, 0]
        rgb[y0:y1, max(x1 - 1, x0)] = [255, 215, 0]
        rgb[y0, x0:x1] = [255, 215, 0]
        rgb[max(y1 - 1, y0), x0:x1] = [255, 215, 0]
    return rgb


def _center_overlay(image: np.ndarray, cy: float, cx: float) -> np.ndarray:
    gray = _scale_gray(image)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    y = int(round(cy))
    x = int(round(cx))
    rgb[max(0, y - 5):min(rgb.shape[0], y + 6), x] = [255, 40, 40]
    rgb[y, max(0, x - 5):min(rgb.shape[1], x + 6)] = [255, 40, 40]
    return rgb


def _radial_overlay(image: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    gray = _scale_gray(image)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    yy, xx = np.indices(image.shape)
    rr = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    for radius in np.linspace(8, min(image.shape) / 2 - 2, 5):
        rgb[np.abs(rr - radius) < 0.5] = [40, 200, 80]
    return rgb
