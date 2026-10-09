"""Second-round MIB detection and provisional pixel-coordinate diagnostics."""
from __future__ import annotations

import argparse
import csv
import inspect
import itertools
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import cdist

from .mib_round1 import digest, export_peak_coordinates, plot_grid, raw_peak_options, read_data, write_json


PEAK_DTYPE = np.dtype([
    ("scan_y", "i4"), ("scan_x", "i4"), ("qx_raw_px", "f8"), ("qy_raw_px", "f8"),
    ("qx_rel_px", "f8"), ("qy_rel_px", "f8"), ("peak_intensity", "f8"),
    ("peak_quality", "u1"), ("signal_counts", "f8"), ("snr_proxy", "f8"),
    ("roi_id", "U16"), ("method", "U8"),
])


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_write(path, rows, fields=None):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return dict(count=int(a.size), median=float(np.median(a)) if a.size else None,
                p95=float(np.percentile(a, 95)) if a.size else None,
                maximum=float(a.max()) if a.size else None)


def fingerprint_tree(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}


def check_baseline(cfg):
    root = Path(cfg["baseline_root"])
    bcfg = yaml.safe_load(Path(cfg["baseline_config"]).read_text(encoding="utf-8"))
    manifest = load_json(root / "run_manifest.json")
    require(manifest["artifact_checks_passed"], "First-round artifact checks did not pass")
    for name, sha in manifest["artifacts"].items():
        require(digest(root / name) == sha, f"Baseline artifact changed: {name}")
    model_path = root / "P1_origin/origin_model.npz"
    qc = load_json(root / "P1_origin/origin_qc.json")
    require(qc["gate1_passed"] and qc["origin_model_sha256"] == digest(model_path), "Origin QC/model mismatch")
    meta = load_json(root / "P0_baseline/metadata.json")
    source = Path(bcfg["input"]["path"])
    require(source.resolve() == Path(meta["input"]).resolve(), "MIB path differs from baseline")
    require(source.stat().st_size == meta["size_bytes"] and source.stat().st_mtime_ns == meta["mtime_ns"],
            "MIB size or modification time changed")
    with np.load(model_path) as f:
        model = {k: f[k] for k in ["origin_x", "origin_y"]}
        original_config = json.loads(str(f["config_json"]))
    require(bcfg["input"] == original_config["input"], "Input reader config differs from origin model")
    require(bcfg["origin"] == original_config["origin"], "Origin config differs from validated model")
    shape = tuple(bcfg["input"]["scan_shape"])
    coverage = np.zeros(shape, dtype=bool)
    ids = []
    for roi in cfg["rois"]:
        y0, y1, x0, x1 = roi["bounds"]
        require(0 <= y0 < y1 <= shape[0] and 0 <= x0 < x1 <= shape[1], "ROI out of bounds")
        require((y1-y0, x1-x0) == (32, 32), "This round requires 32x32 ROIs")
        require(not coverage[y0:y1, x0:x1].any(), "Main ROIs overlap; define a deduplication policy first")
        coverage[y0:y1, x0:x1] = True
        require(roi["id"].isalnum() and len(roi["id"]) <= 16, "ROI id must be 1-16 alphanumeric characters")
        ids.append(roi["id"])
    require(len(set(ids)) == len(ids), "Duplicate ROI ids")
    require(cfg["sampling"]["holdout_per_roi"] >= 20, "At least 20 holdout DPs per ROI required")
    return bcfg, model, fingerprint_tree(root)


def sample_sets(shape, tuning_count, holdout_count, seed):
    require(tuning_count > 0 and holdout_count > 0 and tuning_count+holdout_count <= np.prod(shape),
            "Invalid sampling counts")
    # Four spatial strata: prevent random samples from missing an entire quadrant.
    rng = np.random.default_rng(seed)
    pools = []
    for ya, yb in [(0, shape[0]//2), (shape[0]//2, shape[0])]:
        for xa, xb in [(0, shape[1]//2), (shape[1]//2, shape[1])]:
            pools.append(rng.permutation([(y, x) for y in range(ya, yb) for x in range(xa, xb)]))
    ordered = np.array([pools[i % 4][i // 4] for i in range(tuning_count+holdout_count)])
    return ordered[:tuning_count], ordered[tuning_count:]


def roi_context(cfg, data, out):
    from matplotlib.patches import Rectangle
    adf = np.load(Path(cfg["baseline_root"]) / "P2_virtual/adf_normalized.npy")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(adf)
    ax.set(xlabel="scan_x", ylabel="scan_y", title="Corrected ADF / Total and selected ROIs")
    for number, roi in enumerate(cfg["rois"]):
        y0, y1, x0, x1 = roi["bounds"]
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="red"))
        ax.text(x0, y0, roi["id"], color="white")
        tuning, _ = sample_sets((y1-y0, x1-x0), cfg["sampling"]["tuning_per_roi"],
                                cfg["sampling"]["holdout_per_roi"], cfg["seed"]+number)
        positions = tuning[:12]+[y0, x0]
        plot_grid(out / f"roi_spotcheck_{roi['id']}.png", [data[int(y), int(x)].compute() for y, x in positions],
                  [str(tuple(p)) for p in positions], log=True, columns=4)
    fig.tight_layout(); fig.savefig(out / "roi_selection.png", dpi=140); plt.close(fig)


def centered_kernel(patch, detector_shape, support_radius=8, annulus_outer=12):
    """A compact measured positive lobe plus a unit negative annulus, FFT origin at (0,0)."""
    require(patch.ndim == 2 and patch.shape[0] == patch.shape[1] and patch.shape[0] % 2 == 1,
            "Template patch must be an odd square")
    radius = patch.shape[0]//2
    require(0 < support_radius < annulus_outer <= radius, "Invalid template support/annulus")
    require(min(detector_shape) > 2*radius+1, "Detector too small for template")
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    rr = np.hypot(yy, xx)
    taper = np.clip((support_radius-rr)/2, 0, 1)
    positive = np.maximum(np.asarray(patch, float), 0)*taper
    require(positive.sum() > 0, "Template has no positive mass")
    positive /= positive.sum()
    ring = (rr >= support_radius) & (rr <= annulus_outer)
    compact = positive - ring/ring.sum()
    kernel = np.zeros(detector_shape, dtype=float)
    kernel[yy % detector_shape[0], xx % detector_shape[1]] = compact
    return kernel, positive, compact


def build_template(cfg, data, model, out):
    root = Path(cfg["baseline_root"])
    t = cfg["template"]
    source = np.load(root / "P1_origin/origin_measured_sampled.npz")
    keep = source["valid"] & ~source["outlier"] & (source["residual_px"] <= t["max_origin_residual_px"])
    for roi in cfg["rois"]:
        y0, y1, x0, x1 = roi["bounds"]
        keep &= ~((source["scan_y"] >= y0) & (source["scan_y"] < y1) &
                  (source["scan_x"] >= x0) & (source["scan_x"] < x1))
    pool = np.flatnonzero(keep)
    require(len(pool) >= t["source_count"], "Not enough independent template sources")
    indices = np.sort(np.random.default_rng(cfg["seed"]).choice(pool, t["source_count"], replace=False))
    r = t["crop_radius_px"]
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    rr = np.hypot(yy, xx)
    patches, frames, positions, rows, raw_crops = [], [], [], [], []
    for i in indices:
        sy, sx = int(source["scan_y"][i]), int(source["scan_x"][i])
        cy, cx = float(source["origin_y"][i]), float(source["origin_x"][i])
        dp = data[sy, sx].compute().astype(float)
        patch = map_coordinates(dp, [yy+cy, xx+cx], order=1, mode="constant")
        raw_crops.append(patch.copy())
        background = float(np.median(patch[rr >= r-2]))
        patch = np.maximum(patch-background, 0)
        _, positive, _ = centered_kernel(patch, dp.shape, t["support_radius_px"], t["annulus_outer_px"])
        patches.append(positive)
        frames.append(dp.astype(np.uint16))
        positions.append((sy, sx))
        rows.append(dict(scan_y=sy, scan_x=sx, origin_y=cy, origin_x=cx, background=background,
                         origin_residual_px=float(source["residual_px"][i])))
    patches = np.array(patches)
    # Patches already have a support taper. Do not apply it a second time.
    def from_median(a):
        positive = np.median(a, axis=0)
        positive /= positive.sum()
        ring = (rr >= t["support_radius_px"]) & (rr <= t["annulus_outer_px"])
        compact = positive-ring/ring.sum()
        kernel = np.zeros(data.shape[-2:])
        kernel[yy % kernel.shape[0], xx % kernel.shape[1]] = compact
        return kernel, positive, compact
    kernel, positive, compact = from_median(patches)
    _, half_a, _ = from_median(patches[::2])
    _, half_b, _ = from_median(patches[1::2])
    offset = float(np.hypot((positive*yy).sum(), (positive*xx).sum()))
    l1 = float(abs(half_a-half_b).sum())
    passed = offset <= t["max_centroid_offset_px"] and l1 <= t["max_half_relative_l1"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "empirical_template.npz", kernel=kernel, positive=positive,
        compact=compact, patches=patches, source_positions=positions, source_dp=frames, raw_crops=raw_crops,
        template_kind="specimen_direct_beam_local_estimate_not_vacuum")
    csv_write(out / "template_sources.csv", rows)
    plot_grid(out / "template_shapes.png", [positive, compact, half_a, half_b, abs(half_a-half_b)],
              ["Median positive lobe", "Zero-sum compact kernel", "Sources even", "Sources odd", "Split difference"])
    plot_grid(out / "template_source_dp.png", frames, [str(p) for p in positions], log=True, columns=6)
    plot_grid(out / "template_source_crops.png", raw_crops, [str(p) for p in positions], log=True, columns=6)
    qc = dict(source_count=len(positions), source_points_outside_all_rois=True, positive_centroid_offset_px=offset,
        split_half_relative_l1=l1, numerical_template_gate_passed=passed,
        support_radius_px=t["support_radius_px"], annulus_outer_px=t["annulus_outer_px"],
        zero_sum=float(kernel.sum()), kernel_fft_origin="zero displacement at [0,0] via modulo indexing",
        physical_probe_calibrated=False,
        limitation="Specimen central-beam shape estimate; split similarity does not prove an uncontaminated vacuum probe.")
    write_json(out / "template_qc.json", qc)
    require(passed, "Template numerical QC failed; inspect source patches before correlation detection")
    return kernel, qc


def detect(dc, positions, method, kernel, cfg, relative=None, spacing=None, edge=None):
    c = cfg["detection"]
    opts = raw_peak_options() if method == "raw" else dict(template=kernel)
    return dc.find_Bragg_disks(data=(positions[:, 0], positions[:, 1]), **opts,
        minRelativeIntensity=relative if relative is not None else c["working_relative_intensity"],
        minPeakSpacing=spacing if spacing is not None else c["working_spacing"],
        edgeBoundary=edge if edge is not None else c["working_edge"],
        sigma=c["sigma"], subpixel="poly", corrPower=1, maxNumPeaks=c["max_peaks"])


def local_quality(dp, x, y, snr_min):
    ix, iy = int(round(x)), int(round(y))
    ya, yb = max(0, iy-7), min(dp.shape[0], iy+8)
    xa, xb = max(0, ix-7), min(dp.shape[1], ix+8)
    yy, xx = np.mgrid[ya:yb, xa:xb]
    rr = np.hypot(yy-y, xx-x)
    a = np.asarray(dp[ya:yb, xa:xb], dtype=float)
    aperture, ring = rr <= 3, (rr >= 5) & (rr <= 7)
    bg = float(np.median(a[ring])) if ring.any() else 0.
    mass = float(a[aperture].sum())
    area = int(aperture.sum())
    signal = mass-area*bg
    noise = np.sqrt(max(1., mass+area*bg+area**2*bg/max(int(ring.sum()), 1)))
    snr = signal/noise
    flag = int(snr < snr_min)
    flag |= 4 if min(y, x, dp.shape[0]-1-y, dp.shape[1]-1-x) < 7 else 0
    flag |= 8 if a[aperture].size and a[aperture].max() >= 4095 else 0
    return signal, snr, flag


def correct_coordinates(raw_x, raw_y, scan_y, scan_x, model):
    sy, sx = np.asarray(scan_y), np.asarray(scan_x)
    require(np.issubdtype(sy.dtype, np.integer) and np.issubdtype(sx.dtype, np.integer), "Scan coordinates must be integer")
    require(((sy >= 0) & (sy < model["origin_x"].shape[0])).all() and
            ((sx >= 0) & (sx < model["origin_x"].shape[1])).all(), "Global scan coordinate out of bounds")
    return np.asarray(raw_x)-model["origin_x"][sy, sx], np.asarray(raw_y)-model["origin_y"][sy, sx]


def records_for_dp(dp, peaks, sy, sx, roi_id, method, model, cfg):
    p = peaks.data
    x, y = export_peak_coordinates(p)
    rx, ry = correct_coordinates(x, y, int(sy), int(sx), model)
    records = np.zeros(len(p), dtype=PEAK_DTYPE)
    records["scan_y"], records["scan_x"] = sy, sx
    records["qx_raw_px"], records["qy_raw_px"] = x, y
    records["qx_rel_px"], records["qy_rel_px"] = rx, ry
    records["peak_intensity"] = p["intensity"]
    records["roi_id"], records["method"] = roi_id, method
    for i in range(len(p)):
        signal, snr, flag = local_quality(dp, x[i], y[i], cfg["detection"]["quality_snr_min"])
        flag |= 2 if np.hypot(rx[i], ry[i]) < cfg["detection"]["central_radius_px"] else 0
        if len(p) > 1 and np.count_nonzero(np.hypot(x-x[i], y-y[i]) < 14) > 1:
            flag |= 16
        records["signal_counts"][i], records["snr_proxy"][i], records["peak_quality"][i] = signal, snr, flag
    return records


def mutual_pairs(a, b, max_distance):
    """One-to-one mutual nearest pairs; unmatched peaks are not forced into a match."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.empty((0, 2), dtype=int)
    distance = cdist(a, b)
    forward, backward = distance.argmin(axis=1), distance.argmin(axis=0)
    pairs = [(i, j) for i, j in enumerate(forward) if backward[j] == i and distance[i, j] <= max_distance]
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def xy(peaks, relative=True):
    suffix = "rel_px" if relative else "raw_px"
    return np.column_stack([peaks[f"qx_{suffix}"], peaks[f"qy_{suffix}"]])


def synthetic_check(cfg, kernel, out):
    import py4DSTEM
    shape = kernel.shape
    yy, xx = np.indices(shape)
    # Asymmetric multi-peak positions, fractional coordinates and unequal amplitudes.
    truth = np.array([[shape[1]*.24+.2, shape[0]*.31+.1], [shape[1]*.64+.3, shape[0]*.28+.2],
                      [shape[1]*.37+.4, shape[0]*.66+.3], [shape[1]*.72+.1, shape[0]*.73+.4]])
    dp = np.full(shape, 3.)
    for (x, y), amplitude in zip(truth, [1000, 220, 110, 65]):
        dp += amplitude*np.exp(-((xx-x)**2+(yy-y)**2)/(2*2.4**2))
    dc = py4DSTEM.DataCube(data=dp[None, None])
    result = {}
    for method in ["raw", "template"]:
        peaks = detect(dc, np.array([[0, 0]]), method, kernel, cfg)[0].data
        x, y = export_peak_coordinates(peaks)
        points = np.column_stack([x, y])
        pairs = mutual_pairs(truth, points, cfg["gates"]["synthetic_max_error_px"])
        errors = np.linalg.norm(truth[pairs[:, 0]]-points[pairs[:, 1]], axis=1)
        result[method] = dict(true_peaks=len(truth), detected_peaks=len(points), matches=len(pairs),
                             errors_px=summary(errors), passed=len(pairs) == len(truth) == len(points))
    write_json(out / "synthetic_detection_qc.json", result)
    require(all(r["passed"] for r in result.values()), "Synthetic peak localization/detection failed")
    return result


def p3_roi(cfg, data, model, kernel, roi, number, out):
    import py4DSTEM
    y0, y1, x0, x1 = roi["bounds"]
    frames = np.asarray(data[y0:y1, x0:x1].compute(), dtype=np.uint16)
    dc = py4DSTEM.DataCube(data=frames)
    tuning, holdout = sample_sets(frames.shape[:2], cfg["sampling"]["tuning_per_roi"],
                                  cfg["sampling"]["holdout_per_roi"], cfg["seed"]+number)
    np.savez_compressed(out / f"sampling_{roi['id']}.npz", tuning=tuning, holdout=holdout, bounds=roi["bounds"])
    c = cfg["detection"]
    rows, working, cached = [], {}, {}
    for method in ["raw", "template"]:
        for j, (relative, spacing, edge) in enumerate(itertools.product(c["relative_intensities"], c["min_spacings"], c["edges"])):
            detected = detect(dc, tuning, method, kernel, cfg, relative, spacing, edge)
            for i, (sy, sx) in enumerate(tuning):
                p = records_for_dp(frames[sy, sx], detected[i], y0+sy, x0+sx, roi["id"], method, model, cfg)
                cached[f"{method}_param{j}_dp{i}"] = p
                central = (p["peak_quality"] & 2) != 0
                rows.append(dict(roi_id=roi["id"], method=method, parameter_id=j, scan_y=int(y0+sy), scan_x=int(x0+sx),
                    relative=relative, spacing=spacing, edge=edge, peak_count=len(p), central_count=int(central.sum()),
                    low_snr_count=int(((p["peak_quality"] & 1) != 0).sum()),
                    noncentral_supported_count=int(((p["peak_quality"] & 3) == 0).sum()),
                    peak_cap_reached=len(p) >= c["max_peaks"]))
        print(f"{roi['id']}: {method} 18-parameter sweep complete", flush=True)
        positions = np.array(list(np.ndindex(frames.shape[:2])))
        detected = detect(dc, positions, method, kernel, cfg)
        working[method] = {(int(sy), int(sx)): records_for_dp(frames[sy, sx], peaks, y0+sy, x0+sx,
                            roi["id"], method, model, cfg) for (sy, sx), peaks in zip(positions, detected)}
        all_peaks = np.concatenate(list(working[method].values()))
        np.savez_compressed(out / f"bragg_raw_{roi['id']}_{method}.npz",
            peaks=all_peaks[["scan_y", "scan_x", "qx_raw_px", "qy_raw_px", "peak_intensity", "peak_quality", "signal_counts", "snr_proxy", "roi_id", "method"]],
            bounds=roi["bounds"], status="provisional")
    np.savez_compressed(out / f"sweep_peaks_{roi['id']}.npz", **cached)
    csv_write(out / f"parameter_sweep_{roi['id']}.csv", rows)
    # Persist every holdout DP so review does not need the 8 GiB source online.
    np.savez_compressed(out / f"holdout_dp_{roi['id']}.npz", dp=frames[holdout[:, 0], holdout[:, 1]],
                        scan_y=holdout[:, 0]+y0, scan_x=holdout[:, 1]+x0)
    review = []
    for page, start in enumerate(range(0, len(holdout), 5), 1):
        fig, axes = plt.subplots(5, 2, figsize=(11, 23))
        for row, (sy, sx) in enumerate(holdout[start:start+5]):
            for column, method in enumerate(["raw", "template"]):
                ax = axes[row, column]
                ax.imshow(np.log1p(frames[sy, sx].astype(float)), cmap="gray")
                p = working[method][int(sy), int(sx)]
                ax.scatter(p["qx_raw_px"], p["qy_raw_px"], facecolors="none", edgecolors="lime", s=45)
                low = (p["peak_quality"] & 1) != 0
                ax.scatter(p["qx_raw_px"][low], p["qy_raw_px"][low], marker="x", c="orange", s=25)
                ax.set_title(f"{method} ({y0+sy},{x0+sx}) n={len(p)} low-SNR={low.sum()}")
                review.append(dict(roi_id=roi["id"], method=method, scan_y=int(y0+sy), scan_x=int(x0+sx),
                    detected_count=len(p), central_count=int(((p["peak_quality"] & 2) != 0).sum()),
                    low_snr_count=int(low.sum()), reviewer="", false_peaks="", missed_major_peaks="", verdict="pending"))
        fig.tight_layout()
        fig.savefig(out / f"holdout_overlays_{roi['id']}_{page}.png", dpi=130)
        plt.close(fig)
    # Never overwrite a review already annotated by a person.
    review_path = out / f"review_{roi['id']}.csv"
    if not review_path.exists():
        csv_write(review_path, review)
    csv_write(out / f"holdout_metrics_{roi['id']}.csv", [{k: v for k, v in r.items() if k not in
        ["reviewer", "false_peaks", "missed_major_peaks", "verdict"]} for r in review])
    return working, rows, holdout


def neighbor_diagnostics(points, shape, max_distance):
    before, after = [], []
    supported_pairs = 0
    for sy, sx in np.ndindex(shape):
        a = points[sy, sx]
        a = a[(a["peak_quality"] & 3) == 0]  # Noncentral, local SNR-supported peaks only.
        for ty, tx in [(sy+1, sx), (sy, sx+1)]:
            if ty >= shape[0] or tx >= shape[1]:
                continue
            b = points[ty, tx]
            b = b[(b["peak_quality"] & 3) == 0]
            pairs = mutual_pairs(xy(a), xy(b), max_distance)
            if len(pairs):
                supported_pairs += 1
                pa, pb = a[pairs[:, 0]], b[pairs[:, 1]]
                before.extend(np.linalg.norm(xy(pb, False)-xy(pa, False), axis=1))
                after.extend(np.linalg.norm(xy(pb)-xy(pa), axis=1))
    return dict(raw_displacement_px=summary(before), corrected_displacement_px=summary(after),
        adjacent_position_pairs_with_matches=supported_pairs,
        total_adjacent_position_pairs=shape[0]*(shape[1]-1)+(shape[0]-1)*shape[1],
        interpretation="Mutual proximity matches, not hkl identification; no truth claim near changing orientations."), before, after


def p4_roi(cfg, data, model, kernel, roi, working, out):
    import py4DSTEM
    y0, y1, x0, x1 = roi["bounds"]
    shape = (y1-y0, x1-x0)
    # Independently reread a sub-ROI to exercise nonzero local-to-global offsets.
    sub_bounds = [y0+8, y0+24, x0+8, x0+24]
    sub_data = np.asarray(data[sub_bounds[0]:sub_bounds[1], sub_bounds[2]:sub_bounds[3]].compute(), dtype=np.uint16)
    sub_dc = py4DSTEM.DataCube(data=sub_data)
    sub_positions = np.array(list(np.ndindex(sub_data.shape[:2])))
    result, exported = {}, []
    for method in ["raw", "template"]:
        points = working[method]
        peaks = np.concatenate(list(points.values()))
        exported.append(peaks)
        np.savez_compressed(out / f"peaks_corrected_{roi['id']}_{method}.npz", peaks=peaks, bounds=roi["bounds"],
                            origin_model_sha256=digest(Path(cfg["baseline_root"]) / "P1_origin/origin_model.npz"),
                            units="detector_px", status="provisional")
        counts, central = np.zeros(shape, int), np.full(shape, np.nan)
        duplicate_central, low_snr = 0, 0
        for sy, sx in np.ndindex(shape):
            p = points[sy, sx]
            counts[sy, sx] = len(p)
            mask = (p["peak_quality"] & 2) != 0
            if mask.any():
                central[sy, sx] = np.linalg.norm(xy(p[mask]), axis=1).min()
            duplicate_central += int(mask.sum() > 1)
            low_snr += int(((p["peak_quality"] & 1) != 0).sum())
        np.savez_compressed(out / f"maps_{roi['id']}_{method}.npz", peak_count=counts, central_residual_px=central)
        plot_grid(out / f"maps_{roi['id']}_{method}.png", [counts, central],
                  [f"{roi['id']} {method} peak count", "Central residual (px); blank=missing"], columns=2)
        # The raw BVM is measured relative to the fixed geometric detector centre.
        extent = max(data.shape[-2:])
        bins = 2*extent
        raw_x = peaks["qx_raw_px"]-(data.shape[-1]-1)/2
        raw_y = peaks["qy_raw_px"]-(data.shape[-2]-1)/2
        hist_raw, edges, _ = np.histogram2d(raw_y, raw_x, bins=bins, range=[[-extent, extent]]*2)
        hist_rel, _, _ = np.histogram2d(peaks["qy_rel_px"], peaks["qx_rel_px"], bins=bins, range=[[-extent, extent]]*2)
        np.savez_compressed(out / f"bvm_{roi['id']}_{method}.npz", before=hist_raw, after=hist_rel, edges_px=edges)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        vmax = max(np.log1p(hist_raw).max(), np.log1p(hist_rel).max())
        for ax, image, title in zip(axes, [hist_raw, hist_rel], ["Fixed geometric reference", "Origin model subtracted"]):
            ax.imshow(np.log1p(image), extent=[-extent, extent, extent, -extent], vmin=0, vmax=vmax)
            ax.set(xlim=(-180, 180), ylim=(180, -180), xlabel="x (px)", ylabel="y (px)", title=title)
        fig.tight_layout(); fig.savefig(out / f"bvm_before_after_{roi['id']}_{method}.png", dpi=150); plt.close(fig)
        neighbors, before, after = neighbor_diagnostics(points, shape, cfg["detection"]["neighbor_match_radius_px"])
        np.savez_compressed(out / f"neighbor_displacements_{roi['id']}_{method}.npz", raw=before, corrected=after)
        sub_peaks = detect(sub_dc, sub_positions, method, kernel, cfg)
        overlap_error = 0.
        for (sy, sx), p in zip(sub_positions, sub_peaks):
            observed = records_for_dp(sub_data[sy, sx], p, sub_bounds[0]+sy, sub_bounds[2]+sx,
                                       roi["id"], method, model, cfg)
            expected = points[int(sy)+8, int(sx)+8]
            require(len(observed) == len(expected), "Overlap detection count mismatch")
            if len(observed):
                order_a = np.lexsort((observed["qy_raw_px"], observed["qx_raw_px"]))
                order_b = np.lexsort((expected["qy_raw_px"], expected["qx_raw_px"]))
                for relative in [False, True]:
                    overlap_error = max(overlap_error, float(abs(xy(observed[order_a], relative)-xy(expected[order_b], relative)).max()))
                require(np.array_equal(observed["scan_y"], expected["scan_y"]) and
                        np.array_equal(observed["scan_x"], expected["scan_x"]), "Overlap global scan indices differ")
        center_stats = summary(central)
        fraction = center_stats["count"]/int(np.prod(shape))
        g = cfg["gates"]
        passed = bool(fraction >= g["central_coverage_min"] and
            center_stats["median"] is not None and center_stats["median"] <= g["central_median_max_px"] and
            center_stats["p95"] <= g["central_p95_max_px"] and overlap_error <= g["overlap_max_error_px"])
        result[method] = dict(roi_id=roi["id"], peak_count=len(peaks), central_coverage=fraction,
            central_residual_px=center_stats, multiple_central_patterns=duplicate_central,
            low_snr_peaks=low_snr, peak_cap_patterns=int((counts >= cfg["detection"]["max_peaks"]).sum()),
            neighbor_matches=neighbors, overlap_max_coordinate_error_px=overlap_error,
            overlap_patterns=len(sub_positions), overlap_bounds=sub_bounds, engineering_gate_passed=passed,
            scientific_gate4_passed=False, status="provisional_pending_gate3_review")
    return result, exported


def schema_document(out):
    (out / "coordinates_spec.md").write_text(
        "# Pixel coordinate contract v2\n\n"
        "Input axes: (scan_y,scan_x,det_y,det_x). Zero-based x=column, y=row. "
        "py4DSTEM qx is axis 0; exported qx_raw_px=py4DSTEM.qy, qy_raw_px=py4DSTEM.qx. "
        "Global scan index = local index + ROI (y0,x0). Bounds have exclusive upper ends.\n\n"
        "qx_rel_px=qx_raw_px-origin_x[scan_y,scan_x]; qy_rel_px=qy_raw_px-origin_y[scan_y,scan_x]. "
        "Both raw and relative coordinates are detector pixels, never inverse Angstrom.\n\n"
        "peak_intensity is the native smoothed maximum (raw branch) or correlation score (template branch). "
        "Scores across branches are not photon counts or directly comparable. signal_counts is "
        "the local radius-3 aperture sum minus a radius-5-to-7 ring median background. "
        "snr_proxy is an approximate Poisson local support metric, not a false-positive probability.\n\n"
        "peak_quality bit mask: 1=local SNR below threshold; 2=within 8 px of model central beam; "
        "4=local support clipped at detector edge; 8=aperture includes >=4095 count pixel; "
        "16=another detected peak is within 14 px, so local background may overlap. "
        "Zero means none of these flags; it is not a verified crystallographic peak.\n\n"
        "roi_id preserves region identity. method is raw or template. Merged table contains both "
        "methods as alternative detections, not independent observations to sum. "
        "All corrected products are provisional until Gate 3 expert review.\n",
        encoding="utf-8")


def sweep_summary(rows, out):
    groups = {}
    for row in rows:
        key = (row["roi_id"], row["method"], row["relative"], row["spacing"], row["edge"])
        groups.setdefault(key, []).append(row)
    result = []
    for (roi_id, method, relative, spacing, edge), values in groups.items():
        result.append(dict(roi_id=roi_id, method=method, relative=relative, spacing=spacing, edge=edge,
            mean_peak_count=float(np.mean([v["peak_count"] for v in values])),
            mean_supported_noncentral=float(np.mean([v["noncentral_supported_count"] for v in values])),
            mean_low_snr_count=float(np.mean([v["low_snr_count"] for v in values])),
            central_coverage=float(np.mean([v["central_count"] > 0 for v in values]))))
    csv_write(out / "parameter_summary.csv", result)
    roi_ids = list(dict.fromkeys(r["roi_id"] for r in result))
    fig, axes = plt.subplots(1, len(roi_ids), figsize=(6*len(roi_ids), 4), squeeze=False)
    for ax, roi_id in zip(axes.flat, roi_ids):
        for method in ["raw", "template"]:
            subset = [r for r in result if r["roi_id"] == roi_id and r["method"] == method and r["spacing"] == 8 and r["edge"] == 10]
            ax.plot([r["relative"] for r in subset], [r["mean_supported_noncentral"] for r in subset], "o-", label=method)
        ax.set(title=roi_id, xlabel="Relative intensity threshold", ylabel="Mean SNR-supported noncentral peaks")
        ax.legend()
    fig.tight_layout(); fig.savefig(out / "parameter_sensitivity.png", dpi=150); plt.close(fig)


def verify_output(out):
    manifest = load_json(out / "run_manifest.json")
    require(load_json(out / "status.json")["status"] == "complete", "Run is not complete")
    for name, sha in manifest["artifacts"].items():
        require(digest(out / name) == sha, f"Output changed: {name}")
    # Review CSV files are intentionally mutable and are not automated approval inputs.
    baseline = Path(manifest["config"]["baseline_root"])
    require(fingerprint_tree(baseline) == manifest["baseline_fingerprints"], "First-round outputs changed")
    meta = load_json(baseline / "P0_baseline/metadata.json")
    raw_stat = Path(meta["input"]).stat()
    require(raw_stat.st_size == meta["size_bytes"] and raw_stat.st_mtime_ns == meta["mtime_ns"], "Raw MIB changed")
    with np.load(baseline / "P1_origin/origin_model.npz") as model:
        all_peaks = np.load(out / "P4_bragg_corrected/merged_peaks.npz")["peaks"]
        rx, ry = correct_coordinates(all_peaks["qx_raw_px"], all_peaks["qy_raw_px"],
                                    all_peaks["scan_y"], all_peaks["scan_x"], model)
    require(np.allclose(rx, all_peaks["qx_rel_px"], atol=1e-12, rtol=0) and
            np.allclose(ry, all_peaks["qy_rel_px"], atol=1e-12, rtol=0), "Merged coordinate correction mismatch")
    source_positions = np.load(out / "template/empirical_template.npz")["source_positions"]
    expected_count = 0
    for roi in manifest["config"]["rois"]:
        y0, y1, x0, x1 = roi["bounds"]
        require(not ((source_positions[:, 0] >= y0) & (source_positions[:, 0] < y1) &
                     (source_positions[:, 1] >= x0) & (source_positions[:, 1] < x1)).any(), "Template leakage into ROI")
        sets = np.load(out / f"P3_bragg/sampling_{roi['id']}.npz")
        require(not set(map(tuple, sets["tuning"])).intersection(map(tuple, sets["holdout"])), "Tuning/holdout leakage")
        require(len(sets["holdout"]) >= 20, "Too few holdout DPs")
        for method in ["raw", "template"]:
            p = np.load(out / f"P4_bragg_corrected/peaks_corrected_{roi['id']}_{method}.npz")["peaks"]
            expected_count += len(p)
            require(((p["scan_y"] >= y0) & (p["scan_y"] < y1) &
                     (p["scan_x"] >= x0) & (p["scan_x"] < x1)).all(), "Exported global ROI bounds violated")
            merged = all_peaks[(all_peaks["roi_id"] == roi["id"]) & (all_peaks["method"] == method)]
            require(np.array_equal(merged, p), "Merged table differs from per-ROI table")
    require(expected_count == len(all_peaks), "Merged peak count mismatch")
    print(f"Artifact verification passed: {len(all_peaks)} peaks, baseline unchanged, independent samples and global coordinates valid.", flush=True)


def write_report(out, cfg, template_qc, synthetic, results, sweep_rows):
    lines = ["# MIB 第二轮执行报告", "", "完成三 ROI 双分支扫描与 P4 像素坐标诊断。所有结果为 provisional；Gate 3 领域复核尚未完成。", "",
        "## 模板与合成验证", "",
        f"模板来源 {template_qc['source_count']} 个 ROI 外可靠中央束局部图像；正峰重心偏差 {template_qc['positive_centroid_offset_px']:.4f} px，"
        f"分半模板 L1 差 {template_qc['split_half_relative_l1']:.4f}。这是试样中的局部中央束估计，不是真空探针。",
        "", f"合成检测两分支通过：{all(r['passed'] for r in synthetic.values())}。合成真值只覆盖代码与简化峰形，不证明真实样品检测准确率。", "",
        "## ROI 结果", "", "| ROI | 分支 | 峰数 | 中央束覆盖 | 中位误差 px | P95 px | 重叠最大差 px | 工程阈值 |",
        "|---|---|---:|---:|---:|---:|---:|---|"]
    for roi_id, methods in results.items():
        for method, r in methods.items():
            c = r["central_residual_px"]
            median_text = f"{c['median']:.4f}" if c['median'] is not None else "N/A"
            p95_text = f"{c['p95']:.4f}" if c['p95'] is not None else "N/A"
            lines.append(f"| {roi_id} | {method} | {r['peak_count']} | {r['central_coverage']:.2%} | "
                f"{median_text} | {p95_text} | {r['overlap_max_coordinate_error_px']:.3g} | {r['engineering_gate_passed']} |")
    lines.extend(["", "中央束误差只对模型周围 8 px 内检出的最近峰统计，覆盖率分母是每 ROI 的全部 1,024 张 DP。"
        "近邻匹配统计基于互为最近邻及局部信噪支持，不等同于已确认的同一 hkl。晶界取向变化不会被自动解释为仪器残差。",
        "", f"参数扫描共 {len(sweep_rows)} 条 DP/参数/分支统计；每 ROI 固定 20 个调参点，另有 20 个未用于调参的复核点。"
        "工作参数运行前固定为 0.03/8/10，未按复核集选择最佳参数。",
        "", "同一个相对阈值在两个分支上不代表相同检出灵敏度。原始峰与相关峰分数的分布不同，不能仅用峰数或中央束残差宣布一方总体更准确。"
        "`parameter_summary.csv` 和 `parameter_sensitivity.png` 汇总调参集的局部信噪支持与阈值敏感性。",
        "", "## 复核与交付", "",
        "`P3_bragg/holdout_overlays_*.png` 对照两分支，绿色圈为检测峰，橙叉标记低局部信噪支持；"
        "`review_*.csv` 留待填写 reviewer、false_peaks、missed_major_peaks、verdict。自动指标不填写人工结论。",
        "", "`P4_bragg_corrected/merged_peaks.npz` 保留两个检测分支；不得将两个分支作为独立观察叠加。"
        "原始和校正坐标、区域身份、质量位、原始局部计数和相关分数均分开保存。完整坐标及强度契约见该目录的 `coordinates_spec.md`。",
        "", "第一轮结果通过哈希复核未改变。原始 MIB 只读访问，文件大小与修改时间前后检查。"
        "完整数据未物化，仅逐 ROI 读取；仍未确认扫描物理方向和绝对 q 标定，未进行相/取向/应变计算。",
        "", "复现：`./.conda-phase/python.exe scripts/04_mib_round2.py`；验证："
        "`./.conda-phase/python.exe scripts/04_mib_round2.py --verify`。使用 `--output` 保存不同试验。"])
    (out / "report_zh.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(cfg, out):
    import py4DSTEM
    out.mkdir(parents=True, exist_ok=True)
    require(out.resolve() != Path(cfg["baseline_root"]).resolve() and
            Path(cfg["baseline_root"]).resolve() not in out.resolve().parents and
            out.resolve() not in Path(cfg["baseline_root"]).resolve().parents,
            "Round-two output must be outside the first-round directory")
    bcfg, model, baseline_fingerprints = check_baseline(cfg)
    previous_path = out / "run_manifest.json"
    if previous_path.exists():
        require(load_json(previous_path)["config"] == cfg, "Changed configuration: choose a new output directory")
    for review_path in (out / "P3_bragg").glob("review_*.csv"):
        with review_path.open(encoding="utf-8-sig", newline="") as stream:
            reviews = list(csv.DictReader(stream))
        require(all(r["verdict"] == "pending" and not r["reviewer"] and not r["false_peaks"] and
                    not r["missed_major_peaks"] for r in reviews),
                "Annotated review exists: choose a new output directory to preserve its matching detections")
    write_json(out / "status.json", dict(status="running"))
    try:
        write_json(out / "resolved_config.json", cfg)
        data, _ = read_data(bcfg)
        roi_context(cfg, data, out)
        template_out, p3, p4 = out / "template", out / "P3_bragg", out / "P4_bragg_corrected"
        p3.mkdir(exist_ok=True); p4.mkdir(exist_ok=True)
        kernel, template_qc = build_template(cfg, data, model, template_out)
        print("Template constructed; running synthetic coordinate checks", flush=True)
        synthetic = synthetic_check(cfg, kernel, out)
        results, sweep_rows, exported = {}, [], []
        for number, roi in enumerate(cfg["rois"]):
            working, rows, _ = p3_roi(cfg, data, model, kernel, roi, number, p3)
            sweep_rows.extend(rows)
            results[roi["id"]], peaks = p4_roi(cfg, data, model, kernel, roi, working, p4)
            exported.extend(peaks)
            print(f"{roi['id']}: full detection, correction and overlap audit complete", flush=True)
        csv_write(p3 / "bragg_param_sweep.csv", sweep_rows)
        sweep_summary(sweep_rows, p3)
        csv_write(p3 / "roi_manifest.csv", [dict(roi_id=r["id"], y0=r["bounds"][0], y1=r["bounds"][1],
                  x0=r["bounds"][2], x1=r["bounds"][3], description=r["description"]) for r in cfg["rois"]])
        np.savez_compressed(p4 / "merged_peaks.npz", peaks=np.concatenate(exported), units="detector_px", status="provisional")
        write_json(p4 / "peak_centering_qc.json", results)
        schema_document(p4)
        write_json(p3 / "detection_qc.json", dict(gate3_passed=False, review_status="pending_domain_review",
            template_kind="empirical specimen central beam; not vacuum", parameters=cfg["detection"],
            sample_counts=cfg["sampling"], sweep_rows=len(sweep_rows), source_rois=cfg["rois"]))
        write_report(out, cfg, template_qc, synthetic, results, sweep_rows)
        require(baseline_fingerprints == fingerprint_tree(Path(cfg["baseline_root"])), "Baseline changed during run")
        check_baseline(cfg)  # Also recheck raw input size/mtime after processing.
        artifacts = {name: sha for name, sha in fingerprint_tree(out).items()
                     if name not in ["run_manifest.json", "status.json"] and not Path(name).name.startswith("review_")}
        write_json(out / "run_manifest.json", dict(config=cfg, artifacts=artifacts,
            baseline_fingerprints=baseline_fingerprints, baseline_unchanged=True,
            python=platform.python_version(), py4dstem=py4DSTEM.__version__,
            api_signature=str(inspect.signature(py4DSTEM.DataCube.find_Bragg_disks)),
            source_sha256={str(p): digest(p) for p in [Path(__file__), Path(__file__).with_name("mib_round1.py")]},
            gate3_passed=False, gate4_scientific_passed=False,
            all_geometry_checks_passed=all(r["engineering_gate_passed"] for methods in results.values() for r in methods.values())))
        write_json(out / "status.json", dict(status="complete"))
        verify_output(out)
    except Exception as exc:
        write_json(out / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mib_round2.yaml")
    parser.add_argument("--output")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.output:
        cfg["output"] = args.output
    out = Path(cfg["output"])
    if args.verify:
        verify_output(out)
    else:
        run(cfg, out)


if __name__ == "__main__":
    main()
