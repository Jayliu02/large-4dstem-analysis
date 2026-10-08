"""Read-only MIB first round: auditable origins, virtual detectors and ROI sweep.

All arrays use (scan_y, scan_x, det_y, det_x); x always means column.
No physical reciprocal-space calibration is inferred from the filename.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter, shift
from scipy.optimize import least_squares


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding="utf-8")


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plot_grid(path, arrays, titles, shared=False, log=False, columns=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    images = [np.log1p(a) if log else a for a in arrays]
    limits = np.nanpercentile(np.concatenate([a.ravel() for a in images]), [1, 99.5])
    fig, axes = plt.subplots(int(np.ceil(len(images) / columns)), columns,
                             figsize=(5 * columns, 4 * int(np.ceil(len(images) / columns))),
                             squeeze=False)
    for ax, a, title in zip(axes.flat, images, titles):
        im = ax.imshow(a, **(dict(vmin=limits[0], vmax=limits[1]) if shared else {}))
        ax.set_title(title)
        ax.set_xlabel("x (column)")
        ax.set_ylabel("y (row)")
        fig.colorbar(im, ax=ax)
    for ax in list(axes.flat)[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def read_data(cfg):
    from rsciio.quantumdetector import file_reader
    c = cfg["input"]
    item = file_reader(c["path"], lazy=True, mmap_mode="r",
                       navigation_shape=tuple(c["scan_shape"]),
                       chunks=tuple(c["chunks"]))[0]
    data = item["data"]
    if tuple(data.shape) != tuple(c["scan_shape"] + c["detector_shape"]):
        raise ValueError(f"Unexpected MIB dimensions: {data.shape}")
    if data.dtype != np.dtype(c["dtype_expected"]):
        raise ValueError(f"Unexpected MIB dtype: {data.dtype}")
    return data, item["original_metadata"]


def measure_origin(dp, cfg):
    """Smooth for initial peak only; background-subtracted local raw centroid."""
    dp = np.asarray(dp, dtype=float)
    y0, y1, x0, x1 = cfg["search_window"]
    smooth = gaussian_filter(dp, cfg["smoothing_sigma"])
    iy, ix = np.unravel_index(np.argmax(smooth[y0:y1, x0:x1]), (y1-y0, x1-x0))
    iy, ix = iy+y0, ix+x0
    radius = cfg["local_radius_px"]
    ya, yb = max(0, iy-radius), min(dp.shape[0], iy+radius+1)
    xa, xb = max(0, ix-radius), min(dp.shape[1], ix+radius+1)
    patch = dp[ya:yb, xa:xb]
    border = np.concatenate([patch[0], patch[-1], patch[1:-1, 0], patch[1:-1, -1]])
    background = float(np.median(border))
    weight = np.maximum(patch-background, 0)
    mass = weight.sum()
    yy, xx = np.mgrid[ya:yb, xa:xb]
    cy = float((weight*yy).sum()/mass) if mass > 0 else float(iy)
    cx = float((weight*xx).sum()/mass) if mass > 0 else float(ix)
    width = float(np.sqrt((weight*((yy-cy)**2+(xx-cx)**2)).sum()/max(mass, 1)))
    peak = float(patch.max())
    snr = (peak-background)/max(1., np.sqrt(max(background, 0)), 1.4826*np.median(abs(border-background)))
    edge = min(iy-y0, y1-1-iy, ix-x0, x1-1-ix) < radius
    lo, hi = cfg["width_range_px"]
    valid = bool(mass > 0 and snr >= cfg["min_snr"] and lo <= width <= hi and not edge)
    return dict(origin_x=cx, origin_y=cy, peak=peak, background=background,
                width_px=width, snr=float(snr), near_search_edge=edge, valid=valid)


def design(scan_y, scan_x):
    y, x = np.broadcast_arrays(scan_y, scan_x)
    return np.column_stack([np.ones(x.size), x.ravel(), y.ravel()])


def fit_plane(scan_y, scan_x, origin, valid, scale=2.):
    a = design(scan_y, scan_x)
    keep = np.asarray(valid).ravel() & np.isfinite(np.asarray(origin).ravel())
    if keep.sum() < 6 or np.linalg.matrix_rank(a[keep]) < 3:
        raise ValueError("Insufficient spatially distributed valid origins")
    z = np.asarray(origin).ravel()[keep]
    initial = np.linalg.lstsq(a[keep], z, rcond=None)[0]
    result = least_squares(lambda p: a[keep]@p-z, initial, loss="soft_l1", f_scale=scale)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def validation_positions(shape, excluded, per_quadrant, seed):
    rng = np.random.default_rng(seed)
    ny, nx = shape
    points = []
    for ya, yb in [(0, ny//2), (ny//2, ny)]:
        for xa, xb in [(0, nx//2), (nx//2, nx)]:
            pool = [(y, x) for y in range(ya, yb) for x in range(xa, xb)
                    if (y, x) not in excluded]
            points.extend(pool[i] for i in rng.choice(len(pool), per_quadrant, replace=False))
    return np.array(points)


def measure_positions(data, positions, cfg):
    rows = []
    for i, (y, x) in enumerate(positions):
        rows.append(measure_origin(data[int(y), int(x)].compute(), cfg))
        if (i+1) % 128 == 0:
            print(f"Measured {i+1}/{len(positions)} origins", flush=True)
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def baseline(cfg, data, metadata, root):
    out = root / "P0_baseline"
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    backup = out / "backup"
    backup.mkdir(exist_ok=True)
    for source in sorted(Path(cfg["baseline"]).glob("*")):
        if not source.is_file():
            continue
        target = backup / source.name
        sha = digest(source)
        if target.exists():
            if digest(target) != sha:
                raise ValueError(f"Baseline changed: {source}")
        else:
            shutil.copy2(source, target)
            target.chmod(stat.S_IREAD)
        manifest.append(dict(source=str(source), backup=str(target), sha256=sha,
                             size_bytes=source.stat().st_size, read_only=True))
    write_json(out / "baseline_manifest.json", manifest)
    commands = [[sys.executable, "--version"], [sys.executable, "-m", "pip", "check"],
                [sys.executable, "-m", "pip", "freeze"]]
    env = []
    for cmd in commands:
        r = subprocess.run(cmd, capture_output=True, text=True)
        env.append(f"{' '.join(cmd)}\nexit_code={r.returncode}\n{r.stdout}{r.stderr}")
    (out / "environment.txt").write_text("\n".join(env), encoding="utf-8")
    p = Path(cfg["input"]["path"])
    props = metadata["mib_properties"]
    expected = int(np.prod(data.shape[:2])) * (props["head_size"] + int(np.prod(data.shape[2:]))*data.dtype.itemsize)
    if p.stat().st_size != expected:
        raise ValueError("File size does not match frame count, header and pixels")
    write_json(out / "metadata.json", dict(input=str(p.resolve()), size_bytes=p.stat().st_size,
        mtime_ns=p.stat().st_mtime_ns, shape=list(data.shape), dtype=data.dtype.str,
        reader_metadata=metadata, config=cfg, scan_order="row-major assumed; physical raster/serpentine orientation unconfirmed",
        absolute_q_calibration="uncalibrated", gate0="structural checks passed; scan orientation requires experimental confirmation"))
    rng = np.random.default_rng(cfg["seed"])
    ny, nx = data.shape[:2]
    points = [tuple(divmod(int(i), nx)) for i in rng.choice(ny*nx, 16, replace=False)]
    points += [(0, 0), (0, nx-1), (ny-1, 0), (ny-1, nx-1), (ny//2, nx//2)]
    dps = np.stack([data[y, x].compute() for y, x in points])
    np.savez_compressed(out / "dp_spotcheck.npz", positions=points, dp=dps)
    plot_grid(out / "dp_spotcheck.png", dps, [str(p) for p in points], log=True, columns=5)


def origin_stage(cfg, data, root):
    out = root / "P1_origin"
    out.mkdir(parents=True, exist_ok=True)
    c = cfg["origin"]
    ny, nx = data.shape[:2]
    sy, sx = np.meshgrid(np.arange(0, ny, c["samples_stride"]),
                         np.arange(0, nx, c["samples_stride"]), indexing="ij")
    points = np.column_stack([sy.ravel(), sx.ravel()])
    measured = measure_positions(data, points, c)
    px, py = [fit_plane(points[:, 0], points[:, 1], measured[f"origin_{axis}"],
                       measured["valid"], c["robust_scale_px"]) for axis in ["x", "y"]]
    a = design(points[:, 0], points[:, 1])
    rx, ry = measured["origin_x"]-a@px, measured["origin_y"]-a@py
    residual = np.hypot(rx, ry)
    outlier = residual > c["outlier_threshold_px"]
    np.savez_compressed(out / "origin_measured_sampled.npz", scan_y=points[:, 0], scan_x=points[:, 1],
        **measured, residual_x=rx, residual_y=ry, residual_px=residual, outlier=outlier,
        fit_input=measured["valid"], reliable=measured["valid"] & ~outlier)
    yy, xx = np.indices((ny, nx))
    origin_x, origin_y = [(design(yy, xx)@p).reshape(ny, nx) for p in [px, py]]
    validation = validation_positions((ny, nx), set(map(tuple, points)),
                                      c["validation_points_per_quadrant"], cfg["seed"])
    vm = measure_positions(data, validation, c)
    va = design(validation[:, 0], validation[:, 1])
    vx, vy = vm["origin_x"]-va@px, vm["origin_y"]-va@py
    error = np.hypot(vx, vy)
    # Never remove validation outliers on the basis of model agreement.
    valid_error = error[vm["valid"]]
    median = float(np.median(valid_error)) if valid_error.size else None
    p95 = float(np.percentile(valid_error, 95)) if valid_error.size else None
    gate = bool(median is not None and median <= c["median_target_px"] and p95 <= c["p95_target_px"]
                and vm["valid"].mean() >= c["min_valid_fraction"]
                and measured["valid"].mean() >= c["min_valid_fraction"] and len(validation) >= 40)
    np.savez_compressed(out / "origin_validation.npz", scan_y=validation[:, 0], scan_x=validation[:, 1],
                        **vm, predicted_x=va@px, predicted_y=va@py,
                        residual_x=vx, residual_y=vy, residual_px=error)
    np.savez_compressed(out / "origin_model.npz", origin_x=origin_x, origin_y=origin_y,
        coefficients_x=px, coefficients_y=py, model_version="robust_plane_v1", gate1_passed=gate,
        input_path=cfg["input"]["path"], units="detector_px", axes="scan_y,scan_x",
        config_json=json.dumps(cfg, sort_keys=True))
    qc = dict(gate1_passed=gate, model="soft_l1 plane; all quality-valid training points retained",
        origin_model_sha256=digest(out / "origin_model.npz"),
        training_count=len(points), training_valid_fraction=float(measured["valid"].mean()),
        training_outliers=int(outlier.sum()), training_residual_median_px=float(np.median(residual[measured["valid"]])),
        validation_count=len(validation), validation_valid_count=int(vm["valid"].sum()),
        validation_median_px=median, validation_p95_px=p95,
        validation_all_median_px=float(np.median(error)), validation_all_p95_px=float(np.percentile(error, 95)),
        validation_independent=True, thresholds=c, coefficients_x=px.tolist(), coefficients_y=py.tolist(),
        interpretation="Instrument drift and physical beam deflection are not separated; no global CoM substituted for origin.",
        spatial_review="Residual maps and individual validation overlays require review; grain boundary labels unavailable.")
    qc["validation_quadrants"] = []
    for yhalf, xhalf in itertools.product([False, True], repeat=2):
        mask = ((validation[:, 0] >= ny//2) == yhalf) & ((validation[:, 1] >= nx//2) == xhalf)
        qc["validation_quadrants"].append(dict(y_half=int(yhalf), x_half=int(xhalf),
            count=int(mask.sum()), valid_count=int(vm["valid"][mask].sum()),
            median_px=float(np.median(error[mask])), p95_px=float(np.percentile(error[mask], 95))))
    write_json(out / "origin_qc.json", qc)
    plot_grid(out / "origin_residual_maps.png", [rx.reshape(sy.shape), ry.reshape(sy.shape), residual.reshape(sy.shape),
        measured["valid"].reshape(sy.shape), outlier.reshape(sy.shape), measured["width_px"].reshape(sy.shape)],
        ["Residual x (px)", "Residual y (px)", "Residual magnitude (px)", "Measurement valid", "Residual > 5 px", "Local width (px)"])
    import matplotlib.pyplot as plt
    reliable = measured["valid"] & ~outlier
    plot_grid(out / "origin_residual_inliers.png",
        [np.where(reliable, a, np.nan).reshape(sy.shape) for a in [rx, ry, residual]],
        ["Reliable residual x (px)", "Reliable residual y (px)", "Reliable residual norm (px)"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(residual, bins=50); axes[0].set_title("Training residual, all measurements")
    axes[1].hist(error, bins=20); axes[1].set_title("Independent validation, all measurements")
    im = axes[2].scatter(validation[:, 1], validation[:, 0], c=error, vmin=0, vmax=max(5, np.percentile(error, 95)))
    axes[2].invert_yaxis(); axes[2].set_title("Validation residual (px)"); fig.colorbar(im, ax=axes[2])
    fig.tight_layout(); fig.savefig(out / "origin_validation.png", dpi=150); plt.close(fig)
    # Expose possible wrong-spot measurements rather than silently accepting the fit.
    nrows = int(np.ceil(len(validation)/8))
    fig, axes = plt.subplots(nrows, 8, figsize=(24, 3*nrows), squeeze=False)
    for i, (ax, (y, x)) in enumerate(zip(axes.flat, validation)):
        ax.imshow(np.log1p(data[int(y), int(x)].compute()), cmap="gray")
        ax.plot(vm["origin_x"][i], vm["origin_y"][i], "r+", label="measured")
        ax.plot((va@px)[i], (va@py)[i], "co", fillstyle="none", label="model")
        ax.set_title(f"({y},{x}) err={error[i]:.2f} valid={vm['valid'][i]}", fontsize=8)
        ax.set_xlim(55, 200); ax.set_ylim(200, 55)
    fig.tight_layout(); fig.savefig(out / "validation_overlays.png", dpi=120); plt.close(fig)
    (out / "coordinates_spec.md").write_text(
        "# Coordinate contract v1\n\nArrays: (scan_y, scan_x, det_y, det_x); zero-based pixels. "
        "x is column, y is row. Plane = c0 + c1*scan_x + c2*scan_y. "
        "origin_x[y,x] and origin_y[y,x] are absolute detector column and row. "
        "ROI [y0:y1,x0:x1] uses exclusive upper bounds. Global scan = local + (y0,x0).\n\n"
        "For an untransposed NumPy DP, py4DSTEM raw qx indexes detector axis 0 (row); "
        "raw qy indexes axis 1 (column). Export qx_raw_px=py4DSTEM.qy, qy_raw_px=py4DSTEM.qx. "
        "qx_rel_px=qx_raw_px-origin_x[scan_y,scan_x]; qy_rel_px=qy_raw_px-origin_y[scan_y,scan_x]. "
        "Units remain pixels, never 1/Angstrom. Physical scan orientation is unconfirmed.\n",
        encoding="utf-8")
    print(json.dumps(qc, indent=2), flush=True)
    return gate


def integrate_detectors(frames, origin_y, origin_x, bf_radii, adf_annuli):
    """Integrate raw counts with moving masks; never resample the source DP."""
    yy, xx = np.ogrid[:frames.shape[-2], :frames.shape[-1]]
    r2 = ((yy[None]-np.asarray(origin_y)[:, None, None])**2 +
          (xx[None]-np.asarray(origin_x)[:, None, None])**2)
    result = {f"bf_r{r}": np.sum(frames, axis=(1, 2), where=r2 <= r*r, dtype=np.uint64)
              for r in bf_radii}
    result.update({f"adf_r{lo}_{hi}": np.sum(frames, axis=(1, 2),
        where=(r2 >= lo*lo) & (r2 < hi*hi), dtype=np.uint64) for lo, hi in adf_annuli})
    return result


def distribution(a):
    return dict(zip(["min", "p01", "p05", "median", "p95", "p99", "max"],
                    np.percentile(a, [0, 1, 5, 50, 95, 99, 100]).tolist()))


def virtual_stage(cfg, data, root):
    out = root / "P2_virtual"
    out.mkdir(parents=True, exist_ok=True)
    model = np.load(root / "P1_origin/origin_model.npz")
    c = cfg["virtual"]
    ny, nx, dy, dx = data.shape
    bf = sorted(set(c["bf_radii_px"] + [c["bf_radius_px"]]))
    adf = sorted(set(map(tuple, c["adf_annuli_px"] + [[c["adf_inner_px"], c["adf_outer_px"]]])))
    names = [f"bf_r{r}" for r in bf] + [f"adf_r{lo}_{hi}" for lo, hi in adf]
    maps = {name: np.zeros((ny, nx), dtype=np.uint64) for name in names + ["total", "bf_fixed", "adf_fixed"]}
    maps.update({name: np.zeros((ny, nx), dtype=float) for name in ["com_x", "com_y"]})
    batch = cfg["input"]["chunks"][1]
    qy, qx = np.ogrid[:dy, :dx]
    for y in range(ny):
        for x in range(0, nx, batch):
            stop = min(nx, x+batch)
            frames = np.asarray(data[y, x:stop].compute(), dtype=np.uint16)
            corrected = integrate_detectors(frames, model["origin_y"][y, x:stop],
                model["origin_x"][y, x:stop], bf, adf)
            fixed = integrate_detectors(frames, np.full(len(frames), (dy-1)/2),
                np.full(len(frames), (dx-1)/2), [c["bf_radius_px"]], [(c["adf_inner_px"], c["adf_outer_px"])])
            for name, values in corrected.items():
                maps[name][y, x:stop] = values
            maps["bf_fixed"][y, x:stop] = fixed[f"bf_r{c['bf_radius_px']}"]
            maps["adf_fixed"][y, x:stop] = fixed[f"adf_r{c['adf_inner_px']}_{c['adf_outer_px']}"]
            total = frames.sum(axis=(1, 2), dtype=np.uint64)
            maps["total"][y, x:stop] = total
            maps["com_x"][y, x:stop] = (frames*qx).sum(axis=(1, 2))/np.maximum(total, 1)
            maps["com_y"][y, x:stop] = (frames*qy).sum(axis=(1, 2))/np.maximum(total, 1)
        if (y+1) % 16 == 0:
            print(f"Virtual detectors {y+1}/{ny} rows", flush=True)
    maps["bf_corrected"] = maps[f"bf_r{c['bf_radius_px']}"]
    maps["adf_corrected"] = maps[f"adf_r{c['adf_inner_px']}_{c['adf_outer_px']}"]
    for name in ["bf", "adf"]:
        for mode in ["corrected", "fixed"]:
            target = f"{name}_normalized" if mode == "corrected" else f"{name}_fixed_normalized"
            maps[target] = np.divide(maps[f"{name}_{mode}"], maps["total"],
                out=np.zeros((ny, nx), dtype=float), where=maps["total"] > 0)
    for name, array in maps.items():
        np.save(out / f"{name}.npy", array)
    plot_grid(out / "bf_adf_compare.png", [maps[k] for k in ["bf_fixed", "bf_corrected", "adf_fixed", "adf_corrected"]],
        ["BF fixed", "BF moving", "ADF fixed", "ADF moving"], shared=True, log=True, columns=2)
    plot_grid(out / "detector_radius_sweep.png", [maps[n]/np.maximum(maps["total"], 1) for n in names],
              [n+" / Total" for n in names], shared=True)
    plot_grid(out / "total_com_normalized.png", [maps[k] for k in ["total", "com_x", "com_y", "bf_normalized", "adf_normalized"]],
              ["Total counts", "CoM x (raw px)", "CoM y (raw px)", "BF / Total", "ADF / Total"])
    val = np.load(root / "P1_origin/origin_validation.npz")
    positions = np.column_stack([val["scan_y"], val["scan_x"]])
    raw, aligned = np.zeros((dy, dx)), np.zeros((dy, dx))
    for y, x in positions:
        dp = data[int(y), int(x)].compute().astype(float)
        raw += dp
        aligned += shift(dp, ((dy-1)/2-model["origin_y"][y, x], (dx-1)/2-model["origin_x"][y, x]),
                         order=1, mode="constant", prefilter=False)
    raw /= len(positions); aligned /= len(positions)
    np.savez_compressed(out / "mean_dp.npz", before=raw, after=aligned, positions=positions,
                        reference_y=(dy-1)/2, reference_x=(dx-1)/2)
    plot_grid(out / "mean_dp_before_after.png", [raw, aligned], ["Before (fixed validation set)", "After (display only)"],
              shared=True, log=True, columns=2)
    # Global central-feature second moment measures the broadening of the drifting beam.
    def central_width(dp):
        window = np.zeros(dp.shape, bool); window[64:192, 64:192] = True
        w = np.maximum(dp-np.median(dp), 0)*window
        cy, cx = (w*qy).sum()/w.sum(), (w*qx).sum()/w.sum()
        return dict(centroid_y=float(cy), centroid_x=float(cx),
                    rms_radius_px=float(np.sqrt((w*((qy-cy)**2+(qx-cx)**2)).sum()/w.sum())), peak=float(dp.max()))
    # Save radial intensity for radius selection, including scattering outside the direct beam.
    rr = np.hypot(qy-(dy-1)/2, qx-(dx-1)/2).astype(int)
    radial = np.bincount(rr.ravel(), weights=aligned.ravel())/np.maximum(np.bincount(rr.ravel()), 1)
    np.save(out / "aligned_radial_intensity.npy", radial)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4)); ax.semilogy(radial[:80])
    for r in sorted(set(bf + [v for pair in adf for v in pair])):
        ax.axvline(r, alpha=.3)
    ax.set(xlabel="Radius (detector px)", ylabel="Mean intensity", title="Aligned mean DP radial profile")
    fig.tight_layout(); fig.savefig(out / "radial_intensity.png", dpi=150); plt.close(fig)
    write_json(out / "virtual_qc.json", dict(parameters=c, distributions={k: distribution(v) for k, v in maps.items()},
        origin_model_sha256=digest(root / "P1_origin/origin_model.npz"),
        mean_dp_before=central_width(raw), mean_dp_after=central_width(aligned),
        mean_dp_note="Central 128x128 RMS includes Bragg scattering; not a fitted direct-beam disk radius.",
        aligned_mean_count_retention=float(aligned.sum()/raw.sum()),
        zero_total_count=int((maps["total"] == 0).sum()),
        gate2="Numerical products complete; morphology requires review; no physical defect assignment"))


def export_peak_coordinates(peaks):
    """py4DSTEM names axis 0 qx; our exported x means detector column."""
    return peaks["qy"].copy(), peaks["qx"].copy()


def raw_peak_options():
    """Compensate the installed 0.14.18 unconditional IFFT in template=None.

    Public filter callback returns FFT(DP), so the backend IFFT restores DP.
    Check implementation rather than assuming every 0.14.18 build is identical.
    """
    from py4DSTEM.braggvectors.diskdetection import _find_Bragg_disks_single
    source = inspect.getsource(_find_Bragg_disks_single)
    buggy = "cc = DP" in source and "np.fft.ifft2(cc)" in source
    return dict(template=None, filter_function=np.fft.fft2 if buggy else None)


def bragg_stage(cfg, data, root):
    import csv
    import py4DSTEM
    import matplotlib.pyplot as plt
    out = root / "P3_bragg"
    out.mkdir(parents=True, exist_ok=True)
    c = cfg["bragg"]
    # ROI must be explicitly selected from corrected images and spot checks.
    if "roi" not in c:
        raise ValueError("Set bragg.roi=[y0,y1,x0,x1] after reviewing corrected ADF and DPs")
    y0, y1, x0, x1 = c["roi"]
    if (y1-y0, x1-x0) != tuple(c["roi_size"]) or not (0 <= y0 < y1 <= data.shape[0] and 0 <= x0 < x1 <= data.shape[1]):
        raise ValueError("Invalid ROI bounds or size")
    roi = np.asarray(data[y0:y1, x0:x1].compute(), dtype=np.uint16)
    dc = py4DSTEM.DataCube(data=roi)
    signature = str(inspect.signature(dc.find_Bragg_disks))
    model = np.load(root / "P1_origin/origin_model.npz")
    rng = np.random.default_rng(cfg["seed"])
    indices = np.sort(rng.choice(roi.shape[0]*roi.shape[1], c["review_patterns"], replace=False))
    sy, sx = np.unravel_index(indices, roi.shape[:2])
    rows, cache = [], {}
    params = list(itertools.product(c["min_relative_intensity_candidates"],
                                   c["min_peak_spacing_candidates"], c["edge_boundary_candidates"]))
    common = dict(**raw_peak_options(), corrPower=c["correlation_power"], sigma=1,
                  subpixel=c["subpixel"], maxNumPeaks=70)
    for j, (intensity, spacing, edge) in enumerate(params):
        detected = dc.find_Bragg_disks(data=(sy, sx), minRelativeIntensity=intensity,
            minPeakSpacing=spacing, edgeBoundary=edge, **common)
        for i, peaks in enumerate(detected):
            p = peaks.data
            qx, qy = export_peak_coordinates(p)
            cy, cx = model["origin_y"][y0+sy[i], x0+sx[i]], model["origin_x"][y0+sy[i], x0+sx[i]]
            r = np.hypot(qx-cx, qy-cy)
            cache[f"p{j}_dp{i}"] = p
            rows.append(dict(parameter_id=j, scan_y=int(y0+sy[i]), scan_x=int(x0+sx[i]),
                min_relative_intensity=intensity, min_peak_spacing=spacing, edge_boundary=edge,
                peak_count=len(p), central_peak_count=int((r < 8).sum()),
                noncentral_peak_count=int((r >= 8).sum()),
                max_peak_intensity=float(p["intensity"].max()) if len(p) else 0))
        print(f"Bragg parameter sweep {j+1}/{len(params)}", flush=True)
    with (out / "bragg_param_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out / "sweep_peaks.npz", **cache, local_scan_y=sy, local_scan_x=sx)
    # Keep a documented baseline, not a claim of optimized detection quality.
    selected = (0.05, 8, 10)
    j = params.index(selected)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for i, ax in enumerate(axes.flat):
        p = cache[f"p{j}_dp{i}"]; qx, qy = export_peak_coordinates(p)
        ax.imshow(np.log1p(roi[sy[i], sx[i]]), cmap="gray")
        ax.scatter(qx, qy, facecolors="none", edgecolors="red", s=35)
        ax.set_title(f"({y0+sy[i]}, {x0+sx[i]}) n={len(p)}")
    fig.tight_layout(); fig.savefig(out / "bragg_overlays_01.png", dpi=140); plt.close(fig)
    result = dc.find_Bragg_disks(minRelativeIntensity=selected[0], minPeakSpacing=selected[1],
                               edgeBoundary=selected[2], **common)
    records, counts = [], np.zeros(roi.shape[:2], dtype=int)
    for iy, ix in np.ndindex(roi.shape[:2]):
        peaks = result.raw[iy, ix].data
        qx, qy = export_peak_coordinates(peaks)
        counts[iy, ix] = len(peaks)
        for k in range(len(peaks)):
            records.append((y0+iy, x0+ix, qx[k], qy[k], peaks["intensity"][k]))
    dtype = [("scan_y", "i4"), ("scan_x", "i4"), ("qx_raw_px", "f8"), ("qy_raw_px", "f8"), ("peak_intensity", "f8")]
    np.savez_compressed(out / "bragg_raw_roi_01.npz", peaks=np.array(records, dtype=dtype), roi=c["roi"],
        parameters=selected, units="detector_px", origin_model_sha256=digest(root / "P1_origin/origin_model.npz"))
    np.save(out / "peak_count_map_01.npy", counts)
    plot_grid(out / "peak_count_map_01.png", [counts], ["Raw peak count (includes central beam)"], columns=1)
    with (out / "roi_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["roi_id", "y0", "y1", "x0", "x1", "selection_reason"])
        writer.writerow(["01", y0, y1, x0, x1, c.get("roi_selection_reason", "See review report")])
    write_json(out / "bragg_qc.json", dict(py4dstem_version=py4DSTEM.__version__, api_signature=signature,
        roi=c["roi"], sweep_parameter_count=len(params), review_pattern_count=len(indices),
        full_roi_patterns=int(np.prod(roi.shape[:2])), raw_peak_count=len(records), baseline_parameters=selected,
        baseline_not_optimized=True, template="None: raw maxima baseline",
        template_none_ifft_workaround=raw_peak_options()["filter_function"] is not None,
        template_comparison="deferred: no independently verified vacuum/direct-beam template",
        gate3_passed=False, gate3_status="First ROI sweep complete; multi-ROI and expert false-positive/missed-peak review pending",
        input=cfg["input"]["path"], config=cfg, units="px; uncalibrated"))


def main(stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mib_3-1.yaml")
    parser.add_argument("--output")
    parser.add_argument("--stage", choices=["baseline", "origin", "virtual", "bragg", "all"], default=stage or "all")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.output:
        cfg["output"]["root"] = args.output
    root = Path(cfg["output"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    data, metadata = read_data(cfg)
    if args.stage in ["baseline", "origin", "all"]:
        baseline(cfg, data, metadata, root)
    if args.stage in ["origin", "all"]:
        origin_stage(cfg, data, root)
    if args.stage in ["virtual", "bragg", "all"]:
        qc = json.loads((root / "P1_origin/origin_qc.json").read_text(encoding="utf-8"))
        if not qc["gate1_passed"]:
            raise RuntimeError("Gate 1 failed; inspect origin QC before virtual/Bragg analysis")
        if qc["origin_model_sha256"] != digest(root / "P1_origin/origin_model.npz"):
            raise ValueError("Origin model does not match its QC record")
        with np.load(root / "P1_origin/origin_model.npz") as model:
            previous = json.loads(str(model["config_json"]))
        for section in ["input", "origin", "seed"]:
            if cfg[section] != previous[section]:
                raise ValueError(f"Changed {section}; rerun P1 before continuing")
        meta = json.loads((root / "P0_baseline/metadata.json").read_text(encoding="utf-8"))
        current = Path(cfg["input"]["path"]).stat()
        if current.st_size != meta["size_bytes"] or current.st_mtime_ns != meta["mtime_ns"]:
            raise ValueError("Input file changed since baseline capture")
        if args.stage in ["virtual", "all"]:
            virtual_stage(cfg, data, root)
        if args.stage in ["bragg", "all"]:
            bragg_stage(cfg, data, root)


if __name__ == "__main__":
    main()
