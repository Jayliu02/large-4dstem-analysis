"""Low-threshold peak refinement with synthetic selection and new real-data audits."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .mib_round1 import digest, export_peak_coordinates, read_data, write_json
from .mib_round2 import (
    check_baseline, correct_coordinates, csv_write, detect, fingerprint_tree,
    load_json, local_quality, require, summary, verify_output as verify_round2,
)


DTYPE = np.dtype([
    ("scan_y", "i4"), ("scan_x", "i4"), ("roi_id", "U16"),
    ("candidate_x", "f8"), ("candidate_y", "f8"),
    ("qx_raw_px", "f8"), ("qy_raw_px", "f8"), ("qx_rel_px", "f8"), ("qy_rel_px", "f8"),
    ("source_flags", "u1"), ("raw_score", "f8"), ("template_score", "f8"),
    ("signal_counts", "f8"), ("snr_proxy", "f8"), ("width_px", "f8"),
    ("refine_shift_px", "f8"), ("single_pixel_fraction", "f8"),
    ("rejection_flags", "u2"), ("accepted", "?"),
])


def match_points(a, b, radius):
    """Maximum-cardinality bounded one-to-one matching, then minimum distance."""
    a, b = np.asarray(a).reshape(-1, 2), np.asarray(b).reshape(-1, 2)
    if len(a) == 0 or len(b) == 0:
        return np.empty((0, 2), dtype=int)
    d = cdist(a, b)
    cost = np.where(d <= radius, d, (min(len(a), len(b))+1)*(radius+1))
    i, j = linear_sum_assignment(cost)
    keep = d[i, j] <= radius
    return np.column_stack([i[keep], j[keep]])


def coordinates(p):
    return np.column_stack([p["qx_raw_px"], p["qy_raw_px"]])


def candidate_union(raw, template, radius):
    """Prefer template coordinates for matched candidates; preserve both native scores."""
    rxy = np.column_stack(export_peak_coordinates(raw))
    txy = np.column_stack(export_peak_coordinates(template))
    pairs = match_points(rxy, txy, radius)
    by_template = {int(j): int(i) for i, j in pairs}
    matched_raw = set(pairs[:, 0])
    rows = []
    for j, (x, y) in enumerate(txy):
        i = by_template.get(j)
        rows.append((x, y, 3 if i is not None else 2,
                     float(raw["intensity"][i]) if i is not None else np.nan, float(template["intensity"][j])))
    for i, (x, y) in enumerate(rxy):
        if i not in matched_raw:
            rows.append((x, y, 1, float(raw["intensity"][i]), np.nan))
    return rows


def refine_candidate(dp, x, y, cfg):
    c = cfg["detection"]
    start_x, start_y = x, y
    radius = c["refine_radius_px"]
    flag = 0
    width, fraction = 0., 1.
    # Iterations use raw pixels, never smoothed or correlation intensities.
    for _ in range(3):
        ix, iy = int(round(x)), int(round(y))
        ya, yb = max(0, iy-radius), min(dp.shape[0], iy+radius+1)
        xa, xb = max(0, ix-radius), min(dp.shape[1], ix+radius+1)
        patch = np.asarray(dp[ya:yb, xa:xb], dtype=float)
        yy, xx = np.mgrid[ya:yb, xa:xb]
        border = np.concatenate([patch[0], patch[-1], patch[1:-1, 0], patch[1:-1, -1]])
        weights = np.maximum(patch-np.median(border), 0)
        mass = weights.sum()
        if mass <= 0:
            flag |= 64
            break
        x, y = float((weights*xx).sum()/mass), float((weights*yy).sum()/mass)
        width = float(np.sqrt((weights*((xx-x)**2+(yy-y)**2)).sum()/mass))
        fraction = float(weights.max()/mass)
    movement = float(np.hypot(x-start_x, y-start_y))
    signal, snr, _ = local_quality(dp, x, y, 0)
    if not c["width_min_px"] <= width <= c["width_max_px"]:
        flag |= 2
    if movement > c["max_refine_shift_px"]:
        flag |= 4
    if fraction > c["max_single_pixel_fraction"]:
        flag |= 8
    if min(x, y, dp.shape[1]-1-x, dp.shape[0]-1-y) < radius:
        flag |= 16
    return x, y, signal, snr, width, movement, fraction, flag


def candidates(dp, method, kernel, cfg, round2_cfg):
    import py4DSTEM
    dc = py4DSTEM.DataCube(data=np.asarray(dp, float)[None, None])
    detected = {}
    for branch in (["raw", "template"] if method == "hybrid" else [method]):
        detected[branch] = detect(dc, np.array([[0, 0]]), branch, kernel, round2_cfg,
                                   relative=cfg["detection"]["relative_intensity"])[0].data
    empty = np.empty(0, dtype=[("qx", "f8"), ("qy", "f8"), ("intensity", "f8")])
    rows = candidate_union(detected.get("raw", empty), detected.get("template", empty), cfg["detection"]["merge_radius_px"])
    out = np.zeros(len(rows), dtype=DTYPE)
    for i, (x, y, flags, rs, ts) in enumerate(rows):
        out["candidate_x"][i], out["candidate_y"][i] = x, y
        out["source_flags"][i], out["raw_score"][i], out["template_score"][i] = flags, rs, ts
        values = refine_candidate(dp, x, y, cfg)
        for name, value in zip(["qx_raw_px", "qy_raw_px", "signal_counts", "snr_proxy", "width_px",
                                "refine_shift_px", "single_pixel_fraction", "rejection_flags"], values):
            out[name][i] = value
    return out


def accept_candidates(candidates_, snr_min, radius):
    out = candidates_.copy()
    # Clear only decisions made here, retaining geometric/shape rejection bits.
    out["rejection_flags"] &= np.uint16(65535 ^ (1 | 32))
    out["rejection_flags"][out["snr_proxy"] < snr_min] |= 1
    accepted = []
    for i in np.argsort(-out["snr_proxy"], kind="stable"):
        if out["rejection_flags"][i]:
            continue
        if accepted and np.min(np.linalg.norm(coordinates(out[accepted])-coordinates(out[i:i+1]), axis=1)) < radius:
            out["rejection_flags"][i] |= 32
        else:
            accepted.append(i)
    out["accepted"] = out["rejection_flags"] == 0
    return out


def synthetic_patterns(cfg, shape, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape)
    nsignal, nnull = cfg["synthetic"]["signal_patterns"], cfg["synthetic"]["null_patterns"]
    for index in range(nsignal+nnull):
        center = np.array([shape[1]*.49, shape[0]*.51]) + rng.uniform(-6, 6, 2)
        mean = np.full(shape, rng.uniform(2, 8))
        mean += 14*np.exp(-((xx-center[0])**2+(yy-center[1])**2)/(2*35**2))
        truth, weak = [], []
        if index < nsignal:
            anchors = [(0.49, .51), (.25, .26), (.68, .23), (.28, .74), (.73, .71), (.76, .46)]
            # Alternating isolated peaks and a pair 10-14 px apart.
            if index % 3 == 0:
                anchors[-1] = (.73, .71+12/shape[0])
            for k, (fx, fy) in enumerate(anchors):
                x, y = np.array([fx*shape[1], fy*shape[0]]) + rng.uniform(-1.5, 1.5, 2)
                amplitude = rng.uniform(600, 900) if k == 0 else rng.uniform(14, 85)
                sigma_x, sigma_y = rng.uniform(1.5, 3.0, 2)
                exponent = ((xx-x)/sigma_x)**2+((yy-y)/sigma_y)**2
                # Include flat-topped, approximately disk-shaped spots in one third of patterns.
                mean += amplitude*np.exp(-.5*(exponent if index % 3 else (exponent/2)**2))
                truth.append((x, y)); weak.append(amplitude < 35)
        counts = rng.poisson(mean).astype(np.uint16)
        # Isolated bright-pixel contamination is not a true Bragg peak.
        counts[18+index % 7, shape[1]-24] += 200
        yield index, counts, np.array(truth).reshape(-1, 2), np.array(weak, dtype=bool)


def synthetic_scores(cfg, cfg2, kernel, seed, choices):
    accum = {choice: dict(tp=0, fp=0, fn=0, weak_tp=0, weak_total=0, null_fp=0, errors=[]) for choice in choices}
    for index, dp, truth, weak in synthetic_patterns(cfg, kernel.shape, seed):
        cache = {}
        for method in dict.fromkeys(m for m, _ in choices):
            cache[method] = candidates(dp, method, kernel, cfg, cfg2)
        for choice in choices:
            method, snr_min = choice
            p = accept_candidates(cache[method], snr_min, cfg["detection"]["merge_radius_px"])
            found = coordinates(p[p["accepted"]])
            pairs = match_points(truth, found, cfg["synthetic"]["match_radius_px"])
            a = accum[choice]
            a["tp"] += len(pairs); a["fp"] += len(found)-len(pairs); a["fn"] += len(truth)-len(pairs)
            a["weak_total"] += int(weak.sum())
            a["weak_tp"] += int(weak[pairs[:, 0]].sum())
            if len(truth) == 0:
                a["null_fp"] += len(found)
            a["errors"].extend(np.linalg.norm(truth[pairs[:, 0]]-found[pairs[:, 1]], axis=1))
        if (index+1) % 8 == 0:
            print(f"Synthetic seed {seed}: {index+1} patterns", flush=True)
    results = []
    for (method, snr_min), a in accum.items():
        precision = a["tp"]/max(a["tp"]+a["fp"], 1)
        recall = a["tp"]/max(a["tp"]+a["fn"], 1)
        results.append(dict(method=method, snr_min=snr_min, precision=precision, recall=recall,
            f1=2*precision*recall/max(precision+recall, 1e-12), true_positive=a["tp"], false_positive=a["fp"],
            false_negative=a["fn"], null_false_positive=a["null_fp"],
            weak_recall=a["weak_tp"]/max(a["weak_total"], 1), weak_truth_count=a["weak_total"],
            localization_error_px=summary(a["errors"])))
    return results


def choose_parameters(cfg, cfg2, kernel, out):
    choices = [(m, s) for m in cfg["detection"]["methods"] for s in cfg["detection"]["snr_candidates"]]
    training = synthetic_scores(cfg, cfg2, kernel, cfg["synthetic"]["training_seed"], choices)
    eligible = [r for r in training if r["precision"] >= cfg["synthetic"]["precision_min"]]
    selected = max(eligible or training, key=lambda r: (r["recall"] if eligible else r["f1"], r["precision"]))
    choice = dict(method=selected["method"], snr_min=selected["snr_min"],
                  selection_rule="Maximum training recall subject to precision >= 0.95; otherwise best F1, gate fails",
                  training_constraint_passed=bool(eligible))
    # Freeze the choice before independent synthetic and real-data validation.
    write_json(out / "selected_parameters.json", choice)
    validation = synthetic_scores(cfg, cfg2, kernel, cfg["synthetic"]["validation_seed"],
                                   [(choice["method"], choice["snr_min"])])[0]
    passed = bool(eligible and validation["precision"] >= cfg["synthetic"]["precision_min"] and
                  validation["recall"] >= cfg["synthetic"]["recall_min"])
    write_json(out / "synthetic_qc.json", dict(training=training, validation=validation, gate_passed=passed,
        training_seed=cfg["synthetic"]["training_seed"], validation_seed=cfg["synthetic"]["validation_seed"],
        interpretation="Known simulated peaks only; not experimental precision/recall."))
    return choice, passed


def new_positions(cfg, cfg2, roi, number):
    """New audit positions exclude persisted sampling from both previous rounds."""
    r1, r2 = Path(cfg2["baseline_root"]), Path(cfg["round2_root"])
    excluded = set()
    for name in ["origin_measured_sampled.npz", "origin_validation.npz"]:
        with np.load(r1 / "P1_origin" / name) as a:
            excluded.update(zip(a["scan_y"].tolist(), a["scan_x"].tolist()))
    with np.load(r1 / "P0_baseline/dp_spotcheck.npz") as a:
        excluded.update(map(tuple, a["positions"]))
    with np.load(r1 / "P3_bragg/sweep_peaks.npz") as a:
        b = np.load(r1 / "P3_bragg/bragg_raw_roi_01.npz")["roi"]
        excluded.update(zip(a["local_scan_y"]+b[0], a["local_scan_x"]+b[2]))
    for previous_roi in cfg2["rois"]:
        y0, _, x0, _ = previous_roi["bounds"]
        a = np.load(r2 / f"P3_bragg/sampling_{previous_roi['id']}.npz")
        for group in ["tuning", "holdout"]:
            excluded.update(map(tuple, a[group]+[y0, x0]))
        # Preliminary visual spot checks recorded in round-two planning.
        excluded.update((y0+y, x0+x) for y, x in [(4, 4), (4, 27), (16, 16), (27, 4), (27, 27)])
    excluded.update((y, x) for y in [34, 42, 50, 60] for x in [50, 62, 77])
    excluded.update(map(tuple, np.load(r2 / "template/empirical_template.npz")["source_positions"]))
    count = cfg["validation"]["points_per_roi"]
    require(count >= 20 and count % 4 == 0, "Holdout count must be >=20 and divisible by four")
    y0, y1, x0, x1 = roi["bounds"]
    ym, xm = (y0+y1)//2, (x0+x1)//2
    rng = np.random.default_rng(cfg["seed"]+number)
    positions = []
    for ya, yb in [(y0, ym), (ym, y1)]:
        for xa, xb in [(x0, xm), (xm, x1)]:
            pool = [(y, x) for y in range(ya, yb) for x in range(xa, xb) if (y, x) not in excluded]
            require(len(pool) >= count//4, "Not enough unseen points in a quadrant")
            positions.extend(pool[i] for i in rng.choice(len(pool), count//4, replace=False))
    return np.array(positions, dtype=int), np.array(sorted(excluded), dtype=int)


def process_dp(dp, sy, sx, roi_id, choice, kernel, cfg, cfg2, model):
    p = accept_candidates(candidates(dp, choice["method"], kernel, cfg, cfg2), choice["snr_min"],
                          cfg["detection"]["merge_radius_px"])
    p["scan_y"], p["scan_x"], p["roi_id"] = sy, sx, roi_id
    p["qx_rel_px"], p["qy_rel_px"] = correct_coordinates(p["qx_raw_px"], p["qy_raw_px"], int(sy), int(sx), model)
    return p


def split_counts(dp, seed):
    """Conditional binomial thinning, not a second experimental acquisition."""
    a = np.random.default_rng(seed).binomial(np.asarray(dp, dtype=np.int64), .5).astype(np.uint16)
    return a, np.asarray(dp, dtype=np.uint16)-a


def run_roi(cfg, cfg2, data, model, kernel, choice, roi, number, out):
    import matplotlib.pyplot as plt
    import py4DSTEM
    y0, y1, x0, x1 = roi["bounds"]
    frames = np.asarray(data[y0:y1, x0:x1].compute(), dtype=np.uint16)
    positions, excluded = new_positions(cfg, cfg2, roi, number)
    np.savez_compressed(out / f"sampling_{roi['id']}.npz", positions=positions, excluded=excluded, bounds=roi["bounds"])
    full, count_map = {}, np.zeros(frames.shape[:2], dtype=int)
    for y, x in np.ndindex(frames.shape[:2]):
        p = process_dp(frames[y, x], y+y0, x+x0, roi["id"], choice, kernel, cfg, cfg2, model)
        full[y+y0, x+x0] = p
        count_map[y, x] = int(p["accepted"].sum())
        if x == frames.shape[1]-1 and (y+1) % 8 == 0:
            print(f"{roi['id']}: full extraction {y+1}/{frames.shape[0]} rows", flush=True)
    all_candidates = np.concatenate(list(full.values()))
    accepted = all_candidates[all_candidates["accepted"]]
    np.savez_compressed(out / f"candidates_{roi['id']}.npz", peaks=all_candidates, units="detector_px", status="provisional")
    np.savez_compressed(out / f"peaks_corrected_{roi['id']}.npz", peaks=accepted, bounds=roi["bounds"],
                        status="provisional", units="detector_px")
    np.save(out / f"peak_count_{roi['id']}.npy", count_map)
    radius = cfg["validation"]["split_match_radius_px"]
    metrics, review, cache, baseline_cache = [], [], {}, {}
    for i, (sy, sx) in enumerate(positions):
        dp = frames[sy-y0, sx-x0]
        p = full[int(sy), int(sx)]
        p = p[p["accepted"]]
        dc = py4DSTEM.DataCube(data=dp.astype(float)[None, None])
        old = detect(dc, np.array([[0, 0]]), "template", kernel, cfg2)[0].data
        old_xy = np.column_stack(export_peak_coordinates(old))
        baseline_cache[i] = old_xy
        matched = match_points(old_xy, coordinates(p), radius)
        added = np.ones(len(p), dtype=bool); added[matched[:, 1]] = False
        halves = split_counts(dp, cfg["seed"]+number*10000+i)
        half_points = []
        for j, half in enumerate(halves):
            hp = process_dp(half, int(sy), int(sx), roi["id"], choice, kernel, cfg, cfg2, model)
            half_points.append(coordinates(hp[hp["accepted"]]))
            cache[f"point{i}_half{j}"] = hp
        reproduced = []
        for hp in half_points:
            pairs = match_points(coordinates(p), hp, radius)
            reproduced.append(set(pairs[:, 0]))
        both = reproduced[0] & reproduced[1]
        metrics.append(dict(roi_id=roi["id"], scan_y=int(sy), scan_x=int(sx),
            baseline_peak_count=len(old), selected_peak_count=len(p), matched_to_baseline=len(matched),
            added_count=int(added.sum()), lost_baseline_count=len(old)-len(matched),
            supported_both_halves=len(both), added_supported_both_halves=sum(bool(added[k]) for k in both),
            split_match_fraction=len(both)/max(len(p), 1)))
        review.append(dict(roi_id=roi["id"], scan_y=int(sy), scan_x=int(sx), reviewer="",
                           false_peaks="", missed_major_peaks="", verdict="pending"))
    np.savez_compressed(out / f"split_candidates_{roi['id']}.npz", **cache)
    np.savez_compressed(out / f"holdout_dp_{roi['id']}.npz", dp=frames[positions[:, 0]-y0, positions[:, 1]-x0], positions=positions)
    csv_write(out / f"holdout_metrics_{roi['id']}.csv", metrics)
    if not (out / f"review_{roi['id']}.csv").exists():
        csv_write(out / f"review_{roi['id']}.csv", review)
    for page, start in enumerate(range(0, len(positions), 6), 1):
        fig, axes = plt.subplots(6, 2, figsize=(10, 24))
        for row, i in enumerate(range(start, min(start+6, len(positions)))):
            sy, sx = positions[i]
            p = full[int(sy), int(sx)]; p = p[p["accepted"]]
            for col, (points, title) in enumerate([(baseline_cache[i], "Round 2 template 0.03"), (coordinates(p), "Round 3 selected")]):
                ax = axes[row, col]
                ax.imshow(np.log1p(frames[sy-y0, sx-x0].astype(float)), cmap="gray")
                ax.scatter(points[:, 0], points[:, 1], s=38, facecolors="none", edgecolors="lime")
                ax.set_title(f"{title} ({sy},{sx}) n={len(points)}", fontsize=9)
        fig.tight_layout(); fig.savefig(out / f"holdout_{roi['id']}_{page}.png", dpi=130); plt.close(fig)
    center_errors, center_map = [], np.full(frames.shape[:2], np.nan)
    for (sy, sx), p in full.items():
        p = p[p["accepted"]]
        r = np.hypot(p["qx_rel_px"], p["qy_rel_px"])
        if (r < 8).any():
            value = float(r.min()); center_errors.append(value); center_map[sy-y0, sx-x0] = value
    # Independently read and process an offset 8x8 sub-ROI, including the full quality decision.
    sub = np.asarray(data[y0+8:y0+16, x0+8:x0+16].compute(), dtype=np.uint16)
    overlap_error, overlap_count = 0., 0
    for y, x in np.ndindex(sub.shape[:2]):
        p = process_dp(sub[y, x], y0+y+8, x0+x+8, roi["id"], choice, kernel, cfg, cfg2, model)
        old = full[y0+y+8, x0+x+8]
        require(len(p) == len(old) and np.array_equal(p["accepted"], old["accepted"]), "Overlap decisions differ")
        if len(p):
            for field in ["qx_raw_px", "qy_raw_px", "qx_rel_px", "qy_rel_px"]:
                overlap_error = max(overlap_error, float(abs(p[field]-old[field]).max()))
        overlap_count += 1
    np.save(out / f"central_residual_{roi['id']}.npy", center_map)
    from .mib_round1 import plot_grid
    plot_grid(out / f"maps_{roi['id']}.png", [count_map, center_map], ["Accepted peak count", "Central residual (px)"], columns=2)
    central = summary(center_errors)
    g = cfg["gates"]
    passed = bool(central["count"]/1024 >= g["central_coverage_min"] and central["median"] is not None and
                  central["median"] <= g["central_median_max_px"] and central["p95"] <= g["central_p95_max_px"] and
                  overlap_error <= g["overlap_max_error_px"])
    result = dict(candidate_count=len(all_candidates), accepted_count=len(accepted), central=central,
        central_coverage=central["count"]/1024, geometry_gate_passed=passed,
        overlap_patterns=overlap_count, overlap_max_error_px=overlap_error, holdout_patterns=len(positions),
        baseline_holdout_peaks=sum(r["baseline_peak_count"] for r in metrics),
        selected_holdout_peaks=sum(r["selected_peak_count"] for r in metrics),
        added_holdout_peaks=sum(r["added_count"] for r in metrics),
        lost_baseline_peaks=sum(r["lost_baseline_count"] for r in metrics),
        added_supported_both_halves=sum(r["added_supported_both_halves"] for r in metrics),
        all_supported_both_halves=sum(r["supported_both_halves"] for r in metrics),
        split_note="Conditional binomial thinning of one acquisition; same SNR threshold at half dose. Not experimental truth.",
        rejected_reason_counts={str(bit): int(((all_candidates["rejection_flags"] & bit) != 0).sum()) for bit in [1, 2, 4, 8, 16, 32, 64]})
    write_json(out / f"qc_{roi['id']}.json", result)
    print(f"{roi['id']}: holdout, half-count and overlap diagnostics complete", flush=True)
    return accepted, result


def write_calibration(cfg, out):
    c = cfg["calibration"]
    require(c["status"] == "uncalibrated" and c["g_scale_inv_angstrom_per_px"] is None and
            not c["independent_references"], "This workflow only records uncalibrated status; supplied calibration needs independent validation")
    out.mkdir(exist_ok=True)
    write_json(out / "calibration.json", dict(**c, gate5_passed=False, output_coordinate_units="detector_px",
        missing=["confirmed acceleration voltage", "independent known d-spacing/pixel radius references",
                 "camera length units and detector geometry", "physical scan order"],
        filename_camera_length_nominal=110, filename_camera_length_usable_as_scale=False))
    (out / "missing_information.md").write_text(
        "# P5 未标定\n\n用户于 2026-10-09 确认：暂无标定。\n\n"
        "需要独立已知 d 间距与对应像素半径（包括来源、测量误差、训练/验证分组），"
        "并补充确认的加速电压、相机长度单位/几何与扫描方向。名义 CL 110 不足以建立可靠尺度。"
        "g=1/d 的尺度目前为 null，所有衍射坐标保持 detector_px，Gate 5 未通过。\n", encoding="utf-8")


def report(out, choice, synthetic_passed, results):
    lines = ["# MIB 第三轮执行报告", "", "完成低阈值检出、局部细化、合成参数选择、新复核集和 ROI 坐标诊断。", "",
        f"预先冻结的选择：`{choice['method']}`，候选相对阈值 0.01，局部 SNR ≥{choice['snr_min']}。独立合成验收通过：{synthetic_passed}。", "",
        "| ROI | 接受峰数 | 中央束覆盖 | 中位/P95误差 px | 新复核点基线/本轮峰数 | 新增峰两半均支持/新增峰 | 重读差 px |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, q in results.items():
        c = q["central"]
        error_text = f"{c['median']:.4f}/{c['p95']:.4f}" if c["median"] is not None else "N/A"
        lines.append(f"| {name} | {q['accepted_count']} | {q['central_coverage']:.2%} | {error_text} | "
            f"{q['baseline_holdout_peaks']}/{q['selected_holdout_peaks']} | "
            f"{q['added_supported_both_halves']}/{q['added_holdout_peaks']} | {q['overlap_max_error_px']:.3g} |")
    s = load_json(out / "synthetic_qc.json")["validation"]
    lines.extend(["", f"独立合成集 precision={s['precision']:.4f}、recall={s['recall']:.4f}、"
        f"弱峰 recall={s['weak_recall']:.4f}、纯背景假峰={s['null_false_positive']}；"
        f"定位 P95={s['localization_error_px']['p95']:.4f} px。这些是模拟数据指标，不是实验准确率。", "",
        "新复核点排除前两轮已有抽样、调参、复核及模板来源点。候选选择和阈值不根据这些新位置调整。"
        "两半支持是一次计数的条件二项拆分，使用相同 SNR 门槛，弱峰在半剂量下可能消失；"
        "两半支持也不能证明峰必为真实 Bragg 反射。新增与丢失数量由 2 px 一对一匹配定义。", "",
        "原始/相对坐标、拒绝候选及原因均在 `P3_P4` 保存。所有峰表仍为 provisional；"
        "`review_*.csv` 留待领域人员填写，未自动批准 Gate 3。用户确认暂无标定，P5 明确为 uncalibrated，未产生物理 q/相/应变。", "",
        "复现：`./.conda-phase/python.exe scripts/05_mib_round3.py`；校验加 `--verify`。"])
    (out / "report_zh.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def verify(out):
    manifest = load_json(out / "run_manifest.json")
    require(load_json(out / "status.json")["status"] == "complete", "Third round not complete")
    for name, sha in manifest["artifacts"].items():
        require(digest(out / name) == sha, f"Third-round artifact changed: {name}")
    for root, fingerprints in manifest["previous_outputs"].items():
        require(fingerprint_tree(Path(root)) == fingerprints, f"Previous round changed: {root}")
    cfg2 = manifest["round2_config"]
    meta = load_json(Path(cfg2["baseline_root"]) / "P0_baseline/metadata.json")
    st = Path(meta["input"]).stat()
    require(st.st_size == meta["size_bytes"] and st.st_mtime_ns == meta["mtime_ns"], "Raw MIB changed")
    model = np.load(Path(cfg2["baseline_root"]) / "P1_origin/origin_model.npz")
    merged = np.load(out / "P3_P4/merged_peaks.npz")["peaks"]
    expected_count = 0
    for roi in cfg2["rois"]:
        sets = np.load(out / f"P3_P4/sampling_{roi['id']}.npz")
        require(len(sets["positions"]) >= 20 and not set(map(tuple, sets["positions"])).intersection(map(tuple, sets["excluded"])),
                "New holdout overlap or insufficient count")
        y0, y1, x0, x1 = roi["bounds"]
        p = np.load(out / f"P3_P4/peaks_corrected_{roi['id']}.npz")["peaks"]
        all_p = np.load(out / f"P3_P4/candidates_{roi['id']}.npz")["peaks"]
        expected_count += len(p)
        require(p["accepted"].all() and (p["rejection_flags"] == 0).all(), "Accepted peak carries rejection flags")
        require(len(p) == all_p["accepted"].sum(), "Accepted/candidate count mismatch")
        require(((p["scan_y"] >= y0) & (p["scan_y"] < y1) & (p["scan_x"] >= x0) & (p["scan_x"] < x1)).all(), "Peak outside ROI")
        a = merged[merged["roi_id"] == roi["id"]]
        require(len(a) == len(p), "Merged ROI size mismatch")
        for field in DTYPE.names:
            if p.dtype[field].kind == "f":
                require(np.array_equal(p[field], a[field], equal_nan=True), "Merged floating fields differ")
            else:
                require(np.array_equal(p[field], a[field]), "Merged fields differ")
    require(len(merged) == expected_count, "Merged peak total mismatch")
    rx, ry = correct_coordinates(merged["qx_raw_px"], merged["qy_raw_px"], merged["scan_y"], merged["scan_x"], model)
    require(np.allclose(rx, merged["qx_rel_px"], atol=1e-12, rtol=0) and
            np.allclose(ry, merged["qy_rel_px"], atol=1e-12, rtol=0), "Coordinate correction mismatch")
    calibration = load_json(out / "P5_calibration/calibration.json")
    require(calibration["status"] == "uncalibrated" and calibration["g_scale_inv_angstrom_per_px"] is None,
            "Unexpected physical scale")
    print(f"Verified {len(merged)} provisional peaks; independent samples, coordinate contract and both previous rounds intact.", flush=True)


def main():
    import csv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mib_round3.yaml")
    parser.add_argument("--output")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.output:
        cfg["output"] = args.output
    out = Path(cfg["output"])
    if args.verify:
        verify(out)
        return
    cfg2 = yaml.safe_load(Path(cfg["round2_config"]).read_text(encoding="utf-8"))
    require(cfg2 == load_json(Path(cfg["round2_root"]) / "run_manifest.json")["config"], "Second-round config differs from executed run")
    for root in [Path(cfg["round2_root"]), Path(cfg2["baseline_root"])]:
        require(root.resolve() != out.resolve() and root.resolve() not in out.resolve().parents and
                out.resolve() not in root.resolve().parents, "Output must be separate from previous rounds")
    verify_round2(Path(cfg["round2_root"]))
    bcfg, model, first_fingerprints = check_baseline(cfg2)
    previous = {cfg2["baseline_root"]: first_fingerprints, cfg["round2_root"]: fingerprint_tree(Path(cfg["round2_root"]))}
    if (out / "run_manifest.json").exists():
        require(load_json(out / "run_manifest.json")["config"] == cfg, "Changed config: use a new output directory")
    for review_path in (out / "P3_P4").glob("review_*.csv"):
        with review_path.open(encoding="utf-8-sig", newline="") as stream:
            require(all(r["verdict"] == "pending" and not r["reviewer"] and not r["false_peaks"] and
                        not r["missed_major_peaks"] for r in csv.DictReader(stream)), "Annotated review exists; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "status.json", dict(status="running"))
    try:
        write_json(out / "resolved_config.json", cfg)
        kernel = np.load(Path(cfg["round2_root"]) / "template/empirical_template.npz")["kernel"]
        choice, synthetic_passed = choose_parameters(cfg, cfg2, kernel, out)
        print(f"Frozen selection: {choice['method']} SNR >= {choice['snr_min']}; synthetic gate={synthetic_passed}", flush=True)
        data, _ = read_data(bcfg)
        pdir = out / "P3_P4"; pdir.mkdir(exist_ok=True)
        all_peaks, results = [], {}
        for number, roi in enumerate(cfg2["rois"]):
            p, results[roi["id"]] = run_roi(cfg, cfg2, data, model, kernel, choice, roi, number, pdir)
            all_peaks.append(p)
        np.savez_compressed(pdir / "merged_peaks.npz", peaks=np.concatenate(all_peaks), status="provisional", units="detector_px")
        write_json(pdir / "qc_summary.json", results)
        (pdir / "coordinates_spec.md").write_text(
            "# Third-round pixel schema\n\nscan_y/scan_x are global zero-based row/column indices; "
            "qx_raw_px/qy_raw_px are refined column/row detector coordinates. Relative coordinates subtract the "
            "unchanged first-round origin model. candidate_x/y preserve initial positions. Units: detector_px.\n\n"
            "source_flags: 1=raw maximum, 2=template maximum, 3=matched candidate from both. Native raw_score and "
            "template_score remain separate; absent scores are NaN, never photon counts. signal_counts is local "
            "background-subtracted radius-3 aperture counts, snr_proxy an approximate local Poisson support statistic.\n\n"
            "rejection_flags: 1=low SNR; 2=width outside range; 4=refinement moved too far; "
            "8=single-pixel-dominated; 16=local patch clipped; 32=duplicate after refinement; 64=no positive mass. "
            "accepted means no rejection bits, not a verified reflection. All candidates are retained. "
            "Half-count stability is diagnostic only and is not used to accept or reject full-dose peaks.\n",
            encoding="utf-8")
        write_calibration(cfg, out / "P5_calibration")
        report(out, choice, synthetic_passed, results)
        for root, snapshots in previous.items():
            require(fingerprint_tree(Path(root)) == snapshots, "Previous outputs changed during execution")
        artifacts = {name: sha for name, sha in fingerprint_tree(out).items()
                     if name not in ["run_manifest.json", "status.json"] and not Path(name).name.startswith("review_")}
        write_json(out / "run_manifest.json", dict(config=cfg, round2_config=cfg2, previous_outputs=previous,
            artifacts=artifacts, selected=choice, synthetic_gate_passed=synthetic_passed,
            geometry_gate_passed=all(q["geometry_gate_passed"] for q in results.values()),
            gate3_passed=False, gate5_passed=False, status="provisional",
            source_sha256={str(p): digest(p) for p in [Path(__file__), Path(__file__).with_name("mib_round1.py"),
                                                       Path(__file__).with_name("mib_round2.py")]}))
        write_json(out / "status.json", dict(status="complete"))
        verify(out)
    except Exception as exc:
        write_json(out / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    main()
