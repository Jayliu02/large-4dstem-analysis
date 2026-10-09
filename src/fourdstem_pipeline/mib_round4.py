"""Fixed-detector recovery on real backgrounds and provenance-bound human review."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import cdist

from .mib_round1 import digest, write_json
from .mib_round2 import csv_write, fingerprint_tree, load_json, require, summary
from .mib_round3 import coordinates, match_points, process_dp, verify as verify_round3


def select_targets(dp_shape, all_candidates, accepted, cfg, rng):
    c = cfg["injection"]
    margin = c["detector_margin_px"]
    low, high = [margin, margin], [dp_shape[1]-1-margin, dp_shape[0]-1-margin]
    points = rng.uniform(low, high, size=(4096, 2))
    initial = np.column_stack([all_candidates["candidate_x"], all_candidates["candidate_y"]])
    distances = cdist(points, initial).min(axis=1) if len(initial) else np.full(len(points), np.inf)
    available = np.flatnonzero(distances >= c["isolated_exclusion_px"])
    isolated = points[available[0]] if len(available) else None
    noncentral = accepted[np.hypot(accepted["qx_rel_px"], accepted["qy_rel_px"]) > 16]
    crowded = None
    for index in rng.permutation(len(noncentral)):
        origin = coordinates(noncentral[index:index+1])[0]
        for angle in rng.uniform(0, 2*np.pi, 24):
            target = origin+c["crowded_distance_px"]*np.array([np.cos(angle), np.sin(angle)])
            if (target >= low).all() and (target <= high).all() and np.linalg.norm(coordinates(accepted)-target, axis=1).min() > 3:
                crowded = target
                break
        if crowded is not None:
            break
    return dict(isolated=isolated, crowded=crowded)


def inject_counts(dp, target, shape, amplitude, empirical, sigma, seed):
    """Add a separately sampled nonnegative Poisson signal, never alter source counts."""
    x, y = map(float, target)
    radius = empirical.shape[0]//2
    ix, iy = int(round(x)), int(round(y))
    ya, yb, xa, xb = iy-radius, iy+radius+1, ix-radius, ix+radius+1
    require(ya >= 0 and xa >= 0 and yb <= dp.shape[0] and xb <= dp.shape[1], "Injection support leaves detector")
    yy, xx = np.mgrid[ya:yb, xa:xb]
    if shape == "empirical":
        signal = map_coordinates(empirical, [yy-y+radius, xx-x+radius], order=1, mode="constant")
    elif shape == "gaussian":
        signal = np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma**2))
    else:
        raise ValueError(f"Unknown injection shape: {shape}")
    require(amplitude >= 0 and signal.max() > 0, "Invalid injection amplitude/shape")
    expectation = signal*(amplitude/signal.max())
    increment = np.random.default_rng(seed).poisson(expectation).astype(np.uint32)
    augmented = np.asarray(dp, dtype=np.uint32).copy()
    augmented[ya:yb, xa:xb] += increment
    return augmented, increment, [ya, yb, xa, xb], float(expectation.sum())


def pattern_data(cfg):
    r3 = Path(cfg["round3_root"])
    manifest = load_json(r3 / "run_manifest.json")
    cfg3, cfg2 = manifest["config"], manifest["round2_config"]
    r2 = Path(cfg3["round2_root"])
    t = np.load(r2 / "template/empirical_template.npz")
    model = np.load(Path(cfg2["baseline_root"]) / "P1_origin/origin_model.npz")
    patterns = []
    for roi in cfg2["rois"]:
        rid = roi["id"]
        held = np.load(r3 / f"P3_P4/holdout_dp_{rid}.npz")
        candidates = np.load(r3 / f"P3_P4/candidates_{rid}.npz")["peaks"]
        split = np.load(r3 / f"P3_P4/split_candidates_{rid}.npz")
        for index, ((sy, sx), dp) in enumerate(zip(held["positions"], held["dp"])):
            all_p = candidates[(candidates["scan_y"] == sy) & (candidates["scan_x"] == sx)]
            peaks = all_p[all_p["accepted"]]
            supports = []
            for half in [0, 1]:
                hp = split[f"point{index}_half{half}"]
                pairs = match_points(coordinates(peaks), coordinates(hp[hp["accepted"]]), cfg3["validation"]["split_match_radius_px"])
                supports.append(set(map(int, pairs[:, 0])))
            patterns.append(dict(id=f"{rid}:{sy}:{sx}", roi_id=rid, scan_y=int(sy), scan_x=int(sx),
                                 dp=dp, candidates=all_p, peaks=peaks, both_halves=supports[0] & supports[1]))
    return patterns, cfg3, cfg2, load_json(r3 / "selected_parameters.json"), t["kernel"], t["positive"], model


def recovery_experiment(cfg, patterns, cfg3, cfg2, choice, kernel, empirical, model, out):
    rows, increments, locations = [], {}, []
    c = cfg["injection"]
    for number, pattern in enumerate(patterns):
        dp, peaks = pattern["dp"], pattern["peaks"]
        baseline = process_dp(dp, pattern["scan_y"], pattern["scan_x"], pattern["roi_id"], choice, kernel, cfg3, cfg2, model)
        baseline = baseline[baseline["accepted"]]
        require(len(baseline) == len(peaks) and np.allclose(coordinates(baseline), coordinates(peaks), atol=1e-12, rtol=0),
                "Frozen detector no longer reproduces third-round peaks")
        targets = select_targets(dp.shape, pattern["candidates"], peaks, cfg, np.random.default_rng(cfg["seed"]+number))
        for context, target in targets.items():
            if target is None:
                locations.append(dict(pattern_id=pattern["id"], context=context, available=False))
                continue
            prior_hit = len(match_points([target], coordinates(peaks), c["match_radius_px"])) > 0
            locations.append(dict(pattern_id=pattern["id"], context=context, available=True,
                                  target_x=float(target[0]), target_y=float(target[1]), original_hit=prior_hit))
            for shape in c["shapes"]:
                for amplitude in c["amplitudes"]:
                    case = len(rows)
                    seed = cfg["seed"]*10000+case
                    augmented, increment, bbox, expected = inject_counts(dp, target, shape, amplitude, empirical,
                                                                         c["gaussian_sigma_px"], seed)
                    found = process_dp(augmented, pattern["scan_y"], pattern["scan_x"], pattern["roi_id"],
                                       choice, kernel, cfg3, cfg2, model)
                    found = found[found["accepted"]]
                    pairs = match_points([target], coordinates(found), c["match_radius_px"])
                    hit = len(pairs) > 0
                    error = float(np.linalg.norm(coordinates(found)[pairs[0, 1]]-target)) if hit else None
                    rows.append(dict(case_id=case, pattern_id=pattern["id"], roi_id=pattern["roi_id"],
                        context=context, shape=shape, amplitude=amplitude, target_x=float(target[0]), target_y=float(target[1]),
                        seed=seed, expected_added_counts=expected, realized_added_counts=int(increment.sum()),
                        bbox=bbox, original_hit=prior_hit, recovered=bool(hit and not prior_hit),
                        localization_error_px=error, output_peak_count=len(found), original_peak_count=len(peaks)))
                    increments[f"case_{case}"] = increment
        if (number+1) % 6 == 0:
            print(f"Real-background injections: {number+1}/{len(patterns)} patterns, {len(rows)} cases", flush=True)
    csv_write(out / "injection_cases.csv", [{**row, "bbox": json.dumps(row["bbox"])} for row in rows])
    write_json(out / "injection_cases.json", rows)
    write_json(out / "target_locations.json", locations)
    np.savez_compressed(out / "added_counts.npz", **increments)
    groups = []
    for roi_id in dict.fromkeys(p["roi_id"] for p in patterns):
        for context in ["isolated", "crowded"]:
            for shape in c["shapes"]:
                for amplitude in c["amplitudes"]:
                    subset = [r for r in rows if (r["roi_id"], r["context"], r["shape"], r["amplitude"]) == (roi_id, context, shape, amplitude)]
                    eligible = [r for r in subset if not r["original_hit"]]
                    success = sum(r["recovered"] for r in eligible)
                    groups.append(dict(roi_id=roi_id, context=context, shape=shape, amplitude=amplitude,
                        trials=len(subset), eligible=len(eligible), recovered=success,
                        recovery_fraction=success/len(eligible) if eligible else None,
                        error_px=summary([r["localization_error_px"] for r in eligible if r["recovered"]])))
    strong = [g for g in groups if g["context"] == "isolated" and g["amplitude"] == c["strong_amplitude"]]
    passed = len(strong) == len(cfg["injection"]["shapes"])*len(cfg2["rois"]) and all(
        g["eligible"] == sum(p["roi_id"] == g["roi_id"] for p in patterns) and
        g["recovery_fraction"] >= c["strong_isolated_recovery_min"] for g in strong)
    qc = dict(groups=groups, total_cases=len(rows), available_targets=sum(x["available"] for x in locations),
              target_count=len(locations), diagnostic_gate_passed=bool(passed), threshold=c,
              interpretation="Recovery of specified injected shapes on these backgrounds; not experimental precision or phase identification.")
    write_json(out / "recovery_qc.json", qc)
    return qc


def review_bundle(patterns, source_hash):
    result = dict(schema_version=1, source_manifest_sha256=source_hash, patterns=[])
    for p in patterns:
        peaks = [dict(id=i, x=float(k["qx_raw_px"]), y=float(k["qy_raw_px"]),
                      snr=float(k["snr_proxy"]), signal_counts=float(k["signal_counts"]), both_halves=i in p["both_halves"])
                 for i, k in enumerate(p["peaks"])]
        result["patterns"].append(dict(id=p["id"], roi_id=p["roi_id"], scan_y=p["scan_y"], scan_x=p["scan_x"],
            width=int(p["dp"].shape[1]), height=int(p["dp"].shape[0]), peaks=peaks,
            dp_sha256=hashlib.sha256(p["dp"].tobytes()).hexdigest()))
    result["bundle_id"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result


def make_review_page(bundle, patterns, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .mib_review_ui import HTML
    page = json.loads(json.dumps(bundle))
    for record, pattern in zip(page["patterns"], patterns):
        for name, a in [("image_log", np.log1p(pattern["dp"].astype(float))), ("image_linear", pattern["dp"].astype(float))]:
            buf = io.BytesIO()
            plt.imsave(buf, a, cmap="gray", vmin=0, vmax=max(float(np.percentile(a, 99.7)), 1), format="png")
            record[name] = "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode("ascii")
    embedded = json.dumps(page, ensure_ascii=False).replace("<", "\\u003c")
    path.write_text(HTML.replace("__BUNDLE__", embedded), encoding="utf-8")


def validate_review(review, bundle, rules):
    """Validate annotations as declared by a named reviewer; never fabricate them."""
    from datetime import datetime
    require(review.get("schema_version") == 1 and review.get("bundle_id") == bundle["bundle_id"], "Review bundle/version mismatch")
    require(isinstance(review.get("reviewer"), str) and bool(review["reviewer"].strip()), "Missing reviewer identity")
    try:
        timestamp = datetime.fromisoformat(review["reviewed_at_utc"].replace("Z", "+00:00"))
        require(timestamp.utcoffset() is not None, "Review timestamp must include timezone")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid declared review timestamp") from exc
    catalog = {p["id"]: p for p in bundle["patterns"]}
    require(isinstance(review.get("patterns"), list), "Missing review patterns")
    seen, results = set(), {}
    for p in bundle["patterns"]:
        results.setdefault(p["roi_id"], dict(completed=0, total_peaks=0, false_peaks=0, uncertain=0, missed_major=0))
    for r in review["patterns"]:
        require(isinstance(r, dict) and r.get("id") in catalog and r["id"] not in seen, "Unknown/duplicate pattern ID")
        seen.add(r["id"])
        p = catalog[r["id"]]
        require(type(r.get("completed")) is bool, "Completion flag must be boolean")
        require(isinstance(r.get("peaks"), list), "Missing peak annotations")
        peak_ids = set()
        expected = {k["id"] for k in p["peaks"]}
        labels = []
        for k in r["peaks"]:
            require(isinstance(k, dict) and type(k.get("id")) is int and k["id"] in expected and k["id"] not in peak_ids,
                    "Unknown/duplicate peak ID")
            peak_ids.add(k["id"])
            require(k.get("label") in ["unreviewed", "valid", "false_peak", "uncertain"], "Invalid peak label")
            labels.append(k["label"])
        require(peak_ids == expected, "Missing peak annotations")
        require(isinstance(r.get("missed_major_peaks"), list), "Missing missed-peak list")
        for k in r["missed_major_peaks"]:
            require(isinstance(k, dict) and type(k.get("x")) in [int, float] and type(k.get("y")) in [int, float] and
                    np.isfinite(k["x"]) and np.isfinite(k["y"]) and 0 <= k["x"] < p["width"] and 0 <= k["y"] < p["height"],
                    "Missed-peak coordinate outside detector")
        if r["completed"]:
            require("unreviewed" not in labels, "Completed pattern still has unreviewed peaks")
            q = results[p["roi_id"]]
            q["completed"] += 1; q["total_peaks"] += len(labels)
            q["false_peaks"] += labels.count("false_peak"); q["uncertain"] += labels.count("uncertain")
            q["missed_major"] += len(r["missed_major_peaks"])
    for q in results.values():
        q["false_fraction"] = q["false_peaks"]/max(q["total_peaks"], 1)
        q["mean_missed_major"] = q["missed_major"]/max(q["completed"], 1)
        q["coverage_passed"] = q["completed"] >= rules["minimum_complete_per_roi"]
        q["quality_passed"] = q["uncertain"] == 0 and q["false_fraction"] <= rules["maximum_false_fraction"] and q["mean_missed_major"] <= rules["maximum_mean_missed_major"]
    coverage = all(q["coverage_passed"] for q in results.values())
    passed = coverage and all(q["quality_passed"] for q in results.values())
    return dict(review_status="passed" if passed else ("failed" if coverage else "pending"),
        review_passed=passed, reviewer=review["reviewer"], declared_reviewed_at=review["reviewed_at_utc"],
        identity_note="User-declared reviewer identity, not independently authenticated", rois=results)


def import_review(out, path):
    verify_output(out)
    manifest = load_json(out / "run_manifest.json")
    bundle = load_json(out / "review_bundle.json")
    review = load_json(path)
    decision = validate_review(review, bundle, manifest["config"]["review"])
    decision.update(gate3_passed=bool(decision["review_passed"] and manifest["diagnostic_gate_passed"]),
                    diagnostic_gate_passed=manifest["diagnostic_gate_passed"], gate5_passed=False,
                    calibration_status="uncalibrated", bundle_id=bundle["bundle_id"], source_review_sha256=digest(path))
    audit = out / "reviews" / decision["source_review_sha256"]
    audit.mkdir(parents=True, exist_ok=True)
    # Audit names are content hashes, never supplied path components.
    (audit / "review.json").write_bytes(Path(path).read_bytes())
    write_json(audit / "decision.json", decision)
    write_json(out / "current_review.json", dict(review_sha256=decision["source_review_sha256"], decision=decision))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def recovery_figures(qc, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rois = list(dict.fromkeys(g["roi_id"] for g in qc["groups"]))
    fig, axes = plt.subplots(2, len(rois), figsize=(5*len(rois), 8), squeeze=False)
    for row, context in enumerate(["isolated", "crowded"]):
        for ax, rid in zip(axes[row], rois):
            for shape in qc["threshold"]["shapes"]:
                groups = [g for g in qc["groups"] if (g["roi_id"], g["context"], g["shape"]) == (rid, context, shape)]
                ax.plot([g["amplitude"] for g in groups], [g["recovery_fraction"] for g in groups], "o-", label=shape)
            ax.set(title=f"{rid}: {context}", xlabel="Injected expected peak counts", ylabel="Recovery fraction", ylim=(-.03, 1.03))
            ax.legend()
    fig.tight_layout(); fig.savefig(out / "recovery_curves.png", dpi=150); plt.close(fig)


def write_report(qc, out, bundle):
    lines = ["# MIB 第四轮执行报告", "", "检测器参数保持第三轮选择不变，全部注入结果已保留。", "",
             "| 条件 | 形状 | 增量 counts | 回收/有效试验 | 回收率 |", "|---|---|---:|---:|---:|"]
    for context in ["isolated", "crowded"]:
        for shape in qc["threshold"]["shapes"]:
            for amplitude in qc["threshold"]["amplitudes"]:
                groups = [g for g in qc["groups"] if (g["context"], g["shape"], g["amplitude"]) == (context, shape, amplitude)]
                n, k = sum(g["eligible"] for g in groups), sum(g["recovered"] for g in groups)
                lines.append(f"| {context} | {shape} | {amplitude} | {k}/{n} | {k/max(n,1):.1%} |")
    lines.extend(["", f"共 {qc['total_cases']} 次注入，{qc['available_targets']}/{qc['target_count']} 个目标位置可用；"
        f"强注入孤立位置诊断阈值通过：{qc['diagnostic_gate_passed']}。它不覆盖拥挤峰识别，更不代表实验峰准确率。",
        "", "孤立与拥挤位置、形状、增量使用运行前固定规则。新增计数由独立 Poisson 信号生成；原始背景噪声来自同一张已有 DP。"
        "这些重复使用背景的试验存在相关性，不提供把所有案例当成独立样本的置信区间。相同峰顶增量下两种形状总计数不同，不能据此比较纯粹的峰宽影响。",
        "", "## 人工复核", "", f"`review.html` 含 {len(bundle['patterns'])} 张固定 DP 和来源绑定峰表，直接用浏览器打开，无需联网。"
        "逐峰标记、Shift+点击主要漏峰、填写复核人、完成 DP 后导出 JSON。导出的结果尚未放行验收，需要运行：", "",
        "```powershell", "./.conda-phase/python.exe scripts/06_mib_round4.py --import-review <导出的JSON路径>", "```", "",
        "CLI 验证来源、ID、完整性与阈值，原始导入 JSON 和决定保存在 `reviews/<sha256>/`，当前状态在 `current_review.json`。"
        "未提供人工复核时 Gate 3=pending；即便复核通过，用户确认暂无标定，Gate 5 仍未通过。",
        "", "复现：`./.conda-phase/python.exe scripts/06_mib_round4.py`；核验加 `--verify`。前三轮产物未改写，没有运行全场相/取向/应变分析。"])
    (out / "report_zh.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def verify_output(out):
    manifest = load_json(out / "run_manifest.json")
    require(load_json(out / "status.json")["status"] == "complete", "Fourth round not complete")
    for name, sha in manifest["artifacts"].items():
        require(digest(out / name) == sha, f"Fourth-round artifact changed: {name}")
    for root, fingerprints in manifest["previous_outputs"].items():
        require(fingerprint_tree(Path(root)) == fingerprints, f"Previous output changed: {root}")
    verify_round3(Path(manifest["config"]["round3_root"]))
    bundle = load_json(out / "review_bundle.json")
    canonical = dict(bundle); bundle_id = canonical.pop("bundle_id")
    require(hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == bundle_id,
            "Review bundle identity mismatch")
    cases = load_json(out / "injection/injection_cases.json")
    increments = np.load(out / "injection/added_counts.npz")
    require(len(cases) == len(increments.files), "Injection increment count mismatch")
    for row in cases:
        a = increments[f"case_{row['case_id']}"]
        ya, yb, xa, xb = row["bbox"]
        require(a.shape == (yb-ya, xb-xa) and int(a.sum()) == row["realized_added_counts"], "Injection counts inconsistent")
        require(not row["recovered"] or not row["original_hit"], "Original hit counted as newly recovered")
    current = out / "current_review.json"
    if current.exists():
        pointer = load_json(current)
        sha = pointer["review_sha256"]
        require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "Invalid review audit hash")
        audit = out / "reviews" / sha
        require(digest(audit / "review.json") == sha, "Imported review changed")
        decision = validate_review(load_json(audit / "review.json"), bundle, manifest["config"]["review"])
        decision.update(gate3_passed=bool(decision["review_passed"] and manifest["diagnostic_gate_passed"]),
            diagnostic_gate_passed=manifest["diagnostic_gate_passed"], gate5_passed=False,
            calibration_status="uncalibrated", bundle_id=bundle_id, source_review_sha256=sha)
        require(pointer["decision"] == decision == load_json(audit / "decision.json"), "Review decision changed")
    print(f"Fourth-round verification passed: {len(cases)} injection cases, {len(bundle['patterns'])} review DPs, previous outputs intact.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mib_round4.yaml")
    parser.add_argument("--output")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--import-review", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.output:
        cfg["output"] = args.output
    out = Path(cfg["output"])
    if args.import_review:
        import_review(out, args.import_review)
        return
    if args.verify:
        verify_output(out)
        return
    r3 = Path(cfg["round3_root"])
    verify_round3(r3)
    manifest3 = load_json(r3 / "run_manifest.json")
    previous = dict(manifest3["previous_outputs"])
    previous[str(r3)] = fingerprint_tree(r3)
    for root in map(Path, previous):
        require(root.resolve() != out.resolve() and root.resolve() not in out.resolve().parents and out.resolve() not in root.resolve().parents,
                "Fourth-round output must be separate from previous rounds")
    if (out / "run_manifest.json").exists():
        require(load_json(out / "run_manifest.json")["config"] == cfg, "Changed config: choose a new output directory")
    require(not (out / "current_review.json").exists(), "Imported review exists: use --verify or a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "status.json", dict(status="running"))
    try:
        write_json(out / "resolved_config.json", cfg)
        patterns, cfg3, cfg2, choice, kernel, empirical, model = pattern_data(cfg)
        idir = out / "injection"; idir.mkdir(exist_ok=True)
        qc = recovery_experiment(cfg, patterns, cfg3, cfg2, choice, kernel, empirical, model, idir)
        recovery_figures(qc, idir)
        bundle = review_bundle(patterns, digest(r3 / "run_manifest.json"))
        write_json(out / "review_bundle.json", bundle)
        make_review_page(bundle, patterns, out / "review.html")
        write_json(out / "gate_status.json", dict(gate3_passed=False, review_status="pending",
            diagnostic_gate_passed=qc["diagnostic_gate_passed"], gate5_passed=False, calibration_status="uncalibrated",
            current_review_pointer="current_review.json if a human review has been imported"))
        write_report(qc, out, bundle)
        for root, fingerprints in previous.items():
            require(fingerprint_tree(Path(root)) == fingerprints, "Previous round changed during processing")
        artifacts = {name: sha for name, sha in fingerprint_tree(out).items()
                     if name not in ["run_manifest.json", "status.json", "current_review.json"] and not name.startswith("reviews/")}
        write_json(out / "run_manifest.json", dict(config=cfg, previous_outputs=previous, artifacts=artifacts,
            diagnostic_gate_passed=qc["diagnostic_gate_passed"], frozen_detector=choice, review_bundle_id=bundle["bundle_id"],
            source_sha256={str(p): digest(p) for p in [Path(__file__), Path(__file__).with_name("mib_review_ui.py"),
                Path(__file__).with_name("mib_round3.py"), Path(__file__).with_name("mib_round2.py"), Path(__file__).with_name("mib_round1.py")]}))
        write_json(out / "status.json", dict(status="complete"))
        verify_output(out)
    except Exception as exc:
        write_json(out / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    main()
