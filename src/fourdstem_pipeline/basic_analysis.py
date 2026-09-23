"""Reproducible, bounded-memory basic analysis of single-chip U16 MIB scans.

Run with ``python -m fourdstem_pipeline.basic_analysis --data data``.
Coordinates follow acquisition order: scan y, scan x, detector y, detector x.
"""
from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import numpy as np
import yaml
from scipy import ndimage
from threadpoolctl import threadpool_limits

from .array_utils import iter_navigation_slices
from .loaders import load_dataset
from .masks import build_annular_masks
from .logging import configure_pipeline_logging
from .virtual import compute_virtual_images
from .workflow import run_workflow


def save_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def inspect_mib(path: Path, scan_shape: tuple[int, int]) -> dict:
    """Validate every fixed-length header; reject interrupted or mismatched scans."""
    with path.open("rb") as stream:
        first = stream.read(384).decode("ascii").split(",")
    if first[0] != "MQ1" or first[6] != "U16" or int(first[2]) != 384 or first[7].strip() != "1x1":
        raise ValueError("Only single-chip, processed U16 MIB with 384-byte headers is supported.")
    detector = (int(first[5]), int(first[4]))
    frame_bytes = 384 + int(np.prod(detector)) * 2
    frames, remainder = divmod(path.stat().st_size, frame_bytes)
    if remainder or frames != int(np.prod(scan_shape)):
        raise ValueError(f"File layout has {frames} frames and {remainder} trailing bytes; expected {scan_shape}.")
    frame_dtype = np.dtype([("header", "S384"), ("image", ">u2", detector)])
    mapped = np.memmap(path, mode="r", dtype=frame_dtype, shape=(frames,))
    start_sequence = int(first[1])
    for i, header in enumerate(mapped["header"]):
        fields = bytes(header).decode("ascii").split(",")
        if int(fields[1]) != start_sequence + i or fields[0] != "MQ1" or fields[2:8] != first[2:8]:
            raise ValueError(f"Invalid or discontinuous frame header at frame {i}.")
    # Compare both interpretations for audit; the format specifies big-endian U16.
    sample = np.asarray(mapped["image"][::max(1, frames // 64)])
    swapped = sample.view("<u2")
    audit = {"big_endian_max": int(sample.max()), "little_endian_max": int(swapped.max()),
             "big_endian_fraction_above_4095": float(np.mean(sample > 4095)),
             "little_endian_fraction_above_4095": float(np.mean(swapped > 4095))}
    del sample, swapped, mapped
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns, "scan_shape": list(scan_shape),
            "detector_shape": list(detector), "frames": frames, "header_bytes": 384,
            "dtype": ">u2", "first_sequence": start_sequence, "last_sequence": start_sequence + frames - 1,
            "all_headers_valid": True, "byte_order_audit": audit,
            "byte_order_reference": "https://raw.githubusercontent.com/hyperspy/rosettasciio/main/rsciio/quantumdetector/_api.py",
            "scan_order": "row-major, unidirectional, confirmed by user",
            "filename_step_nm_unverified": float(re.search(r"ss([\d.]+)nm", path.name)[1]) if re.search(r"ss([\d.]+)nm", path.name) else None}


def raw_quality(dataset, output: Path) -> tuple[dict, np.ndarray, dict]:
    """Count actual pixel values, without percentile-based saturation proxies."""
    shape = dataset.navigation_shape
    totals = np.zeros(shape, dtype=np.float64)
    zeros = np.zeros(shape, dtype=np.int32)
    high = np.zeros(shape, dtype=np.int32)
    mean_sum = np.zeros(dataset.signal_shape, dtype=np.float64)
    histogram = np.zeros(65536, dtype=np.int64)
    blocks = list(iter_navigation_slices(shape, (16, 16)))
    for i, (ys, xs) in enumerate(blocks):
        block = np.asarray(dataset.data[ys, xs, :, :], dtype=np.uint16)
        totals[ys, xs] = block.sum(axis=(-2, -1), dtype=np.float64)
        zeros[ys, xs] = (block == 0).sum(axis=(-2, -1))
        high[ys, xs] = (block >= 4095).sum(axis=(-2, -1))
        mean_sum += block.sum(axis=(0, 1), dtype=np.float64)
        histogram += np.bincount(block.ravel(), minlength=65536)
        if i % 32 == 0:
            print(f"  raw QC {i + 1}/{len(blocks)}", flush=True)
    logs = np.log1p(totals)
    median = np.median(logs)
    mad = np.median(np.abs(logs - median))
    outliers = np.abs(logs - median) > max(8 * 1.4826 * mad, 1e-6)
    row_means = totals.mean(axis=1)
    odd_even = abs(float(row_means[::2].mean() - row_means[1::2].mean())) / max(float(totals.mean()), 1)
    n_pixels = int(np.prod(dataset.shape))
    result = {"frames": int(np.prod(shape)), "zero_frames": int(np.sum(totals == 0)),
              "zero_pixel_fraction": int(histogram[0]) / n_pixels,
              "suspected_saturated_frames": int(np.sum(high > 0)),
              "suspected_saturated_pixel_fraction": int(high.sum()) / n_pixels,
              "pixels_above_12bit_range": int(histogram[4096:].sum()),
              "saturation_threshold": 4095, "saturation_interpretation": "12-bit filename assumption; threshold hits are suspected clipping, not proof",
              "maximum_count": int(np.flatnonzero(histogram)[-1]),
              "mean_total_counts": float(totals.mean()),
              "total_count_percentiles": np.percentile(totals, [0, 1, 50, 99, 100]).tolist(),
              "intensity_outlier_frames": int(outliers.sum()), "outlier_rule": "abs(log1p(total)-median) > 8*1.4826*MAD; no deletion",
              "row_mean_cv": float(row_means.std() / max(row_means.mean(), 1)),
              "odd_even_row_relative_difference": odd_even,
              "stripe_interpretation": "Row variation also includes real specimen structure; not an automatic artifact classification."}
    arrays = {"total_counts": totals, "zero_pixel_counts": zeros, "suspected_saturated_pixel_counts": high,
              "intensity_outliers": outliers, "count_histogram": histogram}
    np.savez_compressed(output / "raw_quality_arrays.npz", **arrays)
    save_json(output / "raw_quality.json", result)
    mean = mean_sum / np.prod(shape)
    np.save(output / "raw_mean_diffraction.npy", mean)
    return result, mean, arrays


def estimate_geometry(mean: np.ndarray) -> tuple[list[float], dict, dict]:
    """Estimate the central bright component and use complete detector annuli."""
    smooth = ndimage.gaussian_filter(mean.astype(float), 1.0)
    sy, sx = smooth.shape
    y0, x0 = sy // 4, sx // 4
    inner = smooth[y0:sy - y0, x0:sx - x0]
    py, px = np.unravel_index(np.argmax(inner), inner.shape)
    py, px = py + y0, px + x0
    baseline = float(np.percentile(smooth, 50))
    cutoff = baseline + 0.5 * (float(smooth[py, px]) - baseline)
    labels, _ = ndimage.label(smooth >= cutoff)
    component = labels == labels[py, px]
    if not component.any() or labels[py, px] == 0:
        raise ValueError("No central beam component could be estimated.")
    yy, xx = np.indices(mean.shape)
    weights = np.maximum(smooth - baseline, 0) * component
    center = [float((weights * yy).sum() / weights.sum()), float((weights * xx).sum() / weights.sum())]
    radius = float(np.sqrt(component.sum() / np.pi))
    outer = float(min(center[0], center[1], sy - 1 - center[0], sx - 1 - center[1]))
    bf = max(2.0, 1.3 * radius)
    adf_inner = 1.2 * bf
    if adf_inner >= outer:
        raise ValueError("Estimated central beam too large for reliable annular imaging; inspect mean diffraction.")
    edges = np.linspace(bf, outer, 4)
    masks = {"bf": {"inner_radius": 0.0, "outer_radius": bf},
             "adf": {"inner_radius": adf_inner, "outer_radius": outer},
             "total": {"inner_radius": 0.0, "outer_radius": float(np.hypot(sy, sx))}}
    for i in range(3):
        masks[f"ring_{i + 1}"] = {"inner_radius": float(edges[i]), "outer_radius": float(edges[i + 1])}
    return center, masks, {"center_yx": center, "disk_equivalent_radius_px": radius,
                            "method": "Gaussian sigma=1; central-half peak; connected component above 50% peak-background; BF radius=1.3*equivalent radius, ADF inner=1.2*BF radius",
                            "mask_units": "detector pixels", "masks": masks}


def verify_small_block(dataset, masks: dict) -> dict:
    from .dataset import DatasetHandle
    block = np.asarray(dataset.data[:3, :4, :, :], dtype=np.uint16)
    small = DatasetHandle(block, "verification")
    result = compute_virtual_images(small, masks, block_shape=(2, 3))
    np.testing.assert_allclose(result.mean_diffraction, block.mean(axis=(0, 1)), rtol=1e-6, atol=1e-5)
    np.testing.assert_array_equal(result.max_diffraction, block.max(axis=(0, 1)))
    for name, mask in masks.items():
        np.testing.assert_allclose(result.images[name], block[..., mask].sum(axis=-1, dtype=np.float64), rtol=1e-6)
    return {"status": "passed", "navigation_shape": [3, 4], "block_shape": [2, 3],
            "checks": ["mean diffraction", "maximum diffraction", "all virtual detector sums"]}


def write_report(output: Path, title: str, paragraphs: list[str], figures: list[tuple[str, str]], links: list[tuple[str, str]]) -> None:
    md = [f"# {title}", *paragraphs]
    body = [f"<h1>{html.escape(title)}</h1>"] + [f"<p>{html.escape(p)}</p>" for p in paragraphs]
    for caption, path in figures:
        md.append(f"![{caption}]({path})")
        body.append(f'<figure><img src="{html.escape(path, quote=True)}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    for label, path in links:
        md.append(f"[{label}]({path})")
        body.append(f'<p><a href="{html.escape(path, quote=True)}">{html.escape(label)}</a></p>')
    (output / "report_zh.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    (output / "report_zh.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>' + html.escape(title) + '</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#183047}img{max-width:100%;height:auto}figure{margin:28px 0}a{color:#0764a3}figcaption{color:#52657a}</style><body>' + "\n".join(body) + "</body></html>", encoding="utf-8")


def plot_image(ax, data: np.ndarray, title: str, *, cmap="gray", logarithmic=False):
    shown = np.log1p(data) if logarithmic else data
    low, high = np.percentile(shown, [1, 99.5])
    artist = ax.imshow(shown, cmap=cmap, vmin=low, vmax=max(high, low + 1e-9), origin="upper")
    ax.set_title(title)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.figure.colorbar(artist, ax=ax, shrink=0.75)


def make_figures(result, arrays: dict, geometry: dict, output: Path) -> list[tuple[str, str]]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    virtual, phase, fingerprints = result.virtual_images, result.phase_screening, result.fingerprints
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, key in zip(axes[0], ["bf", "adf", "total"]):
        plot_image(ax, virtual.images[key], f"Virtual {key.upper()} (counts)")
    plot_image(axes[1, 0], virtual.mean_diffraction, "Mean diffraction: log(1+counts)", logarithmic=True)
    cy, cx = geometry["center_yx"]
    for key, color in [("bf", "cyan"), ("adf", "orange")]:
        spec = geometry["masks"][key]
        for radius in [spec["inner_radius"], spec["outer_radius"]]:
            if radius:
                axes[1, 0].add_patch(Circle((cx, cy), radius, fill=False, color=color, linewidth=0.8))
    plot_image(axes[1, 1], virtual.max_diffraction, "Maximum diffraction: log(1+counts)", logarithmic=True)
    axes[1, 2].imshow(phase.labels, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    axes[1, 2].set(title="Diffraction feature classes (0-3)", xlabel="scan x", ylabel="scan y")
    fig.savefig(output / "overview.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    class_means = np.load(output / "05_cluster_diagnostics" / "cluster_mean_dps.npy")
    reps = []
    for label in range(4):
        selected = phase.labels == label
        fraction = float(selected.mean())
        plot_image(axes[0, label], class_means[label], f"Class {label}: {fraction:.1%} (log)", logarithmic=True)
        profile = fingerprints.profiles[selected].mean(axis=0)
        axes[1, label].plot(fingerprints.radii, profile)
        axes[1, label].set(xlabel="radius (detector pixels)", ylabel="mean counts / pixel")
        embedding = phase.embedding[selected]
        distances = np.sum((embedding - embedding.mean(axis=0)) ** 2, axis=1)
        y, x = np.argwhere(selected)[np.argmin(distances)]
        reps.append({"class": label, "fraction": fraction, "representative_yx": [int(y), int(x)],
                     "bbox_y0_y1_x0_x1": [max(0, int(y) - 8), min(phase.labels.shape[0], int(y) + 9), max(0, int(x) - 8), min(phase.labels.shape[1], int(x) + 9)]})
    save_json(output / "representative_regions.json", reps)
    np.savez_compressed(output / "representative_patterns.npz", **{f"class_{r['class']}": np.asarray(result.dataset.data[r["representative_yx"][0], r["representative_yx"][1], :, :]) for r in reps})
    fig.savefig(output / "feature_classes.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    plot_image(axes[0, 0], arrays["zero_pixel_counts"] / np.prod(result.dataset.signal_shape), "Zero-valued pixel fraction", cmap="viridis")
    plot_image(axes[0, 1], arrays["suspected_saturated_pixel_counts"], "Pixels >= 4095 per pattern", cmap="magma")
    totals = arrays["total_counts"]
    axes[1, 0].plot(totals.mean(axis=1))
    axes[1, 0].set(xlabel="scan row", ylabel="mean total counts", title="Row intensity: structure + possible artifacts")
    hist = arrays["count_histogram"]
    occupied = np.flatnonzero(hist)
    axes[1, 1].semilogy(occupied, hist[occupied])
    axes[1, 1].set(xlabel="pixel count", ylabel="frequency (log)", title="Full-data count histogram")
    fig.savefig(output / "quality.png", dpi=150)
    plt.close(fig)
    return [("虚拟成像、衍射图与特征分区；图像独立调整显示范围，定量比较使用原始数值。", "overview.png"),
            ("四类衍射特征及其平均径向曲线；类别不是晶相。", "feature_classes.png"),
            ("全数据计数质量检查；零计数不等于坏像素，行间变化不等于扫描伪影。", "quality.png")]


def analyze_file(path: Path, output: Path, scan_shape: tuple[int, int]) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    metadata = inspect_mib(path, scan_shape)
    save_json(output / "input_metadata.json", metadata)
    data_cfg = {"path": str(path.resolve()), "backend": "mib_memmap", "lazy": True,
                "scan_shape": list(scan_shape), "detector_shape": metadata["detector_shape"],
                "dtype": ">u2", "mib_header_bytes": 384, "chunks": {"navigation": [16, 16]}}
    dataset = load_dataset(**{k: v for k, v in data_cfg.items() if k != "chunks"})
    quality, mean, arrays = raw_quality(dataset, output)
    center, masks, geometry = estimate_geometry(mean)
    save_json(output / "geometry.json", geometry)
    mask_arrays = build_annular_masks(dataset.signal_shape, masks, center=center)
    verification = verify_small_block(dataset, mask_arrays)
    np.savez_compressed(output / "detector_masks.npz", **mask_arrays)
    cfg = {"pipeline": {"stages": ["stage1"]}, "project": {"name": path.stem, "output_dir": str(output.resolve())},
           "data": data_cfg, "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
           "geometry": {"center": center, "radial_bins": 128}, "virtual_images": {"masks": masks},
           "phase_screening": {"method": "pca_nmf_cluster", "n_components": 4, "n_clusters": 4, "candidate_phases": []},
           "orientation": {"enabled": False}, "sample_mask": {"enabled": False}, "roi_bragg": {"enabled": False}}
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    result = run_workflow(config_path)
    if result.errors or result.diagnostics.get("_errors"):
        raise RuntimeError(f"Workflow reported errors: {result.errors}; {result.diagnostics.get('_errors')}")
    assert result.phase_screening.labels.shape == scan_shape
    assert result.fingerprints.profiles.shape == scan_shape + (128,)
    assert np.isfinite(result.fingerprints.profiles).all()
    assert result.orientation is None
    np.testing.assert_allclose(result.virtual_images.mean_diffraction, mean, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(result.virtual_images.images["total"], arrays["total_counts"], rtol=1e-6)
    verification["full_data_checks"] = ["shape", "finite profiles", "orientation disabled", "mean DP matches float64 QC", "total image matches raw counts"]
    save_json(output / "verification.json", verification)
    figures = make_figures(result, arrays, geometry, output)
    fractions = np.bincount(result.phase_screening.labels.ravel(), minlength=4) / np.prod(scan_shape)
    profile = result.fingerprints.profiles.mean(axis=(0, 1))
    np.savez_compressed(output / "comparison_profile.npz", radii=result.fingerprints.radii, mean_profile=profile)
    paragraphs = [f"输入：{path.name}。全部 {quality['frames']:,} 个扫描点参与分析；扫描及探测器尺寸均保留原分辨率。",
                  f"按大端 U16 读取，全部帧头及帧序号检查通过。零强度帧 {quality['zero_frames']}；最大像素计数 {quality['maximum_count']}；零计数像素占 {quality['zero_pixel_fraction']:.2%}。",
                  f"按文件名中的 12-bit 假设，以 4095 为疑似饱和阈值：涉及 {quality['suspected_saturated_frames']} 帧、{quality['suspected_saturated_pixel_fraction']:.6%} 像素。超出 12-bit 范围的像素数 {quality['pixels_above_12bit_range']}。",
                  f"平均亮区中心 (y,x)=({center[0]:.2f}, {center[1]:.2f}) 像素，平均亮区等效半径 {geometry['disk_equivalent_radius_px']:.2f} 像素；掩膜见 geometry.json。此估计用于探索性虚拟探测器，不代表单帧束斑尺寸或角度标定。",
                  "类别 0–3 占比：" + "、".join(f"{v:.2%}" for v in fractions) + "。对各径向曲线按最大值归一化，PCA 四维聚类；NMF 作为补充特征，随机种子为 0。四类是预设探索参数，不代表存在四个晶相。",
                  f"强度离群帧 {quality['intensity_outlier_frames']}；奇偶行均值相对差 {quality['odd_even_row_relative_difference']:.2%}。离群及行间变化可能来自真实结构，未删除或修正任何原始数据。",
                  "扫描坐标为扫描点，衍射坐标为探测器像素。文件名步长仅记录为待核实元数据。未做晶相识别、定量取向、晶格间距或应变分析。",
                  "小块直接计算与分块计算一致；完整数据均值与总强度通过独立校验。代表区域由各类 PCA 特征空间中靠近类中心的扫描点选取，不依赖取向结果。"]
    write_report(output, f"4D-STEM 基础分析：{path.stem}", paragraphs, figures,
                 [("原始计数质量指标", "raw_quality.json"), ("分析配置", "config.yaml"), ("代表区域坐标", "representative_regions.json"), ("数值验证", "verification.json")])
    summary = {"file": path.name, "output": output.name, "status": "complete", "elapsed_seconds": time.perf_counter() - started,
               **quality, "center_yx": center, "class_fractions": fractions.tolist()}
    save_json(output / "basic_summary.json", summary)
    del result, dataset, arrays
    gc.collect()
    return summary


def batch_report(output: Path, summaries: list[dict]) -> None:
    import matplotlib.pyplot as plt
    complete = [s for s in summaries if s["status"] == "complete"]
    columns = ["file", "status", "qc_status", "frames", "zero_frames", "maximum_count", "mean_total_counts", "zero_pixel_fraction",
               "suspected_saturated_frames", "suspected_saturated_pixel_fraction", "intensity_outlier_frames", "row_mean_cv", "odd_even_row_relative_difference", "elapsed_seconds", "error"]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    figures = []
    if complete:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        outer_start = max(json.loads((output / s["output"] / "geometry.json").read_text(encoding="utf-8"))["masks"]["bf"]["outer_radius"] for s in complete)
        for summary in complete:
            with np.load(output / summary["output"] / "comparison_profile.npz") as values:
                radii, profile = values["radii"], values["mean_profile"]
            label = summary["output"]
            axes[0].semilogy(radii, np.maximum(profile, 1e-9), label=label)
            outer = radii >= outer_start
            normalized = profile / max(float(profile[outer].sum()), 1e-12)
            axes[1].plot(radii[outer], normalized[outer], label=label)
        axes[0].set(xlabel="radius (detector pixels)", ylabel="mean counts / pixel", title="Mean radial profiles")
        axes[1].set(xlabel="radius (detector pixels)", ylabel="normalized intensity", title=f"Outer profiles, normalized over r >= {outer_start:.1f} px")
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.2)
        fig.savefig(output / "radial_comparison.png", dpi=150)
        plt.close(fig)
        figures.append((f"跨文件径向强度比较：右图在 r≥{outer_start:.1f} 像素范围内归一化；未标定为散射角或倒空间距离。", "radial_comparison.png"))
    paragraphs = [f"已完成 {len(complete)}/{len(summaries)} 份文件的全扫描基础分析。每份结果及未完成原因见下列链接和 comparison.csv。",
                  "全部文件按已确认的逐行同向扫描排列重建，采用大端 U16 和 384 字节帧头。原始文件保持不变；处理块为 16×16 扫描点。",
                  "不同文件的类别编号独立，不对应同一晶相或结构。图像独立调整显示对比度；定量比较请使用保存的计数数组和径向曲线。束斑估计和掩膜逐文件确定，因此虚拟明暗场绝对强度不直接等同于样品厚度差异。",
                  "缺少样品成分和衍射标定，本报告仅解释计数质量、空间强度及衍射特征差异；没有晶相、定量取向或应变结论。"]
    for s in summaries:
        if s["status"] == "complete":
            paragraphs.append(f"{s['output']}：平均每帧总计数 {s['mean_total_counts']:.1f}；零帧 {s['zero_frames']}；最大像素计数 {s['maximum_count']}；疑似饱和帧 {s['suspected_saturated_frames']}；各类占比 " + " / ".join(f"{v:.1%}" for v in s["class_fractions"]) + "。")
        else:
            paragraphs.append(f"未完成 {s['file']}：{s.get('error', 'unknown error')}")
    links = [(s["file"], f"{s['output']}/report_zh.html") for s in complete] + [("对比数值表", "comparison.csv"), ("批次状态与异常", "batch_summary.json")]
    write_report(output, "三份 4D-STEM 数据基础分析汇总", paragraphs, figures, links)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/basic_analysis"))
    parser.add_argument("--scan-shape", type=int, nargs=2, default=(256, 256))
    args = parser.parse_args()
    paths = sorted(args.data.glob("*.mib"))
    if not paths:
        parser.error("No .mib files found.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; select a new --output to preserve previous results.")
    args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    configure_pipeline_logging()
    file_handler = logging.FileHandler(args.output / "analysis.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger("fourdstem_pipeline").addHandler(file_handler)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (args.output / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
    summaries = []
    with threadpool_limits(limits=4):
        for index, path in enumerate(paths, 1):
            name = f"scan_{index:02d}_{path.stem.rsplit('-', 1)[-1]}"
            print(f"Analyzing {index}/{len(paths)}: {path.name}", flush=True)
            try:
                summary = analyze_file(path, args.output / name, tuple(args.scan_shape))
            except Exception as exc:
                traceback.print_exc()
                summary = {"file": path.name, "output": name, "status": "failed", "error": str(exc)}
                (args.output / f"{name}_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            summaries.append(summary)
            save_json(args.output / "batch_summary.json", summaries)
            print(f"  {name}: {summary['status']}", flush=True)
        batch_report(args.output, summaries)
        from .beam_audit import audit_completed_batch
        audit_completed_batch(args.output)
    return int(any(s["status"] != "complete" for s in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
