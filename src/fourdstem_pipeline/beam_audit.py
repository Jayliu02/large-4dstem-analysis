"""Read-only input audit for scan-correlated beam motion in completed basic runs."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import RANSACRegressor

from .basic_analysis import batch_report, save_json
from .loaders import load_dataset
from .export import save_report
from .qc import QCFlag, QCResult, save_qc_summary


def measure_motion(dataset) -> tuple[dict, dict]:
    """Robustly fit bright-spot motion; Bragg peaks may be outliers, not beam centres."""
    ny, nx = dataset.navigation_shape
    sy, sx = dataset.signal_shape
    rows = np.unique(np.linspace(0, ny - 1, min(ny, 16)).astype(int))
    cols = np.unique(np.linspace(0, nx - 1, min(nx, 16)).astype(int))
    y0, y1, x0, x1 = sy // 4, sy - sy // 4, sx // 4, sx - sx // 4
    positions, peaks = [], []
    for y in rows:
        for x in cols:
            pattern = np.asarray(dataset.data[int(y), int(x), :, :], dtype=np.float32)
            smooth = gaussian_filter(pattern, 1)
            py, px = np.unravel_index(np.argmax(smooth[y0:y1, x0:x1]), (y1 - y0, x1 - x0))
            positions.append([int(y), int(x)])
            peaks.append([int(py + y0), int(px + x0)])
    positions, peaks = np.asarray(positions), np.asarray(peaks)
    model = RANSACRegressor(min_samples=3, residual_threshold=3, random_state=0, max_trials=1000)
    model.fit(positions, peaks)
    predicted = model.predict(positions)
    residual = np.linalg.norm(peaks - predicted, axis=1)
    corners = np.array([[0, 0], [0, nx - 1], [ny - 1, 0], [ny - 1, nx - 1]])
    extent = np.ptp(model.predict(corners), axis=0)
    fraction = float(model.inlier_mask_.mean())
    warning = fraction >= 0.7 and float(np.max(extent)) > 5
    result = {"sample_count": len(positions), "inlier_fraction": fraction,
              "median_residual_px": float(np.median(residual)),
              "inlier_median_residual_px": float(np.median(residual[model.inlier_mask_])),
              "predicted_span_yx_px": extent.tolist(), "affine_coefficients_yx": model.estimator_.coef_.tolist(),
              "affine_intercept_yx": model.estimator_.intercept_.tolist(),
              "scan_correlated_motion_warning": warning,
              "method": "16x16 uniform sample; sigma=1 smoothed central-half maximum; RANSAC affine fit, residual_threshold=3 px, seed=0",
              "interpretation": "Brightest spots are candidates; a coherent scan-dependent trajectory supports beam motion. No calibration or correction applied."}
    arrays = {"scan_yx": positions, "peak_yx": peaks, "predicted_yx": predicted,
              "inlier": model.inlier_mask_, "residual_px": residual}
    return result, arrays


def report_callout(directory: Path, text: str, *, image: str | None = None) -> None:
    """Insert one idempotent quality note in both primary and legacy reports."""
    for stem in ["report_zh", "report"]:
        md_path = directory / f"{stem}.md"
        if md_path.exists():
            content = md_path.read_text(encoding="utf-8")
            start, end = "<!-- beam-audit-start -->", "<!-- beam-audit-end -->"
            if start in content:
                before, rest = content.split(start, 1)
                content = before + rest.split(end, 1)[1].lstrip("\n")
            note = f"{start}\n\n**质量复核：** {text}\n\n"
            if image:
                note += f"![扫描相关亮斑位移]({image})\n\n"
            md_path.write_text(note + end + "\n\n" + content, encoding="utf-8")
        html_path = directory / f"{stem}.html"
        if html_path.exists():
            content = html_path.read_text(encoding="utf-8")
            start, end = "<!-- beam-audit-start -->", "<!-- beam-audit-end -->"
            if start in content:
                before, rest = content.split(start, 1)
                content = before + rest.split(end, 1)[1]
            note = start + '<aside style="padding:20px;background:#fff0d3;border-left:5px solid #b77714"><strong>质量复核：</strong>' + html.escape(text)
            if image:
                note += f'<p><img style="max-width:100%" src="{html.escape(image)}" alt="扫描相关亮斑位移"></p>'
            note += "</aside>" + end
            body = content.find(">", content.find("<body")) + 1
            content = content[:body] + note + content[body:] if body else note + content
            html_path.write_text(content, encoding="utf-8")


def audit_completed_batch(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = json.loads((output / "batch_summary.json").read_text(encoding="utf-8"))
    warnings = []
    for summary in summaries:
        if summary["status"] != "complete":
            continue
        directory = output / summary["output"]
        metadata = json.loads((directory / "input_metadata.json").read_text(encoding="utf-8"))
        source = Path(metadata["path"])
        if source.stat().st_size != metadata["size_bytes"] or source.stat().st_mtime_ns != metadata["mtime_ns"]:
            raise ValueError(f"Input changed since analysis: {source}")
        dataset = load_dataset(source, backend="mib_memmap", scan_shape=metadata["scan_shape"],
                               detector_shape=metadata["detector_shape"], dtype=">u2", mib_header_bytes=384)
        motion, arrays = measure_motion(dataset)
        np.savez_compressed(directory / "beam_motion_samples.npz", **arrays)
        save_json(directory / "beam_motion.json", motion)
        with np.load(directory / "raw_quality_arrays.npz") as qc_arrays:
            abnormal_frames = np.argwhere(qc_arrays["suspected_saturated_pixel_counts"] > 0)
        abnormal = []
        for y, x in abnormal_frames:
            pattern = np.asarray(dataset.data[int(y), int(x), :, :])
            for qy, qx in np.argwhere(pattern >= 4095):
                abnormal.append({"scan_yx": [int(y), int(x)], "detector_yx": [int(qy), int(qx)], "raw_count": int(pattern[qy, qx])})
        save_json(directory / "high_count_pixels.json", abnormal)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        grid_shape = (len(np.unique(arrays["scan_yx"][:, 0])), len(np.unique(arrays["scan_yx"][:, 1])))
        for axis, ax in enumerate(axes[:2]):
            im = ax.imshow(arrays["predicted_yx"][:, axis].reshape(grid_shape), origin="upper", extent=(-0.5, metadata["scan_shape"][1] - 0.5, metadata["scan_shape"][0] - 0.5, -0.5), cmap="viridis")
            ax.set(title=f"Fitted beam {'y' if axis == 0 else 'x'} (detector px)", xlabel="scan x", ylabel="scan y")
            fig.colorbar(im, ax=ax)
        axes[2].scatter(arrays["predicted_yx"][:, 0], arrays["peak_yx"][:, 0], s=9, label="y")
        axes[2].scatter(arrays["predicted_yx"][:, 1], arrays["peak_yx"][:, 1], s=9, label="x")
        axes[2].set(xlabel="predicted detector coordinate", ylabel="measured peak coordinate", title="Sampled spot positions")
        axes[2].legend()
        fig.savefig(directory / "beam_motion.png", dpi=140)
        plt.close(fig)
        span = motion["predicted_span_yx_px"]
        message = (f"抽样 {motion['sample_count']} 个扫描点，亮斑位置线性模型的内点占 {motion['inlier_fraction']:.1%}，"
                   f"内点典型残差 {motion['inlier_median_residual_px']:.2f} 像素；全扫描预测位移跨度 (y,x)=({span[0]:.1f}, {span[1]:.1f}) 像素。")
        flags = []
        if motion["scan_correlated_motion_warning"]:
            message += "存在明显扫描相关束斑位移。固定束心的径向分类受到该位移强烈影响，近同心分区不能解释为材料分区或晶相；宽平均亮区也不是单帧束斑尺寸。当前虚拟探测器采用固定掩膜，保留为探索性图像。后续材料分区应先校正逐点束心。此次未校正或修改原始数据。"
            flags.append(QCFlag("warning", "SCAN_CORRELATED_BEAM_MOTION", message, motion))
            warnings.append(summary["output"])
        else:
            message += "此抽样检查未触发强扫描相关位移阈值；不代表完全不存在束移。"
        if abnormal:
            note = f" 检出 {len(abnormal)} 个 >=4095 的像素，坐标和值见 high_count_pixels.json；超过 4095 的值记为原始计数异常，不直接断言为物理饱和。"
            message += note
            flags.append(QCFlag("warning", "HIGH_RAW_COUNTS", note, {"count": len(abnormal)}))
        qc = json.loads((directory / "qc_summary.json").read_text(encoding="utf-8"))
        preserved = [QCFlag(**f) for f in qc["flags"] if f["code"] not in {"SCAN_CORRELATED_BEAM_MOTION", "HIGH_RAW_COUNTS"}]
        all_flags = preserved + flags
        n_critical = sum(f.severity == "critical" for f in all_flags)
        n_warnings = sum(f.severity == "warning" for f in all_flags)
        status = "FAIL" if n_critical else "PASS_WITH_WARNINGS" if n_warnings else "PASS"
        qc_result = QCResult(status, n_warnings, n_critical, all_flags)
        save_qc_summary(directory, qc_result)
        for filename, field, value in [("stage1_summary.json", "qc_status", status), ("workflow_summary.json", "qc", qc_result.to_dict())]:
            manifest_path = directory / filename
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest[field] = value
            save_json(manifest_path, manifest)
            if filename == "workflow_summary.json":
                save_report(directory, manifest, np.load(directory / "fingerprint_classes/fingerprint_class_labels.npy"))
        report_callout(directory, message, image="beam_motion.png")
        summary["beam_motion"] = motion
        summary["qc_status"] = status
        summary["high_count_pixel_count"] = len(abnormal)
        save_json(directory / "basic_summary.json", summary)
        print(f"{directory.name}: motion span {span}, inliers {motion['inlier_fraction']:.1%}, high pixels {len(abnormal)}", flush=True)
    save_json(output / "batch_summary.json", summaries)
    batch_report(output, summaries)
    if warnings:
        report_callout(output, "、".join(warnings) + " 检出显著扫描相关束斑位移，固定束心分类受到强烈影响，不能将当前类别图视为材料分区。请优先查看各文件报告开头的束移诊断。全部原始数据保持不变；后续定量分区需要逐点束心校正。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    audit_completed_batch(parser.parse_args().output)
