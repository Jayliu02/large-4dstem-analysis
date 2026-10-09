"""Direct Merlin MIB 4D-STEM quick-look analysis.

Data flow: MIB -> RosettaSciIO (lazy Dask) -> virtual images / CoM ->
optional small ROI -> py4DSTEM.  No full-size HDF5 conversion.

Example:
  python scripts/analyze_mib_direct.py --mib "data/my data.mib" --scan 256 256 \
      --roi 112 144 112 144 --output outputs/mib_analysis

Requires: numpy<2 (if py4DSTEM 0.14.18), rosettasciio, dask, matplotlib;
py4DSTEM only for --roi.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Save PNG reliably from PowerShell/VS Code
import matplotlib.pyplot as plt


def save_gray(arr: np.ndarray, filepath: Path, *, log: bool = False) -> None:
    """Save a 2D image; percentile scaling is for display only."""
    image = np.asarray(arr, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    if log:
        image = np.log1p(np.maximum(image, 0))
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError(f"Cannot plot {filepath}: all values are non-finite")
    lo, hi = np.percentile(image[finite], (1.0, 99.5))
    if hi <= lo:
        lo, hi = float(np.min(image[finite])), float(np.max(image[finite]))
    if hi <= lo:
        hi = lo + 1.0
    plt.imsave(filepath, np.ma.masked_invalid(image), cmap="gray", vmin=lo, vmax=hi)
    print(f"Saved image: {filepath.resolve()}")


def detector_masks(
    detector_shape: tuple[int, int],
    center_yx: tuple[float, float],
    bf_radius: float,
    adf_inner: float,
    adf_outer: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0 < bf_radius and 0 <= adf_inner < adf_outer):
        raise ValueError("Require BF radius > 0 and 0 <= ADF inner < ADF outer")
    cy, cx = center_yx
    yy, xx = np.indices(detector_shape, dtype=np.float64)
    r = np.hypot(yy - cy, xx - cx)
    bf_mask = r <= bf_radius
    adf_mask = (r >= adf_inner) & (r < adf_outer)
    if not bf_mask.any() or not adf_mask.any():
        raise ValueError("Detector masks have no pixels; check center/radii")
    return bf_mask, adf_mask, yy - cy, xx - cx


def compute_maps(
    data,
    center_yx: tuple[float, float],
    bf_radius: float = 15.0,
    adf_inner: float = 18.0,
    adf_outer: float = 50.0,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """Read each diffraction frame once, in small navigation-space batches.

    data is a Dask array with (scan_y, scan_x, det_y, det_x) ordering.
    Center-of-mass maps are UNCALIBRATED detector-pixel shifts relative
    to the user-supplied center, integrated over the complete detector.
    """
    sy, sx, dy, dx = map(int, data.shape)
    bf_mask, adf_mask, qy_rel, qx_rel = detector_masks(
        (dy, dx), center_yx, bf_radius, adf_inner, adf_outer
    )
    names = ("bf", "adf", "total", "com_y", "com_x")
    maps = {name: np.empty((sy, sx), dtype=np.float64) for name in names}
    maps["com_y"].fill(np.nan)
    maps["com_x"].fill(np.nan)

    for iy in range(sy):
        for ix in range(0, sx, batch_size):
            right = min(sx, ix + batch_size)
            block = np.asarray(data[iy, ix:right].compute(scheduler="synchronous"))
            expected = (right - ix, dy, dx)
            if block.shape != expected:
                raise RuntimeError(f"Unexpected MIB block shape {block.shape} != {expected}")

            maps["bf"][iy, ix:right] = np.sum(
                block[:, bf_mask], axis=1, dtype=np.float64
            )
            maps["adf"][iy, ix:right] = np.sum(
                block[:, adf_mask], axis=1, dtype=np.float64
            )
            total = np.sum(block, axis=(1, 2), dtype=np.float64)
            maps["total"][iy, ix:right] = total

            valid = total > 0
            com_y = np.full(total.shape, np.nan, dtype=np.float64)
            com_x = np.full(total.shape, np.nan, dtype=np.float64)
            if np.any(valid):
                moment_y = np.einsum("nij,ij->n", block, qy_rel, optimize=True)
                moment_x = np.einsum("nij,ij->n", block, qx_rel, optimize=True)
                np.divide(moment_y, total, out=com_y, where=valid)
                np.divide(moment_x, total, out=com_x, where=valid)
            maps["com_y"][iy, ix:right] = com_y
            maps["com_x"][iy, ix:right] = com_x
        if (iy + 1) % max(1, sy // 8) == 0 or iy + 1 == sy:
            print(f"Virtual maps: {iy + 1}/{sy} scan rows")

    maps["com_magnitude"] = np.hypot(maps["com_x"], maps["com_y"])
    return maps


def sampled_mean_dp(data, stride: int = 8) -> np.ndarray:
    """Compute mean diffraction pattern for a regularly subsampled scan."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    result = data[::stride, ::stride].mean(axis=(0, 1), dtype=np.float64)
    return np.asarray(result.compute(scheduler="synchronous"))


def save_mask_preview(
    mean_dp: np.ndarray,
    center_yx: tuple[float, float],
    bf_radius: float,
    adf_inner: float,
    adf_outer: float,
    filepath: Path,
) -> None:
    from matplotlib.patches import Circle

    cy, cx = center_yx
    im = np.log1p(np.maximum(mean_dp, 0))
    lo, hi = np.percentile(im, [1, 99.5])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(im, cmap="gray", vmin=lo, vmax=max(hi, lo + 1e-9))
    ax.plot(cx, cy, "+", markersize=10, color="yellow")
    for radius, color in (
        (bf_radius, "lime"),
        (adf_inner, "cyan"),
        (adf_outer, "cyan"),
    ):
        ax.add_patch(Circle((cx, cy), radius, fill=False, color=color, lw=1.2))
    ax.set(title="Virtual detector positions (pixel units)", xlabel="detector x", ylabel="detector y")
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)
    print(f"Saved image: {filepath.resolve()}")


def analyze_roi_with_py4dstem(
    data,
    roi: tuple[int, int, int, int],
    center_yx: tuple[float, float],
    bf_radius: float,
    output: Path,
) -> None:
    """Compute a py4DSTEM mean DP and BF on a small ROI only."""
    import py4DSTEM

    y0, y1, x0, x1 = roi
    sy, sx = data.shape[:2]
    if not (0 <= y0 < y1 <= sy and 0 <= x0 < x1 <= sx):
        raise ValueError(f"ROI {roi} lies outside scan {(sy, sx)}")
    roi_data = np.asarray(data[y0:y1, x0:x1].compute(scheduler="synchronous"), dtype=np.uint16)
    roi_data = np.ascontiguousarray(roi_data)
    print(f"py4DSTEM ROI: {roi_data.shape}, {roi_data.nbytes / (1024**2):.1f} MiB")

    dc = py4DSTEM.DataCube(data=roi_data)
    mean_object = dc.get_dp_mean()
    mean_image = np.asarray(getattr(mean_object, "data", mean_object))
    save_gray(mean_image, output / "roi_mean_dp_py4dstem.png", log=True)

    # In py4DSTEM the first detector axis is Qx and the second is Qy.
    # Our ndarray's detector coordinates are (row, column): (cy, cx).
    cy, cx = center_yx
    bf_object = dc.get_virtual_image(
        mode="circle",
        geometry=((cy, cx), bf_radius),
        centered=False,
        calibrated=False,
        verbose=False,
    )
    roi_bf = np.asarray(getattr(bf_object, "data", bf_object))
    save_gray(roi_bf, output / "roi_bf_py4dstem.png")
    np.save(output / "roi_bf_py4dstem.npy", roi_bf)
    np.save(output / "roi_mean_dp_py4dstem.npy", mean_image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mib", type=Path, required=True, help="Original Merlin MIB file")
    parser.add_argument("--scan", type=int, nargs=2, default=(256, 256), metavar=("Y", "X"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mib_analysis"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mean-stride", type=int, default=8)
    parser.add_argument("--center", type=float, nargs=2, default=None, metavar=("Y", "X"))
    parser.add_argument("--bf-radius", type=float, default=15.0)
    parser.add_argument("--adf-inner", type=float, default=18.0)
    parser.add_argument("--adf-outer", type=float, default=50.0)
    parser.add_argument("--point", type=int, nargs=2, default=None, metavar=("Y", "X"))
    parser.add_argument("--roi", type=int, nargs=4, default=None, metavar=("Y0", "Y1", "X0", "X1"),
                        help="Enable optional py4DSTEM analysis on this scan ROI")
    args = parser.parse_args()
    if args.batch_size < 1 or args.mean_stride < 1:
        parser.error("--batch-size and --mean-stride must be >= 1")
    if not args.mib.is_file():
        parser.error(f"MIB file not found: {args.mib}")
    args.output.mkdir(parents=True, exist_ok=True)

    from rsciio.quantumdetector import file_reader
    print(f"Opening {args.mib} (lazy; no HDF5 conversion)")
    signals = file_reader(
        str(args.mib),  # Compatibility with RosettaSciIO 0.15.0
        lazy=True,
        navigation_shape=tuple(args.scan),
        chunks=(1, args.batch_size, 256, 256),
        print_info=True,
    )
    data = signals[0]["data"]
    print(f"Data: shape={data.shape} dtype={data.dtype}, chunks={data.chunks}")
    if data.ndim != 4 or data.shape[:2] != tuple(args.scan):
        raise ValueError(f"Unexpected scan dimensions: {data.shape}, expected {args.scan}")
    sy, sx, dy, dx = map(int, data.shape)
    if (dy, dx) != (256, 256):
        raise ValueError(f"This data-specific script expects 256x256 DPs, got {(dy, dx)}")

    center = tuple(args.center) if args.center is not None else ((dy - 1) / 2, (dx - 1) / 2)
    point = tuple(args.point) if args.point is not None else (sy // 2, sx // 2)
    if not (0 <= point[0] < sy and 0 <= point[1] < sx):
        raise ValueError(f"Requested DP point outside scan: {point}")
    print(f"Detector center (row, col)={center}; single DP scan position={point}")

    single_dp = np.asarray(data[point[0], point[1]].compute(scheduler="synchronous"))
    save_gray(single_dp, args.output / "single_dp.png", log=True)
    np.save(args.output / "single_dp.npy", single_dp)

    mean_dp = sampled_mean_dp(data, args.mean_stride)
    save_gray(mean_dp, args.output / "mean_dp_sampled.png", log=True)
    np.save(args.output / "mean_dp_sampled.npy", mean_dp)
    save_mask_preview(mean_dp, center, args.bf_radius, args.adf_inner, args.adf_outer,
                      args.output / "detector_masks.png")

    maps = compute_maps(data, center, args.bf_radius, args.adf_inner,
                        args.adf_outer, args.batch_size)
    for name, arr in maps.items():
        np.save(args.output / f"{name}.npy", arr)
        save_gray(arr, args.output / f"{name}.png")

    mib_props = signals[0].get("original_metadata", {}).get("mib_properties", {})
    meta = {
        "source_mib": str(args.mib),
        "scan_shape": [sy, sx], "detector_shape": [dy, dx],
        "dtype": str(data.dtype), "frame_header_bytes": mib_props.get("head_size"),
        "nominal_scan_step_nm_from_filename": 5.8,
        "center_yx_pixels": list(center),
        "bf_radius_pixels": args.bf_radius,
        "adf_inner_pixels": args.adf_inner,
        "adf_outer_pixels": args.adf_outer,
        "sampled_mean_stride": args.mean_stride,
        "com_units": "uncalibrated detector pixel offsets",
        "scan_order_verified": False,
    }
    (args.output / "analysis_parameters.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.roi is not None:
        analyze_roi_with_py4dstem(data, tuple(args.roi), center, args.bf_radius, args.output)
    print(f"Done. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()

"""
3-2-ss2.05nm-c2 50-cl 110-sp7-gunlens2-12bit-256x256-0.62ms-190k-1049.mib

python .\scripts\analyze_mib_direct.py `
  --mib ".\data\3-1-ss5.8nm-c2 50-cl 110-sp7-gunlens2-12bit-256x256-0.62ms-67k-1045.mib" `
  --scan 256 256 `
  --batch-size 16 `
  --mean-stride 8 `
  --bf-radius 15 `
  --adf-inner 18 `
  --adf-outer 50 `
  --roi 112 144 112 144 `
  --output ".\outputs\mib_analysis"
  
python .\scripts\analyze_mib_direct.py `
  --mib ".\data\3-2-ss2.05nm-c2 50-cl 110-sp7-gunlens2-12bit-256x256-0.62ms-190k-1049.mib" `
  --scan 256 256 `
  --batch-size 16 `
  --mean-stride 8 `
  --bf-radius 15 `
  --adf-inner 18 `
  --adf-outer 50 `
  --roi 112 144 112 144 `
  --output ".\outputs\mib_analysis"
"""
