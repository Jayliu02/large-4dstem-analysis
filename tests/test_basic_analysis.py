from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from fourdstem_pipeline.basic_analysis import inspect_mib, raw_quality, estimate_geometry, verify_small_block
from fourdstem_pipeline.dataset import DatasetHandle
from fourdstem_pipeline.loaders import load_dataset
from fourdstem_pipeline.masks import build_annular_masks
from fourdstem_pipeline.workflow import run_workflow
from fourdstem_pipeline.qc import _check_saturation, _check_beam_center
from fourdstem_pipeline.virtual import VirtualImageResult


def write_mib(path: Path, patterns: np.ndarray, *, bad_sequence=False):
    with path.open("wb") as stream:
        for i, pattern in enumerate(patterns.reshape((-1,) + patterns.shape[-2:])):
            sequence = i + 1 + (10 if bad_sequence and i == 2 else 0)
            header = f"MQ1,{sequence:06d},00384,01,{pattern.shape[1]:04d},{pattern.shape[0]:04d},U16,   1x1,01,"
            stream.write(header.encode("ascii").ljust(384, b" "))
            stream.write(pattern.astype(">u2").tobytes())


def test_mib_endian_headers_and_truncation(tmp_path):
    data = np.arange(2 * 3 * 8 * 8, dtype=np.uint16).reshape(2, 3, 8, 8)
    path = tmp_path / "test.mib"
    write_mib(path, data)
    metadata = inspect_mib(path, (2, 3))
    assert metadata["frames"] == 6
    handle = load_dataset(path, backend="mib_memmap", scan_shape=(2, 3), detector_shape=(8, 8), dtype=">u2", mib_header_bytes=384)
    np.testing.assert_array_equal(handle.data[:, :, :, :], data)
    del handle
    damaged = tmp_path / "damaged.mib"
    write_mib(damaged, data, bad_sequence=True)
    with pytest.raises(ValueError, match="discontinuous"):
        inspect_mib(damaged, (2, 3))
    short = tmp_path / "short.mib"
    short.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="trailing"):
        inspect_mib(short, (2, 3))
    with pytest.raises(ValueError, match="file size"):
        load_dataset(short, backend="mib_memmap", scan_shape=(2, 3), detector_shape=(8, 8), dtype=">u2", mib_header_bytes=384)


def test_raw_counts_and_blockwise_validation(tmp_path):
    data = np.ones((3, 4, 8, 8), dtype=np.uint16)
    data[0, 0] = 0
    data[1, 2, 3, 4] = 4095
    data[2, 1, 4, 3] = 4096
    handle = DatasetHandle(data, "test")
    qc, mean, arrays = raw_quality(handle, tmp_path)
    assert qc["zero_frames"] == 1
    assert qc["suspected_saturated_frames"] == 2
    assert qc["pixels_above_12bit_range"] == 1
    assert qc["zero_pixel_fraction"] == 1 / 12
    np.testing.assert_array_equal(mean, data.mean(axis=(0, 1)))
    np.testing.assert_array_equal(arrays["total_counts"], data.sum(axis=(-2, -1)))
    masks = build_annular_masks((8, 8), {"bf": {"inner_radius": 0, "outer_radius": 3}, "total": {"inner_radius": 0, "outer_radius": 20}})
    assert verify_small_block(handle, masks)["status"] == "passed"


def test_beam_estimation_tracks_displaced_disk():
    yy, xx = np.indices((128, 128))
    mean = 2 + 1000 * np.exp(-((yy - 58) ** 2 + (xx - 73) ** 2) / 18)
    center, masks, _ = estimate_geometry(mean)
    np.testing.assert_allclose(center, [58, 73], atol=0.05)
    assert masks["bf"]["outer_radius"] < masks["adf"]["inner_radius"]
    assert masks["adf"]["outer_radius"] <= 54.1


def test_broad_central_region_leaves_complete_adf_annulus():
    yy, xx = np.indices((256, 256))
    mean = 2 + 20 * ((abs(yy - 123) < 40) & (abs(xx - 126) < 40))
    center, masks, _ = estimate_geometry(mean)
    np.testing.assert_allclose(center, [123, 126], atol=0.05)
    assert masks["bf"]["outer_radius"] < masks["adf"]["inner_radius"] < masks["adf"]["outer_radius"]


def test_beam_motion_audit_detects_known_scan_shift():
    from fourdstem_pipeline.beam_audit import measure_motion
    yy, xx = np.indices((64, 64))
    patterns = np.empty((8, 8, 64, 64), dtype=np.float32)
    for y in range(8):
        for x in range(8):
            cy, cx = 24 + 0.9 * y + 0.1 * x, 24 + 0.1 * y + 0.8 * x
            patterns[y, x] = 1 + 1000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 3)
    motion, arrays = measure_motion(DatasetHandle(patterns, "test"))
    assert motion["scan_correlated_motion_warning"]
    assert motion["inlier_fraction"] > 0.95
    np.testing.assert_allclose(motion["affine_coefficients_yx"], [[0.9, 0.1], [0.1, 0.8]], atol=0.05)
    assert arrays["peak_yx"].shape == (64, 2)


def test_complete_basic_analysis_writes_reports_and_representatives(tmp_path):
    import json
    import matplotlib
    matplotlib.use("Agg")
    from fourdstem_pipeline.basic_analysis import analyze_file, batch_report
    from fourdstem_pipeline.synthetic import make_synthetic_4dstem

    patterns, _ = make_synthetic_4dstem(navigation_shape=(8, 8))
    path = tmp_path / "test.mib"
    write_mib(path, (patterns * 100).astype(np.uint16))
    output = tmp_path / "scan"
    with threadpool_limits(limits=2):
        summary = analyze_file(path, output, (8, 8))
    assert summary["status"] == "complete"
    assert sum(summary["class_fractions"]) == 1
    assert (output / "report_zh.html").exists()
    assert (output / "overview.png").exists()
    with np.load(output / "representative_patterns.npz") as representatives:
        assert len(representatives.files) == 4
        assert all(representatives[k].shape == (64, 64) for k in representatives.files)
    validation = json.loads((output / "verification.json").read_text())
    assert validation["status"] == "passed"
    batch_report(tmp_path, [summary, {"file": "bad.mib", "status": "failed", "error": "truncated"}])
    assert "truncated" in (tmp_path / "report_zh.html").read_text(encoding="utf-8")
    assert (tmp_path / "radial_comparison.png").exists()
    from fourdstem_pipeline.beam_audit import audit_completed_batch
    from fourdstem_pipeline.basic_analysis import save_json
    save_json(tmp_path / "batch_summary.json", [summary])
    audit_completed_batch(tmp_path)
    audit_completed_batch(tmp_path)
    assert (output / "beam_motion.json").exists()
    assert (output / "report_zh.html").read_text(encoding="utf-8").count("<!-- beam-audit-start -->") == 1


@pytest.mark.parametrize("enabled", [False, None])
def test_orientation_switch_preserves_default_and_disabled_qc(tmp_path, enabled):
    cfg = {"project": {"output_dir": str(tmp_path)}, "data": {"path": "synthetic://demo"},
           "geometry": {"radial_bins": 16}, "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 7}}},
           "phase_screening": {"n_components": 2, "n_clusters": 2}, "sample_mask": {"enabled": False}}
    if enabled is not None:
        cfg["orientation"] = {"enabled": enabled}
    with threadpool_limits(limits=2), patch("fourdstem_pipeline.workflow.run_stage1_diagnostics", return_value={}), patch("fourdstem_pipeline.workflow.run_orientation_preview", return_value=None) as preview:
        result = run_workflow(cfg)
    assert not result.errors
    assert preview.call_count == (0 if enabled is False else 1)
    import json
    qc = json.loads((tmp_path / "qc_summary.json").read_text())
    codes = {f["code"] for f in qc["flags"]}
    assert ("ORIENTATION_MISSING" in codes) == (enabled is None)


def test_qc_uses_actual_saturation_and_configured_radial_center():
    yy, xx = np.indices((32, 32))
    mean = np.exp(-((yy - 12) ** 2 + (xx - 20) ** 2) / 2)
    result = VirtualImageResult({}, np.zeros((2, 2)), np.zeros((2, 2)), mean, mean)
    flags = []
    _check_saturation(result, flags)
    _check_beam_center(result, flags, radial_center=(12, 20))
    assert not flags
    result.saturation_fraction = np.array([[0, 0.01], [0, 0]])
    _check_saturation(result, flags)
    assert flags[0].code == "SATURATION_HIGH"
    assert flags[0].evidence["saturation_fraction"] == 0.25
