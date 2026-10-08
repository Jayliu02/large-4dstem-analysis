"""Validate persisted first-round products without rereading the entire MIB."""
import argparse
import json
from pathlib import Path
import stat
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fourdstem_pipeline.mib_round1 import digest, write_json


def validate(root):
    baseline = json.loads((root / "P0_baseline/baseline_manifest.json").read_text(encoding="utf-8"))
    for entry in baseline:
        assert digest(entry["source"]) == entry["sha256"] == digest(entry["backup"])
        assert not Path(entry["backup"]).stat().st_mode & stat.S_IWRITE
    metadata = json.loads((root / "P0_baseline/metadata.json").read_text(encoding="utf-8"))
    source = Path(metadata["input"])
    assert source.stat().st_size == metadata["size_bytes"]
    assert source.stat().st_mtime_ns == metadata["mtime_ns"]
    qc = json.loads((root / "P1_origin/origin_qc.json").read_text(encoding="utf-8"))
    sha = digest(root / "P1_origin/origin_model.npz")
    assert qc["gate1_passed"] and qc["origin_model_sha256"] == sha
    train = np.load(root / "P1_origin/origin_measured_sampled.npz")
    val = np.load(root / "P1_origin/origin_validation.npz")
    assert len(val["scan_y"]) >= 40
    assert not set(zip(train["scan_y"], train["scan_x"])).intersection(zip(val["scan_y"], val["scan_x"]))
    model = np.load(root / "P1_origin/origin_model.npz")
    shape = tuple(metadata["shape"][:2])
    yy, xx = np.indices(shape)
    for axis in ["x", "y"]:
        p = model[f"coefficients_{axis}"]
        assert model[f"origin_{axis}"].shape == shape
        np.testing.assert_allclose(model[f"origin_{axis}"], p[0]+p[1]*xx+p[2]*yy)
    vqc = json.loads((root / "P2_virtual/virtual_qc.json").read_text(encoding="utf-8"))
    assert vqc["origin_model_sha256"] == sha
    total = np.load(root / "P2_virtual/total.npy")
    assert total.shape == shape and (total > 0).all()
    for name in ["bf", "adf"]:
        counts = np.load(root / f"P2_virtual/{name}_corrected.npy")
        normalized = np.load(root / f"P2_virtual/{name}_normalized.npy")
        assert (counts <= total).all()
        np.testing.assert_allclose(normalized, counts/total)
    peaks = np.load(root / "P3_bragg/bragg_raw_roi_01.npz")
    assert str(peaks["origin_model_sha256"]) == sha
    y0, y1, x0, x1 = peaks["roi"]
    p = peaks["peaks"]
    assert ((p["scan_y"] >= y0) & (p["scan_y"] < y1)).all()
    assert ((p["scan_x"] >= x0) & (p["scan_x"] < x1)).all()
    assert ((p["qx_raw_px"] >= 0) & (p["qx_raw_px"] < metadata["shape"][3])).all()
    assert ((p["qy_raw_px"] >= 0) & (p["qy_raw_px"] < metadata["shape"][2])).all()
    counts = np.load(root / "P3_bragg/peak_count_map_01.npy")
    actual = np.zeros(counts.shape, dtype=int)
    np.add.at(actual, (p["scan_y"]-y0, p["scan_x"]-x0), 1)
    np.testing.assert_array_equal(actual, counts)
    result = dict(artifact_checks_passed=True, baseline_files_preserved=len(baseline),
        validation_count=len(val["scan_y"]), validation_median_px=qc["validation_median_px"],
        validation_p95_px=qc["validation_p95_px"], fullscan_virtual_patterns=int(total.size),
        roi_patterns=int(counts.size), raw_peak_count=len(p),
        origin_model_sha256=sha, raw_mib_size_and_mtime_unchanged=True,
        raw_mib_hash_note="Full 8 GiB hash not computed; read-only mmap, size and mtime checked.",
        artifacts={str(f.relative_to(root)): digest(f) for f in sorted(root.rglob("*"))
                   if f.is_file() and f.name != "run_manifest.json"},
        source_sha256={str(f): digest(f) for f in [Path("src/fourdstem_pipeline/mib_round1.py"),
                          Path("configs/mib_3-1.yaml"), Path("tests/test_mib_round1.py")]})
    write_json(root / "run_manifest.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ["artifacts", "source_sha256"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/mib_pipeline")
    validate(Path(parser.parse_args().output))
