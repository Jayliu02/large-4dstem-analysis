from __future__ import annotations

import unittest
import shutil
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fourdstem_pipeline import (
    Stage1Manifest,
    Stage1ManifestLoadError,
    apply_preprocess,
    build_annular_masks,
    compute_radial_fingerprints,
    compute_virtual_images,
    load_dataset,
    load_roi_candidates,
    load_stage1_manifest,
    run_orientation_preview,
    run_pipeline,
    run_stage1_diagnostics,
    run_workflow,
    screen_phases,
)


WORKSPACE_TMP = Path(__file__).resolve().parents[1] / "test_outputs"
WORKSPACE_TMP.mkdir(exist_ok=True)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.output_dir = WORKSPACE_TMP / self._testMethodName
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

    def _ready_bragg_qc(self):
        return {
            "median_clean_peaks_per_DP": 6,
            "fraction_DP_with_>=4_peaks": 0.75,
            "fraction_DP_with_>=6_peaks": 0.5,
            "fraction_DP_with_>=8_peaks": 0.25,
            "duplicate_fraction_per_pattern_median": 0.0,
            "duplicate_fraction_per_pattern_p90": 0.0,
            "peak_splitting_warning": False,
            "edge_peak_fraction": 0.0,
            "center_tail_peak_fraction": 0.0,
        }

    def test_synthetic_loader_reports_navigation_and_signal_shapes(self):
        dataset = load_dataset("synthetic://demo")
        self.assertEqual(dataset.navigation_shape, (16, 16))
        self.assertEqual(dataset.signal_shape, (64, 64))
        self.assertEqual(dataset.describe()["source"], "synthetic")
        self.assertEqual(dataset.describe()["source_backend"], "synthetic")

    def test_numpy_loaders_do_not_require_pyxem(self):
        data = np.zeros((3, 4, 8, 10), dtype=np.float32)
        npy_path = self.output_dir / "scan.npy"
        npz_path = self.output_dir / "scan.npz"
        np.save(npy_path, data)
        np.savez_compressed(npz_path, data=data + 1)

        npy = load_dataset(npy_path, backend="numpy")
        npz = load_dataset(npz_path, backend="numpy")

        self.assertEqual(npy.navigation_shape, (3, 4))
        self.assertEqual(npy.signal_shape, (8, 10))
        self.assertEqual(npy.describe()["source_backend"], "numpy")
        self.assertEqual(npz.navigation_shape, (3, 4))
        self.assertEqual(npz.signal_shape, (8, 10))
        self.assertEqual(npz.describe()["source_backend"], "numpy")

    def test_raw_export_binning_uses_uint16_mean_not_sum(self):
        from fourdstem_pipeline.preprocess_raw import _bin_Q_mean_uint16, _bin_R_mean_uint16

        class FakeCalibration:
            def __init__(self):
                self.r_pixel_size = 1.0
                self.q_pixel_size = 1.0

            def get_R_pixel_size(self):
                return self.r_pixel_size

            def get_R_pixel_units(self):
                return "nm"

            def set_R_pixel_size(self, value):
                self.r_pixel_size = value

            def get_Q_pixel_size(self):
                return self.q_pixel_size

            def get_Q_pixel_units(self):
                return "A^-1"

            def set_Q_pixel_size(self, value):
                self.q_pixel_size = value

        class FakeCube:
            def __init__(self, data):
                self.data = data
                self.calibration = FakeCalibration()
                self.dims = {}

            def set_dim(self, axis, values, *, units, name):
                self.dims[axis] = (values, units, name)

        raw = np.arange(4 * 4 * 4 * 4, dtype=np.uint16).reshape(4, 4, 4, 4)
        cube = FakeCube(raw.copy())
        cube = _bin_R_mean_uint16(cube, 2)
        cube = _bin_Q_mean_uint16(cube, 2)

        expected = raw.reshape(2, 2, 2, 2, 4, 4).mean(axis=(1, 3))
        expected = expected.reshape(2, 2, 2, 2, 2, 2).mean(axis=(3, 5))
        self.assertEqual(cube.data.shape, (2, 2, 2, 2))
        self.assertEqual(cube.data.dtype, np.uint16)
        np.testing.assert_array_equal(cube.data, np.rint(expected).astype(np.uint16))
        self.assertLess(int(cube.data[0, 0, 0, 0]), int(raw[:2, :2, :2, :2].sum()))

    def test_virtual_images_have_navigation_shape(self):
        dataset = load_dataset("synthetic://demo")
        masks = build_annular_masks(
            dataset.signal_shape,
            {
                "bf": {"inner_radius": 0, "outer_radius": 8},
                "adf": {"inner_radius": 10, "outer_radius": 22},
            },
        )
        result = compute_virtual_images(dataset, masks, output_dir=self.output_dir)
        self.assertEqual(result.images["bf"].shape, dataset.navigation_shape)
        self.assertEqual(result.images["adf"].shape, dataset.navigation_shape)
        self.assertEqual(result.com_x.shape, dataset.navigation_shape)
        self.assertEqual(result.mean_diffraction.shape, dataset.signal_shape)
        self.assertTrue((self.output_dir / "virtual_bf.npy").exists())

    def test_radial_fingerprints_and_phase_screening(self):
        dataset = load_dataset("synthetic://demo")
        fingerprints = compute_radial_fingerprints(dataset, {"center": None}, 24, output_dir=self.output_dir)
        self.assertEqual(fingerprints.profiles.shape, dataset.navigation_shape + (24,))

        reference_a = fingerprints.profiles[:4, :4].mean(axis=(0, 1)).tolist()
        reference_b = fingerprints.profiles[-4:, -4:].mean(axis=(0, 1)).tolist()
        result = screen_phases(
            fingerprints,
            candidate_phases=[
                {"name": "alpha", "reference_profile": reference_a},
                {"name": "beta", "reference_profile": reference_b},
            ],
            n_components=3,
            n_clusters=3,
            output_dir=self.output_dir,
        )
        self.assertEqual(result.labels.shape, dataset.navigation_shape)
        self.assertEqual(result.representative_profiles.shape[1], 24)
        self.assertEqual(set(result.candidate_scores), {"alpha", "beta"})
        self.assertEqual(result.low_confidence_mask.shape, dataset.navigation_shape)

    def test_orientation_preview_roi_path(self):
        dataset = load_dataset("synthetic://demo")
        result = run_orientation_preview(
            dataset,
            binning=(2, 2),
            roi=(4, 12, 4, 12),
            confidence_threshold=0.0,
            output_dir=self.output_dir,
        )
        self.assertEqual(result.orientation_index.shape, (4, 4))
        self.assertEqual(result.phase_label.shape, (4, 4))
        self.assertEqual(result.score.shape, (4, 4))
        self.assertTrue(np.all(result.score >= 0))
        self.assertTrue((self.output_dir / "orientation_index.npy").exists())

    def test_preprocess_changes_navigation_and_signal_shapes(self):
        dataset = load_dataset("synthetic://demo")
        preprocessed = apply_preprocess(dataset, q_crop=[8, 56, 4, 60], q_bin=2, r_bin=4)
        self.assertEqual(preprocessed.navigation_shape, (4, 4))
        self.assertEqual(preprocessed.signal_shape, (24, 28))
        block = preprocessed.data[:2, :2, :, :]
        self.assertEqual(block.shape, (2, 2, 24, 28))

    def test_run_workflow_synthetic_config(self):
        config = {
            "project": {"name": "test", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True, "chunks": {"navigation": [8, 8], "signal": [64, 64]}},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 24},
            "virtual_images": {
                "masks": {
                    "bf": {"inner_radius": 0, "outer_radius": 8},
                    "adf": {"inner_radius": 10, "outer_radius": 22},
                    "ring_1": {"inner_radius": 8, "outer_radius": 16},
                    "ring_2": {"inner_radius": 16, "outer_radius": 28},
                    "ring_3": {"inner_radius": 28, "outer_radius": 31},
                }
            },
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 3, "n_clusters": 3, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        result = run_workflow(config)
        self.assertEqual(result.dataset.navigation_shape, (16, 16))
        self.assertTrue((self.output_dir / "workflow_summary.json").exists())
        self.assertTrue((self.output_dir / "stage1_summary.json").exists())
        self.assertTrue((self.output_dir / "data_contract.json").exists())
        self.assertTrue((self.output_dir / "preprocess_info.json").exists())
        self.assertTrue((self.output_dir / "virtual" / "virtual_bf.npy").exists())
        self.assertTrue((self.output_dir / "virtual" / "virtual_images.npz").exists())
        self.assertTrue((self.output_dir / "fingerprints" / "radial_fingerprints.npy").exists())
        self.assertTrue((self.output_dir / "fingerprints" / "radial_axis.npy").exists())
        self.assertTrue((self.output_dir / "fingerprint_classes" / "fingerprint_class_labels.npy").exists())
        self.assertTrue((self.output_dir / "fingerprint_classes" / "fingerprint_class_labels_cleaned.npy").exists())
        self.assertTrue((self.output_dir / "fingerprint_classes" / "cluster_summary.csv").exists())
        self.assertTrue((self.output_dir / "fingerprint_classes" / "cluster_mean_radial_profiles.npy").exists())
        self.assertTrue((self.output_dir / "orientation" / "orientation_index.npy").exists())
        self.assertTrue((self.output_dir / "05_cluster_diagnostics" / "cluster_summary.csv").exists())
        self.assertTrue((self.output_dir / "05_cluster_diagnostics" / "cluster_cleaned_labels.npy").exists())
        self.assertTrue((self.output_dir / "05_cluster_diagnostics" / "ring_2_over_ring_1.npy").exists())
        self.assertTrue((self.output_dir / "05_cluster_diagnostics" / "cluster_vs_orientation.csv").exists())
        self.assertTrue((self.output_dir / "roi_candidates" / "roi_candidates.yaml").exists())
        self.assertTrue((self.output_dir / "report.html").exists())

    def test_run_workflow_resolves_directory_input(self):
        data_dir = self.output_dir / "data"
        data_dir.mkdir()
        yy, xx = np.indices((16, 16))
        data = np.empty((4, 4, 16, 16), dtype=np.float32)
        for iy in range(4):
            for ix in range(4):
                data[iy, ix] = np.exp(-0.5 * ((np.sqrt((yy - 8) ** 2 + (xx - 8) ** 2) - 3 - iy) / 1.5) ** 2)
        np.save(data_dir / "scan_b.npy", data + 1)
        np.save(data_dir / "scan_a.npy", data)
        config = {
            "project": {"name": "test", "output_dir": str(self.output_dir / "run")},
            "data": {"backend": "hyperspy_pyxem", "directory": str(data_dir), "pattern": "*.npy", "index": 0, "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": None, "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        result = run_workflow(config)
        self.assertTrue(result.dataset.metadata["path"].endswith("scan_a.npy"))
        self.assertTrue((self.output_dir / "run" / "workflow_summary.json").exists())
        summary = json.loads((self.output_dir / "run" / "workflow_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["data_config"]["backend"], "hyperspy_pyxem")
        self.assertEqual(summary["dataset"]["source_backend"], "hyperspy_pyxem")


    def test_parse_roi_rejects_zero_area(self):
        """parse_roi raises ValueError when ROI has zero height or width."""
        from fourdstem_pipeline.array_utils import parse_roi

        # Direct zero area (y1 == y0)
        with self.assertRaises(ValueError):
            parse_roi([64, 64, 192, 256], shape=(512, 512))

        # Zero width (x1 == x0)
        with self.assertRaises(ValueError):
            parse_roi([64, 128, 192, 192], shape=(512, 512))

        # Both zero
        with self.assertRaises(ValueError):
            parse_roi([64, 64, 192, 192], shape=(512, 512))

        # Clamped to zero (all values beyond shape)
        with self.assertRaises(ValueError):
            parse_roi([600, 700, 600, 700], shape=(512, 512))

        # Valid ROI should not raise
        y_s, x_s = parse_roi([64, 128, 192, 256], shape=(512, 512))
        self.assertEqual(y_s, slice(64, 128))
        self.assertEqual(x_s, slice(192, 256))

    def test_run_workflow_skips_zero_area_orientation_roi(self):
        """Workflow skips orientation with info flag when ROI has zero area."""
        config = {
            "project": {"name": "test_zero_roi", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True, "chunks": {"navigation": [8, 8], "signal": [64, 64]}},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 24},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 8}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [64, 64, 192, 192], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        result = run_workflow(config)
        # Orientation should be None
        self.assertIsNone(result.orientation)
        # Workflow should still complete with diagnostics
        self.assertTrue((self.output_dir / "workflow_summary.json").exists())
        self.assertTrue((self.output_dir / "report.md").exists())
        summary = json.loads((self.output_dir / "workflow_summary.json").read_text(encoding="utf-8"))
        # QC should include the ORIENTATION_ROI_INVALID info flag
        qc = summary.get("qc", {})
        flags = qc.get("flags", [])
        self.assertTrue(
            any(f.get("code") == "ORIENTATION_ROI_INVALID" for f in flags),
            f"Expected ORIENTATION_ROI_INVALID flag, got: {[f.get('code') for f in flags]}",
        )
        manifest = Stage1Manifest.load(self.output_dir)
        self.assertTrue(manifest.roi_candidates_path.exists())
        self.assertEqual(manifest.raw["roi_candidates_path"], "roi_candidates/roi_candidates.yaml")

    def test_diagnostics_resilient_to_missing_orientation(self):
        """run_stage1_diagnostics preserves partial results when orientation is None."""
        from fourdstem_pipeline.orientation import OrientationResult
        import numpy as np

        dataset = load_dataset("synthetic://demo")
        fingerprints = compute_radial_fingerprints(dataset, {"center": None}, 8, output_dir=self.output_dir)
        phase = screen_phases(fingerprints, n_components=2, n_clusters=2, output_dir=self.output_dir / "classes")
        masks = build_annular_masks(
            dataset.signal_shape,
            {"bf": {"inner_radius": 0, "outer_radius": 4}, "adf": {"inner_radius": 4, "outer_radius": 8}},
        )
        virtual = compute_virtual_images(dataset, masks, output_dir=self.output_dir)

        # With valid orientation — all outputs present
        orientation = run_orientation_preview(
            dataset, binning=(2, 2), roi=(4, 12, 4, 12),
            confidence_threshold=0.0, output_dir=self.output_dir / "orient",
        )
        result = run_stage1_diagnostics(
            dataset, fingerprints, phase, virtual, orientation,
            output_dir=self.output_dir / "diag",
            png_dir=self.output_dir / "diag_png",
            block_shape=(8, 8),
            confidence_threshold=0.0,
        )
        self.assertIn("cluster_diagnostics", result)
        self.assertIn("cluster_summary_csv", result)
        self.assertIn("beam", result)
        self.assertIn("connected_components", result)
        self.assertIn("orientation_reliability", result)
        self.assertIn("roi_outputs", result)
        self.assertIn("cluster_vs_orientation", result)
        self.assertNotIn("_errors", result)

        # With orientation=None — core outputs present, orientation-dep ones empty
        result_none = run_stage1_diagnostics(
            dataset, fingerprints, phase, virtual, None,
            output_dir=self.output_dir / "diag_none",
            png_dir=self.output_dir / "diag_none_png",
            block_shape=(8, 8),
            confidence_threshold=0.0,
        )
        self.assertIn("cluster_diagnostics", result_none)
        self.assertIn("beam", result_none)
        self.assertEqual(result_none.get("orientation_reliability"), {})
        self.assertEqual(result_none.get("roi_outputs"), {})
        self.assertEqual(result_none.get("cluster_vs_orientation"), {})

    def test_run_workflow_records_load_failure(self):
        """Load-stage exceptions are preserved in WorkflowResult.errors."""
        from unittest.mock import patch

        config = {
            "project": {"name": "test_load_fail", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": None, "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        with patch("fourdstem_pipeline.workflow.load_dataset", side_effect=RuntimeError("boom")):
            result = run_workflow(config)

        self.assertIsNone(result.dataset)
        self.assertTrue(result.errors)
        self.assertEqual(result.errors[0]["stage"], "load")
        self.assertIn("boom", result.errors[0]["error"])
        self.assertTrue((self.output_dir / "qc_summary.json").exists())
        self.assertTrue((self.output_dir / "workflow_summary.json").exists())

    def test_run_workflow_records_preprocess_failure(self):
        """Preprocess-stage exceptions are preserved in WorkflowResult.errors."""
        from unittest.mock import patch

        config = {
            "project": {"name": "test_preprocess_fail", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": None, "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        with patch("fourdstem_pipeline.workflow.apply_preprocess", side_effect=RuntimeError("bad prep")):
            result = run_workflow(config)

        self.assertIsNone(result.dataset)
        self.assertTrue(result.errors)
        self.assertEqual(result.errors[0]["stage"], "preprocess")
        self.assertIn("bad prep", result.errors[0]["error"])
        self.assertTrue((self.output_dir / "qc_summary.json").exists())
        self.assertTrue((self.output_dir / "workflow_summary.json").exists())

    def test_empty_cluster_diagnostics_write_header_only_outputs(self):
        """Diagnostics helpers do not crash when all labels are background."""
        from fourdstem_pipeline.diagnostics_cluster import write_cluster_summary, stats_bar_values, normalisation_comparison
        from fourdstem_pipeline.diagnostics_spatial import connected_component_diagnostics, roi_candidates
        from fourdstem_pipeline.orientation import OrientationResult

        labels = np.full((4, 4), -1, dtype=np.int16)
        images = {"bf": np.ones((4, 4), dtype=np.float32)}
        for path in ("components", "png", "classes", "summary", "norm", "png_norm", "rois"):
            (self.output_dir / path).mkdir(parents=True, exist_ok=True)
        connected = connected_component_diagnostics(
            labels,
            images,
            [],
            self.output_dir / "components",
            self.output_dir / "png",
            self.output_dir / "classes",
        )
        self.assertTrue(Path(connected["connected_components_csv"]).exists())
        self.assertTrue((self.output_dir / "classes" / "fingerprint_class_labels_cleaned.npy").exists())

        write_cluster_summary(self.output_dir / "summary", [])
        self.assertTrue((self.output_dir / "summary" / "cluster_summary.csv").exists())
        self.assertEqual(stats_bar_values([]).shape, (1, 6))
        self.assertEqual(
            normalisation_comparison(
                np.zeros((4, 4, 3), dtype=np.float32),
                0,
                self.output_dir / "norm",
                self.output_dir / "png_norm",
            ),
            {},
        )

        orientation = OrientationResult(
            orientation_index=np.zeros((2, 2), dtype=np.int16),
            phase_label=np.zeros((2, 2), dtype=np.int16),
            score=np.ones((2, 2), dtype=np.float32),
            low_confidence_mask=np.zeros((2, 2), dtype=bool),
            roi=None,
        )
        roi_outputs = roi_candidates(labels, images, orientation, [], self.output_dir / "rois", self.output_dir / "png")
        self.assertTrue(Path(roi_outputs["yaml"]).exists())
        self.assertTrue(Path(roi_outputs["csv"]).exists())

    def test_run_pipeline_smoke_stage1_only_config(self):
        """pipeline_smoke.yaml runs Stage 1 and writes pipeline_summary.json."""
        smoke_root = WORKSPACE_TMP / "pipeline_smoke"
        if smoke_root.exists():
            shutil.rmtree(smoke_root)

        result = run_pipeline("configs/pipeline_smoke.yaml")

        self.assertEqual(result.stages["stage1"].status, "success")
        self.assertEqual(result.stages["stage2a"].status, "skipped")
        self.assertEqual(result.stages["stage2b"].status, "skipped")
        self.assertTrue(result.summary_path.exists())
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["stages"]["stage2a"]["skipped_reason"], "Not listed in pipeline.stages.")

    def test_run_pipeline_injects_stage_paths(self):
        """Unified runner injects stage1_dir and stage2_dir into downstream stages."""
        from types import SimpleNamespace
        from unittest.mock import patch

        stage1_dir = self.output_dir / "stage1"
        stage1_dir.mkdir()
        stage2a_dir = stage1_dir / "stage2" / "roi_bragg"
        stage2a_dir.mkdir(parents=True)
        config = {
            "pipeline": {"stages": ["stage1", "stage2a", "stage2b"], "output_dir": str(self.output_dir / "pipeline")},
            "project": {"name": "pipe_test", "output_dir": str(stage1_dir)},
            "data": {"path": "synthetic://demo"},
            "stage2a": {"thin_r": 2},
            "stage2b": {"candidate_cifs": []},
        }
        fake_stage1 = SimpleNamespace(output_dir=stage1_dir, summary_path=stage1_dir / "workflow_summary.json", errors=[])
        fake_stage2a = SimpleNamespace(output_dir=stage2a_dir, roi_results=[])

        with patch("fourdstem_pipeline.pipeline.run_workflow", return_value=fake_stage1) as wf, \
             patch("fourdstem_pipeline.pipeline.run_stage2", return_value=fake_stage2a) as s2a, \
             patch("fourdstem_pipeline.pipeline.run_stage2_indexing", return_value={"output_dir": str(stage2a_dir / "stage2b_indexing")}) as s2b:
            result = run_pipeline(config)

        self.assertEqual(result.stages["stage1"].status, "success")
        self.assertEqual(result.stages["stage2a"].status, "success")
        self.assertEqual(result.stages["stage2b"].status, "success")
        self.assertEqual(wf.call_args.args[0]["project"]["output_dir"], str(stage1_dir))
        self.assertEqual(s2a.call_args.kwargs["config"]["stage1_dir"], str(stage1_dir))
        self.assertEqual(s2b.call_args.args[0]["stage2_dir"], str(stage2a_dir))

    def test_run_pipeline_skips_after_stage1_failure(self):
        """Stage 2A and 2B are skipped when Stage 1 fails."""
        from types import SimpleNamespace
        from unittest.mock import patch

        stage1_dir = self.output_dir / "stage1_failed"
        config = {
            "pipeline": {"stages": ["stage1", "stage2a", "stage2b"], "output_dir": str(self.output_dir / "pipeline")},
            "project": {"name": "pipe_fail", "output_dir": str(stage1_dir)},
            "data": {"path": "synthetic://demo"},
            "stage2a": {},
            "stage2b": {},
        }
        fake_stage1 = SimpleNamespace(
            output_dir=stage1_dir,
            summary_path=None,
            errors=[{"stage": "load", "error": "boom"}],
        )

        with patch("fourdstem_pipeline.pipeline.run_workflow", return_value=fake_stage1), \
             patch("fourdstem_pipeline.pipeline.run_stage2") as s2a, \
             patch("fourdstem_pipeline.pipeline.run_stage2_indexing") as s2b:
            result = run_pipeline(config)

        self.assertEqual(result.stages["stage1"].status, "failed")
        self.assertEqual(result.stages["stage2a"].status, "skipped")
        self.assertEqual(result.stages["stage2b"].status, "skipped")
        self.assertEqual(result.stages["stage2a"].skipped_reason, "Stage 1 did not complete successfully.")
        s2a.assert_not_called()
        s2b.assert_not_called()

    def test_pipeline_cli_entrypoint_registered_and_callable(self):
        """The unified pipeline command is registered and dispatches run_pipeline."""
        import io
        from types import SimpleNamespace
        from unittest.mock import patch
        from fourdstem_pipeline.cli import pipeline as pipeline_cli
        from fourdstem_pipeline.pipeline import PipelineStageRecord

        fake_result = SimpleNamespace(
            errors=[],
            summary_path=self.output_dir / "pipeline_summary.json",
            stages={"stage1": PipelineStageRecord(name="stage1", status="success")},
        )
        with patch("sys.argv", ["fourdstem-pipeline", "--config", "configs/pipeline_smoke.yaml"]), \
             patch("fourdstem_pipeline.pipeline.run_pipeline", return_value=fake_result) as run_pipe, \
             patch("sys.stdout", new=io.StringIO()) as stdout:
            pipeline_cli()

        self.assertIn("Pipeline finished successfully", stdout.getvalue())
        self.assertEqual(run_pipe.call_args.kwargs["config"], "configs/pipeline_smoke.yaml")
        pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('fourdstem-pipeline = "fourdstem_pipeline.cli:pipeline"', pyproject)

    def test_stage2c_standardises_existing_pyxem_npz(self):
        """Stage 2C converts the standalone pyxem script NPZ into project schema arrays."""
        from fourdstem_pipeline.pyxem_validation import run_stage2c_validation

        npz_path = self.output_dir / "pyxem_ti_phase_orientation_results.npz"
        np.savez_compressed(
            npz_path,
            phase_names=np.array(["Ti-bcc", "Ti-hcp"]),
            best_phase_index=np.array([[0, 1], [0, 1]], dtype=np.int16),
            best_correlation=np.array([[0.8, 0.7], [0.6, 0.5]], dtype=np.float32),
            phase_margin=np.array([[0.2, 0.1], [0.05, np.nan]], dtype=np.float32),
            ambiguous_mask=np.array([[False, False], [True, True]]),
            high_confidence_mask=np.array([[True, True], [False, False]]),
            best_rotation_deg=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )

        out_dir = self.output_dir / "stage2c"
        summary = run_stage2c_validation({
            "output_dir": str(out_dir),
            "input": {"results_npz": str(npz_path)},
            "candidates": {"phases": [{"name": "Ti-bcc"}, {"name": "Ti-hcp"}]},
        })

        self.assertEqual(summary["status"], "success")
        self.assertTrue((out_dir / "stage2c_manifest.json").exists())
        self.assertTrue((out_dir / "phase_index.npy").exists())
        self.assertTrue((out_dir / "score_margin.npy").exists())
        self.assertTrue((out_dir / "pyxem_phase_index.npy").exists())
        self.assertTrue((out_dir / "pyxem_correlation.npy").exists())
        self.assertTrue((out_dir / "pyxem_margin.npy").exists())
        self.assertTrue((out_dir / "pyxem_ambiguous_mask.npy").exists())
        manifest = json.loads((out_dir / "stage2c_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["arrays"]["phase_index"], "phase_index.npy")
        self.assertEqual(manifest["arrays"]["pyxem_phase_index"], "pyxem_phase_index.npy")
        np.testing.assert_array_equal(np.load(out_dir / "phase_index.npy"), np.array([[0, 1], [0, 1]], dtype=np.int16))

    def test_consensus_uses_ti_specific_conflict_labels(self):
        from fourdstem_pipeline.consensus import CONSENSUS_LABELS, _build_consensus_map

        stage2b = {
            "phase_index": np.array([[0, 1, 0], [1, 0, -1]], dtype=np.int16),
            "high_confidence_mask": np.array([[True, True, True], [False, False, False]]),
            "ambiguous_mask": np.array([[False, False, False], [True, True, True]]),
        }
        stage2c = {
            "phase_index": np.array([[0, 1, 1], [1, 0, -1]], dtype=np.int16),
            "high_confidence_mask": np.array([[True, False, True], [True, False, False]]),
            "ambiguous_mask": np.array([[False, True, False], [False, True, True]]),
        }

        consensus = _build_consensus_map(stage2b, stage2c)

        self.assertEqual(CONSENSUS_LABELS[1], "AGREED_Ti_bcc")
        self.assertEqual(CONSENSUS_LABELS[8], "CONFLICT_BCC_HCP")
        self.assertEqual(consensus[0, 0], 1)
        self.assertEqual(consensus[0, 1], 4)
        self.assertEqual(consensus[0, 2], 8)
        self.assertEqual(consensus[1, 0], 6)
        self.assertEqual(consensus[1, 1], 7)
        self.assertEqual(consensus[1, 2], 0)

    def test_run_pipeline_injects_stage2c_paths(self):
        """Unified runner injects Stage 2A/2B paths into Stage 2C."""
        from types import SimpleNamespace
        from unittest.mock import patch

        stage1_dir = self.output_dir / "stage1"
        stage2a_dir = stage1_dir / "stage2" / "roi_bragg"
        stage2b_dir = stage2a_dir / "stage2b_indexing"
        stage2c_dir = stage2a_dir / "stage2c_pyxem_validation"
        stage2a_dir.mkdir(parents=True)
        stage2b_dir.mkdir()
        config = {
            "pipeline": {"stages": ["stage1", "stage2a", "stage2b", "stage2c"], "output_dir": str(self.output_dir / "pipeline")},
            "project": {"name": "pipe_stage2c", "output_dir": str(stage1_dir)},
            "data": {"path": "synthetic://demo"},
            "stage2a": {},
            "stage2b": {"candidate_cifs": []},
            "stage2c": {"input": {"results_npz": "pyxem.npz"}},
        }
        fake_stage1 = SimpleNamespace(output_dir=stage1_dir, summary_path=stage1_dir / "workflow_summary.json", errors=[])
        fake_stage2a = SimpleNamespace(output_dir=stage2a_dir, roi_results=[])
        fake_stage2b = {"output_dir": str(stage2b_dir)}
        fake_stage2c = {
            "status": "success",
            "output_dir": str(stage2c_dir),
            "manifest_path": str(stage2c_dir / "stage2c_manifest.json"),
            "errors": None,
        }

        with patch("fourdstem_pipeline.pipeline.run_workflow", return_value=fake_stage1), \
             patch("fourdstem_pipeline.pipeline.run_stage2", return_value=fake_stage2a), \
             patch("fourdstem_pipeline.pipeline.run_stage2_indexing", return_value=fake_stage2b), \
             patch("fourdstem_pipeline.pipeline.run_stage2c_validation", return_value=fake_stage2c) as s2c:
            result = run_pipeline(config)

        self.assertEqual(result.stages["stage2c"].status, "success")
        self.assertEqual(s2c.call_args.args[0]["stage2_dir"], str(stage2a_dir))
        self.assertEqual(s2c.call_args.args[0]["stage2b_dir"], str(stage2b_dir))

    def test_pyxem_script_roi_uses_project_yx_order(self):
        """Standalone pyxem script follows y0,y1,x0,x1 ROI convention."""
        script = (Path(__file__).resolve().parents[1] / "scripts" / "pyxem_hyperspy_ti_phase_orientation.py").read_text(encoding="utf-8")
        self.assertIn('metavar=("Y0", "Y1", "X0", "X1")', script)
        self.assertIn("s.inav[y0:y1, x0:x1]", script)

    def test_qc_summary_and_report_use_ascii_labels(self):
        """QC summary and report Markdown contain only ASCII labels, no emoji."""
        from fourdstem_pipeline.qc import QCResult, QCFlag, save_qc_summary

        for status, expected_label in [
            ("PASS", "[PASS]"),
            ("PASS_WITH_WARNINGS", "[WARN]"),
            ("FAIL", "[FAIL]"),
        ]:
            result = QCResult(
                stage1_status=status,
                n_warnings=1 if status != "PASS" else 0,
                n_critical=1 if status == "FAIL" else 0,
                flags=[QCFlag(severity="info", code="TEST", message="Test flag.")],
            )
            _, md_path = save_qc_summary(self.output_dir / f"qc_{status}", result)
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn(expected_label, md_text)
            for emoji_char in ["✅", "⚠", "❌", "❓"]:
                self.assertNotIn(emoji_char, md_text,
                                 f"Emoji {repr(emoji_char)} found in QC markdown for {status}")

        # Verify the workflow report Markdown also avoids emoji
        config = {
            "project": {"name": "test_ascii", "output_dir": str(self.output_dir / "run_ascii")},
            "data": {"path": "synthetic://demo", "lazy": True, "chunks": {"navigation": [8, 8], "signal": [64, 64]}},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 24},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 8}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        wf_result = run_workflow(config)
        report_text = wf_result.report_path.read_text(encoding="utf-8")
        for emoji_char in ["✅", "⚠", "❌", "❓"]:
            self.assertNotIn(emoji_char, report_text,
                             f"Emoji {repr(emoji_char)} found in report Markdown")

    def test_no_legacy_diffraction_class_terminology(self):
        """Generated Markdown uses 'fingerprint-class' not 'diffraction-class'."""
        from fourdstem_pipeline.qc import QCResult, QCFlag, save_qc_summary

        # Check QC markdown
        result = QCResult(
            stage1_status="PASS",
            n_warnings=0,
            n_critical=0,
            flags=[QCFlag(
                severity="warning", code="BEAM_CENTER_OFFSET",
                message="Radial fingerprints and fingerprint-class labels may be biased.",
                evidence={},
            )],
        )
        _, md_path = save_qc_summary(self.output_dir / "qc_term", result)
        md_text = md_path.read_text(encoding="utf-8")
        self.assertNotIn("diffraction-class", md_text)

        # Run a full workflow and check the report
        config = {
            "project": {"name": "test_term", "output_dir": str(self.output_dir / "run_term")},
            "data": {"path": "synthetic://demo", "lazy": True, "chunks": {"navigation": [8, 8], "signal": [64, 64]}},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 24},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 8}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        wf_result = run_workflow(config)
        report_text = wf_result.report_path.read_text(encoding="utf-8")
        self.assertNotIn("diffraction-class", report_text)


    # ------------------------------------------------------------------
    # Provenance / dependency reporting
    # ------------------------------------------------------------------

    def test_provenance_reports_scikit_learn_not_sklearn(self):
        """provenance.json reports 'scikit-learn' (the package name), not 'sklearn'."""
        config = {
            "project": {"name": "test_prov", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)
        prov_path = self.output_dir / "provenance.json"
        self.assertTrue(prov_path.exists())
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        packages = prov.get("packages", {})
        # scikit-learn should be reported (not sklearn)
        self.assertIn("scikit-learn", packages)
        self.assertNotIn("sklearn", packages)
        self.assertIsNotNone(packages["scikit-learn"],
                             "scikit-learn should report a version when installed")

    def test_report_has_runtime_dependency_availability_section(self):
        """Both MD and HTML reports use 'Runtime Dependency Availability' not 'Package Versions'."""
        config = {
            "project": {"name": "test_dep", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        result = run_workflow(config)

        md_text = result.report_path.read_text(encoding="utf-8")
        self.assertIn("Runtime Dependency Availability", md_text)
        self.assertNotIn("Package Versions", md_text)

        html_path = result.report_path.with_suffix(".html")
        html_text = html_path.read_text(encoding="utf-8")
        self.assertIn("Runtime Dependency Availability", html_text)
        self.assertNotIn("Package Versions", html_text)

    def test_summary_includes_dependency_fields(self):
        """workflow_summary.json includes 'dependencies' with pyxem/py4DSTEM info."""
        config = {
            "project": {"name": "test_dep2", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)
        summary_path = self.output_dir / "workflow_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        deps = summary.get("dependencies", {})
        self.assertIn("pyxem_available", deps)
        self.assertIn("pyxem_signal_type", deps)
        self.assertIn("py4DSTEM_used", deps)
        self.assertIn("source_backend", deps)
        # py4DSTEM_used should be False since roi_bragg is disabled
        self.assertFalse(deps["py4DSTEM_used"])

        # stage1_summary.json should also include dependencies
        s1_path = self.output_dir / "stage1_summary.json"
        s1 = json.loads(s1_path.read_text(encoding="utf-8"))
        self.assertIn("dependencies", s1)

    def test_provenance_pyxem_reports_not_installed(self):
        """When pyxem is not installed, provenance reports it as null/not-installed."""
        from fourdstem_pipeline.provenance import _installed_packages

        packages = _installed_packages()
        self.assertIn("pyxem", packages)
        # pyxem may or may not be installed in test env, but key should exist
        if packages["pyxem"] is None:
            # Verify the report renders "not installed" label
            from fourdstem_pipeline.export import _render_package_versions
            rendered = _render_package_versions(packages)
            self.assertIn("not installed", rendered)

    # ------------------------------------------------------------------
    # Stage1Manifest contract validation
    # ------------------------------------------------------------------

    def test_stage1_manifest_loads_valid_output(self):
        """Stage1Manifest.load succeeds on a valid Stage-1 output directory."""
        config = {
            "project": {"name": "test_manifest", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)
        manifest = Stage1Manifest.load(self.output_dir)
        self.assertEqual(manifest.run_name, "test_manifest")
        self.assertEqual(manifest.nav_shape, [16, 16])
        self.assertEqual(manifest.sig_shape, [64, 64])
        self.assertEqual(manifest.qc_status, "PASS")
        self.assertTrue(manifest.labels_path.exists())
        self.assertTrue(manifest.roi_candidates_path.exists())
        self.assertTrue(manifest.virtual_images_path.exists())
        self.assertTrue(manifest.fingerprints_path.exists())
        self.assertTrue(manifest.radial_axis_path.exists())
        self.assertTrue(manifest.data_contract_path.exists())
        self.assertTrue(manifest.provenance_path.exists())
        self.assertTrue(manifest.qc_summary_path.exists())
        # Convenience accessors
        self.assertTrue(manifest.has_orientation)
        self.assertTrue(manifest.qc_passed)

    def test_stage1_manifest_missing_file_raises(self):
        """Stage1Manifest.load raises when stage1_summary.json is absent."""
        empty_dir = self.output_dir / "empty"
        empty_dir.mkdir()
        with self.assertRaises(Stage1ManifestLoadError) as ctx:
            Stage1Manifest.load(empty_dir)
        self.assertIn("not found", str(ctx.exception))

    def test_stage1_manifest_missing_required_key_raises(self):
        """Stage1Manifest.load raises when required keys are missing."""
        manifest_dir = self.output_dir / "bad_manifest"
        manifest_dir.mkdir()
        bad = {
            "run_name": "test",
            "nav_shape": [16, 16],
            # Missing sig_shape, labels_path, etc.
        }
        (manifest_dir / "stage1_summary.json").write_text(
            json.dumps(bad), encoding="utf-8"
        )
        with self.assertRaises(Stage1ManifestLoadError) as ctx:
            Stage1Manifest.load(manifest_dir)
        self.assertIn("missing required keys", str(ctx.exception))

    def test_stage1_manifest_fail_qc_raises(self):
        """Stage1Manifest.load raises when qc_status is FAIL."""
        manifest_dir = self.output_dir / "fail_qc"
        manifest_dir.mkdir()
        # Write a minimal manifest with FAIL status
        fail_manifest = {
            "run_name": "test_fail",
            "nav_shape": [16, 16],
            "sig_shape": [64, 64],
            "qc_status": "FAIL",
            "labels_path": "labels.npy",
            "roi_candidates_path": "rois.yaml",
            "virtual_images_path": "virtual.npz",
            "fingerprints_path": "fprints.npy",
            "radial_axis_path": "axis.npy",
            "data_contract_path": "dc.json",
            "provenance_path": "prov.json",
            "qc_summary_path": "qc.json",
        }
        # Create dummy required files with valid content
        for key in ["labels_path", "roi_candidates_path", "virtual_images_path",
                     "fingerprints_path", "radial_axis_path",
                     "provenance_path", "qc_summary_path"]:
            p = manifest_dir / fail_manifest[key]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
        # data_contract.json must have valid bbox_order
        (manifest_dir / "dc.json").write_text(
            json.dumps({"bbox_order": "y0_y1_x0_x1"})
        )
        (manifest_dir / "stage1_summary.json").write_text(
            json.dumps(fail_manifest), encoding="utf-8"
        )
        with self.assertRaises(Stage1ManifestLoadError) as ctx:
            Stage1Manifest.load(manifest_dir)
        self.assertIn("FAIL", str(ctx.exception))

    def test_stage1_manifest_missing_file_on_disk_raises(self):
        """Stage1Manifest.load raises when a required file path does not exist."""
        manifest_dir = self.output_dir / "missing_file"
        manifest_dir.mkdir()
        ok_manifest = {
            "run_name": "test",
            "nav_shape": [16, 16],
            "sig_shape": [64, 64],
            "qc_status": "PASS",
            "labels_path": "labels.npy",
            "roi_candidates_path": "rois.yaml",
            "virtual_images_path": "virtual.npz",
            "fingerprints_path": "fprints.npy",
            "radial_axis_path": "axis.npy",
            "data_contract_path": "dc.json",
            "provenance_path": "prov.json",
            "qc_summary_path": "qc.json",
        }
        # Create only some of the required files (missing labels.npy)
        for key in ["roi_candidates_path", "virtual_images_path",
                     "fingerprints_path", "radial_axis_path", "data_contract_path",
                     "provenance_path", "qc_summary_path"]:
            p = manifest_dir / ok_manifest[key]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
        # labels.npy intentionally not created
        (manifest_dir / "stage1_summary.json").write_text(
            json.dumps(ok_manifest), encoding="utf-8"
        )
        with self.assertRaises(Stage1ManifestLoadError) as ctx:
            Stage1Manifest.load(manifest_dir)
        self.assertIn("does not exist", str(ctx.exception))

    def test_load_stage1_manifest_convenience(self):
        """load_stage1_manifest is a convenience wrapper."""
        config = {
            "project": {"name": "test_conv", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)
        manifest = load_stage1_manifest(self.output_dir)
        self.assertIsInstance(manifest, Stage1Manifest)
        self.assertEqual(manifest.run_name, "test_conv")

    def test_stage1_summary_paths_are_relative(self):
        """All path values in stage1_summary.json are relative (POSIX style)."""
        config = {
            "project": {"name": "test_rel", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)
        s1_path = self.output_dir / "stage1_summary.json"
        s1 = json.loads(s1_path.read_text(encoding="utf-8"))
        for key in ["labels_path", "roi_candidates_path", "virtual_images_path",
                     "fingerprints_path", "radial_axis_path", "data_contract_path",
                     "provenance_path", "qc_summary_path"]:
            val = s1[key]
            self.assertIsNotNone(val, f"{key} should not be None")
            # Paths should use forward slashes (POSIX)
            self.assertNotIn("\\", val, f"{key} should use forward slashes: {val}")
            # Should not be absolute
            self.assertFalse(
                val.startswith("/") or (len(val) > 1 and val[1] == ":"),
                f"{key} should be relative, got: {val}",
            )
            self.assertTrue((self.output_dir / val).exists(), f"{key} should point to an existing file: {val}")

    def test_roi_candidates_loadable(self):
        """ROI candidates YAML can be loaded with load_roi_candidates."""
        config = {
            "project": {"name": "test_roi_load", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)
        yaml_path = self.output_dir / "roi_candidates" / "roi_candidates.yaml"
        self.assertTrue(yaml_path.exists())
        rois = load_roi_candidates(yaml_path)
        self.assertIsInstance(rois, list)
        self.assertGreater(len(rois), 0)
        for roi in rois:
            self.assertIn("name", roi)
            self.assertIn("bbox", roi)
            self.assertIn("center", roi)
            self.assertIn("size", roi)
            # bbox should be [y0, y1, x0, x1]
            bbox = roi["bbox"]
            self.assertEqual(len(bbox), 4)
            self.assertLess(bbox[0], bbox[1], f"y0 < y1 violated: {bbox}")
            self.assertLess(bbox[2], bbox[3], f"x0 < x1 violated: {bbox}")

    # ------------------------------------------------------------------
    # Stage 2A tests (no py4DSTEM required)
    # ------------------------------------------------------------------

    def test_stage2_config_validation(self):
        """Stage 2 config must contain stage1_dir."""
        from fourdstem_pipeline.stage2 import _load_stage2_config

        cfg_dir = self.output_dir / "cfg"
        cfg_dir.mkdir()
        bad_path = cfg_dir / "bad.yaml"
        bad_path.write_text("not_a_dir: true", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            _load_stage2_config(bad_path)
        self.assertIn("stage1_dir", str(ctx.exception))

    def test_stage2_missing_py4dstem_errors_clearly(self):
        """Stage 2 run raises ImportError with a clear message when py4DSTEM is missing."""
        import sys
        from unittest.mock import patch
        from fourdstem_pipeline.stage2 import run_stage2

        # Create a valid Stage-1 output
        config = {
            "project": {"name": "test_s2_err", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)

        # Create a real data file so _resolve_data_path succeeds
        import numpy as np
        data_path = str(self.output_dir / "dummy.npy")
        np.save(data_path, np.zeros((16, 16, 64, 64), dtype=np.float32))

        stage2_cfg = {
            "stage1_dir": str(self.output_dir),
            "output_dir": str(self.output_dir / "stage2"),
            "max_rois": 1,
            "thin_r": 2,
            "bin_q": 2,
            "data_path": data_path,
        }

        # Simulate py4DSTEM not being installed
        with patch.dict(sys.modules, {"py4DSTEM": None}):
            # Also remove any cached import
            for mod_key in list(sys.modules):
                # Preserve the None sentinel that actually blocks the import,
                # including in environments where py4DSTEM is installed.
                if mod_key != "py4DSTEM" and ("py4DSTEM" in mod_key or "py4dstem" in mod_key):
                    sys.modules.pop(mod_key, None)
            with self.assertRaises(ImportError) as ctx:
                run_stage2(stage2_cfg)
            self.assertIn("py4DSTEM", str(ctx.exception))

    def test_stage2_manifest_validation_required(self):
        """Stage 2 rejects a missing stage1_summary.json."""
        from fourdstem_pipeline.stage2 import run_stage2
        from fourdstem_pipeline.contracts import Stage1ManifestLoadError

        empty_dir = self.output_dir / "no_s1"
        empty_dir.mkdir()
        stage2_cfg = {
            "stage1_dir": str(empty_dir),
            "output_dir": str(self.output_dir / "stage2"),
        }
        with self.assertRaises(Stage1ManifestLoadError):
            run_stage2(stage2_cfg)

    def test_stage2_run_creates_output_structure(self):
        """Stage 2 with mocked py4DSTEM creates expected output structure."""
        import sys
        from unittest.mock import MagicMock, patch
        import numpy as np

        # Create valid Stage-1 output first
        config = {
            "project": {"name": "test_s2_struct", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)

        # Mock py4DSTEM to avoid needing the real package
        mock_py4dstem = MagicMock()
        mock_cal = MagicMock()
        mock_cal.get_qy0.return_value = 32.0
        mock_cal.get_qx0.return_value = 32.0
        mock_py4dstem.import_file.return_value = MagicMock(
            data=np.zeros((16, 16, 64, 64), dtype=np.float32),
            calibration=mock_cal,
        )
        mock_py4dstem.__version__ = "0.14.0"
        mock_dc = MagicMock()
        mock_dc.bin_Q.return_value = mock_dc
        mock_dc.data = np.zeros((4, 4, 32, 32), dtype=np.float32)
        mock_bragg = MagicMock()
        mock_hist = MagicMock()
        # Non-zero Bragg vector map so n_peaks > 0
        mock_hist.data = np.eye(64, dtype=np.float32)
        mock_bragg.histogram.return_value = mock_hist
        mock_dc.find_Bragg_disks.return_value = mock_bragg
        mock_py4dstem.DataCube.return_value = mock_dc

        with patch.dict(sys.modules, {"py4DSTEM": mock_py4dstem}):
            from fourdstem_pipeline.stage2 import run_stage2

            stage2_cfg = {
                "stage1_dir": str(self.output_dir),
                "output_dir": str(self.output_dir / "stage2_mock"),
                "max_rois": 1,
                "thin_r": 2,
                "bin_q": 2,
                "scan_shape": [16, 16],
                "data_path": str(self.output_dir / "dummy.mib"),
            }
            # Create a dummy data file so path resolution works
            (self.output_dir / "dummy.mib").write_text("fake mib data")

            result = run_stage2(stage2_cfg)
            self.assertIsNotNone(result)
            self.assertEqual(len(result.roi_results), 1)
            self.assertIsNone(result.roi_results[0].error)

            # Check new coordinate fields
            r0 = result.roi_results[0]
            self.assertEqual(len(r0.stage1_bbox), 4)
            self.assertEqual(len(r0.raw_bbox), 4)
            self.assertEqual(r0.raw_bbox[0], r0.stage1_bbox[0])  # r_bin=1 → identical
            self.assertEqual(r0.raw_bbox[1], r0.stage1_bbox[1])
            self.assertIsNotNone(r0.beam_center_yx)
            self.assertEqual(len(r0.beam_center_yx), 2)
            self.assertIsNotNone(r0.beam_center_source)
            self.assertIsNotNone(r0.reason)
            self.assertGreater(r0.n_peaks, 0)  # non-zero vmap
            mock_py4dstem.import_file.assert_called_once_with(
                str(self.output_dir / "dummy.mib"),
                mem="MEMMAP",
                scan=(16, 16),
            )

        # Check output structure
        s2_dir = self.output_dir / "stage2_mock"
        self.assertTrue((s2_dir / "stage2_summary.json").exists())
        self.assertTrue((s2_dir / "stage2_qc_summary.json").exists())
        self.assertTrue((s2_dir / "provenance.json").exists())

        summary = json.loads((s2_dir / "stage2_summary.json").read_text(encoding="utf-8"))
        self.assertIn("roi_results", summary)
        self.assertEqual(len(summary["roi_results"]), 1)
        r = summary["roi_results"][0]
        self.assertIn("n_bragg_peaks", r)
        self.assertIn("stage1_bbox", r)
        self.assertIn("raw_bbox", r)
        self.assertIn("beam_center_yx", r)
        self.assertIn("beam_center_source", r)
        self.assertIn("bragg_vector_map_path", r)

        # Check per-ROI bragg_summary.json
        roi_dir = s2_dir / ("roi_" + r["name"])
        self.assertTrue((roi_dir / "bragg_summary.json").exists())
        bragg_summary = json.loads((roi_dir / "bragg_summary.json").read_text(encoding="utf-8"))
        self.assertIn("stage1_bbox", bragg_summary)
        self.assertIn("raw_bbox", bragg_summary)
        self.assertIn("beam_center_yx", bragg_summary)
        self.assertIn("beam_center_source", bragg_summary)
        self.assertIn("cluster_validation", bragg_summary)
        self.assertEqual(bragg_summary["dependencies"]["scan_shape"], [16, 16])

        qc = json.loads((s2_dir / "stage2_qc_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(qc["n_rois_total"], 1)
        self.assertEqual(qc["n_rois_success"], 1)
        self.assertEqual(qc["n_rois_failed"], 0)

        self.assertEqual(summary["parameters"]["scan_shape"], [16, 16])
        self.assertEqual(summary["dependencies"]["scan_shape"], [16, 16])

    def test_stage2_report_uses_ascii_verdicts(self):
        """Stage 2A reports use ASCII-safe production labels."""
        from fourdstem_pipeline.export_stage2 import save_stage2_report

        summary = {
            "run_name": "ascii_report",
            "output_dir": str(self.output_dir),
            "manifest": {"run_name": "s1", "nav_shape": [4, 4], "sig_shape": [8, 8], "r_bin": 1, "qc_status": "PASS"},
            "parameters": {"thin_r": 1, "bin_q": 1, "max_rois": 3, "roi_source": "roi_candidates"},
            "beam_center": {"stage1_yx": [4.0, 4.0], "source": "stage1_com"},
            "dependencies": {"data_path": "data.mib"},
            "provenance": {"packages": {}, "pipeline_version": "test"},
            "roi_results": [
                {
                    "name": "ready",
                    "cluster_id": 1,
                    "reason": "largest_cluster",
                    "stage1_bbox": [0, 2, 0, 2],
                    "raw_bbox": [0, 2, 0, 2],
                    "nav_shape": [2, 2],
                    "sig_shape": [8, 8],
                    "n_bragg_peaks": 6,
                    "beam_center_source": "stage1_com",
                    "background_fraction": 0.0,
                    "sample_mask_coverage": 1.0,
                    "bragg_qc": {
                        "median_clean_peaks_per_DP": 6,
                        "fraction_DP_with_>=6_peaks": 0.5,
                        "peak_splitting_warning": False,
                        "edge_peak_fraction": 0.0,
                        "center_tail_peak_fraction": 0.0,
                    },
                    "error": None,
                },
                {
                    "name": "review",
                    "cluster_id": 2,
                    "reason": "warning",
                    "stage1_bbox": [0, 2, 2, 4],
                    "raw_bbox": [0, 2, 2, 4],
                    "nav_shape": [2, 2],
                    "sig_shape": [8, 8],
                    "n_bragg_peaks": 3,
                    "beam_center_source": "detector_center_fallback",
                    "background_fraction": 0.0,
                    "sample_mask_coverage": 1.0,
                    "bragg_qc": {
                        "median_clean_peaks_per_DP": 3,
                        "fraction_DP_with_>=6_peaks": 0.0,
                        "peak_splitting_warning": False,
                        "edge_peak_fraction": 0.0,
                        "center_tail_peak_fraction": 0.0,
                    },
                    "error": None,
                },
                {"name": "failed", "error": "boom"},
            ],
        }

        md_path, html_path = save_stage2_report(self.output_dir, summary)
        md = md_path.read_text(encoding="utf-8")
        html = html_path.read_text(encoding="utf-8")

        self.assertIn("READY_FOR_INDEXING", md)
        self.assertIn("READY_FOR_SCREENING_ONLY", md)
        self.assertIn("[FAIL] boom", md)
        for text in (md, html):
            self.assertTrue(all(ord(ch) < 128 for ch in text))

    def test_stage2_indexing_readiness_contract_shared(self):
        """Stage 2A report and Stage 2B use one shared ROI readiness rule."""
        from fourdstem_pipeline.contracts import is_roi_ready_for_indexing, roi_indexing_readiness
        from fourdstem_pipeline.export_stage2 import _indexing_verdict

        roi = {
            "error": None,
            "n_bragg_peaks": 5,
            "background_fraction": 0.0,
            "sample_mask_coverage": 1.0,
            "beam_center_source": "stage1_com",
            "bragg_qc": {
                "median_clean_peaks_per_DP": 6,
                "fraction_DP_with_>=6_peaks": 0.5,
                "peak_splitting_warning": False,
                "edge_peak_fraction": 0.0,
                "center_tail_peak_fraction": 0.0,
            },
        }
        self.assertTrue(is_roi_ready_for_indexing(roi))
        self.assertEqual(roi_indexing_readiness(roi), "READY_FOR_INDEXING")
        self.assertEqual(_indexing_verdict(roi), "READY_FOR_INDEXING")

        roi["beam_center_source"] = "detector_center_fallback"
        self.assertFalse(is_roi_ready_for_indexing(roi))
        self.assertEqual(roi_indexing_readiness(roi), "READY_FOR_SCREENING_ONLY")
        self.assertEqual(_indexing_verdict(roi), "READY_FOR_SCREENING_ONLY")

        roi["beam_center_source"] = "stage1_com"
        roi["bragg_qc"]["peak_splitting_warning"] = True
        self.assertFalse(is_roi_ready_for_indexing(roi))
        self.assertEqual(roi_indexing_readiness(roi), "NOT_INDEXABLE")

    def test_stage2b_indexing_contract_with_mock_candidates(self):
        """Stage 2B scaffold consumes accepted Stage 2A ROIs and CIF metadata."""
        from fourdstem_pipeline.indexing import run_stage2_indexing

        stage2_dir = self.output_dir / "stage2a"
        stage2_dir.mkdir()
        cif_path = self.output_dir / "candidate.cif"
        cif_path.write_text("data_candidate\n", encoding="utf-8")
        (stage2_dir / "stage2_summary.json").write_text(
            json.dumps({
                "run_name": "mock_stage2a",
                "roi_results": [
                    {
                        "name": "roi_good",
                        "error": None,
                        "n_bragg_peaks": 4,
                        "background_fraction": 0.0,
                        "sample_mask_coverage": 1.0,
                        "beam_center_source": "stage1_com",
                        "bragg_qc": self._ready_bragg_qc(),
                        "bragg_summary_path": str(stage2_dir / "roi_good" / "bragg_summary.json"),
                    },
                    {
                        "name": "roi_skip",
                        "error": None,
                        "n_bragg_peaks": 0,
                        "background_fraction": 0.0,
                        "sample_mask_coverage": 1.0,
                        "beam_center_source": "stage1_com",
                    },
                ],
            }),
            encoding="utf-8",
        )

        output_dir = self.output_dir / "stage2b"
        summary = run_stage2_indexing({
            "stage2_dir": str(stage2_dir),
            "output_dir": str(output_dir),
            "candidate_cifs": [
                {
                    "name": "candidate",
                    "phase": "mock_phase",
                    "path": str(cif_path),
                    "reference_peaks": [1, 2, 3, 4],
                }
            ],
        })

        self.assertTrue((output_dir / "stage2_indexing_summary.json").exists())
        self.assertEqual(summary["status"], "MOCK_SCORED")
        self.assertEqual(summary["accepted_roi_count"], 1)
        self.assertEqual(summary["candidate_cifs"][0]["phase"], "mock_phase")
        self.assertIsNotNone(summary["candidate_cifs"][0]["sha256"])
        self.assertEqual(summary["candidate_cifs"][0]["scoring_mode"], "mock_peak_count")
        self.assertEqual(summary["output_dir"], str(output_dir))
        self.assertEqual(summary["roi_results"][0]["name"], "roi_good")
        self.assertEqual(summary["schema_version"], "stage2b-indexing-v4")
        self.assertEqual(summary["roi_results"][0]["candidate_phase"], "candidate")
        self.assertEqual(summary["roi_results"][0]["match_score"], 1.0)
        self.assertEqual(summary["roi_results"][0]["match_quality"], "mock_scored")
        self.assertEqual(summary["roi_results"][0]["phase_confidence"], "not_scored")
        self.assertIsNone(summary["roi_results"][0]["second_best_candidate"])
        self.assertIsNone(summary["roi_results"][0]["best_zone_axis"])
        self.assertIsNone(summary["roi_results"][0]["score_margin"])
        self.assertTrue((output_dir / "stage2b_standard_manifest.json").exists())
        self.assertEqual(summary["standard_manifest_path"], str(output_dir / "stage2b_standard_manifest.json"))
        self.assertIn(summary["roi_results"][0]["indexability_tier"], {"INDEXABLE", "SCREENING_ONLY", "NOT_INDEXABLE"})

    def test_stage2b_generates_cif_templates_and_matches_roi(self):
        """Stage 2B generates analytic CIF templates and matches ROI mean DPs."""
        from fourdstem_pipeline.indexing import (
            _generate_kinematic_template_stack,
            run_stage2_indexing,
        )

        stage2_dir = self.output_dir / "stage2a"
        roi_dir = stage2_dir / "roi_good"
        roi_dir.mkdir(parents=True)
        cif_path = self.output_dir / "candidate.cif"
        cif_path.write_text(
            "data_candidate\n"
            "_cell_length_a 2.0\n"
            "_cell_length_b 2.0\n"
            "_cell_length_c 2.0\n"
            "_cell_angle_alpha 90\n"
            "_cell_angle_beta 90\n"
            "_cell_angle_gamma 90\n",
            encoding="utf-8",
        )
        stack, _metadata = _generate_kinematic_template_stack(
            {
                "a": 2.0,
                "b": 2.0,
                "c": 2.0,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 90.0,
            },
            sig_shape=(32, 32),
            beam_center_yx=(16.0, 16.0),
            max_index=1,
            orientations_deg=[0.0],
            zone_axis=(0.0, 0.0, 1.0),
            peak_sigma_px=1.0,
            reciprocal_pixels_per_inv_angstrom=8.0,
            intensity_power=2.0,
        )
        roi_data_path = roi_dir / "roi_data.npy"
        np.save(roi_data_path, stack[0][None, None, :, :].astype(np.float32))
        (stage2_dir / "stage2_summary.json").write_text(
            json.dumps({
                "run_name": "template_stage2a",
                "manifest": {"sig_shape": [32, 32]},
                "roi_results": [
                    {
                        "name": "roi_good",
                        "error": None,
                        "n_bragg_peaks": 8,
                        "background_fraction": 0.0,
                        "sample_mask_coverage": 1.0,
                        "beam_center_source": "stage1_com",
                        "beam_center_yx": [16.0, 16.0],
                        "sig_shape": [32, 32],
                        "bragg_qc": self._ready_bragg_qc(),
                        "roi_data_path": str(roi_data_path),
                        "bragg_summary_path": str(roi_dir / "bragg_summary.json"),
                    },
                ],
            }),
            encoding="utf-8",
        )

        output_dir = self.output_dir / "stage2b_template"
        summary = run_stage2_indexing({
            "stage2_dir": str(stage2_dir),
            "output_dir": str(output_dir),
            "template_generation": {
                "max_index": 1,
                "zone_axis": [0, 0, 1],
                "orientations_deg": [0.0],
                "peak_sigma_px": 1.0,
                "reciprocal_pixels_per_inv_angstrom": 8.0,
                "intensity_power": 2.0,
            },
            "candidate_cifs": [
                {"name": "candidate", "phase": "cubic", "path": str(cif_path)}
            ],
        })

        roi_result = summary["roi_results"][0]
        candidate = summary["candidate_cifs"][0]
        self.assertEqual(summary["status"], "TEMPLATE_MATCHED")
        self.assertEqual(summary["schema_version"], "stage2b-indexing-v4")
        self.assertEqual(candidate["scoring_mode"], "template_match")
        self.assertEqual(candidate["template_count"], 1)
        self.assertTrue(Path(candidate["template_stack_path"]).exists())
        self.assertEqual(summary["template_generation"]["zone_axes"], [[0.0, 0.0, 1.0]])
        metadata = json.loads(Path(candidate["template_metadata_path"]).read_text(encoding="utf-8"))
        self.assertEqual(metadata["projections"][0]["mode"], "single_zone_axis_orthographic")
        self.assertEqual(metadata["projections"][0]["zone_axis"], [0.0, 0.0, 1.0])
        self.assertEqual(roi_result["status"], "TEMPLATE_MATCHED")
        self.assertEqual(roi_result["candidate_phase"], "cubic")
        self.assertGreater(roi_result["match_score"], 0.95)
        self.assertEqual(roi_result["match_quality"], "high")
        self.assertEqual(roi_result["orientation_candidate_deg"], 0.0)
        self.assertEqual(roi_result["best_zone_axis"], [0.0, 0.0, 1.0])
        self.assertIn(roi_result["phase_confidence"], ("HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE"))
        self.assertEqual(roi_result["best_phase"], "cubic")
        self.assertIsInstance(roi_result["best_orientation"], dict)
        self.assertIn(roi_result["orientation_confidence"], ("HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE"))
        self.assertIn(roi_result["mapping_confidence"], ("HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE"))
        self.assertIsInstance(roi_result["top_phase_matches"], list)
        self.assertEqual(roi_result["top_phase_matches"][0]["phase"], "cubic")
        self.assertLessEqual(len(roi_result["top_phase_matches"][0]["top_matches"]), summary["matching"]["top_k_per_phase"])
        self.assertIn("radial_support_score", roi_result)
        # Single candidate → no second-best
        self.assertIsNone(roi_result["second_best_candidate"])
        self.assertIsNone(roi_result["score_margin"])
        gallery_html = (stage2_dir / "stage2_gallery.html").read_text(encoding="utf-8")
        self.assertIn("cubic", gallery_html)
        self.assertIn("score:1.000", gallery_html)
        self.assertTrue((roi_dir / "experimental_template_peak_overlay.png").exists())
        self.assertTrue((roi_dir / "radial_q_profile_validation.png").exists())
        self.assertTrue((roi_dir / "phase_orientation_topk.json").exists())
        self.assertTrue((output_dir / "stage2_phase_mapping_report.md").exists())
        self.assertTrue((output_dir / "stage2_phase_mapping_report.html").exists())

    def test_stage2b_missing_candidate_cif_records_null_hash(self):
        """Missing CIF provenance is non-fatal and records sha256 as null."""
        from fourdstem_pipeline.indexing import run_stage2_indexing

        stage2_dir = self.output_dir / "stage2a"
        stage2_dir.mkdir()
        (stage2_dir / "stage2_summary.json").write_text(
            json.dumps({
                "run_name": "mock_stage2a",
                "roi_results": [
                    {
                        "name": "roi_good",
                        "error": None,
                        "n_bragg_peaks": 4,
                        "background_fraction": 0.0,
                        "sample_mask_coverage": 1.0,
                        "beam_center_source": "stage1_com",
                        "bragg_qc": self._ready_bragg_qc(),
                    },
                ],
            }),
            encoding="utf-8",
        )

        summary = run_stage2_indexing({
            "stage2_dir": str(stage2_dir),
            "output_dir": str(self.output_dir / "stage2b_missing_cif"),
            "candidate_cifs": [
                {"name": "missing", "path": str(self.output_dir / "missing.cif")}
            ],
        })

        self.assertIsNone(summary["candidate_cifs"][0]["sha256"])
        self.assertEqual(summary["roi_results"][0]["match_quality"], "not_scored")
        self.assertEqual(summary["roi_results"][0]["phase_confidence"], "not_scored")
        self.assertIsNone(summary["roi_results"][0]["match_score"])
        self.assertIsNone(summary["roi_results"][0]["candidate_phase"])
        self.assertEqual(summary["schema_version"], "stage2b-indexing-v4")

    def test_stage2b_matches_from_bragg_vector_map_without_roi_data(self):
        """Stage 2B still template-matches when Stage 2A skipped roi_data.npy."""
        from fourdstem_pipeline.indexing import (
            _generate_kinematic_template_stack,
            run_stage2_indexing,
        )

        stage2_dir = self.output_dir / "stage2a_no_roi_data"
        roi_dir = stage2_dir / "roi_good"
        roi_dir.mkdir(parents=True)
        cif_path = self.output_dir / "candidate_no_roi_data.cif"
        cif_path.write_text(
            "data_candidate\n"
            "_cell_length_a 2.0\n"
            "_cell_length_b 2.0\n"
            "_cell_length_c 2.0\n"
            "_cell_angle_alpha 90\n"
            "_cell_angle_beta 90\n"
            "_cell_angle_gamma 90\n",
            encoding="utf-8",
        )
        stack, _ = _generate_kinematic_template_stack(
            {"a": 2.0, "b": 2.0, "c": 2.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            sig_shape=(32, 32),
            beam_center_yx=(16.0, 16.0),
            max_index=1,
            orientations_deg=[0.0],
            zone_axis=(0.0, 0.0, 1.0),
            peak_sigma_px=1.0,
            reciprocal_pixels_per_inv_angstrom=8.0,
            intensity_power=2.0,
        )
        bragg_map_path = roi_dir / "bragg_vector_map.npy"
        np.save(bragg_map_path, stack[0].astype(np.float32))
        (stage2_dir / "stage2_summary.json").write_text(
            json.dumps({
                "run_name": "no_roi_data_stage2a",
                "manifest": {"sig_shape": [32, 32]},
                "roi_results": [{
                    "name": "roi_good",
                    "error": None,
                    "n_bragg_peaks": 8,
                    "background_fraction": 0.0,
                    "sample_mask_coverage": 1.0,
                    "beam_center_source": "stage1_com",
                    "beam_center_yx": [16.0, 16.0],
                    "sig_shape": [32, 32],
                    "bragg_qc": self._ready_bragg_qc(),
                    "roi_data_path": None,
                    "bragg_vector_map_path": str(bragg_map_path),
                    "bragg_summary_path": str(roi_dir / "bragg_summary.json"),
                }],
            }),
            encoding="utf-8",
        )

        summary = run_stage2_indexing({
            "stage2_dir": str(stage2_dir),
            "output_dir": str(self.output_dir / "stage2b_no_roi_data"),
            "template_generation": {
                "max_index": 1,
                "zone_axis": [0, 0, 1],
                "orientations_deg": [0.0],
                "peak_sigma_px": 1.0,
                "reciprocal_pixels_per_inv_angstrom": 8.0,
                "intensity_power": 2.0,
            },
            "candidate_cifs": [
                {"name": "candidate", "phase": "cubic", "path": str(cif_path)}
            ],
        })

        roi_result = summary["roi_results"][0]
        self.assertEqual(roi_result["status"], "TEMPLATE_MATCHED")
        self.assertEqual(roi_result["candidate_phase"], "cubic")
        self.assertGreater(roi_result["match_score"], 0.95)

    def test_stage2b_null_candidate_cifs_handled_gracefully(self):
        """Null candidate_cifs (all entries commented out in YAML) is non-fatal."""
        from fourdstem_pipeline.indexing import run_stage2_indexing

        stage2_dir = self.output_dir / "stage2a_null"
        stage2_dir.mkdir()
        (stage2_dir / "stage2_summary.json").write_text(
            json.dumps({
                "run_name": "null_cifs",
                "manifest": {"sig_shape": [32, 32]},
                "roi_results": [
                    {
                        "name": "roi_good",
                        "error": None,
                        "n_bragg_peaks": 4,
                        "background_fraction": 0.0,
                        "sample_mask_coverage": 1.0,
                        "beam_center_source": "stage1_com",
                        "sig_shape": [32, 32],
                        "beam_center_yx": [16.0, 16.0],
                        "bragg_qc": self._ready_bragg_qc(),
                    },
                ],
            }),
            encoding="utf-8",
        )

        summary = run_stage2_indexing({
            "stage2_dir": str(stage2_dir),
            "output_dir": str(self.output_dir / "stage2b_null_cifs"),
            "candidate_cifs": None,  # simulates YAML key with all-commented entries
        })

        self.assertEqual(summary["status"], "NO_TEMPLATES")
        self.assertEqual(summary["accepted_roi_count"], 1)
        self.assertEqual(len(summary["candidate_cifs"]), 0)

    def test_stage2b_cli_entrypoint(self):
        """Stage 2B is available through the CLI module and pyproject script."""
        import io
        import yaml
        from unittest.mock import patch
        from fourdstem_pipeline.cli import stage2b

        stage2_dir = self.output_dir / "stage2a"
        stage2_dir.mkdir()
        (stage2_dir / "stage2_summary.json").write_text(
            json.dumps({
                "run_name": "mock_stage2a",
                "roi_results": [
                    {
                        "name": "roi_good",
                        "error": None,
                        "n_bragg_peaks": 2,
                        "background_fraction": 0.0,
                        "sample_mask_coverage": 1.0,
                        "beam_center_source": "stage1_com",
                        "bragg_qc": self._ready_bragg_qc(),
                    },
                ],
            }),
            encoding="utf-8",
        )
        cfg_path = self.output_dir / "stage2b.yaml"
        cfg_path.write_text(
            yaml.safe_dump({
                "stage2_dir": str(stage2_dir),
                "output_dir": str(self.output_dir / "stage2b_cli"),
                "candidate_cifs": [],
            }),
            encoding="utf-8",
        )

        with patch.object(sys, "argv", ["fourdstem-stage2b", "--config", str(cfg_path)]), \
             patch("sys.stdout", new=io.StringIO()) as stdout:
            stage2b()

        self.assertIn("Stage 2B contract complete", stdout.getvalue())
        summary_path = self.output_dir / "stage2b_cli" / "stage2_indexing_summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["schema_version"], "stage2b-indexing-v4")
        pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('fourdstem-stage2b = "fourdstem_pipeline.cli:stage2b"', pyproject)


    # ------------------------------------------------------------------
    # Stage 2B: multi-zone-axis template matching
    # ------------------------------------------------------------------

    def test_stage2b_multi_zone_axis_template_matching(self):
        """Multi-zone-axis config generates concatenated templates with zone_axis_index."""
        from fourdstem_pipeline.indexing import run_stage2_indexing

        stage2_dir = self.output_dir / "stage2a_mz"
        roi_dir = stage2_dir / "roi_good"
        roi_dir.mkdir(parents=True)
        cif_path = self.output_dir / "candidate_mz.cif"
        cif_path.write_text(
            "data_candidate\n"
            "_cell_length_a 2.0\n"
            "_cell_length_b 2.0\n"
            "_cell_length_c 2.0\n"
            "_cell_angle_alpha 90\n"
            "_cell_angle_beta 90\n"
            "_cell_angle_gamma 90\n",
            encoding="utf-8",
        )
        # Create a small template to use as ROI data (ensures perfect correlation)
        from fourdstem_pipeline.indexing import _generate_kinematic_template_stack
        stack_z0, _ = _generate_kinematic_template_stack(
            {"a": 2.0, "b": 2.0, "c": 2.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            sig_shape=(32, 32), beam_center_yx=(16.0, 16.0),
            max_index=1, orientations_deg=[0.0],
            zone_axis=(0.0, 0.0, 1.0),
            peak_sigma_px=1.0, reciprocal_pixels_per_inv_angstrom=8.0,
            intensity_power=2.0,
        )
        roi_data_path = roi_dir / "roi_data.npy"
        np.save(roi_data_path, stack_z0[0][None, None, :, :].astype(np.float32))
        (stage2_dir / "stage2_summary.json").write_text(
            json.dumps({
                "run_name": "mz_stage2a",
                "manifest": {"sig_shape": [32, 32]},
                "roi_results": [{
                    "name": "roi_good", "error": None, "n_bragg_peaks": 8,
                    "background_fraction": 0.0, "sample_mask_coverage": 1.0,
                    "beam_center_source": "stage1_com",
                    "beam_center_yx": [16.0, 16.0], "sig_shape": [32, 32],
                    "bragg_qc": self._ready_bragg_qc(),
                    "roi_data_path": str(roi_data_path),
                    "bragg_summary_path": str(roi_dir / "bragg_summary.json"),
                }],
            }),
            encoding="utf-8",
        )

        output_dir = self.output_dir / "stage2b_mz"
        summary = run_stage2_indexing({
            "stage2_dir": str(stage2_dir),
            "output_dir": str(output_dir),
            "template_generation": {
                "max_index": 1,
                "zone_axes": [[0, 0, 1], [1, 0, 0]],
                "orientations_deg": [0.0],
                "peak_sigma_px": 1.0,
                "reciprocal_pixels_per_inv_angstrom": 8.0,
                "intensity_power": 2.0,
            },
            "candidate_cifs": [
                {"name": "candidate", "phase": "cubic", "path": str(cif_path)}
            ],
        })

        candidate = summary["candidate_cifs"][0]
        # 2 zone axes × 1 orientation = 2 templates
        self.assertEqual(candidate["template_count"], 2)
        metadata = json.loads(Path(candidate["template_metadata_path"]).read_text(encoding="utf-8"))
        self.assertEqual(metadata["zone_axis_index"], [0, 1])
        self.assertEqual(len(metadata["projections"]), 2)
        self.assertEqual(metadata["zone_axes"], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        roi_result = summary["roi_results"][0]
        self.assertEqual(roi_result["status"], "TEMPLATE_MATCHED")
        self.assertEqual(roi_result["best_zone_axis"], [0.0, 0.0, 1.0])

    def test_stage2b_phase_confidence_thresholds(self):
        """phase_confidence correctly classifies score/margin pairs."""
        from fourdstem_pipeline.indexing import _phase_confidence

        # High: score > 0.60 AND margin > 0.15
        self.assertEqual(_phase_confidence(0.61, 0.16), "HIGH_CONFIDENCE")
        self.assertEqual(_phase_confidence(0.80, 0.20), "HIGH_CONFIDENCE")
        # Medium: score > 0.40 AND margin > 0.08 (but not high)
        self.assertEqual(_phase_confidence(0.61, 0.08), "MEDIUM_CONFIDENCE")   # margin too low for high
        self.assertEqual(_phase_confidence(0.50, 0.20), "MEDIUM_CONFIDENCE")   # score too low for high
        self.assertEqual(_phase_confidence(0.41, 0.07), "MEDIUM_CONFIDENCE")   # just above thresholds
        # Low: everything else
        self.assertEqual(_phase_confidence(0.41, 0.05), "LOW_CONFIDENCE")       # margin too low
        self.assertEqual(_phase_confidence(0.39, 0.20), "LOW_CONFIDENCE")       # score too low
        self.assertEqual(_phase_confidence(0.20, 0.05), "LOW_CONFIDENCE")       # both too low
        # No second-best → low
        self.assertEqual(_phase_confidence(0.90, None), "LOW_CONFIDENCE")

    def test_stage2b_phase_confidence_end_to_end(self):
        """End-to-end: phase_confidence is 'high' when best>>second and 'low' when close."""
        from fourdstem_pipeline.indexing import (
            _generate_kinematic_template_stack,
            run_stage2_indexing,
        )

        stage2_dir = self.output_dir / "stage2a_pc"
        roi_dir = stage2_dir / "roi_good"
        roi_dir.mkdir(parents=True)

        # Candidate A: cubic cell — will match well
        cif_a = self.output_dir / "cand_a.cif"
        cif_a.write_text(
            "data_A\n_cell_length_a 2.0\n_cell_length_b 2.0\n_cell_length_c 2.0\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n",
            encoding="utf-8",
        )
        # Candidate B: different cell — will match poorly
        cif_b = self.output_dir / "cand_b.cif"
        cif_b.write_text(
            "data_B\n_cell_length_a 5.0\n_cell_length_b 5.0\n_cell_length_c 5.0\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n",
            encoding="utf-8",
        )

        # Use candidate A's template as the ROI signal
        stack_a, _ = _generate_kinematic_template_stack(
            {"a": 2.0, "b": 2.0, "c": 2.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            sig_shape=(32, 32), beam_center_yx=(16.0, 16.0),
            max_index=1, orientations_deg=[0.0],
            zone_axis=(0.0, 0.0, 1.0),
            peak_sigma_px=1.0, reciprocal_pixels_per_inv_angstrom=8.0,
            intensity_power=2.0,
        )
        roi_data_path = roi_dir / "roi_data.npy"
        np.save(roi_data_path, stack_a[0][None, None, :, :].astype(np.float32))
        (stage2_dir / "stage2_summary.json").write_text(
            json.dumps({
                "run_name": "pc_stage2a",
                "manifest": {"sig_shape": [32, 32]},
                "roi_results": [{
                    "name": "roi_good", "error": None, "n_bragg_peaks": 8,
                    "background_fraction": 0.0, "sample_mask_coverage": 1.0,
                    "beam_center_source": "stage1_com",
                    "beam_center_yx": [16.0, 16.0], "sig_shape": [32, 32],
                    "bragg_qc": self._ready_bragg_qc(),
                    "roi_data_path": str(roi_data_path),
                    "bragg_summary_path": str(roi_dir / "bragg_summary.json"),
                }],
            }),
            encoding="utf-8",
        )

        output_dir = self.output_dir / "stage2b_pc"
        summary = run_stage2_indexing({
            "stage2_dir": str(stage2_dir),
            "output_dir": str(output_dir),
            "template_generation": {
                "max_index": 1,
                "zone_axis": [0, 0, 1],
                "orientations_deg": [0.0],
                "peak_sigma_px": 1.0,
                "reciprocal_pixels_per_inv_angstrom": 8.0,
                "intensity_power": 2.0,
            },
            "candidate_cifs": [
                {"name": "cand_a", "phase": "cubic_2A", "path": str(cif_a)},
                {"name": "cand_b", "phase": "cubic_5A", "path": str(cif_b)},
            ],
        })

        roi_result = summary["roi_results"][0]
        self.assertEqual(roi_result["candidate_phase"], "cubic_2A")
        self.assertGreater(roi_result["match_score"], 0.95)
        # Candidate A should win by a large margin → high confidence
        self.assertIn(roi_result["phase_confidence"], ("HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE"))
        self.assertIsNotNone(roi_result["score_margin"])
        self.assertGreater(roi_result["score_margin"], 0.15)
        self.assertEqual(roi_result["second_best_candidate"], "cubic_5A")
        self.assertIsNotNone(roi_result["second_best_score"])

    def test_stage2b_score_margin_uses_competing_candidate_not_orientation(self):
        """Score margin compares phase candidates, not symmetry-equivalent orientations."""
        from fourdstem_pipeline.indexing import (
            _generate_kinematic_template_stack,
            run_stage2_indexing,
        )

        stage2_dir = self.output_dir / "stage2a_margin"
        roi_dir = stage2_dir / "roi_good"
        roi_dir.mkdir(parents=True)
        cif_a = self.output_dir / "margin_a.cif"
        cif_a.write_text(
            "data_A\n_cell_length_a 2.0\n_cell_length_b 2.0\n_cell_length_c 2.0\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n",
            encoding="utf-8",
        )
        cif_b = self.output_dir / "margin_b.cif"
        cif_b.write_text(
            "data_B\n_cell_length_a 5.0\n_cell_length_b 5.0\n_cell_length_c 5.0\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n",
            encoding="utf-8",
        )

        stack_a, _ = _generate_kinematic_template_stack(
            {"a": 2.0, "b": 2.0, "c": 2.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            sig_shape=(32, 32), beam_center_yx=(16.0, 16.0),
            max_index=1, orientations_deg=[0.0],
            zone_axis=(0.0, 0.0, 1.0),
            peak_sigma_px=1.0, reciprocal_pixels_per_inv_angstrom=8.0,
            intensity_power=2.0,
        )
        roi_data_path = roi_dir / "roi_data.npy"
        np.save(roi_data_path, stack_a[0][None, None, :, :].astype(np.float32))
        (stage2_dir / "stage2_summary.json").write_text(
            json.dumps({
                "run_name": "margin_stage2a",
                "manifest": {"sig_shape": [32, 32]},
                "roi_results": [{
                    "name": "roi_good", "error": None, "n_bragg_peaks": 8,
                    "background_fraction": 0.0, "sample_mask_coverage": 1.0,
                    "beam_center_source": "stage1_com",
                    "beam_center_yx": [16.0, 16.0], "sig_shape": [32, 32],
                    "bragg_qc": self._ready_bragg_qc(),
                    "roi_data_path": str(roi_data_path),
                    "bragg_summary_path": str(roi_dir / "bragg_summary.json"),
                }],
            }),
            encoding="utf-8",
        )

        summary = run_stage2_indexing({
            "stage2_dir": str(stage2_dir),
            "output_dir": str(self.output_dir / "stage2b_margin"),
            "template_generation": {
                "max_index": 1,
                "zone_axis": [0, 0, 1],
                "orientations_deg": [0.0, 90.0],
                "peak_sigma_px": 1.0,
                "reciprocal_pixels_per_inv_angstrom": 8.0,
                "intensity_power": 2.0,
            },
            "candidate_cifs": [
                {"name": "cand_a", "phase": "phase_a", "path": str(cif_a)},
                {"name": "cand_b", "phase": "phase_b", "path": str(cif_b)},
            ],
        })

        roi_result = summary["roi_results"][0]
        self.assertEqual(roi_result["candidate_phase"], "phase_a")
        self.assertEqual(roi_result["second_best_candidate"], "phase_b")
        self.assertGreater(roi_result["score_margin"], 0.15)

    def test_stage2b_v4_topk_phase_evidence_ranks_two_phase_match(self):
        """v4 records phase-level top-k evidence with radial support for each candidate."""
        from fourdstem_pipeline.indexing import (
            _generate_kinematic_template_stack,
            run_stage2_indexing,
        )

        stage2_dir = self.output_dir / "stage2a_v4_topk"
        roi_dir = stage2_dir / "roi_good"
        roi_dir.mkdir(parents=True)
        cif_a = self.output_dir / "v4_a.cif"
        cif_a.write_text(
            "data_A\n_cell_length_a 2.0\n_cell_length_b 2.0\n_cell_length_c 2.0\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n",
            encoding="utf-8",
        )
        cif_b = self.output_dir / "v4_b.cif"
        cif_b.write_text(
            "data_B\n_cell_length_a 4.5\n_cell_length_b 4.5\n_cell_length_c 4.5\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n",
            encoding="utf-8",
        )
        stack_a, _ = _generate_kinematic_template_stack(
            {"a": 2.0, "b": 2.0, "c": 2.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            sig_shape=(32, 32), beam_center_yx=(16.0, 16.0),
            max_index=1, orientations_deg=[0.0, 45.0, 90.0],
            zone_axis=(0.0, 0.0, 1.0),
            peak_sigma_px=1.0, reciprocal_pixels_per_inv_angstrom=8.0,
            intensity_power=2.0,
        )
        roi_data_path = roi_dir / "roi_data.npy"
        np.save(roi_data_path, stack_a[0][None, None, :, :].astype(np.float32))
        (stage2_dir / "stage2_summary.json").write_text(
            json.dumps({
                "run_name": "v4_topk_stage2a",
                "manifest": {"sig_shape": [32, 32]},
                "roi_results": [{
                    "name": "roi_good", "error": None, "n_bragg_peaks": 8,
                    "background_fraction": 0.0, "sample_mask_coverage": 1.0,
                    "beam_center_source": "stage1_com",
                    "beam_center_yx": [16.0, 16.0], "sig_shape": [32, 32],
                    "bragg_qc": self._ready_bragg_qc(),
                    "roi_data_path": str(roi_data_path),
                    "bragg_summary_path": str(roi_dir / "bragg_summary.json"),
                }],
            }),
            encoding="utf-8",
        )

        summary = run_stage2_indexing({
            "stage2_dir": str(stage2_dir),
            "output_dir": str(self.output_dir / "stage2b_v4_topk"),
            "template_generation": {
                "max_index": 1,
                "zone_axis": [0, 0, 1],
                "orientations_deg": [0.0, 45.0, 90.0],
                "peak_sigma_px": 1.0,
                "reciprocal_pixels_per_inv_angstrom": 8.0,
                "intensity_power": 2.0,
            },
            "matching": {"top_k_per_phase": 2},
            "candidate_cifs": [
                {"name": "cand_a", "phase": "phase_a", "path": str(cif_a)},
                {"name": "cand_b", "phase": "phase_b", "path": str(cif_b)},
            ],
        })

        roi_result = summary["roi_results"][0]
        self.assertEqual(roi_result["best_phase"], "phase_a")
        self.assertEqual(roi_result["top_phase_matches"][0]["phase"], "phase_a")
        self.assertEqual(len(roi_result["top_phase_matches"]), 2)
        self.assertLessEqual(len(roi_result["top_phase_matches"][0]["top_matches"]), 2)
        scores = [m["correlation_score"] for m in roi_result["top_phase_matches"][0]["top_matches"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIsNotNone(roi_result["phase_margin"])
        self.assertIn("radial_support_score", roi_result["top_phase_matches"][0])
        self.assertTrue((roi_dir / "phase_orientation_topk.json").exists())

    def test_stage2b_v4_ambiguous_when_phase_evidence_is_tied(self):
        """v4 keeps ambiguous phase groups when two candidates have tied evidence."""
        from fourdstem_pipeline.indexing import (
            _generate_kinematic_template_stack,
            run_stage2_indexing,
        )

        stage2_dir = self.output_dir / "stage2a_v4_amb"
        roi_dir = stage2_dir / "roi_good"
        roi_dir.mkdir(parents=True)
        cif_a = self.output_dir / "v4_amb_a.cif"
        cif_b = self.output_dir / "v4_amb_b.cif"
        cif_text = (
            "data_same\n_cell_length_a 2.0\n_cell_length_b 2.0\n_cell_length_c 2.0\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        )
        cif_a.write_text(cif_text, encoding="utf-8")
        cif_b.write_text(cif_text, encoding="utf-8")
        stack_a, _ = _generate_kinematic_template_stack(
            {"a": 2.0, "b": 2.0, "c": 2.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            sig_shape=(32, 32), beam_center_yx=(16.0, 16.0),
            max_index=1, orientations_deg=[0.0],
            zone_axis=(0.0, 0.0, 1.0),
            peak_sigma_px=1.0, reciprocal_pixels_per_inv_angstrom=8.0,
            intensity_power=2.0,
        )
        roi_data_path = roi_dir / "roi_data.npy"
        np.save(roi_data_path, stack_a[0][None, None, :, :].astype(np.float32))
        (stage2_dir / "stage2_summary.json").write_text(
            json.dumps({
                "run_name": "v4_amb_stage2a",
                "manifest": {"sig_shape": [32, 32]},
                "roi_results": [{
                    "name": "roi_good", "error": None, "n_bragg_peaks": 8,
                    "background_fraction": 0.0, "sample_mask_coverage": 1.0,
                    "beam_center_source": "stage1_com",
                    "beam_center_yx": [16.0, 16.0], "sig_shape": [32, 32],
                    "bragg_qc": self._ready_bragg_qc(),
                    "roi_data_path": str(roi_data_path),
                    "bragg_summary_path": str(roi_dir / "bragg_summary.json"),
                }],
            }),
            encoding="utf-8",
        )

        summary = run_stage2_indexing({
            "stage2_dir": str(stage2_dir),
            "output_dir": str(self.output_dir / "stage2b_v4_amb"),
            "template_generation": {
                "max_index": 1,
                "zone_axis": [0, 0, 1],
                "orientations_deg": [0.0],
                "peak_sigma_px": 1.0,
                "reciprocal_pixels_per_inv_angstrom": 8.0,
                "intensity_power": 2.0,
            },
            "candidate_cifs": [
                {"name": "cand_a", "phase": "phase_a", "path": str(cif_a)},
                {"name": "cand_b", "phase": "phase_b", "path": str(cif_b)},
            ],
        })

        roi_result = summary["roi_results"][0]
        self.assertEqual(roi_result["phase_call"], "AMBIGUOUS")
        self.assertEqual(roi_result["candidate_phase"], "phase_a / phase_b")
        self.assertLess(roi_result["phase_margin"], summary["matching"]["phase_margin_threshold"])
        self.assertIsNotNone(roi_result["ambiguity_reason"])

    def test_stage2b_backward_compat_zone_axis_singular(self):
        """zone_axis (singular) produces same result as zone_axes with one entry."""
        from fourdstem_pipeline.indexing import _parse_zone_axes

        axes_from_singular = _parse_zone_axes({"zone_axis": [1, 2, 3]})
        self.assertEqual(axes_from_singular, [[1.0, 2.0, 3.0]])

        axes_from_plural = _parse_zone_axes({"zone_axes": [[1, 2, 3]]})
        self.assertEqual(axes_from_plural, [[1.0, 2.0, 3.0]])

        axes_from_default = _parse_zone_axes({})
        self.assertEqual(axes_from_default, [[0.0, 0.0, 1.0]])

        # zone_axes takes precedence when both are present
        axes_both = _parse_zone_axes({"zone_axis": [0, 0, 1], "zone_axes": [[1, 1, 1], [1, 1, 0]]})
        self.assertEqual(axes_both, [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    # ------------------------------------------------------------------
    # Stage 2A correctness: coordinate mapping
    # ------------------------------------------------------------------

    def test_roi_bbox_converts_binned_to_raw(self):
        """ROI bbox in binned coords converts correctly to raw scan coords."""
        # Simulate what _process_one_roi does
        bbox_binned = [10, 20, 30, 40]  # y0, y1, x0, x1
        r_bin = 4
        raw_bbox = [v * r_bin for v in bbox_binned]
        self.assertEqual(raw_bbox, [40, 80, 120, 160])

    def test_roi_bbox_rbin1_is_identity(self):
        """With r_bin=1, binned and raw bbox should be identical."""
        bbox_binned = [5, 15, 8, 24]
        r_bin = 1
        raw_bbox = [v * r_bin for v in bbox_binned]
        self.assertEqual(raw_bbox, bbox_binned)

    def test_roi_bbox_clamping(self):
        """Bbox values are clamped to valid range before conversion."""
        nav_shape_binned = (64, 64)
        bbox = [-5, 70, -10, 80]  # out of bounds
        by0, by1, bx0, bx1 = bbox
        bny, bnx = nav_shape_binned
        by0 = max(0, min(by0, bny))
        by1 = max(by0 + 1, min(by1, bny))
        bx0 = max(0, min(bx0, bnx))
        bx1 = max(bx0 + 1, min(bx1, bnx))
        self.assertEqual([by0, by1, bx0, bx1], [0, 64, 0, 64])

    # ------------------------------------------------------------------
    # Stage 2A correctness: Bragg parameter mapping
    # ------------------------------------------------------------------

    def test_bragg_params_snake_to_camel_mapping(self):
        """Snake-case config params are mapped to py4DSTEM camelCase kwargs."""
        from fourdstem_pipeline.roi_bragg import _convert_bragg_params

        config_kwargs = {
            "corr_power": 1.2,
            "edge_boundary": 20,
            "min_relative_intensity": 0.1,
            "min_peak_spacing": 8,
            "max_num_peaks": 100,
            "cuda": True,
        }
        converted = _convert_bragg_params(config_kwargs)

        self.assertEqual(converted["corrPower"], 1.2)
        self.assertEqual(converted["edgeBoundary"], 20)
        self.assertEqual(converted["minRelativeIntensity"], 0.1)
        self.assertEqual(converted["minPeakSpacing"], 8)
        self.assertEqual(converted["maxNumPeaks"], 100)
        self.assertEqual(converted["CUDA"], True)

    def test_bragg_params_defaults_filled(self):
        """Unspecified params get sensible defaults."""
        from fourdstem_pipeline.roi_bragg import _convert_bragg_params

        converted = _convert_bragg_params({"corr_power": 1.5})
        self.assertEqual(converted["corrPower"], 1.5)
        self.assertEqual(converted["sigma_cc"], 1)
        self.assertEqual(converted["edgeBoundary"], 10)
        self.assertEqual(converted["subpixel"], "poly")

    def test_bragg_params_already_camelcase_passthrough(self):
        """Already-camelCase keys pass through unchanged."""
        from fourdstem_pipeline.roi_bragg import _convert_bragg_params

        converted = _convert_bragg_params({
            "corrPower": 2.0,
            "sigma_cc": 3,
            "CUDA": True,
        })
        self.assertEqual(converted["corrPower"], 2.0)
        self.assertEqual(converted["sigma_cc"], 3)
        self.assertEqual(converted["CUDA"], True)

    # ------------------------------------------------------------------
    # Stage 2A correctness: beam centre
    # ------------------------------------------------------------------

    def test_parse_beam_center_txt(self):
        """beam_center_estimate.txt is parsed correctly."""
        from fourdstem_pipeline.roi_bragg import _parse_beam_center_txt

        txt_path = self.output_dir / "beam_center_estimate.txt"
        txt_path.write_text(
            "estimated_center_yx: [31.523, 32.107]\n"
            "radial_center_yx: [31.500, 31.500]\n"
            "offset_pixels: 0.608\n",
            encoding="utf-8",
        )
        result = _parse_beam_center_txt(txt_path)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 31.523, places=3)
        self.assertAlmostEqual(result[1], 32.107, places=3)

    def test_parse_beam_center_txt_missing_file_returns_none(self):
        """Nonexistent beam_center_estimate.txt returns None."""
        from fourdstem_pipeline.roi_bragg import _parse_beam_center_txt

        result = _parse_beam_center_txt(self.output_dir / "nonexistent.txt")
        self.assertIsNone(result)

    def test_parse_beam_center_txt_malformed_returns_none(self):
        """Malformed beam_center_estimate.txt returns None."""
        from fourdstem_pipeline.roi_bragg import _parse_beam_center_txt

        txt_path = self.output_dir / "bad_beam_center.txt"
        txt_path.write_text("garbage content\nno coordinates here\n", encoding="utf-8")
        result = _parse_beam_center_txt(txt_path)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Stage 2A correctness: Bragg peak QC metrics
    # ------------------------------------------------------------------

    def test_bragg_qc_metrics_empty_vmap(self):
        """Zero-peak vmap returns all-zero fractions."""
        from fourdstem_pipeline.roi_bragg import _compute_bragg_qc_metrics

        vmap = np.zeros((64, 64), dtype=np.float32)
        result = _compute_bragg_qc_metrics(
            vmap, beam_center_yx=(32.0, 32.0), sig_shape=(64, 64),
        )
        self.assertEqual(result["peak_pixel_count"], 0)
        self.assertEqual(result["total_peak_votes"], 0)
        self.assertEqual(result["mean_peak_intensity"], 0.0)
        self.assertIsNone(result["radial_distances"])
        self.assertIsNone(result["radial_distance_mean"])
        self.assertIsNone(result["radial_distance_std"])
        self.assertEqual(result["forbidden_center_zone_fraction"], 0.0)
        self.assertEqual(result["edge_peak_fraction"], 0.0)
        self.assertEqual(result["duplicate_peak_fraction"], 0.0)
        self.assertIsNone(result["beam_center_error_estimate"])

    def test_bragg_qc_metrics_known_peaks(self):
        """Synthetic vmap with peaks at known positions computes correct metrics."""
        from fourdstem_pipeline.roi_bragg import _compute_bragg_qc_metrics

        vmap = np.zeros((64, 64), dtype=np.float32)
        # Place peaks at known positions
        vmap[32, 40] = 5.0   # radius = 8 from center (32,32)
        vmap[32, 24] = 3.0   # radius = 8 from center
        vmap[40, 32] = 2.0   # radius = 8 from center
        vmap[24, 32] = 1.0   # radius = 8 from center
        # Edge peak
        vmap[1, 32] = 1.0    # near top edge
        # Center zone peak
        vmap[32, 33] = 1.0   # radius = 1 from center

        result = _compute_bragg_qc_metrics(
            vmap, beam_center_yx=(32.0, 32.0), sig_shape=(64, 64),
            center_zone_radius=5.0, edge_boundary=10, min_peak_spacing=4.0,
        )
        self.assertEqual(result["peak_pixel_count"], 6)
        self.assertEqual(result["total_peak_votes"], 13)
        self.assertAlmostEqual(result["mean_peak_intensity"], 13 / 6, places=1)
        self.assertIsNotNone(result["radial_distances"])
        self.assertEqual(len(result["radial_distances"]), 6)
        # Centre zone: only the peak at (32,33) with radius 1 is within 5 px
        self.assertAlmostEqual(result["forbidden_center_zone_fraction"], 1 / 6, places=3)
        # Edge: peak at (1, 32) is within 10 px of top edge
        self.assertAlmostEqual(result["edge_peak_fraction"], 1 / 6, places=3)
        # No duplicates (all peaks are > 4 px apart)
        self.assertEqual(result["duplicate_peak_fraction"], 0.0)
        # Beam center error should be small (symmetric pattern centered on beam)
        self.assertIsNotNone(result["beam_center_error_estimate"])

    def test_bragg_qc_metrics_duplicate_peaks(self):
        """Closely-spaced peaks produce high duplicate fraction."""
        from fourdstem_pipeline.roi_bragg import _compute_bragg_qc_metrics

        vmap = np.zeros((64, 64), dtype=np.float32)
        # Cluster of peaks within 2 px of each other
        vmap[32, 32] = 1.0
        vmap[32, 33] = 1.0  # distance 1 from (32,32)
        vmap[33, 32] = 1.0  # distance 1 from (32,32)
        # Isolated peak far away
        vmap[50, 50] = 1.0

        result = _compute_bragg_qc_metrics(
            vmap, beam_center_yx=(32.0, 32.0), sig_shape=(64, 64),
            min_peak_spacing=4.0,
        )
        # 3 out of 4 peaks have a neighbor within min_peak_spacing
        # (32,32), (32,33), (33,32) all have close neighbors; (50,50) is isolated
        self.assertEqual(result["peak_pixel_count"], 4)
        self.assertGreater(result["duplicate_peak_fraction"], 0.5)

    def test_bragg_qc_metrics_no_beam_center(self):
        """Metrics gracefully handle missing beam center."""
        from fourdstem_pipeline.roi_bragg import _compute_bragg_qc_metrics

        vmap = np.zeros((64, 64), dtype=np.float32)
        vmap[32, 40] = 1.0
        result = _compute_bragg_qc_metrics(
            vmap, beam_center_yx=None, sig_shape=(64, 64),
        )
        self.assertIsNone(result["radial_distances"])
        self.assertIsNone(result["radial_distance_mean"])
        self.assertIsNone(result["beam_center_error_estimate"])
        self.assertEqual(result["forbidden_center_zone_fraction"], 0.0)
        # Edge and duplicate should still work without beam center
        self.assertIsNotNone(result["edge_peak_fraction"])
        self.assertIsNotNone(result["duplicate_peak_fraction"])

    def test_bragg_peaks_parquet_empty(self):
        """Empty Bragg detection produces null parquet path."""
        from unittest.mock import MagicMock
        from fourdstem_pipeline.roi_bragg import _save_bragg_peaks_table

        bragg = MagicMock()
        bragg.raw.__getitem__.side_effect = IndexError("no data")
        path, summary = _save_bragg_peaks_table(
            bragg, self.output_dir, scan_shape=(4, 4),
        )
        self.assertIsNone(path)
        self.assertEqual(summary["parquet_rows"], 0)
        self.assertIsNone(summary["peaks_per_pattern_mean"])

    def test_bragg_peaks_parquet_saves_correctly(self):
        """Parquet output contains expected columns and row count."""
        from unittest.mock import MagicMock
        from fourdstem_pipeline.roi_bragg import _save_bragg_peaks_table
        import pandas as pd

        # Build a minimal mock that mimics py4DSTEM's BVects/PointListArray
        class MockBVects:
            def __init__(self, qy_vals, qx_vals, i_vals):
                self.qy = np.asarray(qy_vals, dtype=np.float64)
                self.qx = np.asarray(qx_vals, dtype=np.float64)
                self.I = np.asarray(i_vals, dtype=np.float64)
                self.data = np.zeros(len(qy_vals), dtype=[
                    ("qy", np.float64), ("qx", np.float64), ("intensity", np.float64),
                ])
                self.data["qy"] = self.qy
                self.data["qx"] = self.qx
                self.data["intensity"] = self.I

        class MockRaw:
            def __getitem__(self, key):
                rx, ry = key
                if rx == 0 and ry == 0:
                    return MockBVects([10.0, 20.0], [30.0, 40.0], [0.8, 0.6])
                if rx == 1 and ry == 0:
                    return MockBVects([15.0], [35.0], [0.9])
                return MockBVects([], [], [])

        bragg = MagicMock()
        bragg.raw = MockRaw()

        path, summary = _save_bragg_peaks_table(
            bragg, self.output_dir, scan_shape=(4, 4),
        )
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())
        self.assertEqual(summary["parquet_rows"], 3)

        df = pd.read_parquet(path)
        self.assertEqual(len(df), 3)
        self.assertListEqual(list(df.columns), ["scan_y", "scan_x", "qy", "qx", "intensity", "snr"])
        # First two peaks at scan position (0,0)
        self.assertEqual(df.iloc[0]["scan_y"], 0)
        self.assertEqual(df.iloc[0]["scan_x"], 0)
        self.assertAlmostEqual(df.iloc[0]["qy"], 10.0)
        self.assertAlmostEqual(df.iloc[0]["qx"], 30.0)
        # Third peak at (1,0)
        self.assertEqual(df.iloc[2]["scan_y"], 0)
        self.assertEqual(df.iloc[2]["scan_x"], 1)

    def test_per_dp_duplicates_do_not_pool_across_scan_pixels(self):
        """Repeated q positions in different DPs are not duplicates."""
        from fourdstem_pipeline.roi_bragg import _compute_per_pattern_peak_qc
        import pandas as pd

        df = pd.DataFrame({
            "scan_y": [0, 0, 1, 1],
            "scan_x": [0, 1, 0, 1],
            "qy": [10.0, 10.0, 10.0, 10.0],
            "qx": [20.0, 20.0, 20.0, 20.0],
            "intensity": [1.0, 1.0, 1.0, 1.0],
        })
        qc = _compute_per_pattern_peak_qc(df, scan_shape=(2, 2), min_peak_spacing=4)
        self.assertEqual(qc["duplicate_fraction_per_pattern_median"], 0.0)
        self.assertEqual(qc["duplicate_fraction_per_pattern_p90"], 0.0)
        self.assertFalse(qc["peak_splitting_warning"])
        self.assertEqual(qc["median_clean_peaks_per_DP"], 1.0)

    def test_per_dp_duplicate_warning_for_same_pattern_close_peaks(self):
        """Close peaks in the same DP trigger peak splitting warning."""
        from fourdstem_pipeline.roi_bragg import _compute_per_pattern_peak_qc
        import pandas as pd

        df = pd.DataFrame({
            "scan_y": [0, 0, 0, 1, 1, 1],
            "scan_x": [0, 0, 0, 1, 1, 1],
            "qy": [10.0, 11.0, 40.0, 12.0, 13.0, 41.0],
            "qx": [20.0, 20.5, 40.0, 21.0, 21.5, 41.0],
            "intensity": [1.0] * 6,
        })
        qc = _compute_per_pattern_peak_qc(df, scan_shape=(2, 2), min_peak_spacing=4)
        self.assertGreater(qc["duplicate_fraction_per_pattern_p90"], 0.5)
        self.assertTrue(qc["peak_splitting_warning"])

    def test_calibration_sweep_grid_and_best_selection(self):
        """Calibration sweep uses the fixed grid and ranks by evidence metrics."""
        from fourdstem_pipeline.evidence_qc import calibration_sweep_grid, select_best_calibration_run

        grid = calibration_sweep_grid(0.0192)
        self.assertEqual(len(grid), 45)
        self.assertAlmostEqual(grid[0].inv_ang_per_pixel, 0.0192 * 0.97)
        best = select_best_calibration_run([
            {"radial_support_median": 0.0, "matched_template_fraction_median": 0.2, "unexplained_peak_fraction_median": 0.8, "q_residual_median": 5.0},
            {"radial_support_median": 0.3, "matched_template_fraction_median": 0.1, "unexplained_peak_fraction_median": 0.9, "q_residual_median": 8.0},
        ])
        self.assertEqual(best["radial_support_median"], 0.3)

    def test_stage2_gallery_generates_self_contained_html(self):
        """Gallery HTML uses relative img refs + cross-ROI comparison section."""
        from fourdstem_pipeline.export_stage2 import save_stage2_gallery

        # Create two fake ROI directories with PNGs
        import struct, zlib
        def _tiny_png():
            sig = b'\x89PNG\r\n\x1a\n'
            ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data) & 0xFFFFFFFF)
            idat_raw = zlib.compress(b'\x00\xff\x00\x00')
            idat_crc = struct.pack('>I', zlib.crc32(b'IDAT' + idat_raw) & 0xFFFFFFFF)
            return sig + struct.pack('>I', 13) + b'IHDR' + ihdr_data + ihdr_crc \
                 + struct.pack('>I', len(idat_raw)) + b'IDAT' + idat_raw + idat_crc \
                 + struct.pack('>I', 0) + b'IEND' + struct.pack('>I', zlib.crc32(b'IEND') & 0xFFFFFFFF)

        for name in ("roi_a", "roi_b"):
            roi_dir = self.output_dir / name
            roi_dir.mkdir(parents=True)
            (roi_dir / "mean_dp.png").write_bytes(_tiny_png())

        summary = {
            "run_name": "gallery_test",
            "stage1_dir": str(self.output_dir),
            "roi_results": [
                {"name": "roi_a", "error": None,
                 "bragg_summary_path": str(self.output_dir / "roi_a" / "bragg_summary.json"),
                 "n_bragg_peaks": 42, "beam_center_source": "stage1_com",
                 "background_fraction": 0.05, "candidate_phase": "Ti-bcc", "match_score": 0.43},
                {"name": "roi_b", "error": None,
                 "bragg_summary_path": str(self.output_dir / "roi_b" / "bragg_summary.json"),
                 "n_bragg_peaks": 18, "beam_center_source": "stage1_com",
                 "background_fraction": 0.02},
            ],
        }
        gallery_path = save_stage2_gallery(self.output_dir, summary)
        self.assertIsNotNone(gallery_path)
        self.assertTrue(gallery_path.exists())

        html = gallery_path.read_text(encoding="utf-8")
        # Structural checks
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Per-ROI Detail", html)
        # Two ROIs → cross-ROI comparison section should be present
        self.assertIn("Cross-ROI Comparison", html)
        # Relative img references, NOT base64
        self.assertIn('<img src="roi_a/mean_dp.png"', html)
        self.assertIn('<img src="roi_b/mean_dp.png"', html)
        self.assertNotIn("data:image/png;base64,", html)
        self.assertNotIn('src=""', html)
        # Metadata labels
        self.assertIn("42 peaks", html)
        self.assertIn("Ti-bcc", html)
        self.assertIn("score:0.430", html)
        # Comparison has both ROIs' images
        self.assertIn("comp-card", html)

    # ------------------------------------------------------------------
    # Stage 2A correctness: cluster validation
    # ------------------------------------------------------------------

    def test_cluster_validation_background_fraction(self):
        """Background fraction is computed correctly from labels."""
        from fourdstem_pipeline.roi_bragg import _validate_roi_cluster_binned

        # 10x10 labels with half background (-1)
        labels = np.zeros((10, 10), dtype=np.int16)
        labels[5:, :] = -1  # bottom half = background

        result = _validate_roi_cluster_binned(
            stage1_bbox=[0, 10, 0, 10],
            y0=0, y1=10, x0=0, x1=10,
            labels=labels,
            sample_mask=None,
            r_bin=1,
        )
        self.assertEqual(result["background_fraction"], 0.5)
        self.assertTrue(result["labels_available"])

    def test_cluster_validation_no_background(self):
        """ROI entirely within sample has 0 background fraction."""
        from fourdstem_pipeline.roi_bragg import _validate_roi_cluster_binned

        labels = np.ones((8, 8), dtype=np.int16)  # all cluster 1, no -1
        result = _validate_roi_cluster_binned(
            stage1_bbox=[0, 4, 0, 4],
            y0=0, y1=4, x0=0, x1=4,
            labels=labels,
            sample_mask=None,
            r_bin=1,
        )
        self.assertEqual(result["background_fraction"], 0.0)
        self.assertIsNone(result["warning"])

    def test_cluster_validation_high_background_warns(self):
        """ROI with >50% background produces a warning."""
        from fourdstem_pipeline.roi_bragg import _validate_roi_cluster_binned

        labels = np.full((8, 8), -1, dtype=np.int16)
        labels[:2, :] = 0  # only 2 rows non-background
        result = _validate_roi_cluster_binned(
            stage1_bbox=[0, 8, 0, 8],
            y0=0, y1=8, x0=0, x1=8,
            labels=labels,
            sample_mask=None,
            r_bin=1,
        )
        self.assertAlmostEqual(result["background_fraction"], 0.75, places=2)
        self.assertIsNotNone(result["warning"])
        self.assertIn("75", result["warning"])  # percentage mention

    def test_cluster_validation_sample_mask_coverage(self):
        """Sample mask coverage is computed correctly."""
        from fourdstem_pipeline.roi_bragg import _validate_roi_cluster_binned

        sample_mask = np.zeros((8, 8), dtype=bool)
        sample_mask[:4, :] = True  # top half = sample

        result = _validate_roi_cluster_binned(
            stage1_bbox=[0, 8, 0, 8],
            y0=0, y1=8, x0=0, x1=8,
            labels=None,
            sample_mask=sample_mask,
            r_bin=1,
        )
        self.assertEqual(result["sample_mask_coverage"], 0.5)

    def test_cluster_validation_zero_coverage_warns(self):
        """ROI with 0% sample coverage produces a warning."""
        from fourdstem_pipeline.roi_bragg import _validate_roi_cluster_binned

        sample_mask = np.zeros((8, 8), dtype=bool)  # all background
        result = _validate_roi_cluster_binned(
            stage1_bbox=[0, 4, 0, 4],
            y0=0, y1=4, x0=0, x1=4,
            labels=None,
            sample_mask=sample_mask,
            r_bin=1,
        )
        self.assertEqual(result["sample_mask_coverage"], 0.0)
        self.assertIsNotNone(result["warning"])
        self.assertIn("0%", result["warning"])

    # ------------------------------------------------------------------
    # Stage 2A: real-data smoke test (guarded by env var)
    # ------------------------------------------------------------------

    def test_stage2_real_data_smoke(self):
        """Optional: run one ROI against real data if FOURDSTEM_REAL_DATA=1."""
        import os
        if os.environ.get("FOURDSTEM_REAL_DATA") != "1":
            self.skipTest("Set FOURDSTEM_REAL_DATA=1 to run real-data smoke test.")

        import yaml
        from fourdstem_pipeline.stage2 import run_stage2
        from pathlib import Path

        # Use the unified config's Stage 2A section but cap to 1 ROI.
        config_path = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"
        if not config_path.exists():
            self.skipTest(f"Config not found: {config_path}")

        pipeline_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        stage2_cfg = dict(pipeline_cfg["stage2a"])
        stage2_cfg["stage1_dir"] = pipeline_cfg["project"]["output_dir"]
        stage2_cfg["max_rois"] = 1

        result = run_stage2(stage2_cfg)

        # Acceptance criteria
        self.assertGreater(len(result.roi_results), 0, "No ROIs processed.")
        r = result.roi_results[0]
        self.assertIsNone(r.error, f"ROI failed: {r.error}")

        # Coordinate correctness
        self.assertEqual(len(r.stage1_bbox), 4, "stage1_bbox missing/invalid")
        self.assertEqual(len(r.raw_bbox), 4, "raw_bbox missing/invalid")
        # raw_bbox should be r_bin * stage1_bbox
        r_bin = result.manifest.r_bin
        for i in range(4):
            self.assertEqual(
                r.raw_bbox[i], r.stage1_bbox[i] * r_bin,
                f"raw_bbox[{i}]={r.raw_bbox[i]} != stage1_bbox[{i}]={r.stage1_bbox[i]} * r_bin={r_bin}"
            )

        # Beam centre recorded
        self.assertIsNotNone(r.beam_center_yx, "beam_center_yx not recorded")
        self.assertIsNotNone(r.beam_center_source, "beam_center_source not recorded")
        self.assertIn(
            r.beam_center_source,
            ("stage1_com", "py4dstem_calibration", "detector_center_fallback"),
        )

        # Output files exist
        self.assertTrue(r.bragg_summary_path.exists(), "bragg_summary.json missing")
        self.assertTrue(r.bragg_vector_map_path.exists(), "bragg_vector_map.npy missing")

        # bragg_summary.json has required fields
        import json
        bragg_summary = json.loads(r.bragg_summary_path.read_text(encoding="utf-8"))
        for field in ("stage1_bbox", "raw_bbox", "beam_center_yx",
                       "beam_center_source", "n_bragg_peaks", "dependencies"):
            self.assertIn(field, bragg_summary, f"bragg_summary.json missing '{field}'")

        # Verify scan shape was recorded
        self.assertIn("scan_shape", bragg_summary["dependencies"])
        self.assertIn("py4dstem_version", bragg_summary["dependencies"])


if __name__ == "__main__":
    unittest.main()
