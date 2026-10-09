import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from fourdstem_pipeline.mib_round3 import (
    DTYPE, accept_candidates, candidate_union, match_points, refine_candidate, split_counts, write_calibration,
)


def config():
    return yaml.safe_load(Path("configs/mib_round3.yaml").read_text(encoding="utf-8"))


def test_assignment_maximizes_unique_valid_pairs():
    a = [[0, 0], [.9, 0]]
    b = [[.4, 0], [-.6, 0]]
    pairs = match_points(a, b, 1)
    assert set(map(tuple, pairs)) == {(0, 1), (1, 0)}
    assert match_points(a, [], 1).shape == (0, 2)
    assert len(match_points([[0, 0]], [[5, 5]], 1)) == 0


def test_hybrid_union_preserves_scores_and_column_row_convention():
    dtype = [("qx", "f8"), ("qy", "f8"), ("intensity", "f8")]
    raw = np.array([(20, 40, 100), (50, 70, 20)], dtype=dtype)
    template = np.array([(20.2, 40.1, 3), (80, 90, 1)], dtype=dtype)
    rows = candidate_union(raw, template, 3)
    assert len(rows) == 3
    np.testing.assert_allclose(rows[0], [40.1, 20.2, 3, 100, 3])
    assert rows[1][2] == 2 and np.isnan(rows[1][3])
    assert rows[2][2] == 1 and np.isnan(rows[2][4])


def test_centroid_refines_gaussian_and_flags_single_pixel_contamination():
    y, x = np.indices((64, 80))
    dp = 5+150*np.exp(-((x-42.3)**2+(y-24.7)**2)/8)
    r = refine_candidate(dp, 43., 24., config())
    np.testing.assert_allclose(r[:2], [42.3, 24.7], atol=.08)
    assert r[-1] == 0 and r[3] > 5
    hot = np.full((64, 80), 5.); hot[25, 43] = 1000
    assert refine_candidate(hot, 43., 25., config())[-1] & 8
    assert refine_candidate(np.zeros((64, 80)), 43., 25., config())[-1] & 64


def test_acceptance_retains_rejected_candidates_and_deduplicates():
    a = np.zeros(4, dtype=DTYPE)
    a["qx_raw_px"] = [20, 21, 40, 55]
    a["qy_raw_px"] = 20
    a["snr_proxy"] = [10, 8, 4, 12]
    a["rejection_flags"][3] = 8
    result = accept_candidates(a, 5, 3)
    np.testing.assert_array_equal(result["accepted"], [True, False, False, False])
    np.testing.assert_array_equal(result["rejection_flags"], [0, 32, 1, 8])
    assert len(result) == len(a)
    assert not a["accepted"].any()


def test_count_splitting_conserves_every_pixel_and_is_reproducible():
    dp = np.arange(256, dtype=np.uint16).reshape(16, 16)
    a, b = split_counts(dp, 27)
    np.testing.assert_array_equal(a+b, dp)
    np.testing.assert_array_equal(a, split_counts(dp, 27)[0])
    assert not np.array_equal(a, b)


def test_uncalibrated_workflow_rejects_an_invented_physical_scale(tmp_path):
    c = copy.deepcopy(config())
    c["calibration"]["g_scale_inv_angstrom_per_px"] = .01
    with pytest.raises(ValueError, match="independent validation"):
        write_calibration(c, tmp_path / "calibration")
