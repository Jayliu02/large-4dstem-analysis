import numpy as np
import pytest

from fourdstem_pipeline.mib_round1 import (
    design, export_peak_coordinates, fit_plane, integrate_detectors,
    measure_origin, validation_positions, raw_peak_options,
)


def test_robust_plane_rejects_large_contamination():
    rng = np.random.default_rng(18)
    y, x = np.indices((20, 24))
    truth = np.array([91., -.06, .33])
    z = (design(y, x) @ truth).reshape(y.shape) + rng.normal(0, .02, y.shape)
    z.ravel()[::19] += 40
    z[0, 1] = np.nan
    fitted = fit_plane(y, x, z, np.isfinite(z), scale=.2)
    np.testing.assert_allclose(fitted, truth, atol=.04)
    with pytest.raises(ValueError, match="Insufficient"):
        fit_plane([0, 0], [0, 1], [2, 3], [True, True])


def test_validation_disjoint_reproducible_and_balanced():
    excluded = {(y, x) for y in range(0, 256, 8) for x in range(0, 256, 8)}
    positions = validation_positions((256, 256), excluded, 12, 1045)
    assert len(set(map(tuple, positions))) == 48
    assert not excluded.intersection(map(tuple, positions))
    np.testing.assert_array_equal(positions, validation_positions((256, 256), excluded, 12, 1045))
    for yhalf in [False, True]:
        for xhalf in [False, True]:
            assert np.sum(((positions[:, 0] >= 128) == yhalf) & ((positions[:, 1] >= 128) == xhalf)) == 12


def test_local_centroid_and_invalid_measurements():
    y, x = np.indices((64, 80))
    c = dict(search_window=[5, 60, 5, 75], smoothing_sigma=1.5,
             local_radius_px=6, min_snr=5, width_range_px=[.5, 6])
    dp = 3 + 500*np.exp(-((y-22.4)**2+(x-51.7)**2)/8)
    result = measure_origin(dp, c)
    assert result["valid"]
    np.testing.assert_allclose([result["origin_y"], result["origin_x"]], [22.4, 51.7], atol=.05)
    assert not measure_origin(np.zeros_like(dp), c)["valid"]
    edge = 3 + 500*np.exp(-((y-5)**2+(x-51)**2)/8)
    assert measure_origin(edge, c)["near_search_edge"]


def test_moving_masks_counts_boundaries_and_no_resampling():
    frames = np.zeros((2, 13, 17), dtype=np.uint16)
    frames[0, 3, 9] = 65535
    frames[0, 3, 11] = 5  # BF inclusive radius=2, ADF inclusive inner=2
    frames[0, 3, 13] = 7  # ADF exclusive outer=4
    frames[1, 7, 5] = 65535
    before = frames.copy()
    result = integrate_detectors(frames, [3, 7], [9, 5], [2], [(2, 4)])
    np.testing.assert_array_equal(result["bf_r2"], [65540, 65535])
    np.testing.assert_array_equal(result["adf_r2_4"], [5, 0])
    np.testing.assert_array_equal(frames, before)


def test_py4dstem_coordinate_contract_on_asymmetric_dp():
    py4dstem = pytest.importorskip("py4DSTEM")
    y, x = np.indices((48, 64))
    dp = 100*np.exp(-((y-13.3)**2+(x-42.6)**2)/4)
    dc = py4dstem.DataCube(data=dp[None, None])
    peaks = dc.find_Bragg_disks(**raw_peak_options(), data=(np.array([0]), np.array([0])),
        sigma=1, subpixel="poly", minPeakSpacing=5, edgeBoundary=5, minRelativeIntensity=.1)[0].data
    qx, qy = export_peak_coordinates(peaks)
    np.testing.assert_allclose([qx[0], qy[0]], [42.6, 13.3], atol=.1)
    full = dc.find_Bragg_disks(**raw_peak_options(), sigma=1, subpixel="poly",
        minPeakSpacing=5, edgeBoundary=5, minRelativeIntensity=.1)
    np.testing.assert_array_equal(full.raw[0, 0].data, peaks)
    # Nonzero ROI offset and unequal axes expose transpositions.
    global_y, global_x = 7+2, 19+3
    origin_x, origin_y = np.full((20, 30), 40.), np.full((20, 30), 12.)
    np.testing.assert_allclose([qx[0]-origin_x[global_y, global_x], qy[0]-origin_y[global_y, global_x]],
                               [2.6, 1.3], atol=.1)
