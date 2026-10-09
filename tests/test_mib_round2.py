import numpy as np
import pytest

from fourdstem_pipeline.mib_round2 import (
    centered_kernel, correct_coordinates, detect, export_peak_coordinates,
    local_quality, mutual_pairs, sample_sets,
)


def test_stratified_samples_are_disjoint_balanced_and_reproducible():
    tuning, holdout = sample_sets((32, 32), 20, 20, 2045)
    assert len(set(map(tuple, tuning))) == len(set(map(tuple, holdout))) == 20
    assert not set(map(tuple, tuning)).intersection(map(tuple, holdout))
    for points in [tuning, holdout]:
        for yhalf in [False, True]:
            for xhalf in [False, True]:
                assert np.sum(((points[:, 0] >= 16) == yhalf) & ((points[:, 1] >= 16) == xhalf)) == 5
    np.testing.assert_array_equal(sample_sets((32, 32), 20, 20, 2045)[1], holdout)


def test_fft_origin_and_real_template_localization_on_nonsquare_detector():
    py4dstem = pytest.importorskip("py4DSTEM")
    yy, xx = np.mgrid[-12:13, -12:13]
    kernel, positive, _ = centered_kernel(np.exp(-(xx**2+yy**2)/8), (96, 128))
    assert abs(kernel.sum()) < 1e-12
    assert np.unravel_index(kernel.argmax(), kernel.shape) == (0, 0)
    assert positive.sum() == pytest.approx(1)
    dy, dx = np.indices(kernel.shape)
    truth = np.array([[28.3, 33.6], [85.6, 61.2]])
    dp = np.full(kernel.shape, 5.)
    for (x, y), intensity in zip(truth, [1000, 180]):
        dp += intensity*np.exp(-((dx-x)**2+(dy-y)**2)/8)
    cfg = dict(detection=dict(working_relative_intensity=.03, working_spacing=8,
                             working_edge=10, sigma=1, max_peaks=70))
    dc = py4dstem.DataCube(data=dp[None, None])
    for method in ["raw", "template"]:
        peaks = detect(dc, np.array([[0, 0]]), method, kernel, cfg)[0].data
        x, y = export_peak_coordinates(peaks)
        observed = np.column_stack([x, y])
        pairs = mutual_pairs(truth, observed, .3)
        assert len(pairs) == len(peaks) == 2
        np.testing.assert_allclose(truth[pairs[:, 0]], observed[pairs[:, 1]], atol=.1)


def test_global_coordinate_correction_agrees_between_offset_rois_and_preserves_raw():
    sy, sx = np.indices((64, 80))
    model = dict(origin_x=13+.2*sx+.3*sy, origin_y=27-.1*sx+.4*sy)
    local_a = np.array([[5, 9], [7, 12]])
    global_a = local_a + [20, 30]
    local_b = global_a - [23, 34]
    raw_x = model["origin_x"][global_a[:, 0], global_a[:, 1]] + [11.7, -8.4]
    raw_y = model["origin_y"][global_a[:, 0], global_a[:, 1]] + [-6.3, 14.2]
    preserved = raw_x.copy()
    a = correct_coordinates(raw_x, raw_y, global_a[:, 0], global_a[:, 1], model)
    b = correct_coordinates(raw_x, raw_y, local_b[:, 0]+23, local_b[:, 1]+34, model)
    np.testing.assert_allclose(a, [[11.7, -8.4], [-6.3, 14.2]], atol=1e-12)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(raw_x, preserved)
    with pytest.raises(ValueError, match="out of bounds"):
        correct_coordinates([10.], [12.], [-1], [0], model)
    with pytest.raises(ValueError, match="integer"):
        correct_coordinates([10.], [12.], [3.5], [0], model)


def test_mutual_pairing_no_duplicate_assignments_or_forced_distant_matches():
    a = np.array([[0, 0], [.2, 0], [15, 15]])
    b = np.array([[.05, 0], [40, 40]])
    np.testing.assert_array_equal(mutual_pairs(a, b, 1), [[0, 0]])
    assert mutual_pairs(np.empty((0, 2)), b, 1).shape == (0, 2)


def test_local_support_is_measured_in_raw_counts_not_correlation_score():
    y, x = np.indices((64, 64))
    dp = 4 + 100*np.exp(-((y-24)**2+(x-35)**2)/8)
    signal, snr, flags = local_quality(dp, 35, 24, 5)
    assert signal > 1000 and snr > 5 and flags == 0
    signal, snr, flags = local_quality(np.full((64, 64), 4.), 35, 24, 5)
    assert signal == 0 and snr == 0 and flags & 1
