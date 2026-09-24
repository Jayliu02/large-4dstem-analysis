"""Shared reciprocal scale estimation with spatial holdout and null controls."""
from __future__ import annotations

import numpy as np

from .phase_matching import PhaseMatcher, decisions
from .phase_peaks import atomic_json


def sample_indices(shape, grid):
    ys = np.unique(np.linspace(0, shape[0]-1, min(grid, shape[0])).round().astype(int))
    xs = np.unique(np.linspace(0, shape[1]-1, min(grid, shape[1])).round().astype(int))
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    by, bx = np.indices(yy.shape)
    # Alternating spatial blocks, not alternating adjacent diffraction patterns.
    validation = ((by//4+bx//4) % 2).astype(bool)
    return (yy*shape[1]+xx).ravel(), validation.ravel()


def balanced_score(result, libraries):
    voltages = sorted({int(lib['voltage_kv']) for lib in libraries})
    values = [result[:, [i for i, lib in enumerate(libraries) if int(lib['voltage_kv']) == v], 0].max(axis=1)
              for v in voltages]
    return np.median(values, axis=0)


def assess_scale(scales, training_scores, selected, validation_result, libraries, config, matching):
    """Keep scale ambiguity explicit; this is conditional on the candidate set."""
    scales = np.asarray(scales)
    training_scores = np.asarray(training_scores)
    means = training_scores.mean(axis=1)
    best = int(np.argmin(abs(scales-selected)))
    rng = np.random.default_rng(0)
    indices = rng.integers(training_scores.shape[1], size=(config['bootstrap_repeats'], training_scores.shape[1]))
    boot = training_scores[:, indices].mean(axis=2).argmax(axis=0)
    interval = np.percentile(scales[boot], [2.5, 97.5])
    separated = abs(scales/selected-1) > 0.05
    competitor = float(means[separated].max()) if separated.any() else 0.0
    separation = float((means[best]-competitor)/max(means[best], 1e-12))
    lo, hi = config['scale_range']
    boundary = bool(selected <= lo*1.025 or selected >= hi/1.025)
    # Scale evidence does not require knowing the phase: at least one candidate
    # must explain six non-collinear peaks with a small positional residual.
    support = ((validation_result[:, :, 0] >= matching['min_score'])
               & (validation_result[:, :, 1] >= matching['min_peaks'])
               & (validation_result[:, :, 2] <= matching['max_median_residual_px'])
               & (validation_result[:, :, 6] > 0)).any(axis=1)
    enough = int(support.sum()) >= config['min_validation_patterns']
    reasons = []
    if boundary:
        reasons.append('scale optimum is at the search boundary')
    if separation < 0.03:
        reasons.append('competing scale separated by >5% has comparable score')
    if (interval[1]-interval[0])/selected > 0.04:
        reasons.append('bootstrap reciprocal-scale interval is wider than 4%')
    if not enough:
        reasons.append('too few held-out patterns support the fitted scale')
    return {'status': 'calibrated_conditionally' if not reasons else 'uncalibrated',
            'scale_inv_angstrom_per_pixel': float(selected), 'bootstrap_interval_95': interval,
            'separated_scale_relative_gap': separation, 'boundary_optimum': boundary,
            'validation_supported_patterns': int(support.sum()), 'validation_patterns': len(support),
            'reasons': reasons, 'interpretation': 'candidate-conditioned estimate, not instrument calibration'}


def calibrate(peaks, libraries, visibility, cfg, output):
    shape = peaks['count'].shape
    indices, held_out = sample_indices(shape, cfg['calibration']['sample_grid'])
    counts = peaks['count'].reshape(-1)
    good = peaks['center_valid'].reshape(-1)[indices] & (counts[indices] >= cfg['matching']['min_peaks'])
    train = indices[good & ~held_out]
    validation = indices[good & held_out]
    # Bound template fitting cost, while sampling the whole training domain.
    if len(train) > 128:
        train = train[np.linspace(0, len(train)-1, 128).round().astype(int)]
    if len(train) < 16 or len(validation) < cfg['calibration']['min_validation_patterns']:
        # A complete rejection map is still a meaningful output for weak scans.
        # This working scale is explicitly NOT a fitted calibration.
        middle = float(np.sqrt(np.prod(cfg['calibration']['scale_range'])))
        audit = {'status': 'uncalibrated', 'scale_inv_angstrom_per_pixel': middle,
                 'scale_origin': 'search_range_midpoint_not_fitted', 'bootstrap_interval_95': cfg['calibration']['scale_range'],
                 'separated_scale_relative_gap': 0.0, 'boundary_optimum': False,
                 'validation_supported_patterns': 0, 'validation_patterns': len(validation),
                 'training_patterns': len(train), 'null_score_threshold': cfg['matching']['min_score'],
                 'null_percentile': cfg['matching']['null_percentile'], 'null_patterns': 0,
                 'voltages_kv': cfg['templates']['voltages_kv'], 'phase_specific_scales': False,
                 'reasons': ['too few valid spatial samples for scale fitting and independent validation'],
                 'interpretation': 'no calibration fitted; midpoint used only for rejected diagnostic fits'}
        atomic_json(output/'calibration.json', audit)
        np.savez_compressed(output/'calibration_samples.npz', scales=np.array([middle]), training_scores=np.zeros((1,1)),
            training_indices=train, validation_indices=validation, validation_results=np.empty((0,len(libraries),7)),
            null_max_scores=np.empty(0))
        return audit
    centers = peaks['center_yx'].reshape(-1, 2)
    positions = peaks['peak_yx'].reshape(-1, peaks['peak_yx'].shape[-2], 2)
    observed = positions-centers[:, None, :]
    parameters = cfg['calibration']
    coarse = np.geomspace(*parameters['scale_range'], parameters['scale_steps'])
    curves = {}

    def fit(scale):
        matcher = PhaseMatcher(libraries, scale, visibility, cfg['matching'],
            inner=cfg['peaks']['exclusion_radius'], outer=cfg['peaks']['max_radius'])
        result = matcher.match(observed[train], counts[train], centers[train])
        curves[float(scale)] = balanced_score(result, libraries)
        print(f'  calibration scale={scale:.6f}, score={curves[float(scale)].mean():.4f}', flush=True)

    for scale in coarse:
        fit(scale)
    means = np.array([curves[float(scale)].mean() for scale in coarse])
    maxima = [i for i in range(len(coarse)) if (i == 0 or means[i] >= means[i-1])
              and (i == len(coarse)-1 or means[i] >= means[i+1])]
    finalists = sorted(maxima, key=lambda i: means[i], reverse=True)[:3]
    for i in finalists:
        for scale in np.linspace(coarse[max(0,i-1)], coarse[min(len(coarse)-1,i+1)], parameters['fine_steps']):
            if float(scale) not in curves:
                fit(scale)
    scales = np.array(sorted(curves))
    scores = np.stack([curves[s] for s in scales])
    best = float(scales[scores.mean(axis=1).argmax()])
    matcher = PhaseMatcher(libraries, best, visibility, cfg['matching'],
                           inner=cfg['peaks']['exclusion_radius'], outer=cfg['peaks']['max_radius'])
    validation_result = matcher.match(observed[validation], counts[validation], centers[validation])
    audit = assess_scale(scales, scores, best, validation_result, libraries, parameters, cfg['matching'])
    # Preserve each pattern's radial distribution and peak count, while breaking
    # crystalline angular relationships. This estimates accidental geometric fit.
    null_indices = validation[np.linspace(0, len(validation)-1, min(128, len(validation))).round().astype(int)]
    rng = np.random.default_rng(214)
    radii = np.linalg.norm(observed[null_indices], axis=-1)
    angles = rng.uniform(-np.pi, np.pi, radii.shape)
    null_points = np.stack([radii*np.cos(angles), radii*np.sin(angles)], axis=-1).astype(np.float32)
    null_result = matcher.match(null_points, counts[null_indices], centers[null_indices])
    null_max = null_result[:, :, 0].max(axis=1)
    threshold = max(cfg['matching']['min_score'], float(np.percentile(null_max, cfg['matching']['null_percentile'])))
    audit.update({'null_score_threshold': threshold, 'null_percentile': cfg['matching']['null_percentile'],
                  'scale_origin': 'joint_candidate_fit',
                  'null_patterns': len(null_indices), 'training_patterns': len(train),
                  'voltages_kv': cfg['templates']['voltages_kv'], 'phase_specific_scales': False})
    atomic_json(output/'calibration.json', audit)
    np.savez_compressed(output/'calibration_samples.npz', scales=scales, training_scores=scores,
        training_indices=train, validation_indices=validation, validation_results=validation_result,
        null_max_scores=null_max)
    return audit
