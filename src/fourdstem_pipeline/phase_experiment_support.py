"""Sampling, conditional error estimates and honest acceptance statistics."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import beta

from .phase_refinement import pairs


def binomial_interval(successes, total, confidence=.95):
    """Exact two-sided Clopper-Pearson interval, including zero-event controls."""
    if total == 0:
        return [None, None]
    alpha = (1-confidence)/2
    return [0. if successes == 0 else float(beta.ppf(alpha, successes, total-successes+1)),
            1. if successes == total else float(beta.ppf(1-alpha, successes+1, total-successes))]


def acceptance_metrics(labels, truth=None):
    labels = np.asarray(labels)
    accepted = labels >= 0
    result = {'patterns': len(labels), 'accepted': int(accepted.sum()),
              'accepted_fraction': float(accepted.mean()) if len(labels) else None,
              'phase_counts': {str(int(p)): int((labels == p).sum()) for p in np.unique(labels)}}
    if truth is None:
        result['interpretation'] = 'acceptance coverage only; experimental correctness is unknown'
        return result
    truth = np.asarray(truth)
    wrong = int((accepted & (labels != truth)).sum())
    result.update(wrong_accepted=wrong, wrong_per_pattern_interval_95=binomial_interval(wrong, len(labels)),
                  error_among_accepted_interval_95=binomial_interval(wrong, int(accepted.sum())))
    result['recall_by_phase'] = {}
    for p in np.unique(truth[truth >= 0]):
        n = int((truth == p).sum())
        correct = int(((truth == p) & (labels == p)).sum())
        result['recall_by_phase'][str(int(p))] = {'correct': correct, 'total': n,
            'recall': correct/n, 'interval_95': binomial_interval(correct, n)}
    return result


def stratified_split(counts, valid, reason, signal, cfg, *, legacy_train=(), legacy_validation=(), seed=0):
    """Guarded spatial blocks, stratified without using the predicted phase.

    Test blocks exclude every block that contributed to legacy scale training.
    All legacy calibration samples and their guards are excluded from sampling.
    Spatial neighbors used for averaging cannot cross split boundaries.
    """
    shape = counts.shape
    yy, xx = np.indices(shape)
    block = cfg['block_size']
    by, bx = yy//block, xx//block
    block_id = by*int(np.ceil(shape[1]/block))+bx
    test = (by+bx) % 2 == 1
    legacy_train = np.asarray(legacy_train, dtype=int)
    if len(legacy_train):
        test[np.isin(block_id, np.unique(block_id.ravel()[legacy_train]))] = False
    # -1 = guard/excluded, 0 = train, 1 = test; entire guard belongs to neither.
    domain = test.astype(np.int8)
    guard = cfg['guard_px']
    boundary = np.zeros(shape, bool)
    boundary[1:] |= domain[1:] != domain[:-1]
    boundary[:-1] |= domain[1:] != domain[:-1]
    boundary[:, 1:] |= domain[:, 1:] != domain[:, :-1]
    boundary[:, :-1] |= domain[:, 1:] != domain[:, :-1]
    excluded = np.zeros(shape, bool)
    legacy = np.r_[legacy_train, np.asarray(legacy_validation, dtype=int)]
    excluded.ravel()[legacy] = True
    if guard:
        boundary = binary_dilation(boundary, iterations=guard)
        excluded = binary_dilation(excluded, iterations=guard)
    domain[boundary | excluded] = -1
    # Count, validity, rejection reason and phase-independent signal proxy.
    finite = np.asarray(signal)[np.isfinite(signal)]
    cut = np.quantile(finite, [1/3, 2/3]) if len(finite) else [0, 0]
    strata = (np.digitize(counts, [6, 12])*1000 + (~valid)*100 +
              (np.asarray(reason, int) & 15)*3 + np.digitize(signal, cut))
    rng = np.random.default_rng(seed)
    selected = []
    for split, size in [(0, cfg['train_size']), (1, cfg['test_size'])]:
        pools = [rng.permutation(np.flatnonzero((domain == split) & (strata == s))).tolist()
                 for s in np.unique(strata[domain == split])]
        ids = []
        # Round robin allocates rare failure strata before filling common strata.
        while len(ids) < size and any(pools):
            for pool in pools:
                if pool and len(ids) < size:
                    ids.append(pool.pop())
        selected.append(np.asarray(sorted(ids), dtype=int))
    if not len(selected[0]) or not len(selected[1]):
        raise ValueError('No independent training/test domain remains; reduce block/guard size or inspect legacy sampling.')
    return {'train': selected[0], 'test': selected[1], 'domain': domain,
            'block_id': block_id, 'strata': strata, 'signal_cuts': cut,
            'population_by_split_stratum': {str(split): {str(int(s)): int(((domain == split) & (strata == s)).sum())
                for s in np.unique(strata[domain == split])} for split in [0, 1]}}


def center_bounds(points, center, cfg, fallback):
    """Conditional Friedel-midpoint repeatability, NOT an instrument uncertainty.

    Disjoint pairs avoid treating repeated use of a peak as independent evidence.
    Include disagreement with the current beam center and an explicit floor.
    Missing symmetry support uses the original bound rather than zero error.
    """
    relative = np.asarray(points)-center
    linked = pairs(relative, -relative, 3.)
    used, mids = set(), []
    for a, b in linked:
        if a != b and a not in used and b not in used:
            mids.append((points[a]+points[b])/2)
            used.update((int(a), int(b)))
    if len(mids) < cfg['min_friedel_pairs']:
        return np.full(2, fallback), len(mids), False
    mids = np.asarray(mids)
    middle = np.median(mids, axis=0)
    spread = 1.4826*np.median(abs(mids-middle), axis=0)
    bound = abs(middle-center)+1.96*spread/np.sqrt(len(mids))
    return np.maximum(bound, cfg['center_floor_px']), len(mids), True


def scale_bounds(corrections, block_ids, scale_grid, scale, cfg, fallback, seed=0):
    """Block bootstrap of shared scale correction plus a nonzero grid floor.

    Input corrections must be training-only, non-boundary, well-supported fits.
    Between-block spread is reported separately. The result remains conditional
    on the candidate structures, selected reflections and forward model.
    """
    corrections, block_ids = np.asarray(corrections), np.asarray(block_ids)
    grid = np.sort(np.unique(scale_grid))
    nearest = int(np.argmin(abs(grid-scale)))
    spacing = np.diff(grid[max(0, nearest-1):nearest+2])/scale
    floor = float(np.max(spacing)/2) if len(spacing) else fallback
    if len(corrections) < cfg['min_scale_patterns'] or len(np.unique(block_ids)) < 4:
        return {'bound_fraction': max(fallback, floor), 'status': 'fallback_insufficient_training_support',
                'supported_patterns': len(corrections), 'grid_floor_fraction': floor,
                'shared_interval_95': [-max(fallback, floor), max(fallback, floor)]}
    block_medians = np.array([np.median(corrections[block_ids == b]) for b in np.unique(block_ids)])
    rng = np.random.default_rng(seed)
    samples = rng.choice(block_medians, size=(cfg['bootstrap_repeats'], len(block_medians)), replace=True)
    interval = np.quantile(np.median(samples, axis=1), [.025, .975])
    interval = [float(interval[0]-floor), float(interval[1]+floor)]
    return {'bound_fraction': max(abs(x) for x in interval), 'status': 'candidate_conditioned_repeatability',
            'shared_interval_95': interval, 'grid_floor_fraction': floor,
            'supported_patterns': len(corrections), 'supported_blocks': len(block_medians),
            'between_block_mad_fraction': float(1.4826*np.median(abs(block_medians-np.median(block_medians)))),
            'instrument_systematic_error_known': False}


def random_angles(observed, seed):
    rng = np.random.default_rng(seed)
    radius = np.linalg.norm(observed, axis=-1)
    angle = rng.uniform(-np.pi, np.pi, radius.shape)
    return np.stack([radius*np.cos(angle), radius*np.sin(angle)], axis=-1).astype(np.float32)


def separated_support(results, libraries, counts, matching, threshold):
    from .phase_diagnostics import winning_gates
    _, winners, gates = winning_gates(results, libraries, counts, matching, threshold)
    geometry = np.logical_and.reduce([v for k, v in gates.items() if k != 'phase_margin'])
    return {'geometry_by_voltage': geometry, 'margin_by_voltage': gates['phase_margin'],
            'winner_by_voltage': winners, 'voltage_agreement': (winners == winners[:, :1]).all(axis=1)}
