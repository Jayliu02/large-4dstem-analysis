"""Paired ablations with full-search nulls and separately reported evidence gates."""
from __future__ import annotations

import time
import numpy as np

from .phase_matching import decisions
from .phase_refinement import LocalMatcher, transform, visible_predictions, pairs
from .phase_experiment_support import acceptance_metrics, separated_support, random_angles


VARIANTS = {
    'baseline': (False, False, False, 'baseline'),
    'uncertainty': (False, False, True, 'baseline'),
    'local_zone': (True, False, False, 'baseline'),
    'bounded_geometry': (False, True, False, 'baseline'),
    'adaptive_peaks': (False, False, False, 'adaptive'),
    'nonlocal_peaks': (False, False, False, 'nonlocal'),
    'combined': (True, True, True, 'adaptive'),
}


def matcher_for(variant, libraries, scale, visibility, cfg, experiment, bank):
    zone, geometry, _, _ = VARIANTS[variant]
    return LocalMatcher(libraries, scale, visibility, cfg, experiment,
                        zone=zone, geometry=geometry, bank=bank)


def calibrated_threshold(variant, data, libraries, scale, visibility, cfg, experiment, bank, seed):
    matcher = matcher_for(variant, libraries, scale, visibility, cfg, experiment, bank)
    maxima = []
    for repeat in range(experiment['validation']['null_repeats']):
        randomized = random_angles(data['observed'], seed+repeat)
        fit, _ = matcher.match(randomized, data['count'], data['center'])
        maxima.extend(fit[:, :, 0].max(axis=1))
    maximum = np.asarray(maxima)
    # Conservative order statistic rather than interpolating through the tail.
    threshold = max(cfg['matching']['min_score'], float(np.quantile(maximum,
        cfg['matching']['null_percentile']/100, method='higher')))
    return threshold, maximum


def run_variant(variant, data, libraries, scale, visibility, cfg, experiment, threshold,
                scale_bound, *, bank=None, truth=None, secondary=False):
    started = time.perf_counter()
    bank = bank if bank is not None else {'crystals': {}, 'zones': {}}
    matcher = matcher_for(variant, libraries, scale, visibility, cfg, experiment, bank)
    observed, counts, centers = data['observed'], data['count'], data['center']
    fits, params = matcher.match(observed, counts, centers)
    active = matcher.expanded_libraries
    decision = decisions(fits, libraries, cfg['matching'], observed_counts=counts, min_score=threshold)
    eligible = data['valid'] & (counts >= cfg['matching']['min_peaks'])
    base = decision['phase_id'].copy()
    base[~eligible] = -1
    if not data.get('calibrated', True):
        base[:] = -1
    provisional = np.flatnonzero(base >= 0)
    measured = VARIANTS[variant][2]
    delta = scale_bound if measured else cfg['matching']['scale_perturbation']
    bounds = data['center_bound'] if measured else np.full((len(counts), 2), cfg['matching']['center_perturbation_px'])
    probes = np.full((len(counts), 6), -1, np.int16)
    summaries = {}
    specs = [(f'scale_{sign:+d}', scale*(1+sign*delta), np.zeros_like(centers)) for sign in [-1, 1]]
    specs += [(f'center_{axis}_{sign:+d}', scale,
               np.eye(2)[axis][None, :]*bounds[:, axis, None]*sign) for axis in [0, 1] for sign in [-1, 1]]
    for column, (name, probe_scale, offset) in enumerate(specs):
        if len(provisional):
            probe = matcher_for(variant, libraries, probe_scale, visibility, cfg, experiment, bank)
            shifted = observed[provisional]-offset[provisional, None, :]
            result, _ = probe.match(shifted, counts[provisional], centers[provisional]+offset[provisional])
            labels = decisions(result, libraries, cfg['matching'], observed_counts=counts[provisional], min_score=threshold)['phase_id']
            probes[provisional, column] = labels
            summaries[name] = {'retain_original_phase': int((labels == base[provisional]).sum()),
                              'other_accepted_phase': int(((labels >= 0) & (labels != base[provisional])).sum()),
                              'insufficient_or_ambiguous': int((labels < 0).sum())}
        else:
            summaries[name] = {'retain_original_phase': 0, 'other_accepted_phase': 0, 'insufficient_or_ambiguous': 0}
    stable = (base >= 0) & (probes == base[:, None]).all(axis=1)
    final = base.copy()
    final[(base >= 0) & ~stable] = -2
    support = separated_support(fits, libraries, counts, cfg['matching'], threshold)
    # Persist selected templates so expanded-library indices remain interpretable.
    selected_q = np.zeros((len(counts), len(libraries), cfg['templates']['max_peaks'], 2), np.float32)
    selected_count = np.zeros((len(counts), len(libraries)), np.int16)
    matrices = np.zeros((len(counts), len(libraries), 3, 3), np.float32)
    for i in range(len(counts)):
        for l, lib in enumerate(active):
            t = int(fits[i, l, 4])
            if t >= 0:
                selected_q[i, l] = lib['qxy'][t]
                selected_count[i, l] = lib['count'][t]
                if 'matrices' in lib:
                    matrices[i, l] = lib['matrices'][t]
    saved = {'phase_id': final, 'base_phase_id': base, 'fits': fits, 'pose_parameters': params,
             'probe_labels': probes, 'perturbation_stable': stable, 'probe_evaluated': base >= 0,
             'selected_template_qxy': selected_q, 'selected_template_count': selected_count,
             'orientation_matrix': matrices, 'observed': observed, 'observed_count': counts,
             'center_yx': centers, 'center_bound_px': bounds, **support}
    if truth is not None:
        saved['truth'] = truth
    if secondary:
        # Subtract only one-to-one explained peaks, then require full base evidence
        # for a second component. This is a mixture flag, never a purity decision.
        residual = np.zeros_like(observed)
        residual_count = np.zeros_like(counts)
        residual_anisotropy = np.full(len(counts), np.nan)
        for i in range(len(counts)):
            l = int(fits[i, :, 0].argmax())
            n = selected_count[i, l]
            if not n or fits[i, l, 1] < cfg['matching']['min_peaks']:
                continue
            p = params[i, l]
            predicted = transform(selected_q[i, l, :n]/(scale*np.exp(p[2])), fits[i, l, 3], fits[i, l, 5])
            keep = visible_predictions(predicted, centers[i]+p[:2], visibility,
                                       cfg['peaks']['exclusion_radius'], cfg['peaks']['max_radius'])
            linked = pairs(observed[i, :counts[i]]-p[:2], predicted[keep], cfg['matching']['match_radius_px'])
            unexplained = np.ones(counts[i], bool)
            unexplained[linked[:, 0]] = False
            points = observed[i, :counts[i]][unexplained]
            residual[i, :len(points)] = points
            residual_count[i] = len(points)
            if len(linked) >= 6:
                model = predicted[keep][linked[:, 1]]
                error = observed[i, linked[:, 0]]-p[:2]-model
                # Second harmonic of radial residuals: diagnostic only, not an ellipse correction.
                radius = np.linalg.norm(model, axis=1)
                angle = np.arctan2(model[:, 0], model[:, 1])
                radial = np.sum(error*model, axis=1)/np.maximum(radius, 1)
                design = np.column_stack([np.cos(2*angle), np.sin(2*angle), np.ones(len(angle))])
                coef = np.linalg.lstsq(design, radial, rcond=None)[0]
                residual_anisotropy[i] = np.linalg.norm(coef[:2])
        second_fit, _ = matcher.match(residual, residual_count, centers)
        second = decisions(second_fit, libraries, cfg['matching'], observed_counts=residual_count,
                           min_score=threshold)['phase_id']
        saved.update(secondary_candidate=second, secondary_geometry_support=(second >= 0) & eligible,
                     residual_peak_count=residual_count, residual_anisotropy_px=residual_anisotropy)
    summary = {'variant': variant, 'threshold': threshold, 'scale_bound_fraction': delta,
               'base': acceptance_metrics(base, truth), 'full': acceptance_metrics(final, truth),
               'probes': summaries, 'seconds': time.perf_counter()-started,
               'uncertainty_policy': 'conditional_repeatability_sensitivity_only' if measured else 'legacy_fixed_bounds',
               'voltage_hypotheses': sorted({int(lib['voltage_kv']) for lib in libraries}),
               'promotion_eligible': False}
    return summary, saved
