"""Sparse, rotation invariant shortlist and one-to-one geometric phase scoring.

Coordinates in this module are detector (y, x). The simulated transverse axes
are mapped onto those axes; in-plane angle and handedness are nuisance variables.
Scores describe kinematic geometric agreement, not posterior probabilities.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter1d


def radial_descriptors(points, counts, *, radius=120, inner=8, sigma=1.25):
    radii = np.linalg.norm(points, axis=-1)
    keep = (np.arange(points.shape[1])[None, :] < counts[:, None]) & (radii >= inner) & (radii <= radius)
    rows, cols = np.nonzero(keep)
    bins = np.rint(radii[rows, cols]).astype(int)
    result = np.zeros((len(points), int(radius)+1), np.float32)
    np.add.at(result, (rows, bins), 1)
    result = gaussian_filter1d(result, sigma, axis=1, mode='constant')
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / np.maximum(norms, 1e-12)


@njit(cache=True)
def score_pose(observed, nobs, predicted, npred, center, angle, mirror, visibility,
               inner, outer, tolerance):
    """Greedy global nearest-edge assignment; each peak participates at most once."""
    rotated = np.zeros((npred, 2), np.float32)
    visible = np.zeros(npred, np.bool_)
    co, si = np.cos(angle), np.sin(angle)
    nvisible = 0
    for j in range(npred):
        y, x = predicted[j, 0], predicted[j, 1]*mirror
        ry, rx = co*y-si*x, si*y+co*x
        rotated[j, 0], rotated[j, 1] = ry, rx
        iy, ix = int(round(ry+center[0])), int(round(rx+center[1]))
        r2 = ry*ry+rx*rx
        if (inner*inner <= r2 <= outer*outer and 4 <= iy < visibility.shape[0]-4
                and 4 <= ix < visibility.shape[1]-4 and visibility[iy, ix]):
            visible[j] = True
            nvisible += 1
    # Sparse candidate edges sorted by distance give deterministic one-to-one
    # matching, without allowing several template spots to explain one peak.
    distances = np.empty(nobs*npred, np.float32)
    edges_i = np.empty(nobs*npred, np.int32)
    edges_j = np.empty(nobs*npred, np.int32)
    ne = 0
    for i in range(nobs):
        for j in range(npred):
            if visible[j]:
                dy, dx = observed[i, 0]-rotated[j, 0], observed[i, 1]-rotated[j, 1]
                d2 = dy*dy+dx*dx
                if d2 <= tolerance*tolerance:
                    distances[ne], edges_i[ne], edges_j[ne] = d2, i, j
                    ne += 1
    order = np.argsort(distances[:ne])
    used_i, used_j = np.zeros(nobs, np.bool_), np.zeros(npred, np.bool_)
    residuals = np.empty(min(nobs, npred), np.float32)
    nm, sum2 = 0, 0.0
    direction1 = np.zeros(2, np.float32)
    noncollinear = False
    for k in order:
        i, j = edges_i[k], edges_j[k]
        if not used_i[i] and not used_j[j]:
            used_i[i], used_j[j] = True, True
            residuals[nm] = np.sqrt(distances[k])
            sum2 += distances[k]
            if nm == 0:
                direction1[:] = observed[i]
            else:
                cross = abs(direction1[0]*observed[i, 1]-direction1[1]*observed[i, 0])
                denom = np.sqrt(np.sum(direction1**2)*np.sum(observed[i]**2))
                if denom > 0 and cross/denom > 0.15:
                    noncollinear = True
            nm += 1
    if nm == 0:
        return 0.0, 0, np.inf, False
    f1 = 2.0*nm/max(nobs+nvisible, 1)
    score = f1*np.exp(-0.5*sum2/nm/(1.5**2))
    return score, nm, float(np.median(residuals[:nm])), noncollinear


@njit(cache=True, parallel=True)
def fit_shortlist(observed, counts, centers, predicted, pred_counts, shortlist,
                  visibility, inner, outer, tolerance):
    # columns: score, matches, median residual, angle, template, mirror, 2D support
    result = np.zeros((len(observed), 7), np.float32)
    result[:, 2] = np.inf
    result[:, 4] = -1
    for i in prange(len(observed)):
        nobs = counts[i]
        if nobs < 2:
            continue
        obs = observed[i]
        for template in shortlist[i]:
            pred, npred = predicted[template], pred_counts[template]
            # Use more anchors than the minimum accepted match count, so that
            # a missing strongest reflection does not determine the pose.
            for a in range(min(nobs, 12)):
                ro = np.sqrt(np.sum(obs[a]**2))
                ao = np.arctan2(obs[a, 1], obs[a, 0])
                for b in range(min(npred, 16)):
                    rp = np.sqrt(np.sum(pred[b]**2))
                    if abs(ro-rp) > tolerance:
                        continue
                    for mirror in (-1, 1):
                        angle = ao-np.arctan2(pred[b, 1]*mirror, pred[b, 0])
                        score, nm, residual, support = score_pose(obs, nobs, pred, npred,
                            centers[i], angle, mirror, visibility, inner, outer, tolerance)
                        if score > result[i, 0]:
                            result[i] = np.array([score, nm, residual, angle, template, mirror, support], np.float32)
        if result[i, 4] >= 0:
            template, mirror = int(result[i, 4]), int(result[i, 5])
            for step in (0.02, 0.01, 0.005):
                initial = result[i, 3]
                for sign in (-1, 1):
                    angle = initial+sign*step
                    score, nm, residual, support = score_pose(obs, nobs, predicted[template], pred_counts[template],
                        centers[i], angle, mirror, visibility, inner, outer, tolerance)
                    if score > result[i, 0]:
                        result[i] = np.array([score, nm, residual, angle, template, mirror, support], np.float32)
    return result


class PhaseMatcher:
    def __init__(self, libraries, scale, visibility, config, *, inner=8.0, outer=120.0):
        self.libraries = libraries
        self.scale = float(scale)
        self.visibility = np.asarray(visibility, bool)
        self.config = config
        self.inner, self.outer = float(inner), float(outer)
        self.predicted = [np.ascontiguousarray(lib['qxy']/scale) for lib in libraries]
        self.descriptors = [radial_descriptors(p, lib['count'], radius=outer, inner=inner)
                            for p, lib in zip(self.predicted, libraries)]

    def radial_scores(self, observed, counts):
        features = radial_descriptors(observed, counts, radius=self.outer, inner=self.inner)
        return np.stack([(features @ reference.T).max(axis=1) for reference in self.descriptors], axis=1)

    def match(self, observed, counts, centers):
        observed = np.ascontiguousarray(observed, dtype=np.float32)
        counts = np.ascontiguousarray(counts, dtype=np.int16)
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        features = radial_descriptors(observed, counts, radius=self.outer, inner=self.inner)
        results = []
        for lib, predicted, descriptor in zip(self.libraries, self.predicted, self.descriptors):
            correlations = features @ descriptor.T
            k = min(self.config['shortlist'], len(descriptor))
            shortlist = np.argpartition(correlations, -k, axis=1)[:, -k:].astype(np.int32)
            results.append(fit_shortlist(observed, counts, centers, predicted, lib['count'], shortlist,
                self.visibility, self.inner, self.outer, self.config['match_radius_px']))
        return np.stack(results, axis=1)


def decisions(results, libraries, config, *, observed_counts, min_score=None):
    """Require evidence and the same winning phase under every voltage hypothesis."""
    voltages = sorted({int(lib['voltage_kv']) for lib in libraries})
    phases = sorted({int(lib['phase_id']) for lib in libraries})
    n = len(results)
    winners = np.zeros((n, len(voltages)), np.int16)
    passed = np.zeros((n, len(voltages)), bool)
    margins = np.zeros((n, len(voltages)), np.float32)
    scores = np.zeros((n, len(voltages)), np.float32)
    evidence = np.zeros((n, len(voltages)), bool)
    for v, voltage in enumerate(voltages):
        indices = [next(i for i, lib in enumerate(libraries)
                        if int(lib['voltage_kv']) == voltage and int(lib['phase_id']) == phase) for phase in phases]
        candidates = results[:, indices, :]
        rank = np.argsort(candidates[:, :, 0], axis=1)
        first, second = rank[:, -1], rank[:, -2]
        best = candidates[np.arange(n), first]
        scores[:, v] = best[:, 0]
        margins[:, v] = (best[:, 0]-candidates[np.arange(n), second, 0])/np.maximum(best[:, 0], 1e-12)
        winners[:, v] = np.asarray(phases)[first]
        evidence[:, v] = ((best[:, 1] >= config['min_peaks']) & (best[:, 2] <= config['max_median_residual_px'])
                          & (best[:, 1]/np.maximum(observed_counts,1) >= config['minimum_observed_fraction'])
                          & (best[:, 6] > 0) & (best[:, 0] >= (config['min_score'] if min_score is None else min_score)))
        passed[:, v] = evidence[:, v] & (margins[:, v] >= config['min_phase_margin'])
    stable = np.all(winners == winners[:, :1], axis=1)
    accepted = passed.all(axis=1) & stable
    labels = np.full(n, -1, np.int16)
    labels[evidence.any(axis=1)] = -2
    labels[accepted] = winners[accepted, 0]
    best_candidate = winners[np.arange(n), scores.argmax(axis=1)].copy()
    best_candidate[scores.max(axis=1) <= 0] = -1
    return {'phase_id': labels, 'best_candidate': best_candidate,
            'min_score': scores.min(axis=1), 'min_margin': margins.min(axis=1),
            'voltage_stable': stable, 'accepted': accepted}
