"""Optional research refinements; production matching and cache hashes are unchanged."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .phase_matching import PhaseMatcher, fit_shortlist, score_pose


def transform(points, angle, mirror):
    points = np.asarray(points).copy()
    points[:, 1] *= mirror
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, -s], [s, c]]).T


def visible_predictions(points, center, visibility, inner, outer):
    pixel = np.rint(points + center).astype(int)
    radius = np.linalg.norm(points, axis=1)
    keep = ((radius >= inner) & (radius <= outer) & (pixel[:, 0] >= 4)
            & (pixel[:, 0] < visibility.shape[0]-4) & (pixel[:, 1] >= 4)
            & (pixel[:, 1] < visibility.shape[1]-4))
    idx = np.flatnonzero(keep)
    keep[idx] &= visibility[pixel[idx, 0], pixel[idx, 1]]
    return keep


def pairs(observed, predicted, tolerance):
    """Same one-to-one nearest-edge rule as the baseline, including duplicates."""
    distance = np.linalg.norm(observed[:, None] - predicted[None], axis=-1)
    rows, cols = np.where(distance <= tolerance)
    used_i, used_j, selected = set(), set(), []
    for k in np.argsort(distance[rows, cols], kind='stable'):
        i, j = int(rows[k]), int(cols[k])
        if i not in used_i and j not in used_j:
            selected.append((i, j))
            used_i.add(i)
            used_j.add(j)
    return np.asarray(selected, dtype=int).reshape(-1, 2)


def bounded_pose(observed, predicted, center, initial, visibility, matching, refinement,
                 inner=8., outer=120.):
    """Fit translation, log reciprocal scale and angle, with hard symmetric bounds.

    Parameters are [center_dy, center_dx, log(scale/base_scale), angle_delta].
    Assignment is recomputed after every update. No affine strain is fitted.
    """
    initial = np.asarray(initial, np.float32)
    best = initial.copy()
    parameter = np.zeros(4)
    best_parameter = parameter.copy()
    bound = np.array([refinement['max_center_shift_px']]*2 +
                     [np.log1p(refinement['max_scale_fraction']),
                      np.deg2rad(refinement['max_angle_deg'])])
    if len(observed) < matching['min_peaks'] or initial[4] < 0:
        return best, best_parameter
    for _ in range(refinement['iterations']):
        moved = transform(predicted, initial[3]+parameter[3], initial[5]) / np.exp(parameter[2])
        keep = visible_predictions(moved, center+parameter[:2], visibility, inner, outer)
        visible = np.flatnonzero(keep)
        linked = pairs(observed-parameter[:2], moved[keep], matching['match_radius_px'])
        if len(linked) < matching['min_peaks']:
            break
        oi, pi = linked[:, 0], visible[linked[:, 1]]

        def residual(p):
            model = transform(predicted[pi], initial[3]+p[3], initial[5]) / np.exp(p[2]) + p[:2]
            return (model-observed[oi]).ravel()

        fitted = least_squares(residual, parameter, bounds=(-bound, bound),
                               loss='soft_l1', f_scale=.5, max_nfev=40)
        parameter = fitted.x
        score, n, median, support = score_pose(
            np.asarray(observed-parameter[:2], np.float32), len(observed),
            np.asarray(predicted/np.exp(parameter[2]), np.float32), len(predicted),
            np.asarray(center+parameter[:2], np.float32), float(initial[3]+parameter[3]),
            int(initial[5]), visibility, inner, outer, matching['match_radius_px'])
        if score > best[0]:
            best = np.array([score, n, median, initial[3]+parameter[3], initial[4], initial[5], support], np.float32)
            best_parameter = parameter.copy()
    return best, best_parameter


class LocalMatcher:
    """Refine both phases at every voltage using the same local search budget.

    The coarse best template per library seeds a 5x5 tangent-plane zone grid.
    Candidate reflections are regenerated, not obtained by warping old peaks.
    Returned template indices refer to ``expanded_libraries`` (base indices are
    retained). The extra pose parameters are stored separately from the 7-column
    production fit layout.
    """
    def __init__(self, libraries, scale, visibility, cfg, experiment, *, zone=False, geometry=False,
                 bank=None):
        self.base = PhaseMatcher(libraries, scale, visibility, cfg['matching'],
                                 inner=cfg['peaks']['exclusion_radius'], outer=cfg['peaks']['max_radius'])
        self.cfg, self.experiment = cfg, experiment
        self.zone, self.geometry = zone, geometry
        self.bank = bank if bank is not None else {'crystals': {}, 'zones': {}}
        self.expanded_libraries = libraries

    def _zones(self, lib, seed):
        key = (int(lib['phase_id']), int(lib['voltage_kv']), int(seed))
        if key in self.bank['zones']:
            return self.bank['zones'][key]
        from .phase_structures import read_structure, make_crystal
        phase, voltage, _ = key
        if phase not in self.bank['crystals']:
            candidate = next(c for c in self.cfg['candidates'] if c['id'] == phase)
            self.bank['crystals'][phase] = make_crystal(read_structure(candidate)[0], self.cfg['templates']['k_max'])
        crystal = self.bank['crystals'][phase]
        crystal.setup_diffraction(voltage*1000)
        coords, counts, matrices = [], [], []
        for a in self.experiment['refinement']['zone_offsets_deg']:
            for b in self.experiment['refinement']['zone_offsets_deg']:
                if a == b == 0:
                    continue
                matrix = lib['matrices'][seed] @ Rotation.from_rotvec(np.deg2rad([a, b, 0])).as_matrix()
                pattern = crystal.generate_diffraction_pattern(orientation_matrix=matrix,
                    sigma_excitation_error=self.cfg['templates']['excitation_error'],
                    tol_intensity=1e-7, k_max=self.cfg['templates']['k_max']).data
                q = np.column_stack([pattern['qx'], pattern['qy']])
                keep = np.linalg.norm(q, axis=1) > .05
                q, intensity = q[keep], pattern['intensity'][keep]
                if len(q):
                    order = np.argsort(intensity)[::-1]
                    order = order[intensity[order] >= intensity.max()*self.cfg['templates']['relative_intensity']]
                    q = q[order[:self.cfg['templates']['max_peaks']]]
                padded = np.zeros_like(lib['qxy'][0])
                padded[:len(q)] = q
                coords.append(padded)
                counts.append(len(q))
                matrices.append(matrix)
        value = (np.asarray(coords, np.float32), np.asarray(counts, np.int16), np.asarray(matrices, np.float32))
        self.bank['zones'][key] = value
        return value

    def match(self, observed, counts, centers):
        observed, counts, centers = (np.ascontiguousarray(observed, dtype=np.float32),
                                     np.ascontiguousarray(counts, dtype=np.int16),
                                     np.ascontiguousarray(centers, dtype=np.float32))
        result = self.base.match(observed, counts, centers)
        parameters = np.zeros(result.shape[:2]+(4,), np.float32)
        expanded = []
        for l, lib in enumerate(self.base.libraries):
            active = lib
            if self.zone:
                seeds = result[:, l, 4].astype(int)
                additions, extra_counts, extra_matrices = [], [], []
                local_indices = {}
                offset = len(lib['count'])
                for seed in np.unique(seeds[seeds >= 0]):
                    q, n, matrices = self._zones(lib, seed)
                    local_indices[seed] = np.r_[seed, np.arange(offset, offset+len(n))]
                    additions.append(q)
                    extra_counts.append(n)
                    extra_matrices.append(matrices)
                    offset += len(n)
                if additions:
                    active = {**lib, 'qxy': np.concatenate([lib['qxy']]+additions),
                              'count': np.concatenate([lib['count']]+extra_counts),
                              'matrices': np.concatenate([lib['matrices']]+extra_matrices)}
                    # Only geometry is used; don't leave misaligned hkl/intensity arrays.
                    active = {k: v for k, v in active.items() if k not in ('hkl', 'intensity')}
                    valid = np.flatnonzero(seeds >= 0)
                    shortlist = np.stack([local_indices[seeds[i]] for i in valid]).astype(np.int32)
                    fitted = fit_shortlist(observed[valid], counts[valid], centers[valid],
                        np.ascontiguousarray(active['qxy']/self.base.scale), active['count'], shortlist,
                        self.base.visibility, self.base.inner, self.base.outer, self.cfg['matching']['match_radius_px'])
                    improve = fitted[:, 0] > result[valid, l, 0]
                    result[valid[improve], l] = fitted[improve]
            if self.geometry:
                for i in range(len(observed)):
                    t = int(result[i, l, 4])
                    if t < 0:
                        continue
                    n = int(active['count'][t])
                    result[i, l], parameters[i, l] = bounded_pose(observed[i, :counts[i]],
                        active['qxy'][t, :n]/self.base.scale, centers[i], result[i, l],
                        self.base.visibility, self.cfg['matching'], self.experiment['refinement'],
                        self.base.inner, self.base.outer)
            expanded.append(active)
        self.expanded_libraries = expanded
        return result, parameters
