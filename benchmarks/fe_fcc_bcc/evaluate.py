"""Evaluate pipeline predictions against the benchmark ground truth.

Reads one run dir (benchmarks/Fe_FCC_BCC_v1/runs/<run_id>/): run_manifest.json
points at the datasets, predictions/scan_01_000X/ holds the phase CLI output and
predictions/templates/ the template libraries.  Writes validation_report.json
(gate records with {value, threshold, pass, denominator}), per-dataset peak
pairing CSVs and diagnostic figures, all into the run dir.

Registered gates (plan section 6; see README for the deviations log):
    bragg recall            >= 90% of truth spots paired with a detected peak <= 1.5 px
    position error          median <= 1 px (p90 reported)
    beam center error       median <= 1 px
    reciprocal scale        |scale/0.020 - 1| <= 1% and status calibrated_conditionally
    T01/T02 acceptance      correct >= 95%, rejected <= 5%
    T03 acceptance          overall >= 95%, per grain >= 90%
Orientation metrics are informational: the self-consistency gate (each truth
pattern must reproduce its template-library geometry to <= 0.05 px under some
(angle, mirror) description) is checked first; if it fails, orientation is
marked not_evaluable instead of being scored.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

from bench_geometry import cubic_rotations, rotation_z, rotation_z2

CUBIC = cubic_rotations()
SWAP_XY = np.array([[0.0, 1.0], [1.0, 0.0]])  # the pipeline matches stored (x,y) qxy as (y,x)
REASON_BITS = {1: 'invalid_beam_center', 2: 'insufficient_match_evidence',
               4: 'phase_voltage_ambiguity', 8: 'perturbation_instability',
               16: 'uncalibrated_scale', 32: 'fewer_than_minimum_peaks'}
PHASE_NAMES = {0: 'BCC', 1: 'FCC', -1: 'unindexed', -2: 'ambiguous'}
PHASE_CANDIDATE = {0: 'Fe-BCC', 1: 'Fe-FCC'}


def gate(name, value, threshold, denominator, *, pass_fn=None, units=''):
    fn = pass_fn or (lambda v, t: v >= t)
    return {'name': name, 'value': float(value), 'threshold': threshold,
            'units': units, 'denominator': denominator, 'pass': bool(fn(value, threshold))}


def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def load_truth(path):
    return np.load(path, allow_pickle=True)


def greedy_pair(truth_xy, truth_priority, detected_xy, radius):
    """Greedy one-to-one pairing; truth sorted by descending priority (amplitude)."""
    order = np.argsort(truth_priority)[::-1]
    used = set()
    pairs = []
    for i in order:
        best = None
        for j in range(len(detected_xy)):
            if j in used:
                continue
            d = float(np.hypot(*(truth_xy[i] - detected_xy[j])))
            if d <= radius and (best is None or d < best[1]):
                best = (j, d)
        if best is not None:
            used.add(best[0])
            pairs.append((int(i), best[0], best[1]))
    return pairs


def template_libraries(predictions_dir, candidates):
    """Map (phase_name, voltage_kv) -> npz path from predictions/templates/."""
    template_dir = predictions_dir / 'templates'
    found = {}
    for path in sorted(template_dir.glob('*.npz')):
        for candidate in candidates:
            if path.stem.startswith(f"{candidate['name']}_"):
                try:
                    voltage = int(path.stem.rsplit('_', 1)[-1].removesuffix('kV'))
                except ValueError:
                    continue
                found[(candidate['name'], voltage)] = path
    return found


def gate_pattern(truth_hkl, truth_qxy, library_path, zone, dq, max_zone_deg=0.5):
    """Best (template_row, in-plane angle, mirror, median residual px) describing
    the truth pattern.  Mirrors 1=no flip, -1=qy flip.  Returns None when the
    pattern cannot be represented (gate failure)."""
    library = np.load(library_path)
    target = np.asarray(zone, dtype=float) / np.linalg.norm(zone)
    rows = [i for i in range(len(library['matrices']))
            if float(np.degrees(np.arccos(np.clip(
                np.dot(library['matrices'][i][:, 2], target), -1, 1)))) <= max_zone_deg]
    if not rows:
        return None
    best = None
    for row in rows:
        count = int(library['count'][row])
        template = {}
        for hkl, qxy in zip(library['hkl'][row][:count], library['qxy'][row][:count]):
            template[tuple(int(v) for v in hkl)] = qxy
        pairs = []
        for hkl, qxy in zip(truth_hkl, truth_qxy):
            key = tuple(int(v) for v in hkl)
            if key in template:
                pairs.append((qxy, template[key]))
            elif tuple(-v for v in key) in template:
                pairs.append((qxy, template[tuple(-v for v in key)]))
        if not pairs:
            continue
        truth_px = np.array([r for r, _ in pairs]) / dq
        template_xy = np.array([t for _, t in pairs])
        for mirror in (1, -1):
            t = np.column_stack([template_xy[:, 0], mirror * template_xy[:, 1]]) / dq
            # Optimal 2D rotation taking t onto the truth set: minimize
            # sum |r - Rz2(phi) t|^2 gives phi = atan2(sum(rx*ty - ry*tx), sum(rx*tx + ry*ty))
            # with the cross term r x t, i.e. ry*tx - rx*ty in the numerator.
            num = np.sum(truth_px[:, 1] * t[:, 0] - truth_px[:, 0] * t[:, 1])
            den = np.sum(truth_px[:, 0] * t[:, 0] + truth_px[:, 1] * t[:, 1])
            angle = float(np.arctan2(num, den))
            residual = np.linalg.norm(truth_px - (rotation_z2(angle) @ t.T).T, axis=1)
            median = float(np.median(residual))
            candidate = (median, len(pairs), row, angle, mirror)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        return None
    median, pairs, row, angle, mirror = best
    coverage = pairs / max(1, len(truth_hkl))
    if coverage < 0.9 or median > 0.05:
        return None
    return {'template_row': row, 'in_plane_rad': angle, 'mirror': mirror,
            'median_residual_px': median, 'coverage': coverage}


def evaluate_dataset(entry, candidates, template_paths, run_dir):
    key = entry['key']
    output = run_dir / 'predictions' / entry['scan_name']
    truth = load_truth(entry['truth'])
    dq = float(truth['pixel_transform'][0]['reciprocal_sampling_A_per_px'])
    truth_center = np.asarray(truth['pixel_transform'][0]['center_px'], dtype=float)
    shape = tuple(truth['scan_shape'])
    n = int(np.prod(shape))
    flat = lambda a: np.asarray(a).reshape(n, *np.asarray(a).shape[2:])
    truth_phase = truth['phase_id'].reshape(n)
    grain = truth['grain_id'].reshape(n)
    peak_hkl = truth['peak_hkl'].reshape(n, truth['peak_hkl'].shape[2], 3)
    peak_qxy = truth['peak_qxy_A'].reshape(n, truth['peak_qxy_A'].shape[2], 2)
    peak_amp = truth['peak_amplitude'].reshape(n, truth['peak_amplitude'].shape[2])
    observable = truth['peak_observable'].reshape(n, truth['peak_observable'].shape[2])
    peak_px = truth['peak_px_yx'].reshape(n, truth['peak_px_yx'].shape[2], 2)
    pattern_index = truth['pattern_id'].reshape(n)
    pattern_rot = np.asarray(truth['pattern_rotation'])
    pattern_zone = np.asarray(truth['pattern_zone_axis'])
    n_patterns = len(pattern_rot)

    phase_id = np.load(output / 'phase_id.npy').reshape(n)
    counts = np.load(output / 'peaks' / 'count.npy').reshape(n)
    detected = np.load(output / 'peaks' / 'peak_yx.npy').reshape(n, -1, 2)
    centers = np.load(output / 'peaks' / 'center_yx.npy').reshape(n, 2)
    center_valid = np.load(output / 'peaks' / 'center_valid.npy').reshape(n)
    selected = np.load(output / 'selected_peak_indices.npy').reshape(n, -1)
    selected_count = np.load(output / 'selected_peak_count.npy').reshape(n)
    score = np.load(output / 'score.npy').reshape(n)
    margin = np.load(output / 'margin.npy').reshape(n)
    residual = np.load(output / 'median_residual_px.npy').reshape(n)
    matched_peaks = np.load(output / 'matched_peaks.npy').reshape(n)
    reasons = np.load(output / 'reason_flags.npy').reshape(n)
    all_results = np.load(output / 'all_results.npy').reshape(n, -1, 7)
    schema = json.loads((output / 'array_schema.json').read_text(encoding='utf-8'))
    calibration = json.loads((output / 'calibration.json').read_text(encoding='utf-8'))

    # --- calibration and center gates -------------------------------------
    scale = float(calibration['scale_inv_angstrom_per_pixel'])
    scale_error = abs(scale / dq - 1.0)
    calibrated = calibration['status'] == 'calibrated_conditionally'
    center_error = np.linalg.norm(centers - truth_center, axis=1)
    center_ok = center_valid & (np.linalg.norm(centers - truth_center, axis=1) <= 4.0)
    gates = [gate('reciprocal_scale_error', scale_error, 0.01, n,
                  pass_fn=lambda v, t: v <= t),
             gate('calibration_status', 1.0 if calibrated else 0.0, 1.0, n,
                  pass_fn=lambda v, t: v >= t),
             gate('beam_center_error_median_px', float(np.median(center_error[center_ok])), 1.0,
                  int(center_ok.sum()), pass_fn=lambda v, t: v <= t),
             gate('beam_center_error_p90_px', float(np.percentile(center_error[center_ok], 90)), 2.0,
                  int(center_ok.sum()), pass_fn=lambda v, t: v <= t)]

    # --- recall and position gates ----------------------------------------
    per_pattern = {}
    for p in range(n_patterns):
        rows = np.flatnonzero(pattern_index == p)
        sample = rows[0]
        truth_xy = peak_px[sample][observable[sample]]
        truth_priority = peak_amp[sample][observable[sample]]
        positions, pair_dists = [], []
        for i in rows:
            count = int(counts[i])
            pairs = greedy_pair(truth_xy, truth_priority, detected[i, :count], 1.5)
            positions.append(len(pairs) / max(1, len(truth_xy)))
            pair_dists += [d for _, _, d in pairs]
        per_pattern[p] = {'recall': float(np.mean(positions)), 'pair_dists': pair_dists,
                          'region_size': len(rows)}
    recall = float(np.mean([per_pattern[p]['recall'] for p in per_pattern]))
    pair_dists = np.concatenate([per_pattern[p]['pair_dists'] for p in per_pattern
                                 if per_pattern[p]['pair_dists']])
    gates.append(gate('bragg_recall', recall, 0.90, n_patterns))
    gates.append(gate('position_error_median_px', float(np.median(pair_dists)) if len(pair_dists) else np.inf,
                      1.0, len(pair_dists), pass_fn=lambda v, t: v <= t))
    gates.append(gate('position_error_p90_px', float(np.percentile(pair_dists, 90)) if len(pair_dists) else np.inf,
                      2.0, len(pair_dists), pass_fn=lambda v, t: v <= t))
    # Selection coverage: truth spots recovered among the pipeline-selected observations.
    selection_hits = 0
    for i in range(n):
        sel_positions = detected[i, selected[i, :int(selected_count[i])]]
        pairs = greedy_pair(peak_px[i][observable[i]], peak_amp[i][observable[i]], sel_positions, 1.5)
        selection_hits += len(pairs)
    selection_denominator = int(observable.sum())
    selection_coverage = selection_hits / max(1, selection_denominator)
    gates.append(gate('selection_coverage', selection_coverage, 0.90, selection_denominator))

    # --- acceptance gates ---------------------------------------------------
    correct = (phase_id == truth_phase) & np.isin(phase_id, (0, 1))
    rejected = phase_id == -1
    ambiguous = phase_id == -2
    wrong = np.isin(phase_id, (0, 1)) & (phase_id != truth_phase)
    gates.append(gate('acceptance_correct', correct.mean(), 0.95, n))
    gates.append(gate('acceptance_rejected', rejected.mean(), 0.05, n, pass_fn=lambda v, t: v <= t))
    regions = {}
    for p in range(n_patterns):
        rows = pattern_index == p
        regions[f'pattern_{p}'] = {
            'phase': PHASE_NAMES[int(truth['pattern_phase'][p])],
            'zone_axis': [int(v) for v in pattern_zone[p]],
            'in_plane_deg': float(truth['pattern_in_plane_deg'][p]),
            'region_size': int(rows.sum()),
            'correct': int(correct[rows].sum()),
            'rejected': int(rejected[rows].sum()),
            'ambiguous': int(ambiguous[rows].sum()),
            'wrong': int(wrong[rows].sum()),
            'accuracy': float(correct[rows].mean()),
            'rejected_fraction': float(rejected[rows].mean()),
            'recall': per_pattern[p]['recall'],
            'center_error_median_px': float(np.median(center_error[rows][center_ok[rows]]))
            if center_ok[rows].any() else None,
            'reason_flags': {label: int(((reasons[rows] & bit) > 0).sum())
                             for bit, label in REASON_BITS.items()}}
        if len(per_pattern) > 1:
            gates.append(gate(f'pattern_{p}_accuracy', regions[f'pattern_{p}']['accuracy'], 0.90,
                              int(rows.sum())))
    confusion = np.zeros((2, 4), dtype=int)
    for row_truth in (0, 1):
        mask = truth_phase == row_truth
        confusion[row_truth, 0] = int(((phase_id == 0) & mask).sum())
        confusion[row_truth, 1] = int(((phase_id == 1) & mask).sum())
        confusion[row_truth, 2] = int((rejected & mask).sum())
        confusion[row_truth, 3] = int((ambiguous & mask).sum())
    accepted = np.isin(phase_id, (0, 1))
    worst = np.argsort(np.where(accepted, score, np.inf))[:10]
    worst_entries = [{'position': [int(i // shape[1]), int(i % shape[1])],
                      'score': float(score[i]), 'margin': float(margin[i]),
                      'median_residual_px': float(residual[i]),
                      'matched_peaks': int(matched_peaks[i]),
                      'reason_flags': int(reasons[i])} for i in worst]

    # --- orientation (informational) ----------------------------------------
    orientation = {'status': 'not_evaluable', 'reasons': []}
    gate_records = []
    for p in range(n_patterns):
        candidate_name = PHASE_CANDIDATE[int(truth['pattern_phase'][p])]
        library_key = (candidate_name, 300)
        if library_key not in template_paths:
            orientation['reasons'].append(f'no 300 kV template library for {candidate_name}')
            gate_records.append(None)
            continue
        sample = int(np.flatnonzero(pattern_index == p)[0])
        record = gate_pattern(peak_hkl[sample][observable[sample]],
                              peak_qxy[sample][observable[sample]],
                              template_paths[library_key], pattern_zone[p], dq)
        gate_records.append(record)
        if record is None:
            orientation['reasons'].append(
                f'pattern {p} ({candidate_name} {list(pattern_zone[p])}) not representable '
                f'by its template library at <= 0.05 px')
    if all(record is not None for record in gate_records):
        orientation['status'] = 'evaluable'
        mis, in_plane, row_consistent, mirror_consistent, zone_tilt = [], [], [], [], []
        mis_by_pattern = {p: [] for p in range(n_patterns)}
        library_cache = {}
        for i in np.flatnonzero(accepted):
            winner = int(np.argmax(all_results[i, :, 0]))
            library_spec = schema['result_libraries'][winner]
            name = next(c['name'] for c in candidates if c['id'] == library_spec['phase_id'])
            library_key = (name, int(library_spec['voltage_kv']))
            if library_key not in template_paths:
                continue
            if library_key not in library_cache:
                library_cache[library_key] = np.load(template_paths[library_key])['matrices']
            row = int(round(float(all_results[i, winner, 4])))
            angle = float(all_results[i, winner, 3])
            mirror_flag = float(all_results[i, winner, 5])  # stored as +-1 by score_pose
            p = int(pattern_index[i])
            record = gate_records[p]
            r_truth = pattern_rot[p]
            # Pipeline pose convention, verified numerically against score_pose and
            # the stored template geometry (see findings.md): the matcher treats the
            # stored (x,y) template columns as (y,x), so on the (x,y) plane the pose
            # maps q_t -> P @ Rz2(angle) @ diag(1, mirror) @ q_t with P the x<->y
            # swap (an improper map absorbed by the observed set's mirror symmetry).
            # The stored template geometry is q_t = (M.T @ g)[:2] with the zone axis
            # = M[:, 2] (third column).  A mirror flag makes the implied 3D map
            # improper; for centrosymmetric phases the inversion is absorbed before
            # the proper-rotation disorientation below.
            a2 = SWAP_XY @ rotation_z2(angle) @ np.diag([1.0, mirror_flag])
            a3 = np.eye(3)
            a3[:2, :2] = a2
            r_pred = a3 @ library_cache[library_key][row].T
            delta = r_pred.T @ r_truth
            if np.linalg.det(delta) < 0.0:
                delta = -delta
            traces = np.array([np.trace(delta @ op) for op in CUBIC])
            d_opt = delta @ CUBIC[int(np.argmax(traces))]
            mis_value = np.degrees(np.arccos(np.clip((float(traces.max()) - 1.0) / 2.0, -1.0, 1.0)))
            mis.append(mis_value)
            mis_by_pattern[p].append(mis_value)
            # Residual in-plane deviation after the symmetry-optimal op: the z
            # component of the remaining small rotation.
            omega = np.arctan2(d_opt[1, 0] - d_opt[0, 1], d_opt[0, 0] + d_opt[1, 1])
            in_plane.append(abs(wrap_pi(omega)))
            # Zone axis of the reported template row (third column of M) vs truth.
            row_zone = library_cache[library_key][row][:, 2]
            zone_truth = np.asarray(pattern_zone[p], float)
            zone_truth = zone_truth / np.linalg.norm(zone_truth)
            zone_tilt.append(min(np.degrees(np.arccos(np.clip(
                float(np.dot(row_zone, zone_truth)), -1.0, 1.0))),
                np.degrees(np.arccos(np.clip(
                float(np.dot(row_zone, -zone_truth)), -1.0, 1.0)))))
            # Strict row consistency: is the reported zone cubic-equivalent to the
            # gate row's zone?  (Not expected on the flat score landscape.)
            gate_zone = library_cache[library_key][record['template_row']][:, 2]
            equivalent = any(np.linalg.norm(op @ row_zone - gate_zone) < 1e-3
                             or np.linalg.norm(op @ row_zone + gate_zone) < 1e-3 for op in CUBIC)
            row_consistent.append(equivalent)
            mirror_consistent.append(mirror_flag == record['mirror'])
        if mis:
            orientation.update({
                'accepted_patterns': len(mis),
                'misorientation_deg_median': float(np.median(mis)),
                'misorientation_deg_p90': float(np.percentile(mis, 90)),
                'misorientation_deg_max': float(np.max(mis)),
                'misorientation_deg_by_pattern_median': {str(p): float(np.median(v))
                                                         for p, v in mis_by_pattern.items() if v},
                'template_zone_tilt_deg_median': float(np.median(zone_tilt)),
                'in_plane_error_deg_median': float(np.median(in_plane)),
                'in_plane_error_deg_p90': float(np.percentile(in_plane, 90)),
                'template_row_consistency': float(np.mean(row_consistent)),
                'mirror_flag_consistency': float(np.mean(mirror_consistent)),
                'note': ('the score landscape is flat within the 2.5 px matching '
                         'tolerance: template rows tie over ~4 deg zone tilts and the '
                         'first-found row is reported (zone tilt dominates for 4-fold '
                         'and 2-fold patterns), and the FCC[111] hexagon is 6-fold in '
                         'the detector while the crystal is 3-fold, so a 60 deg '
                         'in-plane rotation (Sigma3) is indistinguishable from the data. '
                         'See findings.md.'),
                'targets_informational': {'misorientation_deg_median': 2.0,
                                          'in_plane_error_deg_median': 1.0}})
        else:
            orientation['status'] = 'not_evaluable'
            orientation['reasons'].append('no accepted patterns with a usable template row')

    # --- artifacts -----------------------------------------------------------
    figures_dir = run_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = figures_dir / key
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, data, title in [(axes[0], truth_phase.reshape(shape), 'truth'),
                            (axes[1], phase_id.reshape(shape), 'predicted'),
                            (axes[2], (phase_id != truth_phase).reshape(shape), 'mismatch')]:
        image = ax.imshow(data, cmap='viridis', interpolation='nearest')
        fig.colorbar(image, ax=ax)
        ax.set_title(f'{key} {title}')
    fig.savefig(stem.with_name(f'{key}_phase_map.png'), dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    image = ax.imshow(confusion, cmap='Blues')
    ax.set_xticks(range(4), ['BCC', 'FCC', 'unindexed', 'ambiguous'])
    ax.set_yticks([0, 1], ['truth BCC', 'truth FCC'])
    for i in range(2):
        for j in range(4):
            ax.text(j, i, str(confusion[i, j]), ha='center', va='center')
    ax.set_title(f'{key} confusion')
    fig.colorbar(image, ax=ax)
    fig.savefig(stem.with_name(f'{key}_confusion.png'), dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(score[accepted], bins=40)
    axes[0].set_title(f'{key} score (accepted)')
    axes[1].hist(margin[accepted], bins=40)
    axes[1].set_title(f'{key} margin (accepted)')
    fig.savefig(stem.with_name(f'{key}_score_margin.png'), dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(centers[:, 1] - truth_center[1], centers[:, 0] - truth_center[0],
                         c=grain, s=4)
    ax.set_xlabel('center dx (px)'); ax.set_ylabel('center dy (px)')
    ax.set_title(f'{key} center error')
    fig.colorbar(scatter, ax=ax, label='grain')
    fig.savefig(stem.with_name(f'{key}_center_error.png'), dpi=130)
    plt.close(fig)
    sample = int(np.flatnonzero(pattern_index == 0)[0])
    pairs_csv = run_dir / f'{key}_peak_pairs.csv'
    with open(pairs_csv, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['h', 'k', 'l', 'qx_A', 'qy_A', 'amplitude', 'detected_index',
                         'distance_px', 'matched'])
        count = int(counts[sample])
        pairs = greedy_pair(peak_px[sample][observable[sample]], peak_amp[sample][observable[sample]],
                            detected[sample, :count], 1.5)
        matched = {truth_i: (det_j, dist) for truth_i, det_j, dist in pairs}
        for index in np.flatnonzero(observable[sample]):
            hit = matched.get(int(index))
            writer.writerow([*peak_hkl[sample, index].tolist(),
                             f'{peak_qxy[sample, index, 0]:.6f}', f'{peak_qxy[sample, index, 1]:.6f}',
                             f'{peak_amp[sample, index]:.1f}',
                             int(hit[0]) if hit else '', f'{hit[1]:.3f}' if hit else '',
                             'true' if hit else 'false'])
    return {'key': key, 'scan_name': entry['scan_name'], 'patterns': n,
            'scale': scale, 'calibration_status': calibration['status'],
            'gates': gates, 'regions': regions,
            'confusion': {'rows': ['truth BCC', 'truth FCC'],
                          'columns': ['BCC', 'FCC', 'unindexed', 'ambiguous'],
                          'matrix': confusion.tolist()},
            'score_margin': {'score_median': float(np.median(score[accepted])) if accepted.any() else None,
                             'margin_median': float(np.median(margin[accepted])) if accepted.any() else None,
                             'residual_median_px': float(np.median(residual[accepted])) if accepted.any() else None,
                             'perturbation_stable_fraction': float(np.load(output / 'perturbation_stable.npy').reshape(n)[accepted].mean()) if accepted.any() else None,
                             'voltage_stable_fraction': float(np.load(output / 'voltage_stable.npy').reshape(n)[accepted].mean()) if accepted.any() else None},
            'worst_10_accepted': worst_entries,
            'selection_coverage': selection_coverage,
            'orientation': orientation,
            'pairing_csv': str(pairs_csv),
            'figures': [str(figures_dir / f'{key}_{suffix}.png') for suffix in
                        ('phase_map', 'confusion', 'score_margin', 'center_error')]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    manifest = json.loads((run_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    candidates = yaml.safe_load((run_dir / 'configs' / 'T01.yaml').read_text(encoding='utf-8'))['candidates']
    template_paths = template_libraries(run_dir / 'predictions', candidates)
    if not template_paths:
        print(f'Warning: no template libraries under {run_dir / "predictions" / "templates"}')
    datasets = []
    for key, entry in manifest['datasets'].items():
        datasets.append(evaluate_dataset(dict(entry, key=key), candidates, template_paths, run_dir))
        report = datasets[-1]
        print(f"{report['key']}: scale {report['scale']:.6f} ({report['calibration_status']}), "
              f"{report['patterns']} patterns")
        for g in report['gates']:
            print(f"  {'PASS' if g['pass'] else 'FAIL'} {g['name']} = {g['value']:.4f} "
                  f"{g['units']} (threshold {g['threshold']}, n={g['denominator']})")
    failed = [g for d in datasets for g in d['gates'] if not g['pass']]
    report = {'generated': datetime.now().isoformat(timespec='seconds'), 'run_id': manifest['run_id'],
              'overall_pass': not failed, 'datasets': datasets,
              'reason_bit_labels': REASON_BITS,
              'orientation_note': 'informational targets: misorientation median <= 2 deg, '
                                  'in-plane median <= 1 deg; scored only after the template '
                                  'self-consistency gate (<= 0.05 px) passes.'}
    target = run_dir / 'validation_report.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(f'Wrote {target}; overall {"PASS" if not failed else "FAIL"} '
          f'({len(failed)} failing gates)')
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
