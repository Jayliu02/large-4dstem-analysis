"""Audit saved phase fits and replay sensitivity checks without changing labels."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .phase_matching import PhaseMatcher, decisions
from .phase_observations import select_observations
from .phase_peaks import atomic_json, read_json


def winning_gates(results, libraries, counts, config, threshold):
    voltages = sorted({int(lib['voltage_kv']) for lib in libraries})
    phases = sorted({int(lib['phase_id']) for lib in libraries})
    winners, bests, margins = [], [], []
    for voltage in voltages:
        indices = [next(i for i, lib in enumerate(libraries)
                        if int(lib['phase_id']) == p and int(lib['voltage_kv']) == voltage) for p in phases]
        candidates = results[:, indices]
        rank = np.argsort(candidates[:, :, 0], axis=1)
        first, second = rank[:, -1], rank[:, -2]
        best = candidates[np.arange(len(results)), first]
        bests.append(best)
        winners.append(np.asarray(phases)[first])
        margins.append((best[:, 0]-candidates[np.arange(len(results)), second, 0])/np.maximum(best[:, 0], 1e-12))
    best = np.stack(bests, axis=1)
    winners = np.stack(winners, axis=1)
    gates = {
        'six_matches': best[:, :, 1] >= config['min_peaks'],
        'observed_fraction': best[:, :, 1]/np.maximum(counts[:, None], 1) >= config['minimum_observed_fraction'],
        'residual': best[:, :, 2] <= config['max_median_residual_px'],
        'score': best[:, :, 0] >= threshold,
        'two_dimensional': best[:, :, 6] > 0,
        'phase_margin': np.stack(margins, axis=1) >= config['min_phase_margin'],
    }
    return voltages, winners, gates


def gate_statistics(results, libraries, counts, center_valid, config, threshold, final):
    voltages, winners, gates = winning_gates(results, libraries, counts, config, threshold)
    same_phase = np.all(winners == winners[:, :1], axis=1)
    eligible = center_valid & (counts >= config['min_peaks'])
    evidence = np.logical_and.reduce([value for key, value in gates.items() if key != 'phase_margin'])
    confidence = evidence & gates['phase_margin']
    provisional = eligible & confidence.all(axis=1) & same_phase
    cascade = {'all_positions': len(counts), 'valid_center': int(center_valid.sum()),
               'eligible_observations': int(eligible.sum()),
               'evidence_any_voltage': int((eligible & evidence.any(axis=1)).sum()),
               'evidence_all_voltages': int((eligible & evidence.all(axis=1)).sum()),
               'phase_margin_all_voltages': int((eligible & confidence.all(axis=1)).sum()),
               'same_phase_all_voltages': int(provisional.sum()),
               'all_perturbations': int((final >= 0).sum())}
    dropped = {}
    for omitted in gates:
        passed = np.logical_and.reduce([value for key, value in gates.items() if key != omitted])
        dropped[omitted] = int((eligible & passed.all(axis=1) & same_phase).sum())
    stats = {'cascade': cascade, 'phase_names': {str(int(lib['phase_id'])): str(lib.get('phase_name', lib['phase_id'])) for lib in libraries},
             'eligible_fail_each_gate_all_voltages_nonexclusive': {
                 key: int((eligible & ~value.all(axis=1)).sum()) for key, value in gates.items()},
             'drop_one_gate_before_perturbation_fixed_fits': dropped,
             'provisional_by_phase': {str(int(p)): int((provisional & (winners[:, 0] == p)).sum()) for p in np.unique(winners)},
             'final_by_phase': {str(int(p)): int((final == p).sum()) for p in np.unique(winners)},
             'single_voltage_before_perturbation': {str(v): int((eligible & confidence[:, i]).sum()) for i, v in enumerate(voltages)},
             'selected_count_percentiles_0_25_50_75_100': np.percentile(counts, [0,25,50,75,100]),
             'insufficient_selected_peaks': int((counts < config['min_peaks']).sum())}
    return stats, provisional


def replay(observed, counts, centers, indices, base_labels, libraries, scale, visibility, cfg, threshold, folder):
    """Evaluate each perturbation separately on exactly the original provisional set."""
    matching = cfg['matching']
    options = {'inner': cfg['peaks']['exclusion_radius'], 'outer': cfg['peaks']['max_radius']}
    base_matcher = PhaseMatcher(libraries, scale, visibility, matching, **options)
    summary, saved = {}, {'indices': indices, 'base_labels': base_labels}
    for suite, delta, shift in [('configured', matching['scale_perturbation'], matching['center_perturbation_px']),
                                ('small_probe', .005, .25)]:
        retained = np.ones(len(indices), bool)
        probes = [(f'scale_{sign:+d}', scale*(1+sign*delta), np.zeros(2)) for sign in [-1, 1]]
        probes += [(f'center_{axis}_{sign:+d}', scale, np.eye(2)[axis]*sign*shift) for axis in [0,1] for sign in [-1,1]]
        suite_result = {'scale_delta': delta, 'center_shift_px': shift, 'probes': {}}
        for name, probe_scale, offset in probes:
            matcher = base_matcher if probe_scale == scale else PhaseMatcher(libraries, probe_scale, visibility, matching, **options)
            r = matcher.match(observed-offset, counts, centers+offset)
            d = decisions(r, libraries, matching, observed_counts=counts, min_score=threshold)
            keep = d['phase_id'] == base_labels
            retained &= keep
            _, _, gates = winning_gates(r, libraries, counts, matching, threshold)
            suite_result['probes'][name] = {
                'same_accepted_phase': int(keep.sum()),
                'rejected_unindexed': int((d['phase_id'] == -1).sum()),
                'rejected_ambiguous': int((d['phase_id'] == -2).sum()),
                'other_accepted_phase': int(((d['phase_id'] >= 0) & ~keep).sum()),
                'fail_each_gate_nonexclusive': {key: int((~value.all(axis=1)).sum()) for key, value in gates.items()}}
            saved[f'{suite}_{name}'] = d['phase_id']
            print(f'    {suite} {name}: {keep.sum()}/{len(indices)} retain phase', flush=True)
        suite_result['retain_all'] = int(retained.sum())
        suite_result['retain_all_by_phase'] = {str(int(p)): int((retained & (base_labels == p)).sum()) for p in [0,1]}
        summary[suite] = suite_result
        saved[f'{suite}_retain_all'] = retained
    np.savez_compressed(folder/'perturbation_replay.npz', **saved)
    return summary


def sample_shortlist(observed, counts, centers, libraries, scale, visibility, cfg, threshold, size):
    """A deterministic spatial sample, independent of phase score or acceptance."""
    indices = np.linspace(0, len(counts)-1, min(size, len(counts))).round().astype(int)
    result = {'sample_size': len(indices), 'cases': {}}
    if not len(indices):
        return result
    options = {'inner': cfg['peaks']['exclusion_radius'], 'outer': cfg['peaks']['max_radius']}
    for k in [cfg['matching']['shortlist'], 32, len(libraries[0]['count'])]:
        config = {**cfg['matching'], 'shortlist': k}
        fits = PhaseMatcher(libraries, scale, visibility, config, **options).match(observed[indices], counts[indices], centers[indices])
        decision = decisions(fits, libraries, config, observed_counts=counts[indices], min_score=threshold)
        result['cases'][str(k)] = {'accepted_before_perturbation': int((decision['phase_id'] >= 0).sum()),
                                   'mean_minimum_winning_score': float(decision['min_score'].mean())}
        print(f'    shortlist={k}: {result["cases"][str(k)]}', flush=True)
    return result


def plot_reports(reports, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stages = ['all_positions','eligible_observations','evidence_any_voltage',
              'evidence_all_voltages','same_phase_all_voltages','all_perturbations']
    fig, axes = plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for report in reports:
        values = [report['cascade'][k] for k in stages]
        axes[0].plot(range(len(stages)),values,'o-',label=report['scan'][-4:])
    axes[0].set_yscale('symlog',linthresh=1)
    axes[0].set_xticks(range(len(stages)),['All','Center +\n6 observations','Evidence at\nany voltage',
        'Evidence at\nall voltages','Before\nperturbation','Final'])
    axes[0].set(ylabel='Positions (symlog scale)',title='Where positions are rejected')
    axes[0].legend(title='Scan')
    axes[0].grid(axis='y',alpha=.2)
    x = np.arange(len(reports))
    for offset,suite,label in [(-.2,'configured','Configured: 2% scale / 1 px center'),
                               (.2,'small_probe','Diagnostic only: 0.5% scale / 0.25 px center')]:
        counts = [r['perturbations'][suite]['retain_all'] for r in reports]
        values = [100*n/max(r['cascade']['same_phase_all_voltages'],1) for n,r in zip(counts,reports)]
        bars = axes[1].bar(x+offset,values,width=.4,label=label)
        axes[1].bar_label(bars,labels=[f'{v:.2f}%\n({n} points)' for v,n in zip(values,counts)],padding=3,fontsize=9)
    axes[1].set_xticks(x,[r['scan'][-4:] for r in reports])
    axes[1].set(ylabel='Retained original provisional positions (%)',ylim=(0,80),
                title='All six perturbations must retain the accepted phase')
    axes[1].legend(loc='upper right',fontsize=8)
    fig.savefig(destination/'rejection_diagnostics.png',dpi=180)
    fig.savefig(destination/'rejection_diagnostics.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/phase_identification_fe'))
    parser.add_argument('--replay', action='store_true', help='Recompute configured and smaller perturbations on all provisional positions.')
    parser.add_argument('--shortlist-sample', type=int, default=0, help='Also test top-8, top-32 and all-template search on this many eligible spatial positions.')
    args = parser.parse_args()
    if args.shortlist_sample < 0:
        parser.error('--shortlist-sample must be nonnegative')
    from numba import set_num_threads
    from threadpoolctl import threadpool_limits
    from .phase_identification import source_signature
    destination = args.root/'diagnostics'
    destination.mkdir(exist_ok=True)
    reports = []
    for scan in sorted(p for p in args.root.glob('scan_*') if p.is_dir()):
        checkpoint = read_json(scan/'matching_checkpoint.json')
        if not checkpoint['complete'] or checkpoint['provenance']['implementation'] != source_signature():
            raise ValueError(f'Incomplete or incompatible matching cache: {scan}')
        cfg = checkpoint['provenance']['config']
        calibration = read_json(scan/'calibration.json')
        libraries = []
        for c in cfg['candidates']:
            for v in cfg['templates']['voltages_kv']:
                with np.load(args.root/'templates'/f"{c['name']}_{v}kV.npz") as data:
                    libraries.append(dict(data))
        fits = np.load(scan/'all_results.npy', mmap_mode='r').reshape(-1,len(libraries),7)
        counts = np.load(scan/'selected_peak_count.npy').ravel()
        valid = np.load(scan/'peaks/center_valid.npy').ravel()
        final = np.load(scan/'phase_id.npy').ravel()
        threshold = calibration['null_score_threshold']
        report, provisional = gate_statistics(fits, libraries, counts, valid, cfg['matching'], threshold, final)
        report['scan'] = scan.name
        report['calibration'] = calibration
        decision = decisions(fits, libraries, cfg['matching'], observed_counts=counts, min_score=threshold)
        np.testing.assert_array_equal(provisional, (decision['phase_id'] >= 0) & valid & (counts >= cfg['matching']['min_peaks']))
        print(scan.name, report['cascade'], flush=True)
        out = destination/scan.name
        out.mkdir(exist_ok=True)
        set_num_threads(cfg['runtime']['threads'])
        with threadpool_limits(limits=cfg['runtime']['threads']):
            if args.replay or args.shortlist_sample:
                peaks = {k: np.load(scan/'peaks'/f'{k}.npy', mmap_mode='r') for k in ['peak_yx','intensity','count','center_yx','center_valid']}
                selected = select_observations(peaks, cfg['matching'])
                centers = peaks['center_yx'].reshape(-1,2)
                observed = selected['peak_yx'].reshape(-1,cfg['peaks']['max_peaks'],2)-centers[:,None,:]
                visibility = np.load(scan/'detector_visibility.npy')
                scale = calibration['scale_inv_angstrom_per_pixel']
                if args.replay:
                    idx = np.flatnonzero(provisional)
                    report['perturbations'] = replay(observed[idx], counts[idx], centers[idx], idx,
                        decision['phase_id'][idx], libraries, scale, visibility, cfg, threshold, out)
                    with np.load(out/'perturbation_replay.npz') as probe:
                        np.testing.assert_array_equal(probe['configured_retain_all'], final[idx] >= 0)
                if args.shortlist_sample:
                    idx = np.flatnonzero(valid & (counts >= cfg['matching']['min_peaks']))
                    report['shortlist_sample'] = sample_shortlist(observed[idx], counts[idx], centers[idx], libraries,
                        scale, visibility, cfg, threshold, args.shortlist_sample)
        atomic_json(out/'gate_diagnostics.json', report)
        reports.append(report)
    atomic_json(destination/'gate_summary.json', {'scans': reports,
        'interpretation': 'Diagnostic counts only; saved phase maps and thresholds are unchanged. Gate failures overlap; cascade order is explicit. Small probes and larger shortlists are sensitivity experiments, not validated replacement settings.'})
    if args.replay:
        plot_reports(reports,destination)


if __name__ == '__main__':
    main()
