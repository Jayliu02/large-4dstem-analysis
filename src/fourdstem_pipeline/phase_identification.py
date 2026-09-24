"""Identify Fe BCC/FCC with per-pattern centering and explicit rejection."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
import yaml

from .phase_peaks import atomic_json, digest, extract_scan, read_json


def source_signature():
    parent = Path(__file__).parent
    return {name: hashlib.sha256((parent/name).read_bytes()).hexdigest() for name in
            ['phase_identification.py', 'phase_matching.py', 'phase_calibration.py',
             'phase_structures.py', 'phase_peaks.py', 'phase_observations.py']}


def identify_scan(path, basic, output, peaks, libraries, cfg):
    from .phase_calibration import calibrate
    from .phase_matching import PhaseMatcher, decisions
    from .phase_report import scan_report
    from .phase_observations import select_observations
    started = time.perf_counter()
    signature_data = {'config': cfg, 'implementation': source_signature(),
                      'peaks': read_json(output/'peaks'/'peaks_checkpoint.json')['signature'],
                      'cifs': {c['name']: hashlib.sha256(Path(c['cif']).read_bytes()).hexdigest() for c in cfg['candidates']},
                      'versions': {name: version(name) for name in ['numpy', 'scipy', 'py4DSTEM', 'pymatgen', 'spglib', 'numba']}}
    signature = digest(signature_data)
    checkpoint_path = output/'matching_checkpoint.json'
    checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else {}
    if checkpoint and checkpoint['signature'] != signature:
        raise ValueError('Matching cache input/configuration/code changed; use a new output directory.')
    if not checkpoint:
        checkpoint = {'signature': signature, 'provenance': signature_data, 'chunks': [], 'complete': False}
        atomic_json(checkpoint_path, checkpoint)
    peaks = select_observations(peaks, cfg['matching'])
    np.save(output/'selected_peak_indices.npy', peaks['selected_peak_indices'])
    np.save(output/'selected_peak_count.npy', peaks['count'])
    visibility = np.load(basic/'raw_mean_diffraction.npy') > 0.05
    np.save(output/'detector_visibility.npy', visibility)
    if (output/'calibration.json').exists() and (output/'calibration_samples.npz').exists():
        calibration = read_json(output/'calibration.json')
    else:
        calibration = calibrate(peaks, libraries, visibility, cfg, output)
    calibrated = calibration['status'] == 'calibrated_conditionally'
    scale = calibration['scale_inv_angstrom_per_pixel']
    shape = peaks['count'].shape
    n = int(np.prod(shape))
    specs = {'phase_id': (shape, 'int16', -1), 'best_candidate': (shape, 'int16', -1),
             'score': (shape, 'float32', 0), 'margin': (shape, 'float32', 0),
             'median_residual_px': (shape, 'float32', np.inf), 'matched_peaks': (shape, 'int16', 0),
             'voltage_stable': (shape, 'bool', False), 'perturbation_stable': (shape, 'bool', False),
             'reason_flags': (shape, 'uint16', 0), 'all_results': (shape+(len(libraries), 7), 'float32', 0)}
    arrays = {}
    for name, (size, dtype, fill) in specs.items():
        target = output/f'{name}.npy'
        if target.exists():
            arrays[name] = np.load(target, mmap_mode='r+')
            if arrays[name].shape != size or arrays[name].dtype != dtype:
                raise ValueError(f'Unexpected cached array layout: {target}')
        else:
            if checkpoint['chunks']:
                raise ValueError(f'Missing cached array: {target}; choose a new output directory.')
            arrays[name] = np.lib.format.open_memmap(target, mode='w+', dtype=dtype, shape=size)
            arrays[name][:] = fill
    flat = {key: arr.reshape((n,)+arr.shape[2:]) for key, arr in arrays.items()}
    centers = peaks['center_yx'].reshape(n, 2)
    counts = peaks['count'].reshape(n)
    valid = peaks['center_valid'].reshape(n)
    positions = peaks['peak_yx'].reshape(n, -1, 2)
    matcher_options = {'inner': cfg['peaks']['exclusion_radius'], 'outer': cfg['peaks']['max_radius']}
    matcher = PhaseMatcher(libraries, scale, visibility, cfg['matching'], **matcher_options)
    delta = cfg['matching']['scale_perturbation']
    scale_matchers = [PhaseMatcher(libraries, scale*(1+sign*delta), visibility, cfg['matching'], **matcher_options)
                      for sign in [-1, 1]] if calibrated else []
    done = set(checkpoint['chunks'])
    block = cfg['runtime']['match_chunk']
    for start in range(0, n, block):
        if start in done:
            continue
        stop = min(start+block, n)
        obs = positions[start:stop]-centers[start:stop, None, :]
        result = matcher.match(obs, counts[start:stop], centers[start:stop])
        decision = decisions(result, libraries, cfg['matching'], observed_counts=counts[start:stop], min_score=calibration['null_score_threshold'])
        labels = decision['phase_id'].copy()
        reason = np.zeros(stop-start, np.uint16)
        reason[labels == -1] |= 2
        reason[labels == -2] |= 4
        reason[~valid[start:stop]] |= 1
        reason[counts[start:stop] < cfg['matching']['min_peaks']] |= 32
        provisional = np.flatnonzero((labels >= 0) & valid[start:stop]) if calibrated else np.array([], dtype=int)
        stable = np.zeros(stop-start, bool)
        stable[provisional] = True
        if len(provisional):
            for altered in scale_matchers:
                fitted = altered.match(obs[provisional], counts[start:stop][provisional], centers[start:stop][provisional])
                other = decisions(fitted, libraries, cfg['matching'], observed_counts=counts[start:stop][provisional], min_score=calibration['null_score_threshold'])
                stable[provisional] &= other['phase_id'] == labels[provisional]
            shift = cfg['matching']['center_perturbation_px']
            for dy, dx in [(shift,0), (-shift,0), (0,shift), (0,-shift)]:
                offset = np.array([dy,dx], np.float32)
                fitted = matcher.match(obs[provisional]-offset, counts[start:stop][provisional], centers[start:stop][provisional]+offset)
                other = decisions(fitted, libraries, cfg['matching'], observed_counts=counts[start:stop][provisional], min_score=calibration['null_score_threshold'])
                stable[provisional] &= other['phase_id'] == labels[provisional]
            rejected = provisional[~stable[provisional]]
            labels[rejected] = -2
            reason[rejected] |= 8
        if not calibrated:
            labels[:] = -1
            reason[:] |= 16
        labels[~valid[start:stop]] = -1
        labels[counts[start:stop] < cfg['matching']['min_peaks']] = -1
        best = result[np.arange(stop-start), result[:, :, 0].argmax(axis=1)]
        values = {'phase_id': labels, 'best_candidate': decision['best_candidate'], 'score': decision['min_score'],
                  'margin': decision['min_margin'], 'voltage_stable': decision['voltage_stable'],
                  'perturbation_stable': stable, 'reason_flags': reason, 'all_results': result,
                  'median_residual_px': best[:, 2], 'matched_peaks': best[:, 1]}
        for name, value in values.items():
            flat[name][start:stop] = value
            arrays[name].flush()
        done.add(start)
        checkpoint['chunks'] = sorted(done)
        atomic_json(checkpoint_path, checkpoint)
        if stop % (16*shape[1]) == 0 or stop == n:
            print(f'  match {output.name}: {stop}/{n}, elapsed {time.perf_counter()-started:.0f}s', flush=True)
    checkpoint['complete'] = True
    atomic_json(checkpoint_path, checkpoint)
    phase_labels = {'-2': 'ambiguous', '-1': 'unindexed',
                    **{str(c['id']): c['label'] for c in cfg['candidates']}}
    atomic_json(output/'array_schema.json', {
        'scan_shape': shape, 'scan_order': 'row-major, unidirectional', 'coordinates': 'detector (y,x), pixels',
        'phase_ids': phase_labels,
        'reason_bits': {'1': 'invalid beam center', '2': 'insufficient match evidence', '4': 'phase/voltage ambiguity',
                        '8': 'perturbation instability', '16': 'uncalibrated scale', '32': 'fewer than minimum peaks'},
        'result_columns': ['score', 'matched_peak_count', 'median_residual_px', 'in_plane_angle_rad',
                           'template_index', 'transverse_mirror', 'non_collinear_support'],
        'result_libraries': [{'phase_id': int(lib['phase_id']), 'voltage_kv': int(lib['voltage_kv'])} for lib in libraries],
        'score': 'minimum winning score across voltage hypotheses; not a probability',
        'best_candidate': 'winner at the voltage hypothesis with highest score; not an accepted phase',
        'median_residual_px': 'residual of highest-scoring fit, including rejected fits; infinity means no match'})
    summary = {'file': str(path), 'output': output.name, 'status': 'complete', 'patterns': n,
               'calibration_status': calibration['status'], 'scale': scale,
               'phase_labels': phase_labels,
               'phase_counts': {k: int(np.count_nonzero(arrays['phase_id'] == int(k))) for k in phase_labels},
               'input_unchanged': path.stat().st_mtime_ns == read_json(output/'peaks'/'peaks_checkpoint.json')['provenance']['mtime_ns']}
    scan_report(path, basic, output, peaks, arrays, libraries, calibration, cfg, summary)
    atomic_json(output/'phase_summary.json', summary)
    return summary


def validate_config(cfg):
    candidates = cfg['candidates']
    if len(candidates) < 2 or [c['id'] for c in candidates] != list(range(len(candidates))):
        raise ValueError('Expected at least two candidates with consecutive phase ids starting at zero.')
    if len({c['name'] for c in candidates}) != len(candidates):
        raise ValueError('Candidate names must be unique.')
    if len(cfg['templates']['voltages_kv']) < 2:
        raise ValueError('At least two voltage hypotheses are required for voltage stability testing.')
    lo, hi = cfg['calibration']['scale_range']
    if not 0 < lo < hi or cfg['calibration']['scale_steps'] < 3:
        raise ValueError('Invalid reciprocal-scale search range/grid.')
    if cfg['matching']['min_peaks'] < 6 or cfg['runtime']['match_chunk'] < 1:
        raise ValueError('Require at least six matched peaks and a positive chunk size.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/phase_identification.yaml'))
    parser.add_argument('--output', type=Path, help='Override output directory; use a new directory after changing inputs/config/code.')
    parser.add_argument('--stage', choices=['all','prepare','extract','identify'], default='all')
    parser.add_argument('--scan', action='append', help='Restrict to named scan, e.g. scan_01_1045 (repeatable).')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if args.output:
        cfg['output_dir'] = str(args.output)
    validate_config(cfg)
    root = Path(cfg['output_dir'])
    root.mkdir(parents=True, exist_ok=True)
    from numba import set_num_threads
    from threadpoolctl import threadpool_limits
    set_num_threads(cfg['runtime']['threads'])
    with threadpool_limits(limits=cfg['runtime']['threads']):
        libraries = []
        if args.stage != 'extract':
            from .phase_structures import prepare_libraries
            libraries = prepare_libraries(cfg['candidates'], cfg['templates'], root/'templates')
        if args.stage == 'prepare':
            return 0
        paths = sorted(Path(cfg['data']['directory']).glob(cfg['data']['pattern']))
        selected = [(path, f'scan_{i:02d}_{path.stem[-4:]}') for i,path in enumerate(paths,1)]
        if args.scan:
            known = {name for _,name in selected}
            if set(args.scan)-known:
                parser.error(f'Unknown scans: {sorted(set(args.scan)-known)}')
            selected = [(path,name) for path,name in selected if name in args.scan]
        if not selected:
            parser.error('No input scans found.')
        for path,name in selected:
            output, basic = root/name, Path(cfg['data']['basic_analysis_dir'])/name
            output.mkdir(parents=True, exist_ok=True)
            print(f'{name}: {path.name}', flush=True)
            if args.stage == 'identify':
                checkpoint = output/'peaks'/'peaks_checkpoint.json'
                if not checkpoint.exists() or not read_json(checkpoint).get('complete'):
                    raise ValueError(f'Complete peak extraction first: {name}')
            peaks = extract_scan(path, basic, output/'peaks', cfg['data']['scan_shape'], cfg['peaks'])
            if args.stage == 'extract':
                continue
            identify_scan(path, basic, output, peaks, libraries, cfg)
        if args.stage != 'extract':
            from .phase_report import batch_report
            batch_report(root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
