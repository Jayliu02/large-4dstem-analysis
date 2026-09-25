"""Build the Fe BCC/FCC benchmark datasets from the reference spot lists.

Per dataset this writes three artifacts under benchmarks/Fe_FCC_BCC_v1/:
    datasets/T0X/<name>.h5     canonical HDF5: /datacube/data uint16 (sy, sx, 256, 256)
                               and /metadata (JSON string); py4DSTEM EMD is NOT claimed
    datasets/T0X/<name>.mib    single-chip MIB with the real 384-byte frame header
                               template so the existing phase CLI reads it unchanged
    truth/T0X_truth.npz        ground truth (phase/grain/pattern maps, spot lists,
                               rotations, pixel transform); the analysis CLI never reads it

Modes:
    (default)        build everything and write build_manifest.json
    --check          re-hash the written payloads and re-derive the truth; exit 0/1
    --write-phase-configs RUN_ID   emit per-dataset phase-CLI configs under
                                   runs/<run_id>/configs + run_manifest.json
    --basic-artifacts              fallback: write beam_motion.json (zero motion) and
                                   raw_mean_diffraction.npy without the basic_analysis
                                   CLI (see basic_artifacts_note.json)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import shutil

import numpy as np
import yaml

import h5py

from bench_geometry import (digest, h5_payload_sha256, mib_header_template,
                            mib_payload_sha256, pattern_rotation, rasterize,
                            rotation_z2, sha256_bytes, write_mib)

PHASE_KEY = {'Fe-BCC': 'bcc', 'Fe-FCC': 'fcc'}
PHASE_ID = {'Fe-BCC': 0, 'Fe-FCC': 1}
MAX_SPOTS = 64


def load_reference(reference_dir, phase, zone):
    key = f'{phase}_{"".join(str(v) for v in zone)}_0'
    rows = list(csv.DictReader(open(reference_dir / f'{key}.csv', encoding='utf-8')))
    return np.array([[int(r['h']), int(r['k']), int(r['l'])] for r in rows], dtype=np.int16), \
        np.array([[float(r['qx_A']), float(r['qy_A'])] for r in rows], dtype=np.float64), \
        np.array([float(r['intensity_norm']) for r in rows]), \
        np.array([float(r['amplitude']) for r in rows])


def dataset_name(key, cfg):
    patterns = cfg['datasets'][key]['patterns']
    sy, sx = cfg['datasets'][key]['scan_shape']
    if len(patterns) == 1:
        p = patterns[0]
        zone = ''.join(str(v) for v in p['zone_axis'])
        return f"{p['phase']}_{zone}_{sy}x{sx}-{cfg['datasets'][key]['mib_suffix']}"
    return f'four_grain_{sy}x{sx}-{cfg['datasets'][key]['mib_suffix']}'


def build_one(key, cfg, reference_dir, root, header_template):
    dataset_cfg = cfg['datasets'][key]
    sy, sx = dataset_cfg['scan_shape']
    geometry = cfg['geometry']
    detector = tuple(geometry['detector_shape'])
    dq = geometry['reciprocal_sampling_A_per_px']
    cx, cy = geometry['center_px'][1], geometry['center_px'][0]
    patterns = []
    for index, p in enumerate(dataset_cfg['patterns']):
        hkl, qxy, norm, amplitude = load_reference(reference_dir, p['phase'], tuple(p['zone_axis']))
        if len(hkl) > MAX_SPOTS:
            raise ValueError(f'{key}: {len(hkl)} spots exceed capacity {MAX_SPOTS}')
        theta = np.deg2rad(p['in_plane_deg'])
        qxy_rot = (rotation_z2(theta) @ qxy.T).T
        pixels = np.column_stack([cy + qxy_rot[:, 1] / dq, cx + qxy_rot[:, 0] / dq])
        image = rasterize(detector, [(cy, cx, cfg['intensity']['direct_beam_amplitude'])] +
                          [(y, x, a) for (y, x), a in zip(pixels, amplitude)],
                          cfg['peak_kernel']['sigma_px'])
        if image.max() > cfg['intensity']['max_count_limit']:
            raise ValueError(f'{key} pattern {index}: max {image.max():.1f} exceeds '
                             f'{cfg['intensity']['max_count_limit']} (clipping)')
        patterns.append({'index': index, 'phase': p['phase'], 'zone_axis': tuple(p['zone_axis']),
                         'in_plane_deg': p['in_plane_deg'], 'region': p['region'],
                         'hkl': hkl, 'qxy': qxy_rot, 'intensity_norm': norm, 'amplitude': amplitude,
                         'pixels': pixels, 'image': np.rint(image).astype(np.uint16)})
    phase_id = np.zeros((sy, sx), dtype=np.int16)
    grain_id = np.zeros((sy, sx), dtype=np.int16)
    pattern_id = np.zeros((sy, sx), dtype=np.int16)
    cube = np.zeros((sy, sx) + detector, dtype=np.uint16)
    for p in patterns:
        region = p['region']
        if region == 'full':
            rows, cols = slice(None), slice(None)
        else:
            rows = slice(region['rows'][0], region['rows'][1])
            cols = slice(region['cols'][0], region['cols'][1])
        cube[rows, cols] = p['image']
        phase_id[rows, cols] = PHASE_ID[p['phase']]
        grain_id[rows, cols] = p['index']
        pattern_id[rows, cols] = p['index']
    name = dataset_name(key, cfg)
    datasets_dir = root / 'datasets' / key
    datasets_dir.mkdir(parents=True, exist_ok=True)
    mib_path = datasets_dir / f'{name}.mib'
    h5_path = datasets_dir / f'{name}.h5'
    write_mib(mib_path, cube, header_template)
    with h5py.File(h5_path, 'w') as handle:
        handle.create_dataset('/datacube/data', data=cube)
        handle.create_dataset('/metadata', data=json.dumps(metadata_block(key, name, cfg), ensure_ascii=False))
    truth = build_truth(key, cfg, patterns, phase_id, grain_id, pattern_id,
                        mib_payload_sha256(mib_path, sy * sx), h5_payload_sha256(h5_path))
    truth_dir = root / 'truth'
    truth_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(truth_dir / f'{key}_truth.npz', **truth)
    return {'key': key, 'name': name, 'mib': str(mib_path), 'h5': str(h5_path),
            'truth': str(truth_dir / f'{key}_truth.npz'),
            'scan_name': f'scan_01_{dataset_cfg['mib_suffix']}',
            'scan_shape': [sy, sx], 'frames': sy * sx,
            'mib_payload_sha256': mib_payload_sha256(mib_path, sy * sx),
            'h5_payload_sha256': h5_payload_sha256(h5_path)}


def metadata_block(key, name, cfg):
    geometry = cfg['geometry']
    dataset_cfg = cfg['datasets'][key]
    return {
        'dataset': key, 'name': name, 'generator': 'benchmarks/fe_fcc_bcc/build_dataset.py',
        'generated': datetime.now().isoformat(timespec='seconds'),
        'config_digest': digest(cfg),
        'axis_order': '(scan_y, scan_x, detector_y, detector_x)', 'dtype': 'uint16',
        'voltage_kv': geometry['voltage_kv'],
        'reciprocal_sampling_A_per_px': geometry['reciprocal_sampling_A_per_px'],
        'center_px': geometry['center_px'], 's_y': geometry['s_y'],
        'coordinate_contract': 'q_x = (col - c_x) * dq; q_y = s_y * (row - c_y) * dq',
        'scan_shape': list(dataset_cfg['scan_shape']),
        'detector_shape': list(geometry['detector_shape']),
        'peak_kernel': {'sigma_px': cfg['peak_kernel']['sigma_px']},
        'intensity_model': cfg['intensity'],
        'noise': 'none (clean quantization to uint16 only)',
        'cif_sha256': {key: sha256_bytes(Path(path).read_bytes()) for key, path in cfg['cifs'].items()},
        'patterns': [{'phase': p['phase'], 'zone_axis': p['zone_axis'], 'in_plane_deg': p['in_plane_deg'],
                      'region': p['region']} for p in dataset_cfg['patterns']],
    }


def build_truth(key, cfg, patterns, phase_id, grain_id, pattern_id, mib_hash, h5_hash):
    sy, sx = phase_id.shape
    m = MAX_SPOTS
    hkl = np.zeros((sy, sx, m, 3), dtype=np.int16)
    qxy = np.zeros((sy, sx, m, 2), dtype=np.float32)
    norm = np.zeros((sy, sx, m), dtype=np.float32)
    amplitude = np.zeros((sy, sx, m), dtype=np.float32)
    observable = np.zeros((sy, sx, m), dtype=bool)
    px = np.zeros((sy, sx, m, 2), dtype=np.float32)
    for p in patterns:
        region = p['region']
        if region == 'full':
            rows, cols = slice(None), slice(None)
        else:
            rows = slice(region['rows'][0], region['rows'][1])
            cols = slice(region['cols'][0], region['cols'][1])
        n = len(p['hkl'])
        hkl[rows, cols, :n] = p['hkl']
        qxy[rows, cols, :n] = p['qxy']
        norm[rows, cols, :n] = p['intensity_norm']
        amplitude[rows, cols, :n] = p['amplitude']
        observable[rows, cols, :n] = True
        px[rows, cols, :n] = p['pixels']
    return {
        'scan_shape': np.array([sy, sx], dtype=np.int64),
        'phase_id': phase_id, 'grain_id': grain_id, 'pattern_id': pattern_id,
        'pattern_zone_axis': np.array([p['zone_axis'] for p in patterns], dtype=np.float32),
        'pattern_in_plane_deg': np.array([p['in_plane_deg'] for p in patterns], dtype=np.float32),
        'pattern_rotation': np.array([pattern_rotation(p['zone_axis'], p['in_plane_deg'])
                                      for p in patterns], dtype=np.float32),
        'pattern_phase': np.array([PHASE_ID[p['phase']] for p in patterns], dtype=np.int16),
        'peak_hkl': hkl, 'peak_qxy_A': qxy, 'peak_intensity_norm': norm,
        'peak_amplitude': amplitude, 'peak_observable': observable, 'peak_px_yx': px,
        'pixel_transform': np.array([{
            'center_px': cfg['geometry']['center_px'],
            'reciprocal_sampling_A_per_px': cfg['geometry']['reciprocal_sampling_A_per_px'],
            's_y': cfg['geometry']['s_y'],
            'formula': 'q_x = (col - c_x) * dq; q_y = s_y * (row - c_y) * dq'}], dtype=object),
        'generator': np.array([{
            'config_digest': digest(cfg),
            'code_sha256': {'build_dataset.py': sha256_bytes(Path(__file__).read_bytes()),
                            'bench_geometry.py': sha256_bytes(Path(__file__).parent.joinpath('bench_geometry.py').read_bytes())},
            'pymatgen_version': version('pymatgen'),
            'h5_payload_sha256': h5_hash, 'mib_payload_sha256': mib_hash}], dtype=object),
    }


def check(root, cfg, reference_dir, header_template):
    manifest_path = root / 'datasets' / 'build_manifest.json'
    if not manifest_path.exists():
        print('FAIL: no build_manifest.json to check against')
        return 1
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    failures = []
    for entry in manifest['datasets']:
        key = entry['key']
        mib_hash = mib_payload_sha256(Path(entry['mib']), entry['frames'])
        h5_hash = h5_payload_sha256(Path(entry['h5']))
        status = []
        if mib_hash != entry['mib_payload_sha256']:
            status.append('mib payload hash mismatch')
        if h5_hash != entry['h5_payload_sha256']:
            status.append('h5 payload hash mismatch')
        # Re-derive the truth arrays and compare with the stored npz.
        dataset_cfg = cfg['datasets'][key]
        sy, sx = dataset_cfg['scan_shape']
        patterns = []
        for index, p in enumerate(dataset_cfg['patterns']):
            hkl, qxy, norm, amplitude = load_reference(reference_dir, p['phase'], tuple(p['zone_axis']))
            qxy_rot = (rotation_z2(np.deg2rad(p['in_plane_deg'])) @ qxy.T).T
            pixels = np.column_stack([cfg['geometry']['center_px'][0] + qxy_rot[:, 1] / cfg['geometry']['reciprocal_sampling_A_per_px'],
                                      cfg['geometry']['center_px'][1] + qxy_rot[:, 0] / cfg['geometry']['reciprocal_sampling_A_per_px']])
            patterns.append({'index': index, 'phase': p['phase'], 'zone_axis': tuple(p['zone_axis']),
                             'in_plane_deg': p['in_plane_deg'], 'region': p['region'],
                             'hkl': hkl, 'qxy': qxy_rot, 'intensity_norm': norm, 'amplitude': amplitude,
                             'pixels': pixels})
        phase_id = np.zeros((sy, sx), dtype=np.int16)
        grain_id = np.zeros((sy, sx), dtype=np.int16)
        pattern_id = np.zeros((sy, sx), dtype=np.int16)
        for p in patterns:
            region = p['region']
            if region == 'full':
                rows, cols = slice(None), slice(None)
            else:
                rows = slice(region['rows'][0], region['rows'][1])
                cols = slice(region['cols'][0], region['cols'][1])
            phase_id[rows, cols] = PHASE_ID[p['phase']]
            grain_id[rows, cols] = p['index']
            pattern_id[rows, cols] = p['index']
        derived = build_truth(key, cfg, patterns, phase_id, grain_id, pattern_id,
                              entry['mib_payload_sha256'], entry['h5_payload_sha256'])
        with np.load(Path(entry['truth']), allow_pickle=True) as stored:
            for array_name, value in derived.items():
                if array_name in ('pixel_transform', 'generator'):
                    continue
                if not np.array_equal(stored[array_name], value):
                    status.append(f'truth array mismatch: {array_name}')
        if status:
            failures.append((key, status))
            print(f'FAIL {key}: {status}')
        else:
            print(f'PASS {key}: mib {entry['mib_payload_sha256'][:16]}... h5 {entry['h5_payload_sha256'][:16]}... '
                  f'{entry['frames']} frames, truth re-derived')
    return 1 if failures else 0


def write_phase_configs(run_id, root, cfg, manifest):
    template = yaml.safe_load(Path('configs/phase_identification.yaml').read_text(encoding='utf-8'))
    run_dir = root / 'runs' / run_id
    configs_dir = run_dir / 'configs'
    configs_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {'run_id': run_id, 'created': datetime.now().isoformat(timespec='seconds'),
                    'datasets': {}}
    snapshot = {'run_id': run_id, 'note': 'Per-dataset phase-CLI configs; identical to '
                                          'configs/phase_identification.yaml except data paths, '
                                          'scan_shape and output_dir.', 'configs': {}}
    for entry in manifest['datasets']:
        key = entry['key']
        conf = copy.deepcopy(template)
        conf['data']['directory'] = f'benchmarks/Fe_FCC_BCC_v1/datasets/{key}'
        conf['data']['scan_shape'] = list(entry['scan_shape'])
        conf['data']['basic_analysis_dir'] = f'benchmarks/Fe_FCC_BCC_v1/basic/{key}'
        conf['output_dir'] = f'benchmarks/Fe_FCC_BCC_v1/runs/{run_id}/predictions'
        (configs_dir / f'{key}.yaml').write_text(
            yaml.safe_dump(conf, allow_unicode=True, sort_keys=False), encoding='utf-8')
        run_manifest['datasets'][key] = {
            'config': f'configs/{key}.yaml', 'mib': entry['mib'], 'h5': entry['h5'],
            'truth': entry['truth'], 'scan_name': entry['scan_name'],
            'scan_shape': entry['scan_shape']}
        snapshot['configs'][key] = conf
    (run_dir / 'run_manifest.json').write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    (run_dir / 'config_snapshot.yaml').write_text(
        yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print(f'Wrote {configs_dir} and {run_dir / 'run_manifest.json'}')


def write_basic_artifacts(root, cfg, manifest):
    for entry in manifest['datasets']:
        key = entry['key']
        scan_dir = root / 'basic' / key / entry['scan_name']
        scan_dir.mkdir(parents=True, exist_ok=True)
        import h5py
        with h5py.File(Path(entry['h5']), 'r') as handle:
            cube = handle['/datacube/data'][:]
        np.save(scan_dir / 'raw_mean_diffraction.npy', cube.mean(axis=(0, 1), dtype=np.float64))
        motion = {
            'sample_count': int(np.prod(entry['scan_shape'])),
            'inlier_fraction': 1.0,
            'median_residual_px': 0.0,
            'inlier_median_residual_px': 0.0,
            'predicted_span_yx_px': [0.0, 0.0],
            'affine_coefficients_yx': [[0.0, 0.0], [0.0, 0.0]],
            'affine_intercept_yx': list(cfg['geometry']['center_px']),
            'scan_correlated_motion_warning': False,
            'method': 'synthetic zero-motion dataset; intercept = config center; no fitting performed',
            'interpretation': 'Synthetic benchmark pattern; the direct beam is exactly at the '
                              'configured center of every scan position.'}
        (scan_dir / 'beam_motion.json').write_text(json.dumps(motion, indent=2), encoding='utf-8')
        (scan_dir / 'basic_artifacts_note.json').write_text(json.dumps({
            'reason': 'basic_analysis CLI may fail on single-pattern synthetic scans: the '
                      'PCA/KMeans class gallery indexes empty clusters (basic_analysis.py '
                      'make_figures). These two files are the only basic-analysis artifacts '
                      'the phase CLI consumes (beam_motion.json, raw_mean_diffraction.npy); '
                      'both are exact for zero-motion synthetic data.',
            'generated': datetime.now().isoformat(timespec='seconds')}, indent=2), encoding='utf-8')
        print(f'Wrote {scan_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('benchmarks/fe_fcc_bcc/config.yaml'))
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--only', help='Comma-separated dataset keys, e.g. T01,T03')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--write-phase-configs', metavar='RUN_ID')
    parser.add_argument('--basic-artifacts', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    root = args.output_root or Path(cfg['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    reference_dir = root / 'reference'
    keys = args.only.split(',') if args.only else list(cfg['datasets'])
    unknown = set(keys) - set(cfg['datasets'])
    if unknown:
        parser.error(f'Unknown datasets: {sorted(unknown)}')
    header_template = mib_header_template(Path(cfg['mib_header_source']))
    if args.check:
        return check(root, cfg, reference_dir, header_template)
    manifest_path = root / 'datasets' / 'build_manifest.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    entries = list(previous.get('datasets', []))
    known = {entry['key']: entry for entry in entries}
    for key in keys:
        entry = build_one(key, cfg, reference_dir, root, header_template)
        if key in known:
            entries = [entry if e['key'] == key else e for e in entries]
        else:
            entries.append(entry)
        print(f'{key}: {entry['frames']} frames, {Path(entry['mib']).stat().st_size / 2**20:.1f} MiB mib, '
              f'{Path(entry['h5']).stat().st_size / 2**20:.1f} MiB h5')
    inputs_dir = root / 'inputs'
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for source in ('bcc', 'fcc'):
        target = inputs_dir / Path(cfg['cifs'][source]).name
        if not target.exists() or target.read_bytes() != Path(cfg['cifs'][source]).read_bytes():
            shutil.copyfile(cfg['cifs'][source], target)
    manifest = {'generated': datetime.now().isoformat(timespec='seconds'),
                'config_digest': digest(cfg), 'datasets': entries}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(f'Wrote {manifest_path}')
    if args.write_phase_configs:
        write_phase_configs(args.write_phase_configs, root, cfg, manifest)
    if args.basic_artifacts:
        write_basic_artifacts(root, cfg, manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
