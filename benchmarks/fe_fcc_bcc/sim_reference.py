"""Independent kinematic reference patterns for the Fe BCC/FCC benchmark.

This replaces the plan's ReciPro SpotInfo() export (ReciPro is not installed on
this machine; substitution recorded per the plan's pause rules).  Patterns come
from pymatgen TEMCalculator (Mott-Bethe electron scattering factors, relativistic
wavelength, ZOLZ selected by beam_direction) - an engine independent of the
py4DSTEM Crystal kinematics the pipeline matches with.  Outputs per orientation:
    <phase>_<zone>_0.csv   h,k,l,qx_A,qy_A,d_hkl_A,intensity_norm,amplitude,observable
    <phase>_<zone>_0.png   log1p rasterized preview (direct beam included)
plus sim_reference_manifest.json with versions, CIF hashes, the wavelength and
the honest non-independence statement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from bench_geometry import (digest, reference_spots, rasterize, rotation_z,
                            sha256_bytes, zone_to_rotation)

PHASE_KEY = {'Fe-BCC': 'bcc', 'Fe-FCC': 'fcc'}

NON_INDEPENDENCE_NOTE = (
    'The reference generator and the analysis side (py4DSTEM Crystal templates) share the same '
    'CIF files, the same pymatgen CIF parser and the same kinematic |F|^2 physics. They differ '
    'in the structure-factor implementation (pymatgen Mott-Bethe vs py4DSTEM), the orientation '
    'machinery and all rasterization/quantization (ours). A defect shared by both kinematic '
    'engines or by pymatgen CIF parsing could escape this benchmark; inspect_cifs.py analytic '
    'extinction rules are the backstop. The geometric spot positions are exact lattice '
    'quantities and are verified against the pipeline template library where available.'
)

SUBSTITUTION_NOTE = (
    'ReciPro is not installed on this machine, so the plan Gate 0 export (SpotInfo CSV/PNG) '
    'was replaced by an in-repo independent kinematic simulator per the plan pause rules: '
    '"先采用其他具有数值峰列表的独立模拟来源，记录替代原因". ReciPro export can be added '
    'later and compared against this reference.'
)


def relativist_wavelength_A(voltage_kv):
    """Relativistic electron wavelength in Angstrom (h*c = 12.39841984 keV*A)."""
    eU = float(voltage_kv)
    hc = 12.39841984
    m0c2 = 510.998950
    return hc / np.sqrt(eU * (eU + 2.0 * m0c2))


def orientation_key(phase, zone):
    return f'{phase}_{"".join(str(v) for v in zone)}_0'


def cross_check_templates(template_dir, orientations, cfg):
    """Verify reference spot geometry against existing pipeline template libraries."""
    records = {}
    for phase, zone in orientations:
        library_path = Path(template_dir) / f'{phase}_300kV.npz'
        if not library_path.exists():
            records[orientation_key(phase, zone)] = {'status': 'skipped', 'reason': 'no template library'}
            continue
        library = np.load(library_path)
        target = np.asarray(zone, dtype=float) / np.linalg.norm(zone)
        row = min(range(len(library['matrices'])),
                  key=lambda i: np.linalg.norm(library['matrices'][i][:, 2] - target))
        count = int(library['count'][row])
        template = {tuple(int(v) for v in hkl): qxy
                    for hkl, qxy in zip(library['hkl'][row][:count], library['qxy'][row][:count])}
        cif = cfg['cifs'][PHASE_KEY[phase]]
        spots = reference_spots(cif, zone, cfg['detector_window']['q_min_A'],
                                cfg['detector_window']['q_max_A'], cfg['simulator']['extinction_floor'])
        strongest = spots[0]['intensity_norm']
        residuals, missing = [], []
        for spot in spots:
            if spot['intensity_norm'] < 0.10 * strongest:
                continue
            if spot['hkl'] in template:
                residuals.append(float(np.linalg.norm(spot['qxy'] - template[spot['hkl']])))
            elif tuple(-v for v in spot['hkl']) in template:
                residuals.append(float(np.linalg.norm(spot['qxy'] - template[tuple(-v for v in spot['hkl'])])))
            else:
                missing.append(spot['hkl'])
        records[orientation_key(phase, zone)] = {
            'status': 'ok' if not missing and residuals else 'failed',
            'template_row': int(row), 'strong_spots': len(residuals) + len(missing),
            'max_residual_A': float(max(residuals)) if residuals else None,
            'median_residual_A': float(np.median(residuals)) if residuals else None,
            'missing_hkl': [list(hkl) for hkl in missing],
        }
        print(f'  cross-check {phase} {zone}: row {row}, '
              f'max residual {max(residuals) * 1000 if residuals else float("nan"):.2f} mAng, '
              f'missing {missing if missing else "none"}')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('benchmarks/fe_fcc_bcc/config.yaml'))
    parser.add_argument('--output', type=Path, default=Path('benchmarks/Fe_FCC_BCC_v1/reference'))
    parser.add_argument('--template-dir', type=Path, default=None,
                        help='Optional pipeline template library dir for the geometry cross-check.')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    geometry = cfg['geometry']
    window = cfg['detector_window']
    intensity = cfg['intensity']
    orientations = sorted({(p['phase'], tuple(p['zone_axis'])) for ds in cfg['datasets'].values()
                           for p in ds['patterns']})
    dq = geometry['reciprocal_sampling_A_per_px']
    cx, cy = geometry['center_px'][1], geometry['center_px'][0]
    wavelength = relativist_wavelength_A(geometry['voltage_kv'])
    records = {}
    for phase, zone in orientations:
        cif = cfg['cifs'][PHASE_KEY[phase]]
        spots = reference_spots(cif, zone, window['q_min_A'], window['q_max_A'],
                                cfg['simulator']['extinction_floor'])
        spots = [s for s in spots if s['intensity_norm'] >= intensity['spot_min_relative']]
        amplitudes = [intensity['strong_spot_amplitude'] * s['intensity_norm'] for s in spots]
        # Dataset invariants from the plan: enough spots to pass the matching gates
        # (matching.min_peaks = 6; Fe-FCC [111] sits exactly at 6 at this cutoff), the
        # weakest retained spot >= spot_min_amplitude, and >= 10x the detection floor
        # (relative_threshold 0.008 x direct-beam signal, background-subtracted).
        assert len(spots) >= 6, f'{phase} {zone}: only {len(spots)} spots at >= {intensity['spot_min_relative']}'
        assert min(amplitudes) >= intensity['spot_min_amplitude'] - 1e-9
        assert min(amplitudes) >= 10.0 * intensity['direct_beam_amplitude'] * 0.008
        pixels = np.array([(cy + s['qxy'][1] / dq, cx + s['qxy'][0] / dq) for s in spots])
        # Window/edge/spacing assertions: the config was chosen so these hold.
        in_plane_q = np.linalg.norm(np.array([s['qxy'] for s in spots]), axis=1)
        assert in_plane_q.min() >= window['q_min_A'] - 1e-12
        assert in_plane_q.max() <= window['q_max_A'] + 1e-12
        margin = window['edge_margin_px']
        assert pixels[:, 0].min() >= margin and pixels[:, 0].max() <= geometry['detector_shape'][0] - 1 - margin
        assert pixels[:, 1].min() >= margin and pixels[:, 1].max() <= geometry['detector_shape'][1] - 1 - margin
        pairwise = np.linalg.norm(pixels[:, None, :] - pixels[None, :, :], axis=-1)
        pairwise[np.diag_indices(len(pixels))] = np.inf
        assert pairwise.min() >= window['min_spot_spacing_px'], f'{phase} {zone}: spacing {pairwise.min():.2f} px'
        key = orientation_key(phase, zone)
        stem = args.output / key
        with open(stem.with_suffix('.csv'), 'w', encoding='utf-8', newline='') as stream:
            stream.write('h,k,l,qx_A,qy_A,d_hkl_A,intensity_norm,amplitude,observable\n')
            for spot, amplitude in zip(spots, amplitudes):
                stream.write(f'{spot["hkl"][0]},{spot["hkl"][1]},{spot["hkl"][2]},'
                             f'{spot["qxy"][0]:.9f},{spot["qxy"][1]:.9f},{spot["d_hkl"]:.9f},'
                             f'{spot["intensity_norm"]:.9f},{amplitude:.6f},true\n')
        image = rasterize(tuple(geometry['detector_shape']),
                          [(cy, cx, intensity['direct_beam_amplitude'])] +
                          [(y, x, a) for (y, x), a in zip(pixels, amplitudes)],
                          cfg['peak_kernel']['sigma_px'])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.log1p(image), cmap='gray')
        ax.set_title(f'{phase} zone {zone}, in-plane 0 deg (log1p preview)')
        fig.savefig(stem.with_suffix('.png'), dpi=150)
        plt.close(fig)
        records[key] = {'phase': phase, 'zone_axis': list(zone), 'cif': cif,
                        'spots': len(spots), 'strong_spots': int(sum(a >= 0.10 * intensity['strong_spot_amplitude']
                                                                     for a in amplitudes)),
                        'amplitude_range': [float(min(amplitudes)), float(max(amplitudes))],
                        'wavelength_A': wavelength}
        print(f'{phase} {zone}: {len(spots)} spots, amplitudes '
              f'{min(amplitudes):.1f}..{max(amplitudes):.1f} counts')
    cross_check = (cross_check_templates(args.template_dir, orientations, cfg)
                   if args.template_dir else {'status': 'not requested'})
    manifest = {
        'generated': datetime.now().isoformat(timespec='seconds'),
        'config_digest': digest(cfg),
        'config_path': str(args.config),
        'pymatgen_version': version('pymatgen'),
        'wavelength_A': wavelength,
        'wavelength_formula': 'relativistic: lambda = h*c / sqrt(eU * (eU + 2*m0*c^2)), h*c = 12.39841984 keV*A',
        'cif_sha256': {key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key, path in cfg['cifs'].items()},
        'code_sha256': {'sim_reference.py': sha256_bytes(Path(__file__).read_bytes()),
                        'bench_geometry.py': sha256_bytes(Path(__file__).parent.joinpath('bench_geometry.py').read_bytes())},
        'orientations': records,
        'template_cross_check': cross_check,
        'substitution_note': SUBSTITUTION_NOTE,
        'non_independence_note': NON_INDEPENDENCE_NOTE,
    }
    target = args.output / 'sim_reference_manifest.json'
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(f'Wrote {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
