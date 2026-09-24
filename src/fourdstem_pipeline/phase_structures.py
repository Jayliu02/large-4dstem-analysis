"""Audited CIF structures and py4DSTEM kinematic sparse template libraries."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path
import warnings
import numpy as np

from .phase_peaks import atomic_json, read_json, digest


def read_structure(candidate):
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.io.cif import CifParser
    path = Path(candidate['cif'])
    with warnings.catch_warnings(record=True) as caught:
        structure = Structure.from_file(path)
    structure.remove_oxidation_states()
    if not structure.is_ordered or set(str(e) for e in structure.elements) != {'Ti'}:
        raise ValueError(f'Expected ordered elemental Ti structure: {path}')
    groups = [SpacegroupAnalyzer(structure, symprec=t).get_space_group_number() for t in [1e-3, 1e-2]]
    if len(set(groups)) != 1 or groups[0] != candidate['expected_space_group']:
        raise ValueError(f'Structure/symmetry mismatch for {path}: inferred {groups}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cif_dict = next(iter(CifParser(str(path)).as_dict().values()))
    declared = cif_dict.get('_symmetry_Int_Tables_number', cif_dict.get('_space_group_IT_number'))
    audit = {'id': candidate['id'], 'name': candidate['name'], 'label': candidate['label'],
             'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'declared_space_group': declared, 'inferred_space_group': groups[0],
             'symmetry_tolerances_angstrom': [1e-3, 1e-2], 'site_count': len(structure),
             'lattice': structure.lattice.matrix, 'abc': structure.lattice.abc,
             'angles': structure.lattice.angles, 'fractional_positions': structure.frac_coords,
             'occupancies': [sum(site.species.values()) for site in structure],
             'parsing_warnings': [str(w.message) for w in caught],
             'interpretation': 'candidate structure only; phase identity comes from expanded structure, not input filename'}
    return structure, audit


def make_crystal(structure, k_max):
    from py4DSTEM.process.diffraction import Crystal
    crystal = Crystal.from_pymatgen_structure(structure, conventional_standard_structure=True)
    crystal.calculate_structure_factors(k_max=k_max)
    return crystal


def prepare_libraries(candidates, cfg, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    libraries = []
    audits = []
    for candidate in candidates:
        structure, audit = read_structure(candidate)
        audits.append(audit)
        signature = digest({'cif': audit['sha256'], 'cfg': cfg, 'py4DSTEM': version('py4DSTEM'),
                            'implementation': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        manifest = output/f"{candidate['name']}_manifest.json"
        if manifest.exists():
            if read_json(manifest)['signature'] != signature:
                raise ValueError('Template cache changed: choose a new output directory.')
            libraries.extend([dict(np.load(output/f"{candidate['name']}_{voltage}kV.npz")) for voltage in cfg['voltages_kv']])
            continue
        crystal = make_crystal(structure, cfg['k_max'])
        crystal.orientation_plan(zone_axis_range='auto', angle_step_zone_axis=cfg['zone_step_deg'],
                                 angle_step_in_plane=2, calculate_correlation_array=False, progress_bar=False)
        matrices = crystal.orientation_rotation_matrices
        print(f"  {candidate['name']}: SG {audit['inferred_space_group']}, {len(matrices)} zone directions", flush=True)
        for voltage in cfg['voltages_kv']:
            crystal.setup_diffraction(voltage*1000)
            coords = np.zeros((len(matrices), cfg['max_peaks'], 2), np.float32)
            intensities = np.zeros(coords.shape[:2], np.float32)
            hkls = np.zeros(coords.shape[:2]+(3,), np.int16)
            counts = np.zeros(len(matrices), np.int16)
            for i, matrix in enumerate(matrices):
                pattern = crystal.generate_diffraction_pattern(orientation_matrix=matrix,
                    sigma_excitation_error=cfg['excitation_error'], tol_intensity=1e-7, k_max=cfg['k_max']).data
                q = np.column_stack([pattern['qx'], pattern['qy']])
                keep = np.linalg.norm(q, axis=1) > 0.05
                pattern, q = pattern[keep], q[keep]
                if len(q) == 0:
                    continue
                keep = pattern['intensity'] >= pattern['intensity'].max()*cfg['relative_intensity']
                pattern, q = pattern[keep], q[keep]
                order = np.argsort(pattern['intensity'])[::-1][:cfg['max_peaks']]
                n = len(order)
                coords[i,:n] = q[order]
                intensities[i,:n] = pattern['intensity'][order]
                hkls[i,:n] = np.column_stack([pattern[k][order] for k in ['h','k','l']])
                counts[i] = n
            library = {'qxy': coords, 'intensity': intensities, 'hkl': hkls, 'count': counts,
                       'matrices': matrices.astype(np.float32), 'phase_id': np.array(candidate['id']),
                       'voltage_kv': np.array(voltage), 'phase_name': np.array(candidate['name'])}
            np.savez_compressed(output/f"{candidate['name']}_{voltage}kV.npz", **library)
            libraries.append(library)
            print(f'    {voltage} kV templates ready', flush=True)
        atomic_json(manifest, {'signature': signature, 'audit': audit, 'config': cfg})
    atomic_json(output/'structure_audit.json', audits)
    return libraries
