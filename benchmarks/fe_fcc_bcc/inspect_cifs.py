"""Audit the benchmark CIFs and independently verify kinematic extinctions.

Reuses the pipeline's own CIF audit (fourdstem_pipeline.phase_structures.read_structure,
the exact check the phase CLI runs) and adds two independent verifications:
  1. analytic structure-factor extinctions from the fractional coordinates
     (BCC I-centering: h+k+l even; FCC F-centering: h,k,l same parity);
  2. a numeric cross-check of |F|^2 for the [001] ZOLZ reflections via
     pymatgen TEMCalculator (a different engine from the pipeline's py4DSTEM).
Exit code is non-zero on any mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fourdstem_pipeline.phase_structures import read_structure

CANDIDATES = [
    {'id': 0, 'name': 'Fe-BCC', 'label': 'Fe-BCC', 'expected_space_group': 229},
    {'id': 1, 'name': 'Fe-FCC', 'label': 'Fe-FCC', 'expected_space_group': 225},
]

# Reflection set for the extinction check, per lattice centering.
CASES = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0)]
ALLOWED = {'bcc': {(1, 1, 0), (2, 0, 0)}, 'fcc': {(1, 1, 1), (2, 0, 0)}}
RELATIVE_TOLERANCE = 1e-6


def analytic_structure_factor(structure, hkl):
    """|F(hkl)|^2 for unit atomic form factors, from fractional coordinates."""
    phase = 2j * np.pi * np.asarray(hkl, dtype=float)
    amplitude = np.exp(phase @ structure.frac_coords.T).sum()
    return float(np.abs(amplitude) ** 2)


def numeric_structure_factors(cif_path):
    """TEMCalculator |F|^2 for the [001] ZOLZ (covers (100),(110),(200) of both phases)."""
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.tem import TEMCalculator
    structure = Structure.from_file(cif_path)
    pattern = TEMCalculator(voltage=300, beam_direction=(0, 0, 1), camera_length=160,
                            debye_waller_factors={}).get_pattern(structure)
    values = {}
    for hkl, raw in zip(pattern['(hkl)'], pattern['Intensity (norm)']):
        hkl = tuple(int(v) for v in hkl)
        if hkl in CASES and hkl not in values:
            values[hkl] = float(raw)
    return values


def check_extinctions(structure, cif_path, kind):
    analytic = {hkl: analytic_structure_factor(structure, hkl) for hkl in CASES}
    numeric = numeric_structure_factors(cif_path)
    reference = max(analytic.values())
    problems = []
    for hkl in CASES:
        must_be_allowed = hkl in ALLOWED[kind]
        analytic_ok = (analytic[hkl] > RELATIVE_TOLERANCE * reference) if must_be_allowed \
            else (analytic[hkl] < RELATIVE_TOLERANCE * reference)
        if not analytic_ok:
            problems.append(f'{hkl}: analytic |F|^2 = {analytic[hkl]:.3e}, expected '
                            f'{"allowed" if must_be_allowed else "forbidden"}')
        # The numeric pattern is the [001] ZOLZ: reflections with a non-zero beam
        # component (l != 0) are legitimately absent; only cross-check l == 0 cases.
        if hkl[2] != 0:
            continue
        if hkl in numeric:
            numeric_ok = (numeric[hkl] > RELATIVE_TOLERANCE * max(numeric.values())) if must_be_allowed \
                else (numeric[hkl] < RELATIVE_TOLERANCE * max(numeric.values()))
            if not numeric_ok:
                problems.append(f'{hkl}: numeric |F|^2 = {numeric[hkl]:.3e}, expected '
                                f'{"allowed" if must_be_allowed else "forbidden"}')
        elif must_be_allowed:
            problems.append(f'{hkl}: missing from numeric [001] ZOLZ pattern')
    return analytic, numeric, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('benchmarks/fe_fcc_bcc/config.yaml'))
    parser.add_argument('--output', type=Path, default=Path('benchmarks/Fe_FCC_BCC_v1/reference'))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    kinds = {'Fe-BCC': ('bcc', cfg['cifs']['bcc']), 'Fe-FCC': ('fcc', cfg['cifs']['fcc'])}
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    failed = []
    for name, (kind, cif) in kinds.items():
        candidate = next(c for c in CANDIDATES if c['name'] == name)
        candidate = dict(candidate, cif=cif)
        structure, audit = read_structure(candidate)
        analytic, numeric, problems = check_extinctions(structure, cif, kind)
        entry = {
            'pipeline_audit': audit,
            'sha256': hashlib.sha256(Path(cif).read_bytes()).hexdigest(),
            'analytic_structure_factors': {str(hkl): value for hkl, value in analytic.items()},
            'numeric_zolz_001_structure_factors': {str(hkl): value for hkl, value in numeric.items()},
            'extinction_rules': {'bcc': 'I-centering: h+k+l even; (100),(111) forbidden; (110),(200) allowed',
                                 'fcc': 'F-centering: h,k,l same parity; (100),(110) forbidden; (111),(200) allowed'}[kind],
            'problems': problems,
        }
        report[name] = entry
        if problems:
            failed.append((name, problems))
        print(f'{name}: SG {audit["inferred_space_group"]}, a = {audit["abc"][0]:.8f} A, '
              f'{audit["site_count"]} sites; extinction check {"PASS" if not problems else "FAIL"}')
        for hkl, value in analytic.items():
            allowed = 'allowed' if hkl in ALLOWED[kind] else 'forbidden'
            print(f'  {hkl}: analytic {value:.3e} ({allowed})')
    target = args.output / 'cif_audit.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(f'Wrote {target}')
    if failed:
        for name, problems in failed:
            print(f'FAILED {name}: {problems}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
