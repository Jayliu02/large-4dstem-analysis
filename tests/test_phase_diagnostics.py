"""Regression checks for the rejection audit, independent of experimental files."""
from pathlib import Path
import numpy as np
import pytest
import yaml

pytest.importorskip('numba')
from fourdstem_pipeline.phase_diagnostics import gate_statistics, winning_gates
from fourdstem_pipeline.phase_matching import decisions


def test_gate_audit_separates_voltage_evidence_from_phase_conflict():
    cfg = yaml.safe_load(Path('configs/phase_identification.yaml').read_text(encoding='utf-8'))['matching']
    libs = [{'phase_id': p, 'voltage_kv': v} for p in [0,1] for v in [80,200]]
    fits = np.zeros((4,4,7),np.float32)
    fits[:,:,0] = .1
    fits[:,:,1] = 8
    fits[:,:,2] = .1
    fits[:,:,6] = 1
    fits[:,:2,0] = .8
    fits[1,1,1] = 4  # Same phase wins both voltages, but evidence fails at 200 kV.
    fits[2,3,0] = .95  # Different phases win at the two voltages.
    counts = np.full(4,8)
    valid = np.array([True,True,True,False])
    final = np.array([0,-1,-2,-1])
    report, provisional = gate_statistics(fits,libs,counts,valid,cfg,.35,final)
    assert provisional.tolist() == [True,False,False,False]
    decision = decisions(fits,libs,cfg,observed_counts=counts)
    np.testing.assert_array_equal(provisional,valid & (decision['phase_id']>=0))
    assert report['cascade'] == {'all_positions':4,'valid_center':3,'eligible_observations':3,
        'evidence_any_voltage':3,'evidence_all_voltages':2,'phase_margin_all_voltages':2,
        'same_phase_all_voltages':1,'all_perturbations':1}
    assert report['single_voltage_before_perturbation'] == {'80':3,'200':2}
    # Removing only the count gate does not also remove the 70% coverage gate.
    assert report['drop_one_gate_before_perturbation_fixed_fits']['six_matches'] == 1


def test_gate_audit_includes_tied_phase_rejection():
    cfg = yaml.safe_load(Path('configs/phase_identification.yaml').read_text(encoding='utf-8'))['matching']
    libs = [{'phase_id':p,'voltage_kv':v} for p in [0,1] for v in [80,200]]
    fits = np.zeros((1,4,7),np.float32)
    fits[:,:,0] = .8; fits[:,:,1] = 8; fits[:,:,2] = .2; fits[:,:,6] = 1
    _,_,gates = winning_gates(fits,libs,np.array([8]),cfg,.35)
    assert not gates['phase_margin'].any()
    report,provisional = gate_statistics(fits,libs,np.array([8]),np.array([True]),cfg,.35,np.array([-2]))
    assert not provisional.any()
    assert report['drop_one_gate_before_perturbation_fixed_fits']['phase_margin'] == 1


def test_retired_cif_example_configuration_has_no_dependencies():
    config = yaml.safe_load(Path('configs/pipeline.yaml').read_text(encoding='utf-8'))
    assert config['pipeline']['stages'] == ['stage1','stage2a']
    assert config['phase_screening']['candidate_phases'] == []
    assert config['stage2b']['candidate_cifs'] == []
    assert not config['stage2c']['enabled']
    assert config['stage2c']['candidates']['phases'] == []
    assert {p.name for p in Path('references/cifs').glob('*.cif')} == {'Fe-BCC.cif','Fe-FCC.cif'}
