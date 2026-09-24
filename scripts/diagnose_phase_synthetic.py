"""Replay the complete robustness gate on the existing held-out Fe controls."""
from pathlib import Path
import argparse
import numpy as np
from numba import set_num_threads
from threadpoolctl import threadpool_limits

from validate_phase_identification import controls
from fourdstem_pipeline.phase_peaks import atomic_json, read_json
from fourdstem_pipeline.phase_matching import PhaseMatcher, decisions
from fourdstem_pipeline.phase_diagnostics import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/phase_identification_fe'))
    args = parser.parse_args()
    root = args.root
    checkpoint = next(root.glob('scan_*/matching_checkpoint.json'))
    cfg = read_json(checkpoint)['provenance']['config']
    libraries = []
    for candidate in cfg['candidates']:
        for voltage in cfg['templates']['voltages_kv']:
            with np.load(root/'templates'/f"{candidate['name']}_{voltage}kV.npz") as data:
                libraries.append(dict(data))
    set_num_threads(cfg['runtime']['threads'])
    reports = []
    visibility = np.ones((256,256),bool)
    with threadpool_limits(cfg['runtime']['threads']):
        for scale in [.009,.015,.025]:
            for degraded in [False,True]:
                points,counts,centers,truth = controls(libraries,cfg,scale,degraded=degraded)
                matcher = PhaseMatcher(libraries,scale,visibility,cfg['matching'],
                    inner=cfg['peaks']['exclusion_radius'],outer=cfg['peaks']['max_radius'])
                decision = decisions(matcher.match(points,counts,centers),libraries,
                    cfg['matching'],observed_counts=counts)
                indices = np.flatnonzero(decision['phase_id'] >= 0)
                np.testing.assert_array_equal(decision['phase_id'][indices],truth[indices])
                destination = root/'diagnostics'/'synthetic_probes'/f'{scale}_{degraded}'
                destination.mkdir(parents=True,exist_ok=True)
                probes = replay(points[indices],counts[indices],centers[indices],indices,
                    decision['phase_id'][indices],libraries,scale,visibility,cfg,cfg['matching']['min_score'],destination)
                report = {'scale':scale,'degraded':degraded,'patterns':len(points),'base_accepted':len(indices),
                          'configured_retained':probes['configured']['retain_all'],
                          'small_probe_retained':probes['small_probe']['retain_all']}
                reports.append(report)
                print(report,flush=True)
    atomic_json(root/'diagnostics'/'synthetic_perturbation.json',reports)


if __name__ == '__main__':
    main()
