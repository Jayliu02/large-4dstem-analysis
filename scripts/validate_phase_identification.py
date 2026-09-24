"""Held-out Fe BCC/FCC diffraction controls (Python 3.12)."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import yaml
from numba import set_num_threads
from threadpoolctl import threadpool_limits

from fourdstem_pipeline.phase_structures import prepare_libraries, read_structure, make_crystal
from fourdstem_pipeline.phase_matching import PhaseMatcher, decisions
from fourdstem_pipeline.phase_peaks import atomic_json
from fourdstem_pipeline.phase_calibration import calibrate


def controls(libraries, cfg, scale, *, degraded=False):
    rng=np.random.default_rng(45)
    observed,counts,centers,truth=[],[],[],[]
    for candidate in cfg['candidates']:
        crystal=make_crystal(read_structure(candidate)[0],cfg['templates']['k_max'])
        crystal.setup_diffraction(200000)
        lib=next(lib for lib in libraries if int(lib['phase_id'])==candidate['id'] and int(lib['voltage_kv'])==200)
        for t in np.linspace(0,len(lib['count'])-1,8).round().astype(int):
            # Generate at a held-out zone direction, not a stored template.
            a=np.deg2rad(.4)
            perturb=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
            matrix=lib['matrices'][t]@perturb
            pattern=crystal.generate_diffraction_pattern(orientation_matrix=matrix,
                sigma_excitation_error=cfg['templates']['excitation_error'],tol_intensity=1e-7,
                k_max=cfg['templates']['k_max']).data
            points=np.column_stack([pattern['qx'],pattern['qy']])/scale
            keep=np.linalg.norm(points,axis=1)>cfg['peaks']['exclusion_radius']
            pattern,points=pattern[keep],points[keep]
            if not len(points):
                continue
            keep=pattern['intensity']>=pattern['intensity'].max()*cfg['templates']['relative_intensity']
            pattern,points=pattern[keep],points[keep]
            order=np.argsort(pattern['intensity'])[::-1][:cfg['templates']['max_peaks']]
            points=points[order]
            angle=rng.uniform(-np.pi,np.pi)
            rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            points=points@rotation.T
            center=np.array([128.,128.])+rng.uniform(-35,35,2)
            keep=(np.linalg.norm(points,axis=1)<=cfg['peaks']['max_radius'])&np.all(points+center>=4,axis=1)&np.all(points+center<252,axis=1)
            points=points[keep][:cfg['peaks']['max_peaks']]
            if len(points)<8:
                continue
            if degraded:
                points=points[rng.random(len(points))>.15]
                if len(points)<6:
                    continue
                points=points[:cfg['peaks']['max_peaks']-2]
                outliers=rng.uniform(-60,60,(2,2))
                points=np.vstack([points,outliers])
            points+=rng.normal(0,.35 if degraded else .15,points.shape)
            padded=np.zeros((cfg['peaks']['max_peaks'],2),np.float32)
            padded[:len(points)]=points
            observed.append(padded);counts.append(len(points));centers.append(center);truth.append(candidate['id'])
    return np.asarray(observed),np.asarray(counts),np.asarray(centers,np.float32),np.asarray(truth)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/phase_identification.yaml'))
    parser.add_argument('--calibration',action='store_true',help='Also recover a known scale from a synthetic spatial grid.')
    args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    root=Path(cfg['output_dir'])
    set_num_threads(cfg['runtime']['threads'])
    reports=[]
    with threadpool_limits(cfg['runtime']['threads']):
        libraries=prepare_libraries(cfg['candidates'],cfg['templates'],root/'templates')
        for scale in [.009,.015,.025]:
            for degraded in [False,True]:
                points,counts,centers,truth=controls(libraries,cfg,scale,degraded=degraded)
                matcher=PhaseMatcher(libraries,scale,np.ones((256,256),bool),cfg['matching'])
                result=matcher.match(points,counts,centers)
                decision=decisions(result,libraries,cfg['matching'],observed_counts=counts)
                accepted=decision['phase_id']>=0
                report={'scale':scale,'degraded':degraded,'patterns':len(points),'truth':truth,
                        'labels':decision['phase_id'],'accepted':int(accepted.sum()),
                        'wrong_accepted':int(np.sum(accepted&(decision['phase_id']!=truth))),
                        'mean_winning_score':float(decision['min_score'].mean())}
                # Independent random-angle negative controls preserve radii.
                rng=np.random.default_rng(213)
                radii=np.linalg.norm(points,axis=-1)
                angles=rng.uniform(-np.pi,np.pi,radii.shape)
                scrambled=np.stack([radii*np.cos(angles),radii*np.sin(angles)],axis=-1)
                null=decisions(matcher.match(scrambled,counts,centers),libraries,cfg['matching'],observed_counts=counts)
                report['random_angle_accepted']=int(np.sum(null['phase_id']>=0))
                # Mixed patterns are diagnostics, not a physical dynamical simulation.
                mixed=points.copy();mixed_counts=counts.copy()
                for i in range(len(points)):
                    other=int(np.flatnonzero(truth!=truth[i])[0])
                    a=points[i,:min(int(counts[i]),24)]
                    b=points[other,:min(int(counts[other]),24)]
                    combined=np.vstack([a,b])
                    mixed[i]=0;mixed[i,:len(combined)]=combined;mixed_counts[i]=len(combined)
                mixture=decisions(matcher.match(mixed,mixed_counts,centers),libraries,cfg['matching'],observed_counts=mixed_counts)
                report['mixed_rejected']=int(np.sum(mixture['phase_id']<0))
                reports.append(report)
                print({k:v for k,v in report.items() if k not in ['truth','labels']},flush=True)
        if args.calibration:
            points,counts,centers,truth=controls(libraries,cfg,.015)
            rng=np.random.default_rng(918)
            indices=rng.integers(len(points),size=256)
            jitter=rng.normal(0,.15,(256,points.shape[1],2))
            centers=centers[indices].reshape(16,16,2)
            peaks={'peak_yx':(points[indices]+jitter).reshape(16,16,-1,2)+centers[:,:,None,:],
                   'count':counts[indices].reshape(16,16),'center_yx':centers,'center_valid':np.ones((16,16),bool)}
            destination=root/'synthetic_calibration';destination.mkdir(exist_ok=True)
            calibration=calibrate(peaks,libraries,np.ones((256,256),bool),cfg,destination)
            atomic_json(destination/'truth.json',{'known_scale':.015,'phase_ids':truth[indices].reshape(16,16),
                'generation':'random resampling of held-out zone controls with independent position noise; numerical scale regression only'})
            assert calibration['status']=='calibrated_conditionally',calibration
            assert abs(calibration['scale_inv_angstrom_per_pixel']/.015-1)<.02,calibration
            print('Known-scale recovery passed:',calibration['scale_inv_angstrom_per_pixel'],flush=True)
    atomic_json(root/'synthetic_validation.json',{'description':'held-out zone directions offset 0.4 deg; random in-plane angles and centers; known scales; 200 kV controls tested against all voltage hypotheses',
        'degradation':'0.35 px noise, 15% random missing peaks, two outliers; clean controls have 0.15 px noise',
        'limitations':'same kinematic forward model, independent zone directions; not experimental material truth; mixed controls are geometric superpositions; baseline decisions before null threshold and perturbation gates',
        'cases':reports})
    if any(r['wrong_accepted'] for r in reports):
        raise SystemExit('Synthetic controls exposed incorrect accepted labels; inspect synthetic_validation.json.')


if __name__=='__main__':
    main()
