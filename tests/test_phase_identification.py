from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip('numba')
from fourdstem_pipeline.phase_matching import PhaseMatcher, decisions, score_pose
from fourdstem_pipeline.phase_peaks import find_pattern_peaks, extract_scan, atomic_json
from fourdstem_pipeline.phase_calibration import assess_scale, sample_indices


@pytest.fixture
def config():
    return yaml.safe_load(Path('configs/phase_identification.yaml').read_text(encoding='utf-8'))


def library(points, phase=0, voltage=200):
    return {'qxy': np.asarray(points,dtype=np.float32)[None,...], 'count': np.array([len(points)],np.int16),
            'phase_id':np.array(phase), 'voltage_kv':np.array(voltage)}


def test_one_to_one_matching_and_two_dimensional_support():
    obs=np.array([[20,0],[20,.1],[-20,0],[0,30],[0,-30],[25,25]],np.float32)
    pred=np.array([[20,0],[-20,0],[0,30],[0,-30],[25,25]],np.float32)
    score,n,residual,support=score_pose(obs,6,pred,5,np.array([64,64]),0,1,np.ones((128,128),bool),8,60,2.5)
    assert n == 5
    assert score == pytest.approx(10/11)
    assert residual == 0 and support
    line=np.array([[20,0],[-20,0],[30,0],[-30,0],[40,0],[-40,0]],np.float32)
    assert not score_pose(line,6,line,6,np.array([64,64]),0,1,np.ones((128,128),bool),8,60,2.5)[3]


def test_rotation_translation_missing_peak_and_noise(config):
    q=np.array([[.3,.1],[-.3,-.1],[.12,.6],[-.12,-.6],[.75,-.35],[-.75,.35],[.5,.8],[-.5,-.8]],np.float32)
    scale=.02
    angle=.713
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    observed=(q/scale)@rotation.T
    observed=observed[:-1]+np.random.default_rng(1).normal(0,.1,(7,2))
    counts=np.array([7,7])
    centers=np.array([[128,128],[100,146]],np.float32)
    matcher=PhaseMatcher([library(q)],scale,np.ones((256,256),bool),config['matching'])
    result=matcher.match(np.stack([observed,observed]),counts,centers)
    assert np.all(result[:,0,1] == 7)
    assert np.all(result[:,0,0] > .85)
    np.testing.assert_allclose(result[0],result[1],atol=1e-5)


def test_visibility_and_duplicate_reflections_do_not_inflate_score():
    points=np.array([[20,0],[-20,0],[0,30],[0,-30]],np.float32)
    visibility=np.ones((128,128),bool)
    visibility[84,64]=False
    score,n,_,_=score_pose(points,4,points,4,np.array([64,64]),0,1,visibility,8,60,2.5)
    assert n == 3 and score == pytest.approx(6/7)


@pytest.mark.parametrize('phase_count',[2,3])
def test_decisions_reject_voltage_disagreement_small_margin_and_weak_support(config,phase_count):
    libs=[library([[1,1]],phase,v) for phase in range(phase_count) for v in [80,300]]
    results=np.zeros((5,phase_count*2,7),np.float32)
    results[:,:,1]=8
    results[:,:,2]=.5
    results[:,:,6]=1
    results[:,0:2,0]=.8
    results[1,3,0]=.95  # second phase wins only at 300 kV
    results[2,2:4,0]=.79
    results[3,:,1]=5
    results[4,:,6]=0
    d=decisions(results,libs,config['matching'],observed_counts=np.full(5,8))
    assert d['phase_id'].tolist() == [0,-2,-2,-1,-1]
    assert decisions(results,libs,config['matching'],observed_counts=np.full(5,8),min_score=.99)['phase_id'].tolist() == [-1]*5
    assert decisions(results,libs,config['matching'],observed_counts=np.full(5,16))['phase_id'].tolist() == [-1]*5


def test_peak_center_shift_noise_and_invalid_pixel(config):
    rng=np.random.default_rng(12)
    yy,xx=np.indices((128,128))
    center=np.array([62.3,70.6])
    image=1000*np.exp(-((yy-center[0])**2+(xx-center[1])**2)/3)
    offsets=np.array([[20,10],[-20,-10],[12,-26],[-12,26],[30,0],[-30,0]])
    for y,x in offsets+center:
        image+=200*np.exp(-((yy-y)**2+(xx-x)**2)/2)
    image=rng.poisson(image+1).astype(float)
    image[20,20]=16385
    original=image.copy()
    p,intensity,fitted,valid,pairs,residual=find_pattern_peaks(image,center+[.8,-.9],config['peaks'])
    assert valid and pairs >= 3
    np.testing.assert_allclose(fitted,center,atol=.25)
    assert len(p) == 6
    assert np.linalg.norm(p-[20,20],axis=1).min() > 5
    np.testing.assert_array_equal(image,original)


def test_calibration_holdout_and_ambiguity(config):
    indices,heldout=sample_indices((256,256),32)
    assert len(indices)==1024 and heldout.sum()==512
    assert not set(indices[heldout]) & set(indices[~heldout])
    assert indices.min()==0 and indices.max()==65535
    scales=np.array([.005,.01,.0198,.02,.0202,.03,.05])
    scores=np.full((7,40),.2)
    scores[3]=.8
    fit=np.zeros((40,6,7))
    fit[:,:,0]=.8;fit[:,:,1]=8;fit[:,:,2]=.3;fit[:,:,6]=1
    libs=[library([[1,1]],p,v) for p in range(3) for v in [80,300]]
    result=assess_scale(scales,scores,.02,fit,libs,config['calibration'],config['matching'])
    assert result['status']=='calibrated_conditionally'
    scores[5]=.79
    result=assess_scale(scales,scores,.02,fit,libs,config['calibration'],config['matching'])
    assert result['status']=='uncalibrated' and result['separated_scale_relative_gap']<.03
    assert assess_scale(scales,scores,.005,fit,libs,config['calibration'],config['matching'])['boundary_optimum']


def test_cif_interpretation_and_cubic_extinctions(config,tmp_path):
    pytest.importorskip('pymatgen')
    pytest.importorskip('py4DSTEM')
    from fourdstem_pipeline.phase_structures import read_structure,make_crystal
    expected=[(229,2,2.86303550),(225,4,3.65555117)]
    for candidate,(group,sites,a) in zip(config['candidates'],expected):
        structure,audit=read_structure(candidate)
        assert audit['inferred_space_group']==group and len(structure)==sites
        assert structure.is_ordered and {str(e) for e in structure.elements}=={'Fe'}
        np.testing.assert_allclose(structure.lattice.abc,[a]*3)
        crystal=make_crystal(structure,1)
        if group==229:
            assert np.all(np.sum(crystal.hkl,axis=0)%2==0)
        else:
            assert np.all(crystal.hkl%2 == crystal.hkl[:1]%2)
        assert crystal.hkl.shape[1]>0
    wrong=deepcopy(config['candidates'][1]);wrong['expected_space_group']=229
    with pytest.raises(ValueError,match='symmetry mismatch'):
        read_structure(wrong)
    # Generate a wrong-element fixture; do not depend on retired reference CIFs.
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter
    wrong_path=tmp_path/'wrong_element.cif'
    CifWriter(Structure(Lattice.cubic(2.86),['Cu','Cu'],[[0,0,0],[.5,.5,.5]])).write_file(wrong_path)
    wrong['cif']=str(wrong_path)
    with pytest.raises(ValueError,match='ordered elemental Fe'):
        read_structure(wrong)


def test_candidate_config_validation(config):
    from fourdstem_pipeline.phase_identification import validate_config
    validate_config(config)
    wrong=deepcopy(config);wrong['candidates'][1]['id']=2
    with pytest.raises(ValueError,match='consecutive phase ids'):
        validate_config(wrong)
    wrong=deepcopy(config);wrong['candidates']=wrong['candidates'][:1]
    with pytest.raises(ValueError,match='at least two'):
        validate_config(wrong)


def test_peak_cache_resume_and_source_change_rejection(tmp_path,config):
    shape=(2,2)
    yy,xx=np.indices((64,64))
    image=(1000*np.exp(-((yy-32)**2+(xx-32)**2)/3)).astype('>u2')
    raw=tmp_path/'test.mib'
    with raw.open('wb') as stream:
        for i in range(4):
            stream.write(f'MQ1,{i+1:06d},00384,01,0064,0064,U16,   1x1,01,'.encode().ljust(384,b' '))
            stream.write(image.tobytes())
    basic=tmp_path/'basic';basic.mkdir()
    atomic_json(basic/'beam_motion.json',{'affine_coefficients_yx':[[0,0],[0,0]],'affine_intercept_yx':[32,32]})
    out=tmp_path/'peaks'
    first=extract_scan(raw,basic,out,shape,config['peaks'])
    second=extract_scan(raw,basic,out,shape,config['peaks'])
    np.testing.assert_array_equal(first['center_yx'],second['center_yx'])
    from fourdstem_pipeline.phase_peaks import read_json,digest,_IO_ONLY_PREDECESSOR
    checkpoint_path=out/'peaks_checkpoint.json'
    predecessor=read_json(checkpoint_path)
    predecessor['provenance']['implementation']=_IO_ONLY_PREDECESSOR
    predecessor['signature']=digest(predecessor['provenance'])
    atomic_json(checkpoint_path,predecessor)
    changed=deepcopy(config['peaks']);changed['relative_threshold']=.1
    with pytest.raises(ValueError,match='cache input/configuration changed'):
        extract_scan(raw,basic,out,shape,changed)
    migrated=extract_scan(raw,basic,out,shape,config['peaks'])
    np.testing.assert_array_equal(migrated['center_yx'],first['center_yx'])
    assert read_json(checkpoint_path)['migrations'][0]['previous_signature']==predecessor['signature']


def test_observation_selection_preserves_indices_and_inputs(config):
    from fourdstem_pipeline.phase_observations import select_observations
    positions=np.arange(10,dtype=np.float32).reshape(1,1,5,2)
    peaks={'peak_yx':positions,'intensity':np.array([[[50,1000,150,5000,0]]],np.float32),
           'count':np.array([[3]],np.int16)}
    selected=select_observations(peaks,config['matching'])
    # Padding with a large arbitrary value must not change the threshold.
    assert selected['count'][0,0]==2
    assert selected['selected_peak_indices'][0,0].tolist()==[1,2,-1,-1,-1]
    np.testing.assert_array_equal(selected['peak_yx'][0,0,:2],positions[0,0,[1,2]])
    np.testing.assert_array_equal(positions,np.arange(10).reshape(1,1,5,2))


def test_empty_observations_have_no_best_candidate(config):
    libs=[library([[1,1]],phase,v) for phase in range(3) for v in [80,300]]
    result=PhaseMatcher(libs,.02,np.ones((256,256),bool),config['matching']).match(
        np.zeros((2,8,2),np.float32),np.zeros(2,np.int16),np.full((2,2),128,np.float32))
    decision=decisions(result,libs,config['matching'],observed_counts=np.zeros(2,np.int16))
    assert decision['phase_id'].tolist()==[-1,-1]
    assert decision['best_candidate'].tolist()==[-1,-1]


@pytest.mark.parametrize('calibrated',[False,True])
def test_calibration_gate_and_perturbation_checks(tmp_path,config,monkeypatch,calibrated):
    pytest.importorskip('py4DSTEM')
    from fourdstem_pipeline.phase_identification import identify_scan
    from fourdstem_pipeline import phase_calibration,phase_report
    raw=tmp_path/'raw.mib';raw.write_bytes(b'preserved raw input')
    basic=tmp_path/'basic';basic.mkdir()
    np.save(basic/'raw_mean_diffraction.npy',np.ones((256,256)))
    out=tmp_path/'scan';(out/'peaks').mkdir(parents=True)
    atomic_json(out/'peaks'/'peaks_checkpoint.json',{'signature':'test-peaks','provenance':{'mtime_ns':raw.stat().st_mtime_ns}})
    q=np.array([[.3,.1],[-.3,-.1],[.12,.6],[-.12,-.6],[.75,-.35],[-.75,.35],[.5,.8],[-.5,-.8]],np.float32)
    libs=[library(q*(1+.3*phase),phase,v) for phase in range(2) for v in [80,300]]
    peaks={'center_yx':np.full((2,2,2),128,np.float32),'center_valid':np.ones((2,2),bool),
           'peak_yx':np.broadcast_to(q/.02+128,(2,2,8,2)).copy(),
           'intensity':np.full((2,2,8),100,np.float32),'count':np.full((2,2),8,np.int16)}
    monkeypatch.setattr(phase_calibration,'calibrate',lambda *args: {
        'status':'calibrated_conditionally' if calibrated else 'uncalibrated',
        'scale_inv_angstrom_per_pixel':.02,'null_score_threshold':.35})
    monkeypatch.setattr(phase_report,'scan_report',lambda *args: None)
    summary=identify_scan(raw,basic,out,peaks,libs,config)
    expected=0 if calibrated else -1
    assert summary['phase_counts'][str(expected)]==4 and summary['input_unchanged']
    assert np.all(np.load(out/'phase_id.npy') == expected)
    if calibrated:
        assert np.all(np.load(out/'perturbation_stable.npy'))
        assert not np.any(np.load(out/'reason_flags.npy'))
    else:
        assert np.all(np.load(out/'reason_flags.npy') & 16)
    assert np.all(np.load(out/'matched_peaks.npy') == 8)
    assert raw.read_bytes()==b'preserved raw input'
    from fourdstem_pipeline.phase_peaks import read_json
    schema=read_json(out/'array_schema.json')
    assert schema['phase_ids']=={'-2':'ambiguous','-1':'unindexed','0':'Fe-BCC','1':'Fe-FCC'}
    assert set(summary['phase_counts'])==set(schema['phase_ids'])
    assert np.load(out/'all_results.npy').shape==(2,2,4,7)
    assert identify_scan(raw,basic,out,peaks,libs,config)==summary
    changed=deepcopy(config);changed['candidates'][0]['name']='changed-candidate'
    with pytest.raises(ValueError,match='Matching cache'):
        identify_scan(raw,basic,out,peaks,libs,changed)


def test_fe_batch_report_labels_and_mixed_candidates(tmp_path):
    from fourdstem_pipeline.phase_report import batch_report
    labels={'-2':'ambiguous','-1':'unindexed','0':'Fe-BCC','1':'Fe-FCC'}
    summary={'output':'scan_01_test','patterns':10,'calibration_status':'uncalibrated',
             'scale':.02,'phase_labels':labels,'phase_counts':{'-2':0,'-1':10,'0':0,'1':0}}
    target=tmp_path/'scan_01_test';target.mkdir()
    atomic_json(target/'phase_summary.json',summary)
    batch_report(tmp_path)
    csv=(tmp_path/'phase_comparison.csv').read_text(encoding='utf-8-sig')
    assert 'Fe-BCC,Fe-FCC,ambiguous,unindexed' in csv
    html=(tmp_path/'report_zh.html').read_text(encoding='utf-8')
    assert 'Fe-BCC' in html and 'Fe-FCC' in html and 'Ti' not in html
    other=tmp_path/'scan_02_test';other.mkdir()
    summary['phase_labels']['1']='old-phase'
    atomic_json(other/'phase_summary.json',summary)
    with pytest.raises(ValueError,match='different candidate'):
        batch_report(tmp_path)


def test_atomic_json_retries_transient_windows_lock(tmp_path,monkeypatch):
    from fourdstem_pipeline import phase_peaks
    target=tmp_path/'checkpoint.json'
    atomic_json(target,{'old':True})
    replace=Path.replace
    calls=[]
    def briefly_locked(source,destination):
        calls.append(source)
        if len(calls)<3:
            assert phase_peaks.read_json(target)=={'old':True}
            raise PermissionError('transient Windows file lock')
        return replace(source,destination)
    monkeypatch.setattr(Path,'replace',briefly_locked)
    monkeypatch.setattr(phase_peaks.time,'sleep',lambda _:None)
    atomic_json(target,{'new':True})
    assert len(calls)==3 and phase_peaks.read_json(target)=={'new':True}
