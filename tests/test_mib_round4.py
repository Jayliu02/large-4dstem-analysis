import copy
import json
import shutil
import subprocess

import numpy as np
import pytest

from fourdstem_pipeline.mib_round4 import inject_counts, validate_review
from fourdstem_pipeline.mib_review_ui import HTML


def fixture_review():
    bundle = dict(bundle_id="test-bundle", patterns=[dict(id="grain:1:2", roi_id="grain", width=64, height=48,
        peaks=[dict(id=0), dict(id=1)])])
    review = dict(schema_version=1, bundle_id="test-bundle", reviewer="Synthetic test reviewer",
        reviewed_at_utc="2026-10-09T01:00:00Z", patterns=[dict(id="grain:1:2", completed=True,
            peaks=[dict(id=0, label="valid"), dict(id=1, label="valid")], missed_major_peaks=[])])
    rules = dict(minimum_complete_per_roi=1, maximum_false_fraction=.05, maximum_mean_missed_major=.5)
    return bundle, review, rules


def test_injection_conserves_source_and_records_exact_added_counts():
    dp = np.full((64, 80), 6, dtype=np.uint16)
    y, x = np.mgrid[-12:13, -12:13]
    empirical = np.exp(-(x*x+y*y)/8)
    for shape in ["empirical", "gaussian"]:
        augmented, increment, bbox, expected = inject_counts(dp, [38.3, 30.7], shape, 16, empirical, 2, 19)
        ya, yb, xa, xb = bbox
        delta = augmented.astype(float)-dp
        np.testing.assert_array_equal(delta[ya:yb, xa:xb], increment)
        assert delta.sum() == increment.sum() and (delta >= 0).all()
        assert expected > 0 and (dp == 6).all()
        np.testing.assert_array_equal(augmented, inject_counts(dp, [38.3, 30.7], shape, 16, empirical, 2, 19)[0])
    np.testing.assert_array_equal(inject_counts(dp, [38.3, 30.7], "gaussian", 0, empirical, 2, 19)[0], dp)
    with pytest.raises(ValueError, match="leaves detector"):
        inject_counts(dp, [1, 1], "gaussian", 16, empirical, 2, 19)


def test_review_pass_fail_and_pending_are_separate():
    bundle, review, rules = fixture_review()
    assert validate_review(review, bundle, rules)["review_passed"]
    review["patterns"][0]["peaks"][0]["label"] = "false_peak"
    assert validate_review(review, bundle, rules)["review_status"] == "failed"
    review["patterns"][0]["completed"] = False
    assert validate_review(review, bundle, rules)["review_status"] == "pending"


@pytest.mark.parametrize("mutation", ["bundle", "duplicate_pattern", "unknown_peak", "duplicate_peak", "missing_peak", "out_of_bounds", "incomplete", "identity", "timestamp"])
def test_review_rejects_wrong_provenance_or_malformed_annotations(mutation):
    bundle, review, rules = fixture_review()
    r = review["patterns"][0]
    if mutation == "bundle": review["bundle_id"] = "other"
    elif mutation == "duplicate_pattern": review["patterns"].append(copy.deepcopy(r))
    elif mutation == "unknown_peak": r["peaks"][0]["id"] = 9
    elif mutation == "duplicate_peak": r["peaks"][1]["id"] = 0
    elif mutation == "missing_peak": r["peaks"].pop()
    elif mutation == "out_of_bounds": r["missed_major_peaks"] = [dict(x=80, y=3)]
    elif mutation == "incomplete": r["peaks"][0]["label"] = "unreviewed"
    elif mutation == "identity": review["reviewer"] = ""
    elif mutation == "timestamp": review["reviewed_at_utc"] = "yesterday"
    with pytest.raises(ValueError): validate_review(review, bundle, rules)


def test_offline_ui_completion_export_and_provenance_checks():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable for offline UI event smoke test")
    bundle = dict(schema_version=1, bundle_id="ui-test", patterns=[dict(id="grain:1:2", roi_id="grain", width=256,
        height=256, peaks=[dict(id=0, x=80, y=90, snr=8, both_halves=True)], image_log="a", image_linear="b")])
    code = HTML.rsplit("<script>", 1)[1].split("</script>", 1)[0]
    harness = r'''
const assert=require('node:assert/strict');
const elements={};let exported=null;
function element(){return {value:'',checked:false,textContent:'',files:[],append(){},replaceChildren(){},click(){},getBoundingClientRect(){return {left:0,top:0,width:768,height:768};},width:768,height:768,getContext(){return new Proxy({}, {get(t,k){return t[k]||(()=>{});},set(t,k,v){t[k]=v;return true;}});}};}
global.document={getElementById(id){return elements[id]||(elements[id]=element());},createElement:element};
global.localStorage={getItem(){return null;},setItem(){}};
global.Image=class{set src(v){if(this.onload)this.onload();}};
global.Blob=class{constructor(parts){exported=JSON.parse(parts.join(''));}};
global.URL={createObjectURL(){return 'mock';},revokeObjectURL(){}};
global.setTimeout=fn=>fn();
document.getElementById('bundle').textContent=__JSON__;
'''.replace("__JSON__", json.dumps(json.dumps(bundle)))
    assertions = r'''
el('complete').checked=true;el('complete').onchange();assert.equal(current().completed,false);
el('allvalid').onclick();el('complete').checked=true;el('complete').onchange();assert.equal(current().completed,true);
el('canvas').onclick({shiftKey:true,clientX:150,clientY:150});assert.equal(current().missed_major_peaks.length,1);assert.equal(current().completed,false);
el('reviewer').value='Test Reviewer';el('complete').checked=true;el('complete').onchange();el('export').onclick();
assert.equal(exported.bundle_id,'ui-test');assert.equal(exported.patterns[0].peaks[0].label,'valid');assert.equal(exported.patterns[0].missed_major_peaks[0].x,49.5);
assert.throws(()=>validateDraft({...exported,bundle_id:'other'}));
const malformed=JSON.parse(JSON.stringify(exported));malformed.patterns[0].peaks[0].id=99;assert.throws(()=>validateDraft(malformed));
'''
    result = subprocess.run([node], input=harness+code+assertions, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_review_import_audits_original_json_and_cannot_bypass_failed_diagnostic(tmp_path, monkeypatch):
    from fourdstem_pipeline import mib_round4 as module
    bundle, review, rules = fixture_review()
    output = tmp_path / "synthetic_review_output"
    output.mkdir()
    module.write_json(output / "review_bundle.json", bundle)
    module.write_json(output / "run_manifest.json", dict(config=dict(review=rules), diagnostic_gate_passed=False))
    source = tmp_path / "synthetic_annotations.json"
    module.write_json(source, review)
    # Synthetic fixture has no MIB; this test isolates import audit and gate conjunction.
    monkeypatch.setattr(module, "verify_output", lambda out: None)
    module.import_review(output, source)
    pointer = module.load_json(output / "current_review.json")
    assert pointer["decision"]["review_passed"]
    assert not pointer["decision"]["gate3_passed"]
    assert not pointer["decision"]["gate5_passed"]
    audit = output / "reviews" / module.digest(source)
    assert (audit / "review.json").read_bytes() == source.read_bytes()
    assert module.load_json(audit / "decision.json") == pointer["decision"]
