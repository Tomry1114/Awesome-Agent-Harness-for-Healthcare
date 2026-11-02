"""GT verdicts, failure semantics, immutable replay bindings and offline aggregation."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'runner'), str(ROOT / 'runner/scorers'), str(ROOT / 'external_data/scripts')]
import aggregate_report as AR
import scoring
import run
import workspace
import judge_outcome
from adapters.external_data import conversion as C, outcome_judge as J
from adapters.external_data.scoring import fingerprint, reviewed_result


def task(bench='MedRBench'):
    return C.task_base(bench, 'fixture', 'Question?', 'GROUND_TRUTH_SECRET',
                       kind='diagnosis' if bench == 'MedRBench' else 'pathvqa', group='fixture')


def evaluate(monkeypatch, reply, bench='MedRBench', answer='candidate answer', t=None):
    t = t or task(bench)
    monkeypatch.setattr(J, 'call_judge', lambda *args: reply)
    cp = scoring.run_checkpoint(J.checkpoint(), {'task': t, 'final_texts': [answer], 'reference': t['reference']})
    result = scoring.build_result(t, [], [cp], {})
    return result, AR.native_task_outcome_detail(result, bench)


@pytest.mark.parametrize('reply,expected', [('Correct', True), ('[Correct]', True), ('Wrong', False), ('[Wrong]', False)])
def test_medr_strict_official_verdict(monkeypatch, reply, expected):
    r, detail = evaluate(monkeypatch, {'ok': True, 'content': reply})
    assert detail['outcome'] is expected
    assert AR.native_task_score(r, 'MedRBench') == float(expected)
    assert r['checkpoints'][0]['detail']['protocol'] == 'medr_official_diagnosis_rubric'


@pytest.mark.parametrize('raw', ['Incorrect', 'Not Correct', 'Correct|Wrong', '', None, '{}', '[Correct'])
def test_malformed_judge_is_not_an_agent_failure(monkeypatch, raw):
    r, detail = evaluate(monkeypatch, {'ok': True, 'content': raw})
    assert detail['outcome'] is None and detail['outcome_status'] == 'judge_error'
    assert r['checkpoints'][0]['score'] is None
    assert AR.native_task_score(r, 'MedRBench') is None


def test_gateway_error_is_unresolved_not_zero(monkeypatch):
    r, detail = evaluate(monkeypatch, {'ok': False, 'error_type': 'timeout'})
    assert detail['n_judge_errors'] == 1
    assert AR._outcome_metric([r], 'MedRBench')['score'] is None
    assert r['checkpoints'][0]['detail']['gateway_error_type'] == 'timeout'


@pytest.mark.parametrize('verdict,expected', [('Correct', True), ('Wrong', False)])
def test_vqa_semantic_rubric_and_reasons(monkeypatch, verdict, expected):
    r, detail = evaluate(monkeypatch, {'ok': True, 'content': json.dumps({'verdict': verdict, 'reason': 'Fixture explanation'})}, 'MeissaEval')
    assert detail['outcome'] is expected
    assert r['checkpoints'][0]['detail']['reason'] == 'Fixture explanation'
    assert AR._outcome_metric([r], 'MeissaEval')['metric'] == 'gt_judged_task_success'


@pytest.mark.parametrize('problem', ['missing_gt', 'empty_answer', 'agent_error', 'teacher', 'unsupported'])
def test_ineligible_inputs_never_call_judge(monkeypatch, problem):
    t = task(); answer = 'candidate'
    if problem == 'missing_gt': t['reference']['gold_answer'] = ''
    if problem == 'empty_answer': answer = ''
    if problem == 'agent_error': answer = 'API_BRAIN_ERROR'
    if problem == 'teacher': t['external_metadata']['reference_kind'] = 'teacher_response_not_independent_ground_truth'
    if problem == 'unsupported': t['external_metadata']['task_kind'] = 'unsupported'
    monkeypatch.setattr(J, 'call_judge', lambda *args: pytest.fail('Must not call the Judge'))
    cp = J.score(J.checkpoint(), {'task': t, 'final_texts': [answer]}, {})
    assert cp['score'] is None and not cp['score_eligible'] and cp['judge']['calls'] == 0


def test_legacy_task_is_upgraded_only_at_scoring(monkeypatch):
    t = task(); original = copy.deepcopy(t)
    legacy = [{'id': 'human_outcome', 'type': 'deterministic', 'dimension': 'Outcome', 'check': {'method': 'external_human_review'}}]
    seen = []
    def judge(messages, config):
        seen.append(messages)
        return {'ok': True, 'content': 'Correct'}
    monkeypatch.setattr(J, 'call_judge', judge)
    assert not J.effective_checkpoints('MeissaSFT', legacy)[0]['id'] == 'gt_outcome'
    assert J.effective_checkpoints('MedRBench', []) == []
    cps = run.score_checkpoints(legacy, {'task': t, 'final_texts': ['answer']}, 'external_data')
    assert cps[0]['score'] == 1 and len(seen) == 1 and t == original
    assert 'GROUND_TRUTH_SECRET' in json.dumps(seen)
    assert 'GROUND_TRUTH_SECRET' not in json.dumps(workspace.public_task(t))


def test_rescore_pure_overlay_rejects_changed_reference_and_answer(monkeypatch, tmp_path):
    monkeypatch.setenv('MH_JUDGE_MODEL', 'fixture-judge')
    t = task(); answer = 'candidate'
    th, ah = fingerprint(t, answer)
    original = {'task_id': t['task_id'], 'provenance': {'fidelity': {'source_benchmark': 'MedRBench'}},
                'external_review': {'answer': answer, 'task_sha256': th, 'answer_sha256': ah},
                'checkpoints': [], 'dimension_scores': {'Verification': None}, 'harness': {'unchanged': True}}
    (tmp_path / 'task.json').write_text(json.dumps(t))
    (tmp_path / 'result.json').write_text(json.dumps(original))
    monkeypatch.setattr(J, 'call_judge', lambda *a: {'ok': True, 'content': 'Correct'})
    path, updated = judge_outcome.rescore(tmp_path)
    assert json.loads((tmp_path / 'result.json').read_text()) == original
    assert updated['dimension_scores'] == original['dimension_scores']
    assert updated['harness'] == original['harness']
    monkeypatch.setattr(J, 'call_judge', lambda *a: pytest.fail('Pure read must not call Judge'))
    loaded = J.overlay_result(original, tmp_path)
    assert AR.native_task_score(loaded, 'MedRBench') == 1
    altered = copy.deepcopy(original); altered['external_review']['answer'] = 'different answer'
    with pytest.raises(ValueError): J.overlay_result(altered, tmp_path)
    t['reference']['gold_answer'] = 'changed GT'
    (tmp_path / 'task.json').write_text(json.dumps(t))
    with pytest.raises(ValueError): J.overlay_result(original, tmp_path)


def test_expert_review_is_supplemental(monkeypatch, tmp_path):
    result, _ = evaluate(monkeypatch, {'ok': True, 'content': 'Correct'})
    result.update(provenance={'fidelity': {'source_benchmark': 'MedRBench'}}, external_review={'answer': 'candidate'})
    reviewed = copy.deepcopy(result)
    reviewed['checkpoints'] = [{'evaluator_kind': 'external_human_review', 'score': 0, 'checkpoint_status': 'failed'}]
    reviewed['success'] = False
    (tmp_path / 'result.reviewed.json').write_text(json.dumps(reviewed))
    merged = reviewed_result(result, tmp_path)
    assert AR.native_task_score(merged, 'MedRBench') == 1
    assert merged['expert_review'][0]['score'] == 0


def test_resume_key_tracks_judge_configuration(monkeypatch):
    t = task()
    monkeypatch.setenv('MH_JUDGE_MODEL', 'judge-a')
    first = run.resume_key(t['task_id'], 'workspace', {}, 'scripted', 12, t)
    monkeypatch.setenv('MH_JUDGE_MODEL', 'judge-b')
    second = run.resume_key(t['task_id'], 'workspace', {}, 'scripted', 12, t)
    assert first != second


def test_batch_preserves_gt_judge_model():
    import judges
    for bench in J.BENCHMARKS:
        with pytest.raises(judges.JudgeConfigError):
            judges.require(bench, {'MH_HARNESS_JUDGE_MODEL': 'harness-only'})
        binding = judges.require(bench, {'MH_HARNESS_JUDGE_MODEL': 'harness', 'MH_JUDGE_MODEL': 'outcome-judge'})
        assert judges.subprocess_env(binding)['MH_JUDGE_MODEL'] == 'outcome-judge'


def test_treatment_protocol_is_explicitly_adapted(monkeypatch):
    t = task(); t['external_metadata']['task_kind'] = 'treatment'
    r, detail = evaluate(monkeypatch, {'ok': True, 'content': '{"verdict":"Wrong","reason":"Essential intervention missing"}'}, t=t)
    assert detail['outcome'] is False
    assert r['checkpoints'][0]['detail']['protocol'] == 'medr_reference_treatment_adapted'
    assert 'not_upstream_metric' in r['checkpoints'][0]['detail']['metric_fidelity']
