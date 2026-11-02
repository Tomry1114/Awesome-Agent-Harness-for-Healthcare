"""MedR single/multiple request protocols through the real environment and harness loop."""
import copy
import json
from pathlib import Path
import socket
import sys

import pytest
pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / 'external_data/MedR-Bench/src/Inference/instructions').is_dir(),
    reason='Pinned MedRBench upstream source is not installed')

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'runner'), str(ROOT / 'runner/scorers')]
import environments
import gateway
import harness
import run
import run_batch
import workspace as W
from adapters.external_data import conversion as C
from adapters.external_data.medr_patient import UPSTREAM_MODEL, patient_settings

CASE = {'case_summary': 'Patient has a cough.\n- **Ancillary Tests:** CXR_PRIVATE_SENTINEL; lab_PRIVATE_SENTINEL.',
        'diagnosis_results': 'REFERENCE_SECRET_DIAGNOSIS', 'differential_diagnosis': 'TEACHER_REASONING_SECRET'}


def make_task(mode):
    return C.medr_interactive_task('fixture', copy.deepcopy(CASE), mode)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from adapters.external_data import outcome_judge as J
    monkeypatch.setenv("MH_JUDGE_MODEL", "fixture-judge")
    monkeypatch.setattr(J, "call_judge", lambda *a: {"ok": True, "content": "Wrong"})
    def deny_network(*args, **kwargs):
        raise AssertionError('Interactive adapter tests must not connect to a network')
    monkeypatch.setattr(socket.socket, 'connect', deny_network)
    for name in ('MH_MEDR_PATIENT_MODEL', 'MH_MEDR_PATIENT_BASE', 'MH_MEDR_PATIENT_KEY',
                 'MH_MEDR_PATIENT_TIMEOUT', 'MH_MEDR_PATIENT_MAX_TOKENS'):
        monkeypatch.delenv(name, raising=False)


def test_release_conversion_matches_upstream_split_and_keeps_oracle():
    tasks = list(C.medr_tasks())
    assert len(tasks) == 3367
    variants = {p: [t for t in tasks if t['external_metadata']['protocol'] == p]
                for p in ('oracle_case_summary', 'one_turn', 'free_turn')}
    assert [len(v) for v in variants.values()] == [1453, 957, 957]
    assert len({t['task_id'] for t in tasks}) == len(tasks)
    rows = json.loads((C.SOURCES / 'MedR-Bench/data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json').read_text())
    for t in variants['one_turn'] + variants['free_turn']:
        private = t['external_runtime']['medr_interaction']
        lines = rows[t['external_metadata']['source_id']]['generate_case']['case_summary'].strip().split('\n')
        cut = next(i for i, line in enumerate(lines) if 'Ancillary Tests' in line)
        assert private['initial_case'] == '\n'.join(lines[:cut])
        assert private['ancillary_tests'] == '\n'.join(lines[cut:])
        assert private['ancillary_tests'] not in json.dumps(W.public_task(t))
        assert t['external_metadata']['source_group'] == 'MedRBench:' + t['external_metadata']['source_id']
    assert len(list(C.medr_tasks(protocol='oracle'))) == 1453
    assert len(list(C.medr_tasks(protocol='one_turn'))) == 957
    assert len(list(C.medr_tasks(protocol='free_turn'))) == 957


def test_split_keeps_inline_findings_and_rejects_missing_boundaries():
    initial, hidden = C.split_medr_summary(CASE['case_summary'])
    assert initial == 'Patient has a cough.'
    assert 'CXR_PRIVATE_SENTINEL' in hidden
    with pytest.raises(ValueError):
        C.split_medr_summary('Patient and all tests are mixed together.')
    with pytest.raises(ValueError):
        C.split_medr_summary('Ancillary Tests: no initial case')


@pytest.mark.parametrize('mode,limit', [('one_turn', 1), ('free_turn', 4)])
def test_source_information_released_only_on_request(monkeypatch, mode, limit):
    t = make_task(mode); calls = []
    def patient(messages, **kwargs):
        calls.append((copy.deepcopy(messages), kwargs))
        return {'ok': True, 'content': 'Requested evidence from the record.'}
    monkeypatch.setattr(gateway, 'chat', patient)
    env = environments.ExternalDataEnv(t); env.reset()
    visible = json.dumps([W.public_task(t), env.initial_observation(), env.call_tool('ReadCase', {})])
    assert 'PRIVATE_SENTINEL' not in visible and 'REFERENCE_SECRET' not in visible
    assert calls == []
    for index in range(limit):
        result = env.call_tool('RequestInformation', {'request': 'Chest imaging and laboratory findings?'})
        assert result['output']['interaction']['information_requests_remaining'] == limit - index - 1
        assert result['output']['interaction']['must_finalize'] is (index == limit - 1)
        assert 'PRIVATE_SENTINEL' not in json.dumps(env.call_tool('ReadCase', {}))
    refused = env.call_tool('RequestInformation', {'request': 'Another test?'})
    assert refused['error_type'] == 'interaction_limit'
    assert len(calls) == limit
    for messages, settings in calls:
        assert 'CXR_PRIVATE_SENTINEL' in messages[0]['content']
        assert 'REFERENCE_SECRET' not in json.dumps(messages)
        assert 'TEACHER_REASONING_SECRET' not in json.dumps(messages)
        assert len(messages) == 2  # upstream service is stateless across requests
        assert settings['model'] == UPSTREAM_MODEL and settings['judge'] is False
    assert env.external_provenance()['calls_attempted'] == limit
    env.reset()
    assert env.initial_observation()['output']['interaction']['information_requests_remaining'] == limit


def test_failed_backend_is_not_a_negative_finding_and_does_not_leak(monkeypatch):
    env = environments.ExternalDataEnv(make_task('one_turn')); env.reset()
    monkeypatch.setattr(gateway, 'chat', lambda *a, **k: {'ok': False, 'error_type': 'timeout',
                                                       'raw': 'CXR_PRIVATE_SENTINEL credential-secret'})
    failure = env.call_tool('RequestInformation', {'request': 'CXR?'})
    assert failure['environment_failure'] and failure['error_type'] == 'timeout'
    assert 'output' not in failure
    assert 'PRIVATE_SENTINEL' not in json.dumps(failure) and 'credential-secret' not in json.dumps(failure)
    assert env.patient.state()['information_requests_remaining'] == 1
    assert env.capabilities()['RequestInformation']['healthy'] is False
    monkeypatch.setattr(gateway, 'chat', lambda *a, **k: {'ok': True, 'content': 'There is no relevant ancillary test information available for this request.'})
    response = env.call_tool('RequestInformation', {'request': 'Unavailable test?'})
    assert response['output']['interaction']['must_finalize']
    assert env.capabilities()['RequestInformation']['healthy']
    assert env.external_provenance()['calls_attempted'] == 2


def test_bad_requests_and_oracle_cannot_open_service(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('No service call expected')
    monkeypatch.setattr(gateway, 'chat', unexpected)
    env = environments.ExternalDataEnv(make_task('one_turn')); env.reset()
    for args in ({}, {'request': ''}, {'request': ' '}, {'request': 'CXR', 'extra': 'reference'}):
        assert 'error' in env.call_tool('RequestInformation', args)
    assert env.patient.calls_attempted == 0
    oracle = environments.ExternalDataEnv(next(C.medr_tasks(protocol='oracle'))); oracle.reset()
    assert 'error' in oracle.call_tool('RequestInformation', {'request': 'CXR?'})
    assert oracle.capabilities()['RequestInformation']['available'] is False
    assert oracle.external_provenance() is None


def test_resume_changes_with_patient_model_and_case(monkeypatch, tmp_path):
    import judges
    t = make_task('free_turn')
    monkeypatch.setenv('MH_HARNESS_JUDGE_MODEL', 'fixture-judge')
    binding = judges.require('MedRBench')
    monkeypatch.setenv('MH_MEDR_PATIENT_KEY', 'key-must-not-be-recorded')
    monkeypatch.setenv('MH_MEDR_PATIENT_BASE', 'https://example.invalid/v1/')
    args = (t['task_id'], 'digest', binding, 'scripted', 12)
    first = run.resume_key(*args, task=t)
    assert 'key-must-not-be-recorded' not in json.dumps(first)
    assert first['medr_patient_service']['base'] == 'https://example.invalid'
    monkeypatch.setenv('MH_MEDR_PATIENT_MODEL', 'different-patient-model')
    second = run.resume_key(*args, task=t)
    assert second != first
    changed = copy.deepcopy(t); changed['external_runtime']['medr_interaction']['ancillary_tests'] += '\nNew evidence.'
    assert run.resume_key(*args, task=changed) != second
    plan, reuse = run_batch.resume_plan('MedRBench', [t['task_id']], {t['task_id']: t}, 'scripted', None,
                                         binding, lambda _: 12, str(tmp_path))
    assert plan[t['task_id']][1]['medr_patient_service'] == patient_settings()
    assert not reuse


def test_batch_protocol_selection_keeps_conditions_separate(monkeypatch, tmp_path):
    from types import SimpleNamespace
    rows = [make_task('one_turn'), make_task('free_turn'), next(C.medr_tasks(protocol='oracle'))]
    C.dump_jsonl(tmp_path / 'benchmark_dataprocess/MedRBench/tasks_unified.jsonl', rows)
    monkeypatch.setattr(run_batch, 'ROOT', str(tmp_path))
    args = SimpleNamespace(task_id=None, has_dimension=None, has_subdimension=None, governance_only=False,
                           task_ids_file=None, offset=0, limit=100, external_protocol='free_turn')
    assert run_batch.select_tasks('MedRBench', args) == [rows[1]['task_id']]
    args.external_protocol = 'one_turn'
    assert run_batch.select_tasks('MedRBench', args) == [rows[0]['task_id']]
    args.external_protocol = 'oracle_case_summary'
    assert run_batch.select_tasks('MedRBench', args) == [rows[2]['task_id']]


def test_expert_form_includes_acquired_and_private_source_evidence(monkeypatch, tmp_path):
    import importlib.util
    from adapters.external_data.scoring import fingerprint, review_key
    spec = importlib.util.spec_from_file_location('medr_review_test', ROOT / 'external_data/scripts/review.py')
    review = importlib.util.module_from_spec(spec); spec.loader.exec_module(review)
    t = make_task('one_turn'); answer = 'A fixture answer.'
    th, ah = fingerprint(t, answer)
    result = {'task_id': t['task_id'], 'provenance': {'fidelity': {'source_benchmark': 'MedRBench'}},
              'external_review': {'answer': answer, 'task_sha256': th, 'answer_sha256': ah, 'review_key': review_key(t, answer)}}
    path = tmp_path / 'result.json'; path.write_text(json.dumps(result))
    (tmp_path / 'task.json').write_text(json.dumps(t))
    C.dump_jsonl(tmp_path / 'trajectory.jsonl', [{'event_type': 'tool_call', 'tool': 'RequestInformation',
                    'args': {'request': 'CXR?'}, 'result': {'output': {'text': 'A supplied imaging result.'}}}])
    monkeypatch.setattr(sys, 'argv', ['review.py', 'export', '--result', str(path), '--reviews', str(tmp_path / 'reviews')])
    review.main()
    form = json.loads((tmp_path / 'reviews' / (review_key(t, answer) + '.json')).read_text())
    assert 'CXR_PRIVATE_SENTINEL' in form['source_ancillary_findings']
    assert form['information_requests'][0]['result']['output']['text'] == 'A supplied imaging result.'
    assert 'CXR_PRIVATE_SENTINEL' not in json.dumps(W.public_task(t))


@pytest.mark.parametrize('mode,limit', [('one_turn', 1), ('free_turn', 4)])
@pytest.mark.parametrize('protocol', ['function_calling', 'text'])
def test_workspace_agent_and_real_runner_acquire_then_finalize(monkeypatch, tmp_path, mode, limit, protocol):
    t = make_task(mode)
    ws = W.Workspace(W.BENCH_H0['MedRBench'], protocol)
    assert ws.check([x['name'] for x in t['available_tools']]) == []
    messages_seen, patient_calls = [], []
    def patient(messages, **kwargs):
        patient_calls.append(copy.deepcopy(messages))
        return {'ok': True, 'content': 'RELEASED_RESPONSE_%d' % len(patient_calls)}
    monkeypatch.setattr(gateway, 'chat', patient)
    def chat(messages, *args, **kwargs):
        messages_seen.append(copy.deepcopy(messages))
        index = len(messages_seen)
        if index <= limit:
            request = {'name': 'RequestInformation', 'arguments': {'request': 'Diagnostic request %d' % index}}
            if protocol == 'text':
                return '<tool_call>' + json.dumps(request) + '</tool_call>'
            return {'content': None, 'tool_calls': [{'id': 'call_%d' % index, 'type': 'function',
                    'function': {'name': request['name'], 'arguments': json.dumps(request['arguments'])}}]}
        return {'content': 'Evidence is limited.'} if protocol == 'function_calling' else '<answer>Evidence is limited.</answer>'
    program = ws.program_class()(W.public_task(t), ws, chat)
    class Agent:
        name = 'study-test-brain'
        def act(self, state):
            return program.act(state)
    monkeypatch.setattr(run.agents, 'make_agent', lambda *a, **k: Agent())
    monkeypatch.setattr(run, 'load_task', lambda *a: copy.deepcopy(t))
    monkeypatch.setattr(harness, '_build_judge', lambda: (None, None))
    monkeypatch.setenv('MH_HARNESS_JUDGE_MODEL', 'fixture-judge')
    result = run.run_task('MedRBench', t['task_id'], agent_name='scripted', max_steps=12,
                          job_dir=str(tmp_path / 'run'))
    assert result.get('end_reason') != 'harness_error', result.get('harness_error')
    assert result['schema_validation']['valid'], result['schema_validation']
    assert result['harness']['effective_mode'] == 'enforce'
    assert len(patient_calls) == limit
    assert 'PRIVATE_SENTINEL' not in json.dumps(messages_seen)
    assert 'REFERENCE_SECRET' not in json.dumps(messages_seen)
    assert 'RELEASED_RESPONSE_1' not in json.dumps(messages_seen[0])
    assert 'RELEASED_RESPONSE_%d' % limit in json.dumps(messages_seen[limit])
    info = result['provenance']['fidelity']['external_environment']
    assert info['information_requests_used'] == limit and info['must_finalize']
    assert result['provenance']['tool_backend_model'] == UPSTREAM_MODEL
    assert result['outcome_status'] == 'resolved'
    calls = [e for e in result['_trajectory'] if e.get('tool') == 'RequestInformation']
    assert len(calls) == limit and all(e['result']['mode'] == 'medr_live_patient_service' for e in calls)
