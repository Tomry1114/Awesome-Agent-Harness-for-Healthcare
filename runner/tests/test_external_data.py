"""Study adapters: leakage boundaries, real image operations, outcome eligibility and runner integration."""
import copy
import io
import json
from pathlib import Path
import socket
import sys

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'runner'), str(ROOT / 'runner/scorers')]
import environments
import workspace as W
import run
import harness
import scoring
import aggregate_report as AR
from adapters.external_data import conversion as C
from adapters.external_data.scoring import fingerprint, review_key


def task(bench='MeissaSFT'):
    prefix = {'MedRBench': 'MEDR', 'MeissaSFT': 'MEISSA'}[bench]
    return C.task_base(bench, prefix + '-fixture', 'A synthetic study case. Explain uncertainty.', 'REFERENCE SECRET',
                       kind='fixture', group='fixture', names=list(C.TOOLS))


@pytest.mark.skipif(not (ROOT / 'external_data/MedR-Bench/data/MedRBench').is_dir() or
                    not (ROOT / 'external_data/Meissa-SFT/data/train.jsonl').is_file(),
                    reason='Pinned upstream study sources are not installed')
def test_real_conversion_public_boundary():
    medr = next(C.medr_tasks())
    assert medr['external_metadata']['protocol'] == 'oracle_case_summary'
    for t in [medr, next(C.meissa_tasks())]:
        public = W.public_task(t)
        assert not {'reference', 'external_runtime', 'external_metadata', 'checkpoints'} & public.keys()
        assert '<think>' not in t['reference']['gold_answer']
        assert set(public) <= {'task_id', 'goal', 'context', 'environment', 'available_tools', 'domain', 'modality'}


def test_image_crop_and_frozen_request_binding(monkeypatch):
    t = task()
    buf = io.BytesIO(); Image.new('RGB', (20, 10), 'red').save(buf, format='PNG')
    original = buf.getvalue()
    def asset(path):
        assert path == 'initial.png'
        return original, 'image/png'
    monkeypatch.setattr(environments.ExternalDataEnv, '_asset', staticmethod(asset))
    t['external_runtime'].update(initial_images=[{'asset_id': 'img_0', 'path': 'initial.png'}], recorded_observations=[
        {'tool': 'BioMedParseTextSeg', 'args': {'image': 'img_0', 'param': 'lung'},
         'input_image_path': 'initial.png', 'text': 'Recorded region.', 'images': []}])
    env = environments.ExternalDataEnv(t); env.reset()
    assert env.initial_observation()['output']['images'][0]['url'].startswith('data:image/png;base64,')
    assert env.call_tool('RequestTest', {'test': 'unrecorded'})['unavailable_request']
    assert env.call_tool('BioMedParseTextSeg', {'image': 'img_0', 'param': 'lung'})['output']['text'] == 'Recorded region.'
    crop = env.call_tool('ZoomInSubfigure', {'image': 'img_0', 'param': '[0,0,500,500]'})
    crop_id = crop['output']['images'][0]['image_id']
    with Image.open(io.BytesIO(env.images[crop_id][0])) as image:
        assert image.size == (10, 5)
    assert env.call_tool('BioMedParseTextSeg', {'image': crop_id, 'param': 'lung'})['unavailable_request']
    assert 'error' in env.call_tool('ReadImage', {'image': 'future_teacher_image'})
    assert 'error' in env.call_tool('ZoomInSubfigure', {'image': 'img_0', 'param': '[500,0,0,1000]'})


def test_asset_path_cannot_escape():
    with pytest.raises(ValueError):
        environments.ExternalDataEnv._asset('/etc/passwd')


@pytest.mark.parametrize('bench', C.BENCHMARKS.values())
@pytest.mark.parametrize('protocol', ['function_calling', 'text'])
def test_workspace_delivers_images_without_oracle(bench, protocol):
    t = task(bench); ws = W.Workspace(W.BENCH_H0[bench], protocol)
    assert ws.check(list(C.TOOLS)) == []
    calls = []
    def chat(messages, *args, **kwargs):
        calls.append(copy.deepcopy(messages))
        return {'content': 'done'} if protocol == 'function_calling' else '<answer>done</answer>'
    program = ws.program_class()(W.public_task(t), ws, chat)
    obs = {'output': {'images': [{'image_id': 'img_0', 'sha256': 'test', 'url': 'data:image/png;base64,fixture'}]}}
    action = program.act({'last_result': obs, 'turn_inputs': [{'kind': 'initial_observation', 'result': obs}]})
    assert action['type'] == 'final'
    assert 'REFERENCE SECRET' not in json.dumps(calls)
    assert any(isinstance(m['content'], list) and any(c.get('type') == 'image_url' for c in m['content']) for m in calls[0])
    if protocol == 'function_calling':
        crop = next(x['function'] for x in program.tools_schema if x['function']['name'] == 'ZoomInSubfigure')
        assert crop['parameters']['required'] == ['image', 'param']


def test_expert_review_binding_and_proxy_exclusion(monkeypatch, tmp_path):
    t = task(); answer = 'REFERENCE SECRET'
    ctx = {'task': t, 'reference': t['reference'], 'final_texts': [answer]}
    monkeypatch.setenv('MH_EXTERNAL_REVIEWS_DIR', str(tmp_path))
    def evaluate():
        cps = [scoring.run_checkpoint(cp, ctx) for cp in t['checkpoints']]
        return scoring.build_result(t, [], cps, {})
    result = evaluate()
    assert result['checkpoints'][1]['score'] == 1
    assert AR.native_task_outcome_detail(result, 'MeissaSFT')['outcome_status'] == 'human_review_pending'
    assert AR.native_task_score(result, 'MeissaSFT') is None
    panel = AR._outcome_metric([result], 'MeissaSFT')
    assert panel['score'] is None
    assert panel['outcome_diagnostics']['n_outcome_checkpoints'] == 0
    th, ah = fingerprint(t, answer)
    review = dict(task_id=t['task_id'], task_sha256=th, answer_sha256=ah, status='reviewed',
                  reviewer='fixture', reviewer_role='clinician', score=0.75)
    path = tmp_path / (review_key(t, answer) + '.json'); path.write_text(json.dumps(review))
    result = evaluate()
    assert AR.native_task_outcome(result, 'MeissaSFT') is True
    assert AR.native_task_score(result, 'MeissaSFT') == 0.75
    review['answer_sha256'] = 'stale'; path.write_text(json.dumps(review))
    assert AR.native_task_outcome(evaluate(), 'MeissaSFT') is None


@pytest.mark.parametrize('bench', C.BENCHMARKS.values())
def test_real_runner_external_adapters(monkeypatch, tmp_path, bench):
    t = task(bench)
    def no_network(*args, **kwargs):
        raise AssertionError('Offline study tests must not connect to the network')
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setenv('MH_HARNESS_JUDGE_MODEL', 'fixture-judge')
    monkeypatch.setenv('MH_JUDGE_MODEL', 'fixture-judge')
    monkeypatch.setattr(harness, '_build_judge', lambda: (None, None))
    monkeypatch.setattr(run, 'load_task', lambda b, tid: copy.deepcopy(t))
    monkeypatch.delenv('MH_SCRIPT_FILE', raising=False)
    monkeypatch.setenv('MH_SCRIPT', json.dumps([{'tool': 'ReadCase', 'args': {}}, {'type': 'final', 'answer': 'Evidence is limited.'}]))
    result = run.run_task(bench, t['task_id'], agent_name='scripted', max_steps=4, job_dir=str(tmp_path / 'run'))
    assert result.get('end_reason') != 'harness_error', result.get('harness_error')
    assert result['schema_validation']['valid'], result['schema_validation']
    assert result['outcome_status'] == ('evaluation_error' if bench == 'MedRBench' else 'human_review_pending')
    assert result['external_review']['task_sha256'] == fingerprint(t, result['external_review']['answer'])[0]
    assert any(e.get('tool') == 'ReadCase' for e in result['_trajectory'])
    assert result['workspace']['path'].endswith('/h0')
    assert result['harness']['effective_mode'] == 'enforce'
    bundle = tmp_path / 'agent' / t['task_id']; bundle.mkdir(parents=True)
    (bundle / 'task.json').write_text(json.dumps(t))
    (bundle / 'result.json').write_text(json.dumps({k: v for k, v in result.items() if not k.startswith('_')}))
    C.dump_jsonl(bundle / 'trajectory.jsonl', result['_trajectory'])
    report = AR.build(str(bundle.parent), bench)
    assert set(report['harness_dimensions']) == set(AR.MODULES)
    assert report['outcome']['score'] is None


@pytest.mark.skipif(not (ROOT / 'external_data/Meissa-SFT/data/train.jsonl').is_file(),
                    reason='Meissa-SFT source is not installed')
def test_downloaded_meissa_image_and_clinical_record():
    first_image = first_clinical = None
    for t in C.meissa_tasks():
        if first_image is None and t['external_runtime']['initial_images']:
            first_image = t
        if first_clinical is None and t['external_metadata']['task_kind'] == 'clinical_simulation':
            first_clinical = t
        if first_image and first_clinical:
            break
    env = environments.ExternalDataEnv(first_image); env.reset()
    assert len(env.initial_observation()['output']['images']) == len(first_image['context']['images'])
    crop = env.call_tool('ZoomInSubfigure', {'image': 'img_last', 'param': '[0,0,500,500]'})
    assert crop['mode'] == 'deterministic_pixel_crop'
    env.teardown()

    env = environments.ExternalDataEnv(first_clinical); env.reset()
    recorded = next(r for r in first_clinical['external_runtime']['recorded_observations'] if r['tool'] == 'RequestTest')
    result = env.call_tool(recorded['tool'], recorded['args'])
    assert result['output']['text'] == recorded['text']
    assert first_clinical['reference']['gold_answer'] not in json.dumps(env.initial_observation())
    env.teardown()


def test_review_export_and_rescore_cli(monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('external_review_cli', ROOT / 'external_data/scripts/review.py')
    review = importlib.util.module_from_spec(spec); spec.loader.exec_module(review)
    t = task(); t['external_metadata']['reference_kind'] = 'synthetic_fixture'
    answer = 'A synthetic answer.'
    ctx = {'task': t, 'reference': t['reference'], 'final_texts': [answer]}
    monkeypatch.setenv('MH_EXTERNAL_REVIEWS_DIR', str(tmp_path / 'reviews'))
    result = scoring.build_result(t, [], [scoring.run_checkpoint(cp, ctx) for cp in t['checkpoints']],
                                  {'fidelity': {'source_benchmark': 'MeissaSFT'}})
    th, ah = fingerprint(t, answer)
    result['external_review'] = dict(answer=answer, task_sha256=th, answer_sha256=ah, review_key=review_key(t, answer))
    output = tmp_path / 'result.json'; output.write_text(json.dumps(result))
    C.dump_jsonl(tmp_path / 'benchmark_dataprocess/MeissaSFT/tasks_unified.jsonl', [t])
    monkeypatch.setattr(review, 'ROOT', tmp_path)
    argv = ['review.py', 'export', '--result', str(output), '--reviews', str(tmp_path / 'reviews')]
    monkeypatch.setattr(sys, 'argv', argv); review.main()
    form_path = tmp_path / 'reviews' / (review_key(t, answer) + '.json')
    form = json.loads(form_path.read_text())
    assert form['answer'] == answer
    assert form['reference'] == t['reference']
    with pytest.raises(FileExistsError):
        review.main()
    form.update(status='reviewed', reviewer='fixture expert', reviewer_role='clinician', score=0.8)
    form_path.write_text(json.dumps(form))
    argv[1] = 'rescore'; review.main()
    rescored = json.loads(output.with_name('result.reviewed.json').read_text())
    assert rescored['external_outcome']['outcome'] is True
    assert json.loads(output.read_text()) == result
    from adapters.external_data.scoring import reviewed_result
    result['Governance'] = {'score': 0.6}
    merged = reviewed_result(result, tmp_path)
    assert merged['Governance'] == result['Governance']
    assert AR.native_task_outcome(merged, 'MeissaSFT') is True
