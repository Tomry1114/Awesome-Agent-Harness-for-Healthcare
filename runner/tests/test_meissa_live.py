"""Real local HTTP contracts and runner image delivery, without model/API calls."""
import base64
import copy
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import subprocess
import threading

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'runner'), str(ROOT / 'runner/scorers')]
import environments
import harness
import run
import workspace as W
import aggregate_report as AR
from adapters.user_study import conversion as C
from adapters.user_study.meissa_image import MeissaImageBackend, settings, zoom


def png(color='red'):
    out = io.BytesIO()
    Image.new('RGB', (100, 80), color).save(out, format='PNG')
    return out.getvalue()


@pytest.fixture(autouse=True)
def clear_config(monkeypatch):
    from adapters.user_study import outcome_judge as J
    monkeypatch.setenv("MH_JUDGE_MODEL", "fixture-judge")
    monkeypatch.setattr(J, "call_judge", lambda *a: {"ok": True, "content": json.dumps({"verdict": "Wrong", "reason": "Offline fixture"})})
    for key in ('TOOL_SERVER', 'SAM_URL', 'BIOMEDPARSE_URL', 'SAM_REVISION', 'BIOMEDPARSE_REVISION', 'TOOL_TIMEOUT', 'RUNTIME_CONFIG', 'TOOL_PROXY'):
        monkeypatch.delenv('MH_MEISSA_' + key, raising=False)


@pytest.fixture
def server():
    state = {'requests': [], 'response': {'text': 'Fixture segmentation only', 'error_code': 0,
                                         'edited_image': base64.b64encode(png('blue')).decode()}}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state['requests'].append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            self.send_response(state.get('status', 200))
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(state['response']).encode())
        def log_message(self, *args):
            pass
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    state['url'] = 'http://127.0.0.1:%d' % http.server_port
    try:
        yield state
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=2)


def live_task(bench='MeissaEval'):
    t = C.task_base(bench, 'MEISSAEVAL-fixture', 'Inspect the supplied image.', 'PRIVATE_REFERENCE',
                    kind='pathvqa', group='fixture', images=[{'asset_id': 'img_original', 'path': 'fixture.png'}])
    t['study_metadata']['reference_kind'] = 'synthetic_fixture'
    return C.enable_live_image(t)


@pytest.mark.parametrize('transport', ['worker', 'call'])
def test_remote_contract_and_current_bytes(monkeypatch, server, transport):
    key = 'MH_MEISSA_SAM_URL' if transport == 'worker' else 'MH_MEISSA_TOOL_SERVER'
    monkeypatch.setenv(key, server['url'])
    backend = MeissaImageBackend()
    data = png()
    result = backend.execute('SegmentRegionAroundPoint', data, '[[500,500]]')
    assert result['data'] == png('blue')
    path, payload = server['requests'][0]
    assert path == ('/worker_generate' if transport == 'worker' else '/call')
    arguments = payload if transport == 'worker' else payload['arguments']
    assert arguments == {'image': base64.b64encode(data).decode(), 'param': '[[500,500]]'}
    if transport == 'call':
        assert payload['tool'] == 'SegmentRegionAroundPoint'
    assert backend.provenance()['execution'][0]['status'] == 'success'


@pytest.mark.parametrize('response', [
    {'error_code': 1, 'text': 'server secret', 'edited_image': base64.b64encode(png()).decode()},
    {'error_code': 0, 'text': 'no result', 'edited_image': None},
    {'error_code': 0, 'edited_image': 'not-base64'},
    {'error_code': 0, 'edited_image': base64.b64encode(b'not an image').decode()},
    ['unexpected response'],
])
def test_backend_failures_are_not_evidence(monkeypatch, server, response):
    monkeypatch.setenv('MH_MEISSA_BIOMEDPARSE_URL', server['url'])
    server['response'] = response
    result = MeissaImageBackend().execute('BioMedParseTextSeg', png(), 'lung')
    assert len(server['requests']) == 1
    assert result['environment_failure'] and 'data' not in result
    assert 'server secret' not in json.dumps(result)


def test_no_backend_never_replays_teacher(monkeypatch):
    t = live_task()
    t['study_runtime']['recorded_observations'] = [
        {'tool': 'BioMedParseTextSeg', 'args': {'image': 'img_original', 'param': 'lung'},
         'text': 'PRIVATE_TEACHER', 'images': []}]
    monkeypatch.setattr(environments.UserStudyEnv, '_asset', staticmethod(lambda path: (png(), 'image/png')))
    env = environments.UserStudyEnv(t)
    env.reset()
    assert env.capabilities()['BioMedParseTextSeg']['available'] is False
    result = env.call_tool('BioMedParseTextSeg', {'image': 'img_original', 'param': 'lung'})
    assert result['environment_failure'] and 'PRIVATE_TEACHER' not in json.dumps(result)
    assert env.last_image == 'img_original'
    assert 'error' in env.call_tool('ReadImage', {'image': 'future_teacher_image'})


def test_failed_service_can_recover_without_publishing_failure_image(monkeypatch, server):
    monkeypatch.setenv('MH_MEISSA_SAM_URL', server['url'])
    monkeypatch.setattr(environments.UserStudyEnv, '_asset', staticmethod(lambda path: (png(), 'image/png')))
    env = environments.UserStudyEnv(live_task())
    env.reset()
    server['status'] = 503
    result = env.call_tool('SegmentRegionAroundPoint', {'image': 'img_last', 'param': '[[500,500]]'})
    assert result['environment_failure'] and env.last_image == 'img_original'
    assert env.capabilities()['SegmentRegionAroundPoint']['healthy'] is False
    server['status'] = 200
    result = env.call_tool('SegmentRegionAroundPoint', {'image': 'img_last', 'param': '[[500,500]]'})
    assert 'output' in result and env.last_image == 'live_image_1'
    assert env.capabilities()['SegmentRegionAroundPoint']['healthy'] is True


def test_changed_source_image_is_rejected(monkeypatch):
    t = live_task()
    t['study_runtime']['initial_images'][0]['sha256'] = 'wrong'
    monkeypatch.setattr(environments.UserStudyEnv, '_asset', staticmethod(lambda path: (png(), 'image/png')))
    with pytest.raises(ValueError, match='changed after task preparation'):
        environments.UserStudyEnv(t).reset()


def test_zoom_upstream_bbox_semantics():
    text, data, mime = zoom(png(), '[250,250,750,750]')
    assert text == 'Zoomed in successfully to region [20, 16, 80, 64].'
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (60, 48) and image.format == 'JPEG'
    assert mime == 'image/jpeg'
    for value in ('[100,0,10,100]', '[0,0,1,1]', '[0,0,2000,3000]', '"mask"'):
        with pytest.raises(ValueError):
            zoom(png(), value)


@pytest.mark.parametrize('protocol', ['function_calling', 'text'])
def test_real_runner_image_loop_and_seven_dimensions(monkeypatch, tmp_path, server, protocol):
    monkeypatch.setenv('MH_MEISSA_TOOL_SERVER', server['url'])
    monkeypatch.setenv('MH_HARNESS_JUDGE_MODEL', 'fixture-judge')
    monkeypatch.setattr(harness, '_build_judge', lambda: (None, None))
    monkeypatch.setattr(environments.UserStudyEnv, '_asset', staticmethod(lambda path: (png(), 'image/png')))
    t = live_task()
    monkeypatch.setattr(run, 'load_task', lambda *args: copy.deepcopy(t))
    ws = W.Workspace(W.BENCH_H0['MeissaEval'], protocol)
    assert ws.check([x['name'] for x in t['available_tools']]) == []
    messages_seen = []
    actions = [('ZoomInSubfigure', '[250,250,750,750]'), ('BioMedParseTextSeg', 'lung'),
               ('SegmentRegionAroundPoint', '[[500,500]]')]
    def chat(messages, *args, **kwargs):
        messages_seen.append(copy.deepcopy(messages))
        index = len(messages_seen) - 1
        if index >= len(actions):
            return {'content': 'Evidence remains limited.'} if protocol == 'function_calling' else '<answer>Evidence remains limited.</answer>'
        name, param = actions[index]
        call = {'name': name, 'arguments': {'image': 'img_last', 'param': param}}
        if protocol == 'text':
            return '<tool_call>' + json.dumps(call) + '</tool_call>'
        return {'content': None, 'tool_calls': [{'id': 'call_%d' % index, 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(call['arguments'])}}]}
    program = ws.program_class()(W.public_task(t), ws, chat)
    class Agent:
        name = 'fixture-brain'
        def act(self, state):
            return program.act(state)
    monkeypatch.setattr(run.agents, 'make_agent', lambda *args, **kwargs: Agent())
    result = run.run_task('MeissaEval', t['task_id'], agent_name='scripted', max_steps=10,
                          job_dir=str(tmp_path / 'run'))
    assert result.get('end_reason') != 'harness_error', result.get('harness_error')
    assert result['schema_validation']['valid'], result['schema_validation']
    provenance = result['provenance']['fidelity']
    assert provenance['environment_fidelity'] == 'meissa_live_image_tools'
    events = provenance['study_environment']['execution']
    assert len(events) == 3 and all(x['status'] == 'success' for x in events)
    assert events[0]['output_sha256'] == events[1]['input_sha256']
    assert events[1]['output_sha256'] == events[2]['input_sha256']
    assert 'PRIVATE_REFERENCE' not in json.dumps(messages_seen)
    assert 'live_image_3' in json.dumps(messages_seen[-1])
    for messages in messages_seen:
        assert any(isinstance(m['content'], list) and any(c.get('type') == 'image_url' for c in m['content']) for m in messages)
    assert result['outcome_status'] == 'resolved'
    bundle = tmp_path / 'agent' / t['task_id']
    bundle.mkdir(parents=True)
    (bundle / 'task.json').write_text(json.dumps(t))
    (bundle / 'result.json').write_text(json.dumps({k: v for k, v in result.items() if not k.startswith('_')}))
    C.dump_jsonl(bundle / 'trajectory.jsonl', result['_trajectory'])
    report = AR.build(str(bundle.parent), 'MeissaEval')
    assert set(report['harness_dimensions']) == set(AR.MODULES)
    assert report['outcome']['score'] == 0.0


def test_eval_manifest_split_paths_and_answer_boundary(monkeypatch, tmp_path):
    monkeypatch.setattr(C, 'ROOT', tmp_path)
    monkeypatch.setattr(C, 'SOURCES', tmp_path / 'user_study')
    image = C.SOURCES / 'Meissa-Evaluation/images/sample.png'
    image.parent.mkdir(parents=True)
    image.write_bytes(png())
    row = dict(id='sample', question='Synthetic question', answer='PRIVATE_REFERENCE', image='sample.png',
               dataset='PathVQA', split='test', source_url='https://example.org/test-fixture')
    manifest = tmp_path / 'manifest.jsonl'
    def convert():
        manifest.write_text(json.dumps(row) + '\n')
        return list(C.meissa_evaluation_tasks(manifest))
    t = convert()[0]
    assert 'PRIVATE_REFERENCE' not in json.dumps(W.public_task(t))
    assert t['source_benchmark'] == 'MeissaEval'
    assert t['study_metadata']['declared_source_split'] == 'test'
    assert t['study_metadata']['held_out_test'] is False
    assert t['study_metadata']['independent_evaluation'] is None
    assert t['study_runtime']['recorded_observations'] == []
    output = tmp_path / 'prepared'
    monkeypatch.setattr(sys, 'argv', ['prepare.py', '--dataset', 'meissa-eval',
                        '--meissa-eval-manifest', str(manifest), '--output-root', str(output)])
    assert C.main()['MeissaEval']['tasks'] == 1
    check = subprocess.run([sys.executable, str(ROOT / 'benchmark_dataprocess/validate_tasks.py'),
                            str(output / 'MeissaEval/tasks_unified.jsonl'), '--strict'],
                           capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    row['split'] = 'train'
    with pytest.raises(ValueError, match='test records'):
        convert()
    row.update(split='test', image='../../../outside.png')
    with pytest.raises(ValueError):
        convert()


@pytest.mark.skipif(not (C.SOURCES / 'Meissa-SFT/data/train.jsonl').is_file(),
                    reason='Meissa-SFT source is not installed')
def test_sft_live_remains_training_data():
    t = next(C.meissa_tasks(image_mode='live'))
    assert t['study_metadata']['protocol'] == 'meissa_live_image'
    assert t['study_metadata']['data_role'] == 'training_trajectory'
    assert not t['study_metadata']['held_out_test']
    assert not t['study_metadata']['independent_evaluation']
    assert t['study_runtime']['recorded_observations'] == []
    env = environments.UserStudyEnv(t)
    env.reset()
    result = env.call_tool('ZoomInSubfigure', {'image': 'img_original', 'param': '[250,250,750,750]'})
    assert result['mode'] == 'meissa_live_image' and 'output' in result
    with Image.open(io.BytesIO(env.images[env.last_image][0])) as image:
        assert image.format == 'JPEG'
    env.teardown()


def test_resume_changes_with_backend_and_task(monkeypatch):
    import judges
    monkeypatch.setattr(judges, 'signature', lambda binding: {})
    t = live_task()
    first = run.resume_key(t['task_id'], 'workspace', {}, 'scripted', 10, t)
    monkeypatch.setenv('MH_MEISSA_SAM_URL', 'http://localhost:9001')
    second = run.resume_key(t['task_id'], 'workspace', {}, 'scripted', 10, t)
    assert first != second
    monkeypatch.setenv('MH_MEISSA_SAM_REVISION', 'new-weights')
    assert second != run.resume_key(t['task_id'], 'workspace', {}, 'scripted', 10, t)
    t['context']['text'] += ' changed'
    assert first['task_sha256'] != run.resume_key(t['task_id'], 'workspace', {}, 'scripted', 10, t)['task_sha256']
    monkeypatch.setenv('MH_MEISSA_SAM_URL', 'http://user:secret@localhost:9001')
    with pytest.raises(ValueError):
        settings()


@pytest.mark.parametrize('transport', ['worker', 'call'])
def test_persistent_direct_proxy_policy_for_compute_hostname(monkeypatch, server, transport):
    import socket
    original = socket.getaddrinfo
    def resolve(host, *args, **kwargs):
        return original('127.0.0.1' if host == 'worker.internal.test' else host, *args, **kwargs)
    monkeypatch.setattr(socket, 'getaddrinfo', resolve)
    for key in ('http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY'):
        monkeypatch.setenv(key, 'http://127.0.0.1:9')
    for key in ('no_proxy', 'NO_PROXY'):
        monkeypatch.setenv(key, '')
    key = 'MH_MEISSA_SAM_URL' if transport == 'worker' else 'MH_MEISSA_TOOL_SERVER'
    monkeypatch.setenv(key, server['url'].replace('127.0.0.1', 'worker.internal.test'))
    assert settings()['tool_proxy'] == 'direct'  # checked-in runtime configuration
    result = MeissaImageBackend().execute('SegmentRegionAroundPoint', png(), 'x=500 y=500')
    assert result['data'] == png('blue') and len(server['requests']) == 1
    monkeypatch.setenv('MH_MEISSA_TOOL_PROXY', 'environment')
    result = MeissaImageBackend().execute('SegmentRegionAroundPoint', png(), 'x=500 y=500')
    assert result['environment_failure'] and len(server['requests']) == 1


def test_runtime_proxy_override_and_resume(monkeypatch, tmp_path):
    import judges
    monkeypatch.setattr(judges, 'signature', lambda binding: {})
    task = live_task()
    before = run.resume_key(task['task_id'], 'workspace', {}, 'scripted', 10, task)
    config = tmp_path / 'runtime.json'
    config.write_text(json.dumps({'tool_proxy': 'environment'}))
    monkeypatch.setenv('MH_MEISSA_RUNTIME_CONFIG', str(config))
    assert settings()['tool_proxy'] == 'environment'
    assert before != run.resume_key(task['task_id'], 'workspace', {}, 'scripted', 10, task)
    config.write_text(json.dumps({'tool_proxy': 'typo'}))
    with pytest.raises(ValueError, match='tool_proxy'):
        settings()


@pytest.mark.parametrize('value', ['x=500 y=500', '[500,500]', [0,0,1000,1000],
                                    '[0,0,1001,1000]', '[500,0,100,1000]'])
def test_zoom_invalid_arguments_leave_image_unchanged(monkeypatch, value):
    monkeypatch.setattr(environments.UserStudyEnv, '_asset', staticmethod(lambda path: (png(), 'image/png')))
    env = environments.UserStudyEnv(live_task())
    env.reset()
    result = env.call_tool('ZoomInSubfigure', {'image': 'img_original', 'param': value})
    assert result['error_type'] == 'invalid_arguments' and result['environment_failure'] is False
    assert env.last_image == 'img_original' and len(env.images) == 1
    if value == 'x=500 y=500':
        assert '[250,250,750,750]' in result['error'] and 'SAM2 only' in result['error']


@pytest.mark.parametrize('bench', ['MeissaSFT', 'MeissaEval'])
@pytest.mark.parametrize('protocol', ['text', 'function_calling'])
def test_zoom_contract_reaches_agent(monkeypatch, bench, protocol):
    ws = W.Workspace(W.BENCH_H0[bench], protocol)
    seen = []
    def chat(messages, *args, **kwargs):
        seen.append((messages, args, kwargs))
        return {'content': 'Done.'} if protocol == 'function_calling' else '<answer>Done.</answer>'
    program = ws.program_class()(W.public_task(live_task(bench)), ws, chat)
    program.act({})
    rendered = json.dumps(seen)
    assert '[250,250,750,750]' in rendered
    assert 'SAM2 only' in rendered
    assert 'PRIVATE_REFERENCE' not in rendered
