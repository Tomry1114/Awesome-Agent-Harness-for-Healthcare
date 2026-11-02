"""Pixel protocol, GT isolation, and native-result eligibility (no model API)."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'runner'), str(ROOT / 'runner/scorers')]
from adapters.medcua.worker import validate_action, Session
from adapters.medcua.scoring import score
from adapters.medcua.conversion import TOOL
from adapters.medcua import source
import workspace as W
import substrate


def task():
    return {'task_id': 'MEDCUA-fixture.intent', 'source_benchmark': 'MedCUABench',
            'domain': 'medical_external_data', 'modality': 'gui_web',
            'environment': {'type': 'medcua_native'}, 'goal': 'Use the visible browser.',
            'context': {'text': 'Use the visible browser.', 'action_reference': 'mouse_click(x, y)'},
            'available_tools': [TOOL], 'reference': {'expected_values': 'PRIVATE_GT'},
            'native_runtime': {'env_id': 'medgym.fixture.intent', 'seed': 42, 'max_steps': 30,
                               'source_tree_sha256': 'fixture'},
            'external_metadata': {'source_commit': 'fixture', 'scored': True}}


@pytest.mark.parametrize('action', ["page.locator('button').click()", "__import__('os')", "mouse_click(1+2, 3)",
    'mouse_click(1,2);mouse_click(3,4)', 'mouse_click(**{})', "goto('file:///etc/passwd')",
    "goto('javascript:alert(1)')", "mouse_upload_file(1,2,'private.json')"])
def test_pixel_action_boundary(action):
    with pytest.raises((ValueError, SyntaxError)):
        validate_action(action, ['mouse_click','goto','mouse_upload_file'])


def test_native_chat_and_scores_do_not_cross_observation():
    import numpy as np
    obs = {'screenshot': np.zeros((5,5,3),dtype=np.uint8), 'goal': 'Public goal',
           'last_action': '', 'last_action_error': '', 'chat_messages': [{'message': 'PRIVATE_GT'}]}
    packed = Session().pack(obs, {'task_info': {'expected_values': 'PRIVATE_GT'}}, reward=1)
    private = packed.pop('private')
    assert 'PRIVATE_GT' in json.dumps(private)
    assert 'PRIVATE_GT' not in json.dumps(packed)
    assert 'reward' not in json.dumps(packed)
    assert packed['output']['images'][0]['url'].startswith('data:image/png;base64,')


@pytest.mark.parametrize('protocol', ['function_calling', 'text'])
def test_workspace_images_and_terminal_boundary(protocol):
    t=task(); ws=W.Workspace(W.BENCH_H0['MedCUABench'], protocol)
    assert ws.check(['BrowserAction']) == []
    calls=[]
    def chat(messages, *args, **kwargs):
        calls.append(copy.deepcopy(messages))
        return {'content':'done'} if protocol=='function_calling' else '<answer>done</answer>'
    public=W.public_task(t)
    assert not {'reference','native_runtime','external_metadata'} & public.keys()
    program=ws.program_class()(public,ws,chat)
    for i in range(2):
        obs={'output':{'images':[{'image_id':'screen_'+str(i),'sha256':str(i),'url':'data:image/png;base64,fixture'+str(i)}]}}
        program.act({'last_result':obs,'turn_inputs':[{'kind':'initial_observation','result':obs}]})
    assert 'PRIVATE_GT' not in json.dumps(calls)
    assert 'mouse_click(x, y)' in calls[0][1]['content']
    assert 'fixture0' in json.dumps(calls[0]) and 'fixture1' in json.dumps(calls[1])
    action=program.act({'last_result':{'output':{'terminated':True}}})
    assert action['type']=='final' and len(calls)==2


@pytest.mark.parametrize('complete,reward,status', [(False,0,'failed'), (True,1,'passed'), (True,.7,'failed'), (False,-1,'failed')])
def test_native_scoring_and_identity(tmp_path,complete,reward,status):
    t=task(); folder=tmp_path/'workspace/output';folder.mkdir(parents=True)
    record={'task_id':t['task_id'],**t['native_runtime'],
            'native_info':{'task_completed':complete,'final_reward':reward,'safety_penalty':1-reward,'safety_dimensions':{}}}
    path=folder/'medcua_native.json';path.write_text(json.dumps(record))
    ctx={'job_dir':str(tmp_path),'task':t}
    result=score({},ctx,{})
    assert result['checkpoint_status']==status and result['score_eligible']
    assert result['detail']['native']['final_reward']==reward
    record['source_tree_sha256']='changed';path.write_text(json.dumps(record))
    assert score({},ctx,{})['failure_mode']=='verifier_error'
    (folder/'medcua_native_error.json').write_text('{}')
    assert score({},ctx,{})['failure_mode']=='environment_error'


def test_source_change_refused(monkeypatch,tmp_path):
    pin=tmp_path/'pin.json';pin.write_text(json.dumps({'tree_sha256':'pinned'}))
    monkeypatch.setattr(source,'PIN',pin)
    monkeypatch.setattr(source,'source_identity',lambda:{'tree_sha256':'modified'})
    with pytest.raises(ValueError,match='source changed'):
        source.verify_source()


def test_process_evidence_is_not_medical_ground_truth():
    from plugins.medcua import evidence
    row={'event_type':'tool_call','tool':'BrowserAction','result':{'output':{'text':'Image returned','images':[{'sha256':'x'}]}}}
    unit=evidence([row])[0]
    assert unit['clinical_quality_status']=='unassessed' and not unit['claim_support_eligible']
    assert 'MedicareAppeals' not in substrate.list_plugins()
    assert 'MedCUABench' in substrate.list_plugins()
