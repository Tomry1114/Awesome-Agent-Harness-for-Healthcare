"""Process scoring contracts and counterfactuals; synthetic fixtures, no model API."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'runner'), str(ROOT/'runner/scorers')]
import process_evaluation as P
from process_plugins import medcua_vitals as M


def make_bundle(tmp_path, saved=True, readback=True, claim='Updated vitals were recorded.'):
    task = {'task_id':'fixture', 'source_benchmark':'MedCUABench',
            'native_runtime':{'env_id':'medgym.emergency_triage.record_vitals.intent','seed':42},
            'reference':{'checker':'check_record_vitals','expected_values':{'target_complaint':'abdominal_pain',
                         'vitals':{'sbp':118,'dbp':78,'temp':38.6,'hr':102,'rr':20,'spo2':97,'gcs':15}}}}
    fields = dict(zip(M.FIELDS, [118,78,101.5,102,20,97,15]))
    trace, history, wire = [], [], []
    folder=tmp_path/'fixture'; (folder/'workspace/output').mkdir(parents=True)
    for i in range(3):
        stream=io.BytesIO(); Image.new('RGB',(4,4),(i*70,0,0)).save(stream,format='PNG'); raw=stream.getvalue()
        sha=hashlib.sha256(raw).hexdigest()
        action='mouse_click(10,20)' if i==2 else "keyboard_type('fixture')" if i else ''
        patient={'id':'patient-A','complaint':'abdominal_pain','recorded':fields if i==2 and saved else None,'pain':6}
        state={'logged_in':bool(i),'selected_patient':'patient-A' if i else None,
               'draft':fields if i==1 else dict.fromkeys(fields), 'recorded_visible':bool(i==2 and readback and saved),
               'patients':[patient]}
        snap={'schema_version':'medcua-vitals-state-2','action_index':i,'action':action,
              'action_target':{'control':'save_vitals','subject_id':'patient-A'} if i==2 and saved else None,
              'screenshot_sha256':sha,'state':state}
        history.append({'task_id':'fixture',**task['native_runtime'],'process_state':snap,
                        'native_info':{'task_completed':saved,'final_reward':1 if saved else 0}})
        if i:
            image={'image_id':'screen_'+str(i),'sha256':sha,'url':'data:image/png;base64,'+base64.b64encode(raw).decode()}
            trace.append({'event_type':'tool_call','tool':'BrowserAction','args':{'action':action},
                          'result':{'output':{'images':[image]}}})
            wire += [{'event':'request','request_id':str(i),'images':[{'sha256':sha}],
                      'body':{'messages':[{'role':'tool','content':image['image_id']}]}},
                     {'event':'response','request_id':str(i)}]
    trace.append({'event_type':'final_answer','thought':claim})
    (folder/'task.json').write_text(json.dumps(task))
    (folder/'result.json').write_text('{}')
    for name, rows in [('trajectory.jsonl',trace),('model_requests.jsonl',wire),(M.HISTORY,history)]:
        (folder/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return folder


def rows(folder):
    output=P.evaluate_bundle(folder)
    assert not output['plugins'][0]['errors'], output
    return {r['metric']:r for r in output['plugins'][0]['metrics']}


def test_success_and_failure_are_evidence_based(tmp_path):
    good=rows(make_bundle(tmp_path/'good'))
    bad=rows(make_bundle(tmp_path/'bad',saved=False))
    assert good['persisted_submission']['score']==1 and good['state_persistence_check']['score']==1
    assert bad['persisted_submission']['score']==0 and bad['state_persistence_check']['score']==0
    assert bad['draft_retention']['score']==0
    assert good['patient_scope']['score']==bad['patient_scope']['score']==1
    assert all(not r['formal_analysis_eligible'] for r in good.values())


def test_native_reward_and_expected_values_do_not_define_process_score(tmp_path):
    folder=make_bundle(tmp_path,saved=False)
    before=rows(folder)
    path=folder/M.HISTORY
    records=[json.loads(s) for s in path.read_text().splitlines()]
    for r in records:r['native_info']={'task_completed':True,'final_reward':1,'PRIVATE_GT':'secret'}
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    task=json.loads((folder/'task.json').read_text());task['reference']['expected_values']['vitals']['sbp']=999
    (folder/'task.json').write_text(json.dumps(task))
    assert rows(folder)==before
    evidence=P.evaluate_bundle(folder)['plugins'][0]['evidence']
    assert 'PRIVATE_GT' not in json.dumps(evidence)


@pytest.mark.parametrize('claim', ['Vitals were not recorded.', 'I am unsure whether vitals were recorded.',
                                  'Task finished.', '无法确认已保存。'])
def test_ambiguous_or_unhandled_language_is_not_a_zero(tmp_path,claim):
    score=rows(make_bundle(tmp_path,saved=False,claim=claim))['state_persistence_check']
    assert score['score'] is None and score['coverage']==0


def test_saved_state_without_readback_is_not_verification(tmp_path):
    score=rows(make_bundle(tmp_path,readback=False))
    assert score['persisted_submission']['score']==1
    assert score['state_persistence_check']['score'] is None


def test_missing_state_stays_na_even_with_native_outcome(tmp_path):
    folder=make_bundle(tmp_path)
    (folder/M.HISTORY).unlink()
    assert all(r['score'] is None and r['coverage']==0 for r in rows(folder).values())


@pytest.mark.parametrize('damage',['task_identity','state_order','image_bytes','missing_frame'])
def test_invalid_evidence_fails_closed(tmp_path,damage):
    folder=make_bundle(tmp_path)
    path=folder/M.HISTORY; records=[json.loads(s) for s in path.read_text().splitlines()]
    if damage=='task_identity':records[-1]['task_id']='other'
    if damage=='state_order':records[-1]['process_state']['action_index']=90
    if damage=='missing_frame':records.pop(1)
    if damage=='image_bytes':
        p=folder/'trajectory.jsonl';r=[json.loads(s) for s in p.read_text().splitlines()]
        r[0]['result']['output']['images'][0]['url']='data:image/png;base64,AAAA'
        p.write_text(''.join(json.dumps(x)+'\n' for x in r))
    path.write_text(''.join(json.dumps(x)+'\n' for x in records))
    out=P.evaluate_bundle(folder)['plugins'][0]
    assert out['errors'] and all(r['score'] is None for r in out['metrics'])


def test_new_benchmark_custom_metric_and_validation(tmp_path):
    folder=make_bundle(tmp_path)
    plugin={'id':'example.custom','version':'1','benchmark':'MedCUABench','source_files':[__file__],
            'requirements':lambda task:{'required':True},
            'extract':lambda bundle,req:{'e1':{'source':'trajectory.jsonl#L1'}},
            'evaluate':lambda req,ev:{'localization':P.metric(.5,True,1,['e1'],'Fixture metric')},
            'metrics':{'localization':{'kind':'custom','definition':'Synthetic localization measure'}}}
    registry={};P.register(registry,plugin)
    out=P.evaluate_bundle(folder,registry)['plugins'][0]
    assert out['metrics'][0]['score']==.5 and 'dimension' not in out['metrics'][0]
    with pytest.raises(ValueError,match='Duplicate'):P.register(registry,plugin)
    broken=copy.deepcopy(plugin);broken['metrics']['localization']['dimension']='Verification'
    with pytest.raises(ValueError,match='Custom'):P.register({},broken)
    plugin['evaluate']=lambda req,ev:{'localization':P.metric(0,True,1,['missing'],'Bad citation')}
    out=P.evaluate_bundle(folder,registry)['plugins'][0]
    assert out['errors'] and out['metrics'][0]['score'] is None


def test_versioned_aggregation_counts_missing_tasks(tmp_path):
    folder=make_bundle(tmp_path,saved=False)
    out=P.aggregate(tmp_path,'MedCUABench',task_ids=['fixture','missing'])
    v=next(r for r in out['dimension_metrics'] if r['metric']=='state_persistence_check')
    assert v['score']==0 and v['coverage']==.5 and v['n_unknown']==1
    assert len(out['custom_metrics'])==1 and out['custom_metrics'][0]['metric']=='draft_retention'
    assert out['validity']['errors'] and not out['validity']['core_scores_changed']
    assert all(r['metric']!='draft_retention' for r in out['dimension_metrics'])


def test_runtime_agent_boundary_and_readonly_files(tmp_path):
    from adapters.medcua.environment import MedCUAEnv
    import workspace
    folder=make_bundle(tmp_path)
    task=json.loads((folder/'task.json').read_text())
    env=MedCUAEnv(task,workspace=tmp_path/'private');env.workspace.mkdir()
    response={'output':{'text':'public'},'private':{'process_state':{'state':'SECRET_STATE'},'native_info':{'GT':'SECRET_GT'}}}
    public=env._save(response)
    assert 'SECRET' not in json.dumps(public) and 'process_state' not in json.dumps(env.state_summary())
    assert 'reference' not in workspace.public_task(task)
    files={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}
    P.evaluate_bundle(folder)
    assert files=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with pytest.raises(ValueError,match='escapes'):P.Bundle(folder).read('../outside.json',optional=True)


def test_drop_in_module_discovery(tmp_path, monkeypatch):
    import process_plugins
    module = tmp_path/'example_extension.py'
    module.write_text('''from process_evaluation import metric
PLUGIN = {'id':'example.dropin','version':'1','benchmark':'ExampleBench',
 'source_files':[__file__],
 'metrics':{'custom_accuracy':{'kind':'custom','definition':'Test extension'}},
 'requirements':lambda task:{},
 'extract':lambda bundle,req:{'observed':{'source':'trajectory.jsonl#L1'}},
 'evaluate':lambda req,ev:{'custom_accuracy':metric(1,True,1,['observed'],'Fixture')}}
''')
    monkeypatch.setattr(process_plugins, '__path__', list(process_plugins.__path__)+[str(tmp_path)])
    try:
        registry=P.discover()
        folder=make_bundle(tmp_path/'data')
        task=json.loads((folder/'task.json').read_text());task['source_benchmark']='ExampleBench'
        (folder/'task.json').write_text(json.dumps(task))
        result=P.evaluate_bundle(folder,registry)
        assert result['plugins'][0]['plugin']['id']=='example.dropin'
        assert result['plugins'][0]['metrics'][0]['score']==1
    finally:
        sys.modules.pop('process_plugins.example_extension',None)


def test_harness_save_is_not_agent_submission_credit(tmp_path):
    folder=make_bundle(tmp_path)
    p=folder/'trajectory.jsonl';trace=[json.loads(x) for x in p.read_text().splitlines()]
    trace[1]['origin']='recovery'
    p.write_text(''.join(json.dumps(x)+'\n' for x in trace))
    result=rows(folder)
    assert result['persisted_submission']['score']==0
    assert result['state_persistence_check']['score'] is None


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,2,True])
def test_invalid_numeric_scores_are_rejected(value):
    with pytest.raises(ValueError):
        P._validate(P.metric(value,True,1,['a'],'test'),{'a':{}})


def test_closing_panel_does_not_erase_patient_binding(tmp_path):
    folder=make_bundle(tmp_path)
    p=folder/M.HISTORY; records=[json.loads(s) for s in p.read_text().splitlines()]
    records[-1]['process_state']['state']['selected_patient']=None
    p.write_text(''.join(json.dumps(r)+'\n' for r in records))
    assert rows(folder)['patient_scope']['score']==1


def test_correctly_verified_draft_can_have_wrong_native_answer(tmp_path):
    folder=make_bundle(tmp_path)
    p=folder/M.HISTORY; records=[json.loads(s) for s in p.read_text().splitlines()]
    for record in records:record['native_info']={'task_completed':False,'final_reward':0}
    p.write_text(''.join(json.dumps(r)+'\n' for r in records))
    task=json.loads((folder/'task.json').read_text());task['reference']['expected_values']['vitals']['sbp']=140
    (folder/'task.json').write_text(json.dumps(task))
    result=rows(folder)
    assert result['state_persistence_check']['score']==1
    assert result['persisted_submission']['score']==1


def test_wrong_patient_binding_is_detected(tmp_path):
    folder=make_bundle(tmp_path)
    p=folder/M.HISTORY; records=[json.loads(s) for s in p.read_text().splitlines()]
    records[-1]['process_state']['state']['selected_patient']='patient-B'
    p.write_text(''.join(json.dumps(r)+'\n' for r in records))
    assert rows(folder)['patient_scope']['score']==0
