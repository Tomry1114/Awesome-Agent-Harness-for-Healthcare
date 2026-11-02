"""Seven-dimension contract/admission, frozen evidence, and existing Judge-route integration."""
import copy
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'runner'),str(ROOT/'runner/scorers')]
import seven_profile as S
import seven_specs as R
import process_evaluation as P
import harness
import gateway
from adapters.user_study import conversion as C


def verdict(profile,evidence,score=1):
    ref=next(k for k,v in evidence.items() if v.get('kind')=='behavior')
    return {d:{'criteria':{k:{'score':score,'evidence':[ref],'reason':'Fixture observed behavior'} for k in spec['obligations']}}
            for d,spec in profile['dimensions'].items() if spec['method']=='judge'}


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    monkeypatch.setenv('MH_HARNESS_JUDGE_MODEL','fixture-judge')
    lock=tmp_path/'lock.json';S.freeze(lock)
    folder=tmp_path/'agent'/'MEDR-fixture';folder.mkdir(parents=True)
    task=C.task_base('MedRBench','MEDR-fixture','Assess the given fever case.','HIDDEN_GT_NEVER_SEND',kind='diagnosis',group='fixture')
    trace=[{'event_type':'tool_call','tool':'ReadCase','args':{},'status':'ok','origin':'agent','audience':'agent',
            'result':{'output':{'text':'Patient has fever.'}}},
           {'event_type':'final_answer','thought':'Fever is present; its cause remains uncertain.','origin':'agent'}]
    result={'task_id':task['task_id'],'provenance':{'api_model_string':'fixture-judge','scorer_validation_only':False},
            'seven_dimension_run':S.run_binding(task,lock)}
    for name,data in [('task.json',task),('result.json',result)]:
        (folder/name).write_text(json.dumps(data))
    (folder/'trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))
    wire=[{'event':'request','request_id':'1','body':{'messages':[{'role':'user','content':'Patient has fever.'}]}},
          {'event':'response','request_id':'1'}]
    (folder/'model_requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in wire))
    return folder,lock


def test_registry_requires_all_seven():
    profiles=P.discover_profiles()
    assert set(profiles)=={'MedRBench','MeissaEval','MedCUABench'}
    wrong=copy.deepcopy(profiles['MedRBench']);wrong['dimensions'].pop('Context')
    with pytest.raises(ValueError):P.register_profile({},wrong)
    assert profiles['MeissaEval']['dimensions']['Verification']['method']=='judge'


def test_pixel_hash_is_not_a_patient():
    from plugins import medcua
    from dim_context import _subject_from_unit
    trace=[{'event_type':'tool_call','tool':'BrowserAction','result':{'output':{'images':[{'sha256':str(i)}]}}} for i in range(26)]
    ev=medcua.evidence(trace)
    assert len({e['frame_id'] for e in ev})==26
    assert all(_subject_from_unit(e) is None for e in ev)
    assert _subject_from_unit({'source_channel':'browser_screenshot','source_instance_id':'frame',
                              'subject_token':'subject:Patient/P1'})==('subject:Patient','P1')


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),-1,2,'1'])
def test_reject_invalid_judge_scores(value):
    profile=R.profile('MedRBench');ev={'e':{'kind':'behavior'}};rows=verdict(profile,ev)
    rows['Context']['criteria']['use']['score']=value
    with pytest.raises(S.IncompleteEvaluation):S.assess_rows(rows,profile,ev)


def test_partial_coverage_cannot_be_a_complete_dimension():
    p=R.profile('MedRBench');ev={'e':{'kind':'behavior'}};rows=verdict(p,ev)
    rows['Verification']['criteria']['support']['score']=None
    scored=S.assess_rows(rows,p,ev)
    assert scored['Verification']['score'] is None and scored['Verification']['coverage']==.5
    with pytest.raises(S.IncompleteEvaluation):S.validate_seven_dimensions(scored,ev)


def test_visual_judgement_requires_pixels():
    p=R.profile('MeissaEval');ev={'e':{'kind':'behavior'}}
    with pytest.raises(S.IncompleteEvaluation):S.assess_rows(verdict(p,ev),p,ev)
    ev['img']={'kind':'image'};rows=verdict(p,ev)
    for d in ('Context','Verification'):rows[d]['criteria'][next(iter(rows[d]['criteria']))]['evidence'].append('img')
    assert S.assess_rows(rows,p,ev)['Verification']['score']==1


def test_success_admission_and_zero_is_valid(bundle,monkeypatch):
    folder,lock=bundle;before=S.bundle_hashes(folder)
    p=R.profile('MedRBench');ev,_,_=S.evidence_packet(P.Bundle(folder))
    rows=verdict(p,ev,0)
    calls=[]
    def call(messages,**kwargs):
        calls.append((messages,kwargs))
        return {'ok':True,'content':json.dumps({'dimensions':rows})}
    monkeypatch.setattr(gateway,'chat',call)
    output=S.evaluate(folder,lock)
    assert not output['errors'],output['errors']
    assert output['admission']['formal_analysis_eligible'],output['admission']
    assert output['dimensions']['Verification']['score']==0
    assert len(calls)==1 and calls[0][1]['judge'] is True
    assert calls[0][1]['model']=='fixture-judge'
    schema=calls[0][1]['extra']['response_format']['json_schema']
    assert schema['strict'] is True
    assert set(schema['schema']['properties']['dimensions']['properties'])==set(rows)
    assert 'HIDDEN_GT_NEVER_SEND' not in json.dumps(calls)
    assert S.bundle_hashes(folder)==before
    monkeypatch.setattr(gateway,'chat',lambda *a,**k:pytest.fail('Formal report must be pure-read'))
    report=S.aggregate(folder.parent,'MedRBench',lock_path=lock)
    assert report['n_admitted']==1 and report['formal_dimensions']['Verification']['score']==0
    with pytest.raises(FileExistsError):S.evaluate(folder,lock)
    output['dimensions']['Context']['score']=1
    (folder/S.ARTIFACT).write_text(json.dumps(output))
    assert not S.load_profile(folder,lock)['admission']['formal_analysis_eligible']


@pytest.mark.parametrize('failure',['gateway','malformed','missing_evidence','old_run','scripted','changed_code','missing_profile'])
def test_fail_closed_without_fabricated_zero(bundle,monkeypatch,failure):
    folder,lock=bundle
    p=R.profile('MedRBench');ev,_,_=S.evidence_packet(P.Bundle(folder));rows=verdict(p,ev)
    if failure=='missing_evidence':(folder/'model_requests.jsonl').unlink()
    if failure in ('old_run','scripted'):
        result=json.loads((folder/'result.json').read_text())
        if failure=='old_run':result.pop('seven_dimension_run')
        else:result['provenance']['scorer_validation_only']=True
        (folder/'result.json').write_text(json.dumps(result))
    if failure=='changed_code':monkeypatch.setattr(S,'code_manifest',lambda:{'changed':'sha'})
    if failure=='missing_profile':
        task=json.loads((folder/'task.json').read_text());task['source_benchmark']='Unknown'
        (folder/'task.json').write_text(json.dumps(task))
    monkeypatch.setattr(gateway,'chat',lambda *a,**k:{'ok':failure!='gateway','content':'invalid' if failure=='malformed' else json.dumps({'dimensions':rows}),'error_type':'timeout'})
    result=S.evaluate(folder,lock)
    assert not result['admission']['formal_analysis_eligible']
    if failure not in ('old_run','scripted'):
        assert result['dimensions']['Verification']['score'] is None
    assert S.aggregate(folder.parent,'MedRBench',lock_path=lock)['n_admitted']==0


def test_stale_inputs_and_incomplete_formal_denominator(bundle,monkeypatch):
    folder,lock=bundle;p=R.profile('MedRBench');ev,_,_=S.evidence_packet(P.Bundle(folder))
    monkeypatch.setattr(gateway,'chat',lambda *a,**k:{'ok':True,'content':json.dumps({'dimensions':verdict(p,ev)})})
    S.evaluate(folder,lock)
    (folder/'trajectory.jsonl').write_text((folder/'trajectory.jsonl').read_text()+'{}\n')
    report=S.aggregate(folder.parent,'MedRBench',lock_path=lock)
    assert report['n_tasks']==1 and report['n_admitted']==0 and report['n_evaluation_incomplete']==1
    assert all(v['score'] is None for v in report['formal_dimensions'].values())


def test_redacted_initial_image_is_recovered_without_forcing_tools(bundle,monkeypatch,tmp_path):
    import hashlib
    from PIL import Image
    folder,_=bundle
    asset=tmp_path/'image.png';Image.new('RGB',(8,8),'red').save(asset)
    h=hashlib.sha256(asset.read_bytes()).hexdigest()
    task=C.task_base('MeissaEval','MEISSAEVAL-fixture','What is visible?','PRIVATE_GT',kind='pathvqa',group='fixture',
                     images=[{'path':'image.png','sha256':h}])
    (folder/'task.json').write_text(json.dumps(task))
    trace=[{'event_type':'final_answer','thought':'A red field; no anatomy can be established.','origin':'agent'}]
    (folder/'trajectory.jsonl').write_text(json.dumps(trace[0])+'\n')
    wire=[{'event':'request','request_id':'1','images':[{'sha256':h}], 'body':{'messages':[
           {'role':'user','content':[{'type':'image_url','image_url':{'url':{'image_data_omitted':{'sha256':h}}}}]}]}},
          {'event':'response','request_id':'1'}]
    (folder/'model_requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in wire))
    monkeypatch.setattr(S,'ROOT',tmp_path)
    ev,images,_=S.evidence_packet(P.Bundle(folder))
    assert h in images and ev['image:'+h]['agent_observed']
    assert 'PRIVATE_GT' not in json.dumps(ev)
    core=S._core(P.Bundle(folder),R.profile('MeissaEval'),ev)
    assert core['Execution']['score'] is not None and core['Observability']['score'] is not None
    asset.write_bytes(b'changed image')
    with pytest.raises(S.IncompleteEvaluation):S.evidence_packet(P.Bundle(folder))
