"""Frozen seven-dimensional scoring and fail-closed formal admission.

Evaluation is explicit and post-run. Report loading never invokes a Judge.
Legacy Core Scores and Outcome are not overwritten.
"""
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

from process_evaluation import DIMENSIONS, Bundle, digest, discover_profiles
from seven_specs import DEFINITIONS, SYSTEM, VERSION

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / 'external_data/seven_dim/rubric_lock.json'
ARTIFACT = 'seven_profile.json'
FILES = ('task.json', 'trajectory.jsonl', 'result.json', 'model_requests.jsonl',
         'workspace/output/medcua_native_history.jsonl')


class IncompleteEvaluation(ValueError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_manifest():
    files = [Path(__file__), ROOT/'runner/scorers/seven_specs.py', ROOT/'runner/scorers/process_evaluation.py',
             ROOT/'runner/scorers/dim_execution.py', ROOT/'runner/scorers/dim_observability.py',
             ROOT/'runner/scorers/dimension_core.py', ROOT/'runner/substrate.py',
             ROOT/'runner/scorers/dim_context.py', ROOT/'runner/run.py', ROOT/'runner/run_batch.py',
             ROOT/'runner/harness/__init__.py', ROOT/'runner/judges.py', ROOT/'runner/gateway.py',
             ROOT/'runner/adapters/medcua/process_capture.py', ROOT/'runner/adapters/medcua/environment.py']
    files += sorted((ROOT/'runner/scorers/process_plugins').glob('*.py'))
    files += sorted((ROOT/'runner/plugins').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def freeze(path=LOCK):
    import judges
    spec = judges.spec('process_profile')
    lock = {'version': VERSION, 'frozen_at': time.time(), 'profiles': discover_profiles(),
            'files': code_manifest(), 'judge': spec,
            'validation_scope': 'Engineering admission; not certification of clinical validity or model independence.'}
    lock['rule_sha256'] = digest(lock)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x') as stream:
        json.dump(lock, stream, indent=2)
    return lock


def read_lock(path=LOCK):
    lock = json.loads(Path(path).read_text())
    body = {k:v for k,v in lock.items() if k != 'rule_sha256'}
    if digest(body) != lock.get('rule_sha256') or lock.get('files') != code_manifest():
        raise IncompleteEvaluation('Frozen scoring code or rubric has changed')
    if lock.get('profiles') != discover_profiles():
        raise IncompleteEvaluation('Plugin contracts differ from the frozen registry')
    return lock


def run_binding(task, path=None):
    """Stamp before the first agent action. No score, GT or private evidence enters the agent."""
    path = Path(path or os.environ.get('MH_SEVEN_RUBRIC_LOCK', str(LOCK)))
    record = {'started_at': time.time(), 'task_sha256': digest(task), 'status': 'unregistered'}
    if not path.exists():
        return dict(record, status='rubric_not_frozen')
    try:
        lock = read_lock(path)
        return dict(record, status='frozen' if task.get('source_benchmark') in lock['profiles'] else 'unregistered',
                    rule_sha256=lock['rule_sha256'])
    except (ValueError, OSError) as error:
        return dict(record, status='invalid_freeze', error_type=type(error).__name__)


def bundle_hashes(folder):
    return {name: sha(Path(folder)/name) for name in FILES if (Path(folder)/name).is_file()}


def _walk_images(value):
    if isinstance(value, str) and value.startswith('data:image/'):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _walk_images(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk_images(v)


def evidence_packet(bundle):
    task = bundle.task
    bench = task.get('source_benchmark')
    kind = (task.get('external_metadata') or {}).get('task_kind')
    if ((bench == 'MedRBench' and kind not in ('diagnosis', 'treatment'))
            or (bench == 'MeissaEval' and kind != 'pathvqa')
            or (bench == 'MedCUABench' and 'record_vitals.' not in task.get('native_runtime', {}).get('env_id', ''))):
        raise IncompleteEvaluation('Task outside the registered profile scope')
    wire = bundle.read('model_requests.jsonl', lines=True, optional=True)
    if not wire:
        raise IncompleteEvaluation('Accepted model-request audit is required')
    accepted = {r.get('request_id') for r in wire if r.get('event') == 'response'}
    requests = [r for r in wire if r.get('event') == 'request' and r.get('request_id') in accepted]
    if not requests:
        raise IncompleteEvaluation('No accepted real model request')
    evidence = {'task': {'kind': 'task_requirement', 'goal': task.get('goal') or task.get('context', {}).get('text'),
                         'tools': task.get('available_tools'), 'source': 'task.json'}}
    images = {}
    observed = {}
    for r in requests:
        for image in r.get('images', []):
            if image.get('sha256'):
                observed[image['sha256']] = r.get('request_id')
        for url in _walk_images((r.get('body') or {}).get('messages', [])):
            raw = base64.b64decode(url.split(',', 1)[1], validate=True)
            h = hashlib.sha256(raw).hexdigest()
            images[h] = url
            observed[h] = r.get('request_id')
    for url in _walk_images(bundle.trace):
        raw = base64.b64decode(url.split(',', 1)[1], validate=True)
        h = hashlib.sha256(raw).hexdigest()
        if h in observed:
            images[h] = url
    for asset in (task.get('context') or {}).get('images', []):
        if not isinstance(asset,dict) or not asset.get('path'):
            continue
        path = (ROOT/asset['path']).resolve()
        if ROOT not in path.parents:
            raise IncompleteEvaluation('Task image path escapes the repository')
        raw = path.read_bytes();h = hashlib.sha256(raw).hexdigest()
        if h in observed:
            import mimetypes
            mime = mimetypes.guess_type(path.name)[0] or 'image/png'
            images[h] = 'data:'+mime+';base64,'+base64.b64encode(raw).decode()
    for h in set(observed)-set(images):
        if len(h)!=64 or any(c not in '0123456789abcdef' for c in h):
            raise IncompleteEvaluation('Invalid image hash in request audit')
        path=bundle.folder/'workspace/output/medcua_frames'/(h+'.png')
        if path.is_file():
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=h:
                raise IncompleteEvaluation('Cached screenshot bytes changed')
            images[h]='data:image/png;base64,'+base64.b64encode(raw).decode()
    if set(observed)-set(images):
        raise IncompleteEvaluation('Some agent-observed images cannot be recovered and hash-verified')
    for h in images:
        evidence['image:'+h]={'kind':'image','sha256':h,'agent_observed':True,
                              'source':'hash-verified source asset / trace / screenshot cache', 'request_id':observed[h]}
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k,v in value.items() if k not in ('_private', 'native_info', 'native_reward', 'reference', 'gold_answer')}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str) and value.startswith('data:image/'):
            raw = base64.b64decode(value.split(',', 1)[1], validate=True)
            h = hashlib.sha256(raw).hexdigest()
            # Returned but unconsumed pixels cannot be claimed as agent observations.
            return {'image_ref': 'image:'+h, 'agent_observed': h in images}
        return value
    evidence['initial_input'] = {'kind':'behavior','event_type':'initial_observation','source':'model_requests.jsonl',
                                 'request_id':requests[0].get('request_id'), 'agent_observed':True,
                                 'messages':clean([m for m in requests[0].get('body',{}).get('messages',[]) if m.get('role')!='system'])}
    finals = []
    for i,e in enumerate(bundle.trace):
        if e.get('event_type') not in ('tool_call', 'final_answer', 'shell_refusal', 'answer_attempt', 'harness_decision'):
            continue
        key = 'event:'+str(i)
        record = {k:e.get(k) for k in ('event_type','tool','args','result','status','origin','audience','thought','canonical_action') if k in e}
        evidence[key] = dict(clean(record), kind='behavior', source='trajectory.jsonl#L'+str(i+1))
        if e.get('event_type') == 'final_answer':
            finals.append(key)
    if not finals:
        raise IncompleteEvaluation('Missing final answer in the execution trace')
    if bench in ('MeissaEval','MedCUABench') and not images:
        raise IncompleteEvaluation('Visual process evaluation requires actual agent-observed pixels')
    if len(images) > 64 or sum(len(u) for u in images.values()) > 32_000_000:
        raise IncompleteEvaluation('Evidence exceeds the frozen multimodal evaluation budget; no silent truncation')
    deterministic = {}
    if bench == 'MedCUABench':
        from process_plugins import medcua_vitals as M
        req = M.requirements(task)
        private = M.extract(bundle, req)
        if not any(k.startswith('state:') for k in private):
            raise IncompleteEvaluation('Complete private GUI state capture is required for this task')
        deterministic = M.evaluate(req, private)
        # These are observed runtime states, never native rewards or expected answer values.
        for key, value in private.items():
            if key.startswith('state:'):
                evidence[key] = dict(value, kind='private_state', audience='evaluator_only')
            elif key == 'final':
                evidence[key] = dict(value, kind='behavior')
    return evidence, images, deterministic


def assess_rows(rows, profile, evidence):
    """Strict complete-criterion assessment; never average away an unscored obligation."""
    output = {}
    expected = {d for d,s in profile['dimensions'].items() if s['method'] == 'judge'}
    if not isinstance(rows, dict) or set(rows) != expected:
        raise IncompleteEvaluation('Judge must return all requested dimensions exactly once')
    for dim in expected:
        criteria = rows[dim].get('criteria') if isinstance(rows[dim], dict) else None
        obligations = profile['dimensions'][dim]['obligations']
        if not isinstance(criteria, dict) or set(criteria) != set(obligations):
            raise IncompleteEvaluation(dim + ': incomplete obligation set')
        normalized = {}
        for key, row in criteria.items():
            if not isinstance(row, dict) or set(row) != {'score','evidence','reason'}:
                raise IncompleteEvaluation(dim + ': invalid criterion schema')
            score, refs = row['score'], row['evidence']
            if score is not None and (type(score) not in (int,float) or not math.isfinite(score) or score not in (0,.5,1)):
                raise IncompleteEvaluation(dim + ': invalid score')
            if not isinstance(refs,list) or any(ref not in evidence for ref in refs):
                raise IncompleteEvaluation(dim + ': unknown evidence reference')
            if not isinstance(row['reason'],str) or not row['reason'].strip():
                raise IncompleteEvaluation(dim + ': reason missing')
            if score is not None and not any(evidence[ref].get('kind') in ('behavior','private_state','image') for ref in refs):
                raise IncompleteEvaluation(dim + ': numeric judgement requires execution evidence')
            normalized[key] = dict(row, obligation=obligations[key])
        values = [v['score'] for v in normalized.values()]
        complete = all(v is not None for v in values)
        refs = sorted({r for v in normalized.values() for r in v['evidence']})
        if profile['benchmark'] == 'MeissaEval' and dim in ('Context','Verification') and complete:
            if not any(evidence[r].get('kind') == 'image' for r in refs):
                raise IncompleteEvaluation(dim + ': visual judgement did not cite actual pixels')
        output[dim] = {'score': sum(values)/len(values) if complete else None,
                       'status': 'assessed' if complete else 'evaluation_incomplete',
                       'coverage': sum(v is not None for v in values)/len(values), 'method': 'harness_judge',
                       'evidence': refs, 'criteria': normalized, 'definition': DEFINITIONS[dim]}
    return output


def response_schema(requested):
    """Constrain syntax and requested fields; semantic validity is still checked locally."""
    def obj(properties):
        return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
    criterion = obj({'score':{'type':['number','null'],'enum':[0,0.5,1,None]},
                     'evidence':{'type':'array','items':{'type':'string'}},'reason':{'type':'string'}})
    return obj({'dimensions':obj({d:obj({'criteria':obj({k:criterion for k in spec['obligations']})})
                                  for d,spec in requested.items()})})


def validate_seven_dimensions(scores, evidence):
    problems = []
    if set(scores) != set(DIMENSIONS):
        problems.append('Exactly seven dimensions are required')
    for dim in DIMENSIONS:
        row = scores.get(dim) or {}
        n = row.get('score')
        if type(n) not in (int,float) or not math.isfinite(n) or not 0 <= n <= 1:
            problems.append(dim + ': missing valid assessment')
        if row.get('coverage') != 1 or row.get('status') != 'assessed':
            problems.append(dim + ': incomplete obligation coverage')
        refs = row.get('evidence')
        if not isinstance(refs,list) or not refs or any(r not in evidence for r in refs):
            problems.append(dim + ': missing or invalid evidence')
        if not row.get('criteria'):
            problems.append(dim + ': missing task obligations')
        else:
            values = []
            for criterion in row['criteria'].values():
                v = criterion.get('score')
                refs = criterion.get('evidence')
                if (type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= 1
                        or not criterion.get('obligation') or not criterion.get('reason')
                        or not isinstance(refs,list) or not refs or any(r not in evidence for r in refs)):
                    problems.append(dim + ': invalid or unscored criterion')
                else:
                    values.append(v)
            if len(values) == len(row['criteria']) and type(n) in (int,float) and abs(n-sum(values)/len(values)) > 1e-8:
                problems.append(dim + ': aggregate does not match criterion scores')
    if problems:
        raise IncompleteEvaluation('; '.join(problems))
    return True


def _core(bundle, profile, evidence):
    import substrate as S
    import dim_execution as E
    import dim_observability as O
    plugin, problem = S.require_plugin(bundle.task['source_benchmark'])
    if problem:
        raise IncompleteEvaluation(problem)
    result = bundle.read('result.json')
    sem = S.map_trace(bundle.trace, plugin)
    policy = S.dimension_policy(bundle.task, plugin)
    units = S.evidence_view(bundle.trace,plugin)
    if evidence.get('initial_input',{}).get('agent_observed'):
        units = [dict(id='initial_input',delivered_to_agent=True,delivery_fidelity=1.0,error_visible=False,
                      payload=json.dumps(evidence['initial_input']),actor='agent')] + units
    raw = {'Execution': E.execution(sem, policy, S.capability_manifest(result.get('provenance',{}))),
           'Observability': O.observability(units,sem,contract=plugin.get('observability_contract'))}
    refs = [k for k,v in evidence.items() if v.get('kind') == 'behavior']
    out = {}
    for dim, value in raw.items():
        score = value.get('score')
        valid = type(score) in (int,float) and value.get('reportable',True) and bool(refs)
        key, obligation = next(iter(profile['dimensions'][dim]['obligations'].items()))
        out[dim] = {'score': score if valid else None, 'coverage': 1 if valid else 0,
                    'status': 'assessed' if valid else 'evaluation_incomplete', 'method': profile['dimensions'][dim]['scorer'],
                    'definition': DEFINITIONS[dim], 'evidence': refs, 'criteria': {key:{'score':score if valid else None,
                    'obligation':obligation,'reason':'Registered deterministic scorer; raw submetrics retained','evidence':refs}}, 'raw':value}
    return out


def admission(profile, lock, result, task):
    reasons = []
    try:
        validate_seven_dimensions(profile.get('dimensions',{}),profile.get('evidence',{}))
        contract = lock.get('profiles',{}).get(task.get('source_benchmark'))
        if not contract:
            raise IncompleteEvaluation('Unregistered benchmark')
        for dim,spec in contract['dimensions'].items():
            if set(profile['dimensions'][dim]['criteria']) != set(spec['obligations']):
                raise IncompleteEvaluation(dim + ': obligations differ from frozen contract')
        complete = True
    except IncompleteEvaluation as error:
        complete = False
        reasons.append(str(error))
    record = result.get('seven_dimension_run') or {}
    if (record.get('status') != 'frozen' or record.get('rule_sha256') != lock['rule_sha256']
            or record.get('task_sha256') != digest(task) or record.get('started_at',0) < lock['frozen_at']):
        reasons.append('Execution was not bound to this frozen rubric before the agent started')
    prov = result.get('provenance') or {}
    if prov.get('scorer_validation_only') or prov.get('uses_hidden_reference'):
        reasons.append('Scripted, replay or hidden-reference execution cannot enter formal agent comparisons')
    if result.get('end_reason') in ('harness_error','missing_result','capability_excluded'):
        reasons.append('Execution did not produce a valid episode')
    return {'seven_dim_complete': complete, 'formal_analysis_eligible': complete and not reasons,
            'status':'admitted' if complete and not reasons else 'evaluation_incomplete', 'reasons':reasons,
            'scope':'Complete engineering assessment, not independent clinical validation'}


def evaluate(folder, lock_path=LOCK):
    """Explicit paid evaluation; all failures are persisted as incomplete, not zero."""
    folder = Path(folder)
    if (folder/ARTIFACT).exists():
        raise FileExistsError('Existing seven-dimensional assessment must be archived explicitly')
    bundle = Bundle(folder)
    result = bundle.read('result.json')
    output = {'version':VERSION,'task_id':bundle.task['task_id'],'benchmark':bundle.task.get('source_benchmark'),
              'created_at':time.time(),'input_sha256':bundle_hashes(folder),'dimensions':{},'evidence':{},'errors':[],
              'historical_core_preserved':True,'outcome_modified':False}
    lock = None
    try:
        lock = read_lock(lock_path)
        output['rule_sha256'] = lock['rule_sha256']
        profile = lock['profiles'].get(output['benchmark'])
        if profile is None:
            raise IncompleteEvaluation('Benchmark has no complete registered seven-dimensional profile')
        output['plugin'] = profile
        evidence, images, deterministic = evidence_packet(bundle)
        output['evidence'] = evidence
        output['dimensions'].update(_core(bundle,profile,evidence))
        import harness
        judge, spec = harness.build_process_judge()
        if spec != lock['judge']:
            raise IncompleteEvaluation('Judge configuration differs from the frozen rubric')
        output['judge'] = spec
        output['judge_independence'] = ('shared_model_with_agent' if spec['model'] == (result.get('provenance') or {}).get('api_model_string')
                                         else 'not_established')
        requested = {d:s for d,s in profile['dimensions'].items() if s['method']=='judge'}
        text = json.dumps({'definitions':{d:DEFINITIONS[d] for d in requested},'requirements':requested,'evidence':evidence},ensure_ascii=False)
        content = [{'type':'text','text':text}]
        for h,url in sorted(images.items()):
            content.extend([{'type':'text','text':'Evidence image:'+h}, {'type':'image_url','image_url':{'url':url}}])
        messages = [{'role':'system','content':SYSTEM},{'role':'user','content':content}]
        schema = response_schema(requested)
        audit = {'spec':spec,'messages':messages,'response_schema':schema,
                 'request_sha256':digest({'messages':messages,'response_schema':schema})}
        # Exclusive write: every raw attempt remains reviewable, even failed evaluations.
        with (folder/'seven_judge_request.json').open('x') as stream:
            json.dump(audit,stream,ensure_ascii=False)
        reply = judge(messages, schema=schema)
        response = {'ok':reply.get('ok'),'content':reply.get('content'),'error_type':reply.get('error_type')}
        (folder/'seven_judge_response.json').write_text(json.dumps(response,ensure_ascii=False,indent=2))
        output['judge_artifacts'] = {n:sha(folder/n) for n in ('seven_judge_request.json','seven_judge_response.json')}
        if not reply.get('ok'):
            raise IncompleteEvaluation('Process Judge gateway failed: '+str(reply.get('error_type')))
        parsed = json.loads(reply.get('content') or '')
        judged = assess_rows(parsed.get('dimensions'),profile,evidence)
        # A captured persisted-save check can settle these obligations directly.
        for name,dim,criterion in [('persisted_submission','Lifecycle','completion'),
                                   ('state_persistence_check','Verification','support')]:
            row = deterministic.get(name,{})
            if row.get('score') is not None:
                judged[dim]['criteria'][criterion].update(score=row['score'],evidence=row['evidence'],reason=row['reason'],
                                                          method='medcua_private_state_check')
                values = [r['score'] for r in judged[dim]['criteria'].values()]
                judged[dim].update(score=sum(values)/len(values) if all(v is not None for v in values) else None,
                                   coverage=sum(v is not None for v in values)/len(values))
                judged[dim]['evidence'] = sorted({r for v in judged[dim]['criteria'].values() for r in v['evidence']})
                judged[dim]['status'] = 'assessed' if judged[dim]['coverage']==1 else 'evaluation_incomplete'
                judged[dim]['method'] = 'harness_judge_and_deterministic_state'
        output['dimensions'].update(judged)
    except Exception as error:
        # Framework/HTTP exceptions can contain credentials: retain only safe contract failures and type.
        output['errors'].append({'type':type(error).__name__, 'reason':str(error)[:800] if isinstance(error,IncompleteEvaluation)
                                  else 'Process evaluation failed; see raw judge artifacts if available'})
    for dim in DIMENSIONS:
        output['dimensions'].setdefault(dim, {'score':None,'coverage':0,'status':'evaluation_incomplete',
                                             'evidence':[],'criteria':{},'definition':DEFINITIONS[dim]})
    gate = admission(output,lock,result,bundle.task) if lock else {
        'seven_dim_complete':False,'formal_analysis_eligible':False,'status':'evaluation_incomplete','reasons':['No valid frozen rubric']}
    if output['errors']:
        gate.update(formal_analysis_eligible=False,status='evaluation_incomplete')
        gate['reasons'].extend(e['reason'] for e in output['errors'])
    output['admission'] = gate
    output['assessment_sha256'] = digest(output)
    with (folder/ARTIFACT).open('x') as stream:
        json.dump(output,stream,indent=2,ensure_ascii=False)
    return output


def load_profile(folder, lock_path=LOCK):
    """Revalidate every formal read, including hashes, frozen rules and gate; no model calls."""
    folder = Path(folder)
    try:
        p = json.loads((folder/ARTIFACT).read_text())
        if p.get('assessment_sha256') != digest({k:v for k,v in p.items() if k != 'assessment_sha256'}):
            raise IncompleteEvaluation('Saved assessment changed')
        lock = read_lock(lock_path)
        task = json.loads((folder/'task.json').read_text())
        if p.get('input_sha256') != bundle_hashes(folder) or p.get('task_id') != task.get('task_id'):
            raise IncompleteEvaluation('Evaluation input changed')
        if p.get('rule_sha256') != lock['rule_sha256'] or p.get('plugin') != lock['profiles'].get(task.get('source_benchmark')):
            raise IncompleteEvaluation('Evaluation rubric differs from frozen registry')
        for name,h in p.get('judge_artifacts',{}).items():
            if name not in ('seven_judge_request.json','seven_judge_response.json') or sha(folder/name) != h:
                raise IncompleteEvaluation('Judge artifact changed')
        gate = admission(p,lock,json.loads((folder/'result.json').read_text()),task)
        if p.get('errors'):
            gate.update(formal_analysis_eligible=False,status='evaluation_incomplete')
        return dict(p,admission=gate)
    except (ValueError,OSError,KeyError) as error:
        return {'task_id':folder.name,'admission':{'seven_dim_complete':False,'formal_analysis_eligible':False,
                'status':'evaluation_incomplete','reasons':[str(error) if isinstance(error,IncompleteEvaluation)
                                                          else 'Missing or invalid saved seven-dimensional assessment']}}


def aggregate(agent_dir, bench, task_ids=None, lock_path=LOCK):
    root = Path(agent_dir)
    ids = sorted(set(task_ids)) if task_ids is not None else sorted(p.parent.name for p in root.glob('*/result.json'))
    tasks = []
    for tid in ids:
        if (root/tid).resolve().parent != root.resolve():
            raise ValueError('Task path escapes agent directory')
        p = load_profile(root/tid,lock_path)
        if p.get('benchmark') not in (None,bench):
            p['admission'].update(formal_analysis_eligible=False,status='evaluation_incomplete')
        tasks.append(p)
    admitted = [p for p in tasks if p['admission']['formal_analysis_eligible']]
    return {'contract_version':VERSION,'tasks':tasks,'n_tasks':len(ids),'n_admitted':len(admitted),
            'n_evaluation_incomplete':len(ids)-len(admitted),
            'admission_coverage':len(admitted)/len(ids) if ids else 0,
            'formal_dimensions':{d:{'score':sum(p['dimensions'][d]['score'] for p in admitted)/len(admitted)
                                      if admitted else None,'n_tasks':len(admitted)} for d in DIMENSIONS},
            'note':'Formal means include only complete frozen seven-dimensional assessments. Historical core and Outcome remain separate.'}
