"""Development rules for record_vitals, using private state + delivered pixels.

The known failed episode motivated this plugin. Fixtures/replays validate its
mechanics, not independent clinical or cross-benchmark construct validity.
"""
import base64
import hashlib
import re
from pathlib import Path
from process_evaluation import metric

FIELDS = ('sbp', 'dbp', 'tempf', 'hr', 'rr', 'spo2', 'gcs')
HISTORY = 'workspace/output/medcua_native_history.jsonl'


def requirements(task):
    applicable = task.get('native_runtime', {}).get('env_id') in (
        'medgym.emergency_triage.record_vitals.intent', 'medgym.emergency_triage.record_vitals.step')
    ref = task.get('reference', {})
    expected = ref.get('expected_values', {})
    return {'applicable': applicable, 'task_id': task['task_id'],
            'runtime': task.get('native_runtime', {}),
            'target_complaint': expected.get('target_complaint'),
            'required_fields': ['tempf' if k == 'temp' else k for k in expected.get('vitals', {})],
            'checker_supported': ref.get('checker') == 'check_record_vitals',
            'scope': 'Patient binding, saved draft, readback support; not medical correctness of entered values.'}


def extract(bundle, req):
    if not req['applicable']:
        return {}
    evidence = {}
    calls = [(i,e) for i,e in enumerate(bundle.trace) if e.get('event_type') == 'tool_call'
             and e.get('tool') == 'BrowserAction']
    history = bundle.read(HISTORY, lines=True, optional=True)
    if history is None or any(not r.get('process_state') for r in history):
        return {'availability': {'status': 'missing_private_state', 'source': HISTORY}}
    if len(history) != len(calls) + 1:
        raise ValueError('State history is incomplete')
    wire = bundle.read('model_requests.jsonl', lines=True, optional=True) or []
    responses = {r['request_id'] for r in wire if r.get('event') == 'response'}
    for index, row in enumerate(history):
        if row.get('task_id') != req['task_id'] or any(row.get(k) != v for k,v in req['runtime'].items()):
            raise ValueError('Private state belongs to a different task/runtime')
        snap = row['process_state']
        if snap.get('schema_version') != 'medcua-vitals-state-2' or snap.get('action_index') != index or not snap.get('state'):
            raise ValueError('Invalid state capture or action order')
        used = False
        agent_action = False
        if (snap.get('action_target') or {}).get('capture_error'):
            raise ValueError('Action target capture failed')
        if index:
            ti, event = calls[index-1]
            agent_action = event.get('origin', 'agent') == 'agent' and event.get('audience', 'agent') == 'agent'
            if snap['action'] != event['args']['action']:
                raise ValueError('State/action mismatch')
            images = event.get('result', {}).get('output', {}).get('images', [])
            image = next((im for im in images if im.get('sha256') == snap['screenshot_sha256']), None)
            if image is None:
                raise ValueError('State/screenshot mismatch')
            data = base64.b64decode(image['url'].split(',', 1)[1], validate=True)
            if hashlib.sha256(data).hexdigest() != snap['screenshot_sha256']:
                raise ValueError('Screenshot bytes changed')
            # Require both this observation's unique image ID and actual bytes
            # in an accepted request. A hash alone can match an earlier frame.
            used = any(r.get('event') == 'request' and r.get('request_id') in responses
                       and any(im.get('sha256') == snap['screenshot_sha256'] for im in r.get('images', []))
                       and image['image_id'] in str(r.get('body', {}).get('messages', [])) for r in wire)
        evidence['state:%d' % index] = dict(snap, source=HISTORY + '#L%d' % (index+1),
                                           consumed_by_model=used, agent_action=agent_action)
    finals = [(i,e) for i,e in enumerate(bundle.trace) if e.get('event_type') == 'final_answer']
    if finals:
        index, event = finals[-1]
        evidence['final'] = {'source': 'trajectory.jsonl#L%d' % (index+1),
                             'text': event.get('canonical_action', {}).get('content', event.get('thought', ''))}
    return evidence


def _same(recorded, draft, fields):
    return bool(recorded and draft and fields) and all(draft.get(f) is not None and recorded.get(f) == draft[f] for f in fields)


def evaluate(req, evidence):
    names = PLUGIN['metrics']
    if not req['applicable']:
        return {n: metric(None, False, 0, [], 'Task outside record_vitals scope') for n in names}
    if not req['checker_supported'] or not req['target_complaint'] or not req['required_fields']:
        return {n: metric(None, None, 0, [], 'Task requirements unavailable') for n in names}
    states = [(k,v) for k,v in evidence.items() if k.startswith('state:')]
    if not states:
        return {n: metric(None, True, 0, [], 'Private process snapshots unavailable; native reward cannot substitute') for n in names}
    fields = req['required_fields']
    if not set(fields) <= set(FIELDS):
        raise ValueError('Unsupported vitals fields')
    candidates = [p['id'] for p in states[0][1]['state']['patients'] if p['complaint'] == req['target_complaint']]
    if len(candidates) != 1:
        return {n: metric(None, True, 0, [], 'Target patient not uniquely identified') for n in names}
    target = candidates[0]
    final_id, final = states[-1]
    current = next((p for p in final['state']['patients'] if p['id'] == target), None)
    if current is None:
        raise ValueError('Target patient missing from final state')
    bindings = [(k,v) for k,v in states if v['state']['selected_patient'] is not None]
    binding_id, binding = bindings[-1] if bindings else states[-1]
    rows = {'patient_scope': metric(float(binding['state']['selected_patient'] == target), True, 1,
                                    [binding_id], 'Last observed patient binding compared with the task-defined subject')}
    saves, drafts, losses = [], [], []
    for (before_id,before),(after_id,after) in zip(states,states[1:]):
        b, a = before['state'], after['state']
        draft = b.get('draft')
        if b['selected_patient'] != target or not draft or not all(draft.get(f) is not None for f in fields):
            continue
        drafts.append(before_id)
        patient = next((p for p in a['patients'] if p['id'] == target), {})
        landed = _same(patient.get('recorded'), draft, fields)
        hit = after.get('action_target') or {}
        if after.get('agent_action') and hit.get('control') == 'save_vitals' and hit.get('subject_id') == target and landed:
            saves.append((before_id, after_id, dict(draft)))
        if a['selected_patient'] == target and not landed and not _same(a.get('draft'), draft, fields):
            # Changed/partially edited values are not lost drafts. Require all
            # required fields to have been cleared by the next observation.
            if not a.get('draft') or all(a['draft'].get(f) is None for f in fields):
                losses.extend([before_id, after_id])
    persisted = bool(saves and _same(current.get('recorded'), saves[-1][2], fields))
    refs = list(dict.fromkeys([states[0][0], final_id] + [r for s in saves for r in s[:2]]))
    rows['persisted_submission'] = metric(float(persisted), True, 1, refs,
        'Observed Save Vitals and final stored values match the submitted draft' if persisted else
        'Complete captured history contains no matching persisted submission at finalization')
    rows['draft_retention'] = metric(float(not losses) if drafts else None, bool(drafts), 1 if drafts else 0,
                                     list(dict.fromkeys(losses or drafts)),
                                     'Unsaved populated draft disappeared' if losses else 'No observed loss of a populated draft' if drafts else 'No populated draft opportunity')
    claim = evidence.get('final', {}).get('text', '')
    # Narrow supported assertion form, not a general-purpose clinical judge.
    positive = bool(re.search(r'\bvitals\s+(?:(?:were|are|have been|had been)\s+)?(?:recorded|saved|updated)\b', claim, re.I))
    ambiguous = bool(re.search(r'\b(not|never|cannot|unable|failed|unsure|uncertain|maybe|might|could)\b', claim, re.I))
    if not positive or ambiguous:
        rows['state_persistence_check'] = metric(None, True, 0, ['final'] if 'final' in evidence else [],
                                                 'No unambiguous supported completion assertion; language assessment unavailable')
    elif not current.get('recorded'):
        rows['state_persistence_check'] = metric(0., True, 1, [final_id, 'final'],
                                                 'Agent asserted recorded vitals; final captured target record has no saved vitals')
    else:
        readback = [k for k,v in states if saves and v['action_index'] >= int(saves[-1][1].split(':')[1])
                    and v['state']['selected_patient'] == target and v['state']['recorded_visible']
                    and v['consumed_by_model']]
        rows['state_persistence_check'] = metric(1. if persisted and readback else None, True,
                                                 1 if persisted and readback else 0,
                                                 [final_id, 'final'] + readback,
                                                 'Saved draft and visible readback support the assertion' if persisted and readback else
                                                 'Saved state alone does not establish agent verification; readback evidence insufficient')
    return rows


PLUGIN = {'id': 'medcua.record_vitals', 'version': '0.1.0-dev', 'benchmark': 'MedCUABench',
          'source_files': [__file__, Path(__file__).resolve().parents[2] / 'adapters/medcua/process_capture.py',
                           Path(__file__).resolve().parents[2] / 'adapters/medcua/worker.py'],
          'requirements': requirements, 'extract': extract, 'evaluate': evaluate,
          'metrics': {
              'patient_scope': {'kind': 'dimension', 'dimension': 'Context', 'definition': 'Last observed patient binding agrees with the task-defined subject.'},
              'persisted_submission': {'kind': 'dimension', 'dimension': 'Lifecycle', 'definition': 'A captured Save Vitals operation persisted the submitted draft through finalization.'},
              'state_persistence_check': {'kind': 'dimension', 'dimension': 'Verification', 'definition': 'A supported completion assertion has saved state and observed readback; a known false assertion fails.'},
              'draft_retention': {'kind': 'custom', 'definition': 'A populated unsaved draft was not lost between observations; diagnostic, not clinical correctness.'}}}
