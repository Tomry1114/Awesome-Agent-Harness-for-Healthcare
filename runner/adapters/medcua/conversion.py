"""Convert registered native tasks without moving expected values into public goals."""
import json
import subprocess
from .source import ROOT, WORKER, python, runtime_env, verify_source

TOOL = {'name': 'BrowserAction', 'provenance': 'converted',
        'description': 'Execute one official MedCUA screenshot-based browser action. Use coordinates/keyboard, never DOM or element IDs. Submit answers with send_msg_to_user(text).',
        'parameters': {'type': 'object', 'properties': {'action': {'type': 'string', 'minLength': 1}},
                       'required': ['action'], 'additionalProperties': False}}


def prepare(output_root=None):
    pin = verify_source()
    catalog = json.loads(subprocess.check_output([python(), str(WORKER), '--catalog'], env=runtime_env(), text=True))
    rows = []
    for entry in catalog['tasks']:
        goal = entry['goal']
        rows.append({'task_id': 'MEDCUA-' + entry['env_id'][len('medgym.'):] ,
                     'source_benchmark': 'MedCUABench', 'domain': 'medical_external_data', 'modality': 'gui_web',
                     'environment': {'type': 'medcua_native', 'adapter': 'medcua_native', 'config': {}},
                     'goal': goal, 'context': {'text': goal, 'action_reference': catalog['action_description']},
                     'available_tools': [TOOL],
                     'reference': {'checker': entry['checker'], 'expected_values': entry['expected_values']},
                     'checkpoints': [{'id': 'native_strict_success', 'type': 'deterministic',
                                      'dimension': 'Outcome', 'subdimension': 'native_strict_success',
                                      'provenance': 'native', 'measurement_family': 'source_outcome',
                                      'check': {'method': 'medcua_native'}}],
                     'scoring': {'mode': 'all_pass'}, 'expected_outcome': {'answer_kind': 'browser_state_and_chat'},
                     'external_metadata': {**{k: entry[k] for k in ('scenario', 'setting', 'backend', 'difficulty', 'scored')},
                                        'source_group': entry['base_task_id'], 'protocol': 'screenshot_only',
                                        'source_commit': pin['commit'], 'source_tree_sha256': pin['tree_sha256'],
                                        'deviations': ['Native checker feedback chat and rewards are hidden from the agent.',
                                                       'One literal action per tool call; local-file navigation and file uploads disabled.',
                                                       'Final prose is not submitted to the native checker; use send_msg_to_user.']},
                     'native_runtime': {'env_id': entry['env_id'], 'seed': 42, 'max_steps': 30,
                                        'source_tree_sha256': pin['tree_sha256']}})
    if len(rows) != 432 or len({x['external_metadata']['source_group'] for x in rows}) != 216:
        raise ValueError('Unexpected MedCUA release task universe')
    out = (output_root or ROOT / 'benchmark_dataprocess') / 'MedCUABench'
    out.mkdir(parents=True, exist_ok=True)
    temp = out / 'tasks_unified.jsonl.tmp'
    temp.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    temp.replace(out / 'tasks_unified.jsonl')
    report = {'benchmark': 'MedCUABench', 'base_tasks': 216, 'task_variants': len(rows),
              'settings': ['intent', 'step'], 'source_commit': pin['commit'], 'source_tree_sha256': pin['tree_sha256'],
              'native_checker': True, 'native_safety_dimensions': 5, 'native_safety_is_harness_seven': False,
              'backend_tasks': {b: sum(x['external_metadata']['backend'] == b for x in rows) for b in ('html', 'openemr', 'ohif')},
              'readiness': 'HTML requires installed Chromium; OpenEMR/OHIF require additional services. Imported does not mean agent-evaluated.'}
    (out / 'conversion_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return report
