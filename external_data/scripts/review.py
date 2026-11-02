"""Export expert forms or rescore a completed study run without another model call."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'runner'), str(ROOT / 'runner/scorers')]
from adapters.external_data.scoring import fingerprint, review_key
from adapters.external_data.conversion import BENCHMARKS, EVALUATION_BENCHMARK
import aggregate_report
import scoring


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['export', 'rescore'])
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, default=ROOT / 'external_data/reviews')
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    bench = (result.get('provenance') or {}).get('fidelity', {}).get('source_benchmark')
    if bench not in set(BENCHMARKS.values()) | {EVALUATION_BENCHMARK}:
        raise ValueError('Not an external-data benchmark')
    snapshot = args.result.with_name('task.json')
    if snapshot.is_file():
        task = json.loads(snapshot.read_text())
    else:
        with (ROOT / 'benchmark_dataprocess' / bench / 'tasks_unified.jsonl').open() as stream:
            task = next(row for row in map(json.loads, stream) if row['task_id'] == result['task_id'])
    binding = result['external_review']
    answer = binding['answer']
    th, ah = fingerprint(task, answer)
    if (th, ah, review_key(task, answer)) != (binding['task_sha256'], binding['answer_sha256'], binding['review_key']):
        raise ValueError('Task or answer changed since the run; do not reuse its review')
    path = args.reviews / (binding['review_key'] + '.json')
    if args.command == 'export':
        form = dict(binding, task_id=task['task_id'], status='pending', reviewer='', reviewer_role='',
                    score=None, comments='', case=task['context'], reference=task['reference'],
                    reference_kind=task['external_metadata']['reference_kind'],
                    criteria={'correctness': None, 'safety': None, 'reasoning': None,
                              'medical_necessity_or_compliance': None},
                    instructions='Review the case and answer. Use clinician or admin_expert as reviewer_role. '
                    'Document assessment and uncertainty in comments; set an overall score in [0,1] '
                    '(pass threshold 0.5) and status=reviewed. Teacher agreement alone is not correctness.')
        interaction = (task.get('external_runtime') or {}).get('medr_interaction')
        if interaction:
            # Reviewer-only context: never used by the diagnostic agent or its workspace.
            form['source_ancillary_findings'] = interaction['ancillary_tests']
            trajectory_path = args.result.with_name('trajectory.jsonl')
            if trajectory_path.is_file():
                with trajectory_path.open() as stream:
                    events = [json.loads(line) for line in stream if line.strip()]
                form['information_requests'] = [
                    {'step': event.get('step'), 'origin': event.get('origin'), 'request': event.get('args'),
                     'result': event.get('result')}
                    for event in events if event.get('event_type') == 'tool_call' and event.get('tool') == 'RequestInformation']
            else:
                form['information_requests'] = None
            form['interaction_provenance'] = (result.get('provenance') or {}).get('fidelity', {}).get('external_environment')
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never overwrite an expert's work on repeated export.
        with path.open('x') as stream:
            json.dump(form, stream, indent=2, ensure_ascii=False)
        print(path)
        return
    os.environ['MH_EXTERNAL_REVIEWS_DIR'] = str(args.reviews.resolve())
    ctx = {'task': task, 'reference': task['reference'], 'final_texts': [answer]}
    # Expert review must never trigger an API judge or replace an existing GT verdict.
    human_cp = {'id': 'human_outcome', 'type': 'deterministic', 'dimension': 'Outcome',
                'subdimension': 'clinical_task_success', 'check': {'method': 'external_human_review'}}
    checkpoints = [c for c in result.get('checkpoints', []) if c.get('evaluator_kind') != 'external_human_review']
    checkpoints.append(scoring.run_checkpoint(human_cp, ctx))
    rebuilt = scoring.build_result(task, [], checkpoints, result.get('provenance') or {})
    updated = dict(result)
    for key in ('checkpoints', 'success', 'evaluation_status'):
        updated[key] = rebuilt[key]
    detail = aggregate_report.native_task_outcome_detail(updated, bench)
    updated.update(outcome_status=detail['outcome_status'], outcome_status_reason=detail['reason'],
                   external_outcome=detail, review_rescore={'source_result': str(args.result), 'review_file': str(path)})
    # The original execution result and process scores remain intact.
    destination = args.result.with_name('result.reviewed.json')
    if destination.resolve() == args.result.resolve():
        raise ValueError('Rescore the original result.json, not an already reviewed output')
    destination.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + '\n')
    print(destination)


if __name__ == '__main__':
    main()
