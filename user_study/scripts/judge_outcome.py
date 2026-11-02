"""Judge an existing final answer against its saved GT; never rerun the agent or seven dimensions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'runner'), str(ROOT / 'runner/scorers')]
import aggregate_report
import scoring
from adapters.user_study import outcome_judge as J
from adapters.user_study.scoring import fingerprint


def rescore(bundle):
    bundle = Path(bundle)
    source = bundle / 'result.rescored.json'
    if not source.exists():
        source = bundle / 'result.json'
    result = json.loads(source.read_text())
    task = json.loads((bundle / 'task.json').read_text())
    if task.get('source_benchmark') not in J.BENCHMARKS:
        raise ValueError('Only MedRBench and MeissaEval have GT outcome protocols')
    binding = result['study_review']
    answer = binding['answer']
    th, ah = fingerprint(task, answer)
    if (th, ah) != (binding['task_sha256'], binding['answer_sha256']) or task['task_id'] != result['task_id']:
        raise ValueError('Saved task or answer changed since execution')
    if (bundle / 'result.outcome.json').exists():
        raise FileExistsError('A judgement already exists; archive it explicitly before rejudging')
    import judges
    judges.spec('study_gt_outcome')  # Same explicit Judge model binding as a normal run.
    cp = scoring.run_checkpoint(J.checkpoint(), {'task': task, 'reference': task['reference'], 'final_texts': [answer]})
    # Use normal serialization, including error_class and all evaluator provenance.
    cp = scoring.build_result(task, [], [cp], {})['checkpoints'][0]
    updated = J.merge_checkpoint(result, cp)
    updated['outcome_rescore'] = {'source_result': str(source), 'task_sha256': th, 'answer_sha256': ah,
                                 'created_at': datetime.now(timezone.utc).isoformat(), 'version': J.VERSION,
                                 'agent_rerun': False, 'process_scores_recomputed': False}
    output = bundle / 'result.outcome.json'
    with output.open('x') as stream:
        json.dump(updated, stream, indent=2, ensure_ascii=False)
    return output, updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--report', type=Path, help='Write a report using persisted judgements; no additional API calls')
    args = parser.parse_args()
    path, result = rescore(args.bundle)
    if args.report:
        bench = result['provenance']['fidelity']['source_benchmark']
        report = aggregate_report.build(str(args.bundle.parent), bench, task_ids=[result['task_id']])
        with args.report.open('x') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps({'result': str(path), 'outcome': result['study_outcome']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
