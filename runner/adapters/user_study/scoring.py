"""Human outcomes bound to one exact task and answer; teacher agreement stays a proxy."""
import hashlib
import json
import math
import os
from pathlib import Path
import re


def fingerprint(task, answer):
    task_hash = hashlib.sha256(json.dumps(task, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    answer_hash = hashlib.sha256(answer.encode()).hexdigest()
    return task_hash, answer_hash


def review_key(task, answer):
    th, ah = fingerprint(task, answer)
    return hashlib.sha256((th + ':' + ah).encode()).hexdigest()


def reviewed_result(result, bundle):
    """Overlay expert outcomes on the process-score bundle, binding to its exact run input."""
    path = Path(bundle) / 'result.reviewed.json'
    if not result.get('study_review') or not path.is_file():
        return result
    reviewed = json.loads(path.read_text())
    if reviewed.get('task_id') != result.get('task_id') or reviewed.get('study_review') != result['study_review']:
        raise ValueError('Expert result belongs to another task or answer')
    merged = dict(result)
    bench = (result.get('provenance') or {}).get('fidelity', {}).get('source_benchmark')
    if bench in ('MedRBench', 'MeissaEval'):
        merged['expert_review'] = [c for c in reviewed.get('checkpoints', [])
                                   if c.get('evaluator_kind') == 'study_human_review']
        merged['review_rescore'] = reviewed.get('review_rescore')
        return merged
    for key in ('checkpoints', 'success', 'evaluation_status', 'outcome_status', 'outcome_status_reason',
                'study_outcome', 'review_rescore'):
        merged[key] = reviewed[key]
    return merged


def score(cp, ctx, base):
    answer = '\n'.join(ctx.get('final_texts') or [])
    task = ctx.get('task') or {}
    method = cp['check']['method']
    if any(text.strip() == 'API_BRAIN_ERROR' for text in (ctx.get('final_texts') or [])):
        return dict(base, checkpoint_status='error', failure_mode='environment_error', score_eligible=False,
                    evaluator_kind='study_human_review' if method == 'study_human_review' else 'study_reference_proxy',
                    detail={'error': 'agent_gateway_failure; no evaluable answer'})
    if method == 'study_reference_agreement':
        target = str((ctx.get('reference') or {}).get('gold_answer') or '')
        normalize = lambda s: re.sub(r'\s+', ' ', s).strip().casefold()
        if not target.strip() or not answer.strip():
            return dict(base, checkpoint_status='skipped', failure_mode=None, score_eligible=False,
                        skip_reason='missing_answer_or_reference', evaluator_kind='study_reference_proxy')
        equal = normalize(answer) == normalize(target)
        return dict(base, checkpoint_status='passed' if equal else 'failed', score=float(equal),
                    failure_mode=None, score_eligible=False, evaluator_kind='study_reference_proxy',
                    detail={'meaning': 'normalized exact reference agreement; not clinical correctness'})
    folder = os.environ.get('MH_STUDY_REVIEWS_DIR')
    path = Path(folder) / (review_key(task, answer) + '.json') if folder else None
    if path is None or not path.is_file():
        return dict(base, checkpoint_status='skipped', failure_mode=None, score_eligible=False,
                    skip_reason='human_review_pending', evaluator_kind='study_human_review')
    try:
        review = json.loads(path.read_text())
        if not isinstance(review, dict):
            raise ValueError('Review must be a JSON object')
        task_hash, answer_hash = fingerprint(task, answer)
        if (review.get('task_sha256') != task_hash or review.get('answer_sha256') != answer_hash
                or review.get('task_id') != task.get('task_id') or review.get('status') != 'reviewed'
                or not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()
                or review.get('reviewer_role') not in ('clinician', 'admin_expert')):
            raise ValueError('Review identity, author or status is invalid')
        value = review.get('score')
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Human score must be a finite number in [0,1]')
        passed = value >= 0.5
        return dict(base, checkpoint_status='passed' if passed else 'failed', score=value,
                    failure_mode=None if passed else 'agent_failure', score_eligible=True,
                    evaluator_kind='study_human_review', detail={'reviewer': review['reviewer'],
                    'reviewer_role': review['reviewer_role'], 'threshold': 0.5, 'review_key': path.stem})
    except (ValueError, OSError) as error:
        return dict(base, checkpoint_status='error', failure_mode='verifier_error', score_eligible=False,
                    evaluator_kind='study_human_review', detail={'error': str(error)})
