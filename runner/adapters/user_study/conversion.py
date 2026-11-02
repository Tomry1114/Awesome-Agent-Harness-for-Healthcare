"""Convert public source data without exposing teacher answers or adjudications."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from .tool_contracts import ZOOM_DESCRIPTION, ZOOM_PARAM

ROOT = Path(__file__).resolve().parents[3]
SOURCES = ROOT / 'user_study'
BENCHMARKS = {'medr': 'MedRBench', 'meissa': 'MeissaSFT'}
EVALUATION_BENCHMARK = 'MeissaEval'


def tool(name, properties, description):
    return {'name': name, 'description': description, 'provenance': 'converted',
            'parameters': {'type': 'object', 'properties': properties, 'required': list(properties),
                           'additionalProperties': False}}


STRING = {'type': 'string'}
TOOLS = {
    'ReadCase': tool('ReadCase', {}, 'Read the supplied case facts; this does not reveal a reference answer.'),
    'RequestInformation': tool('RequestInformation', {'request': {'type': 'string', 'minLength': 1}},
                               'Request additional diagnostic information from the MedR case service. '
                               'Group all tests needed this round in one request. Only source case findings are available.'),
    'ReadImage': tool('ReadImage', {'image': STRING}, 'View an already available image by ID.'),
    'ZoomInSubfigure': tool('ZoomInSubfigure', {'image': STRING, 'param': ZOOM_PARAM}, ZOOM_DESCRIPTION),
    'RequestPhysicalExam': tool('RequestPhysicalExam', {'exam': STRING},
                               'Retrieve a recorded examination. Unrecorded findings are unavailable.'),
    'RequestTest': tool('RequestTest', {'test': STRING},
                       'Retrieve a recorded test result. Unrecorded results are unavailable.'),
    'BioMedParseTextSeg': tool('BioMedParseTextSeg', {'image': STRING, 'param': STRING},
                             'Retrieve a recorded segmentation for exactly this image and request; no live model.'),
    'SegmentRegionAroundPoint': tool('SegmentRegionAroundPoint', {'image': STRING, 'param': STRING},
                                    'Retrieve a recorded point segmentation for exactly this image and request.'),
}


def dump_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    count = 0
    with tmp.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    tmp.replace(path)
    return count


def final_answer(value):
    value = re.sub(r'<think>.*?</think>', '', str(value), flags=re.S).strip()
    return value.rsplit('[FINAL]', 1)[-1].strip()


def task_base(bench, tid, text, answer, *, kind, group, images=None, names=None):
    task = {
        'task_id': tid, 'source_benchmark': bench, 'domain': 'medical_user_study',
        'modality': 'image_text' if images else 'text',
        'environment': {'type': 'study_case', 'adapter': 'study_case', 'config': {}},
        'context': {'text': text, 'images': images or [], 'case_id': tid},
        'goal': text, 'available_tools': [TOOLS[n] for n in (names or ['ReadCase'])],
        'checkpoints': [
            {'id': 'human_outcome', 'type': 'deterministic', 'dimension': 'Outcome',
             'subdimension': 'clinical_task_success', 'provenance': 'augmented',
             'measurement_family': 'source_outcome', 'check': {'method': 'study_human_review'}},
            {'id': 'reference_agreement', 'type': 'deterministic', 'dimension': 'Outcome',
             'subdimension': 'reference_agreement', 'provenance': 'augmented',
             'measurement_family': 'proxy', 'evidence_tier': 'proxy',
             'check': {'method': 'study_reference_agreement'}},
        ],
        'scoring': {'mode': 'all_pass'}, 'expected_outcome': {'answer_kind': 'final_text'},
        'reference': {'gold_answer': answer},
        'study_metadata': {'task_kind': kind, 'source_group': group, 'outcome_review': 'pending',
                           'held_out_test': False},
        'study_runtime': {'initial_images': images or [], 'recorded_observations': []},
    }
    from .outcome_judge import effective_checkpoints
    task['checkpoints'] = effective_checkpoints(bench, task['checkpoints'])
    if bench in ('MedRBench', 'MeissaEval'):
        task['study_metadata']['outcome_review'] = 'gt_judge; expert_review_supplemental'
    return task


def split_medr_summary(summary):
    """Match the upstream split, including findings on the Ancillary Tests heading line."""
    lines = summary.strip().split('\n')
    index = next((i for i, line in enumerate(lines) if 'Ancillary Tests' in line), None)
    if index is None or not '\n'.join(lines[:index]).strip():
        raise ValueError('MedR interactive case lacks a separable Ancillary Tests section')
    return '\n'.join(lines[:index]), '\n'.join(lines[index:])


def medr_interactive_task(case_id, case, protocol, source=SOURCES / 'MedR-Bench'):
    if protocol not in ('one_turn', 'free_turn'):
        raise ValueError('Unknown MedR interaction protocol')
    initial, ancillary = split_medr_summary(case['case_summary'])
    directory = source / 'src/Inference/instructions'
    filename = '1turn_prompt_examination_recommend.txt' if protocol == 'one_turn' else 'free_turn_first_turn_prompt.txt'
    upstream_prompt = (directory / filename).read_text()
    # Keep the clinical task instruction; replace upstream textual control sections with native tools/final.
    prompt = upstream_prompt.split('Always following the response format', 1)[0].strip().format(case=initial)
    limit = 1 if protocol == 'one_turn' else 4
    prompt += ('\n\nUse RequestInformation to request additional diagnostic information. '
               'You may make at most %d information request(s); each may list several tests. '
               'After the last permitted response, give your final diagnosis with a concise evidence-based rationale. '
               'You may finish earlier if the initial evidence is sufficient. '
               'Missing test information is not a negative finding.' % limit)
    task = task_base('MedRBench', 'MEDR-%s-%s' % (protocol, case_id), prompt, case['diagnosis_results'],
                     kind='diagnosis', group='MedRBench:' + case_id, names=['ReadCase', 'RequestInformation'])
    task['context']['interaction'] = {'protocol': protocol, 'max_information_requests': limit,
                                      'max_diagnostic_turns': limit + 1}
    template = (directory / 'patient_agent_prompt.txt').read_text()
    task['study_runtime']['medr_interaction'] = {
        'protocol': protocol, 'initial_case': initial, 'ancillary_tests': ancillary,
        'max_information_requests': limit, 'patient_prompt_template': template,
    }
    task['study_metadata'].update(
        source_file='diagnosis_957_cases_with_rare_disease_491.json', source_id=case_id,
        protocol=protocol, reference_kind='published_case_target',
        patient_prompt_sha256=hashlib.sha256(template.encode()).hexdigest(),
        upstream_initial_prompt_sha256=hashlib.sha256(upstream_prompt.encode()).hexdigest(),
        deviations=['Upstream case split and patient-service prompt retained; native tool calls replace textual request sections.',
                    'One/four successful information responses; harness repairs and retries can add agent turns.',
                    'Early final answers are allowed; no mandatory query when evidence suffices.',
                    'GT Judge outcome: upstream diagnosis rubric with configured Judge; expert review supplemental.'])
    return task


def medr_tasks(source=SOURCES / 'MedR-Bench', protocol='all'):
    if protocol not in ('all', 'oracle', 'one_turn', 'free_turn'):
        raise ValueError('Unknown MedR protocol')
    for kind, filename, target in (
        ('diagnosis', 'diagnosis_957_cases_with_rare_disease_491.json', 'diagnosis_results'),
        ('treatment', 'treatment_496_cases_with_rare_disease_165.json', 'treatment_plan_results'),
    ):
        rows = json.loads((source / 'data/MedRBench' / filename).read_text())
        for case_id, row in rows.items():
            case = row['generate_case']
            instruction = ('Provide a diagnosis, supporting evidence and relevant uncertainty.' if kind == 'diagnosis'
                           else 'Propose a treatment plan, supporting evidence and relevant safety considerations.')
            task = task_base('MedRBench', 'MEDR-%s-%s' % (kind, case_id),
                             case['case_summary'] + '\n\n' + instruction, case[target],
                             kind=kind, group='MedRBench:' + case_id)
            task['study_metadata'].update(source_file=filename, source_id=case_id,
                                           protocol='oracle_case_summary', reference_kind='published_case_target',
                                           deviations=['Oracle summary input; no simulated examination acquisition.'])
            if protocol in ('all', 'oracle'):
                yield task
            if kind == 'diagnosis':
                for mode in ('one_turn', 'free_turn'):
                    if protocol in ('all', mode):
                        yield medr_interactive_task(case_id, case, mode, source)


def parse_call(value):
    value = re.sub(r'<think>.*?</think>', '', value, flags=re.S).strip()
    start = value.find('{')
    if start < 0:
        return None
    try:
        call, _ = json.JSONDecoder().raw_decode(value[start:])
    except ValueError:
        return None
    if isinstance(call, dict) and isinstance(call.get('arguments'), dict) and isinstance(call.get('name'), str):
        return call
    return None


def meissa_task(row, index, source=SOURCES / 'Meissa-SFT'):
    meta = row.get('meta') or {}
    turns = row['conversations']
    if not turns or turns[0].get('from') != 'human':
        raise ValueError('Meissa row %d does not start with a human prompt' % index)
    paths = row.get('images') or []
    # Image slots are consumed in conversation order, not all shown at episode start.
    cursor = turns[0]['value'].count('<image>')
    if cursor > len(paths):
        raise ValueError('Missing initial images in row %d' % index)
    initial = [{'asset_id': 'img_original' if i == 0 else 'img_input_%d' % i,
                'path': str((source / path).relative_to(ROOT))} for i, path in enumerate(paths[:cursor])]
    records, pending, warnings = [], None, []
    known_images = {x['asset_id']: x['path'] for x in initial}
    last_image = initial[-1]['path'] if initial else None
    for turn in turns[1:]:
        if turn['from'] == 'function_call':
            pending = parse_call(turn['value'])
            if pending is None:
                warnings.append('unparsed_function_call')
        elif turn['from'] == 'observation':
            n = turn['value'].count('<image>')
            output_paths = paths[cursor:cursor + n]
            cursor += n
            output_images = []
            named = re.findall(r'\[Output Image ID:\s*([^\]]+)\]', turn['value'])
            for j, path in enumerate(output_paths):
                image_id = named[j].strip() if j < len(named) else 'recorded_%d' % cursor
                output_images.append({'asset_id': image_id, 'path': str((source / path).relative_to(ROOT))})
            if pending and pending['name'] in TOOLS:
                image_arg = pending['arguments'].get('image', 'img_last')
                input_path = last_image if image_arg == 'img_last' else known_images.get(image_arg)
                records.append({'tool': pending['name'], 'args': pending['arguments'],
                                'input_image_path': input_path,
                                'text': turn['value'].replace('<image>', '').strip(), 'images': output_images})
            for item in output_images:
                known_images[item['asset_id']] = item['path']
                last_image = item['path']
            pending = None
    if cursor != len(paths):
        warnings.append('unconsumed_image_slots:%d' % (len(paths) - cursor))
    gold = next((final_answer(t['value']) for t in reversed(turns) if t['from'] == 'gpt'), '')
    if not gold:
        raise ValueError('No teacher answer in Meissa row %d' % index)
    names = ['ReadCase']
    if initial:
        names += ['ReadImage', 'ZoomInSubfigure']
    names += sorted({r['tool'] for r in records} - set(names))
    group = str(meta.get('sample_id') or meta.get('idx') or meta.get('scenario_id') or index)
    task = task_base('MeissaSFT', 'MEISSA-%05d' % index, turns[0]['value'].replace('<image>', '').strip(), gold,
                     kind=meta.get('type') or meta.get('framework'),
                     group='MeissaSFT:%s:%s' % (meta.get('dataset'), group), images=initial, names=names)
    task['study_runtime']['recorded_observations'] = records
    task['study_metadata'].update(source_row=index, source_meta=meta, warnings=warnings,
                                  data_role='training_trajectory', independent_evaluation=False,
                                  reference_kind='teacher_response_not_independent_ground_truth',
                                  protocol='fresh_agent_on_sft_prompt_with_frozen_observations',
                                  deviations=['Zoom is a live pixel crop; segmentation/exam/test results are frozen exact-request lookups.',
                                              'This replay adapter only serves recorded requests; the official Meissa repository separately provides live environments.',
                                              'Terminate is converted to the harness final-answer action.'])
    # Publish the factual acquisition menu, never results, teacher plans or generated images.
    menu = {name: sorted({str(r['args'].get(key, '')) for r in records if r['tool'] == name})
            for name, key in [('RequestTest', 'test'), ('RequestPhysicalExam', 'exam')] if name in names}
    if menu:
        task['context']['available_recorded_requests'] = menu
    return task


def enable_live_image(task):
    """Select a backend explicitly; never use frozen observations as its fallback."""
    task['study_runtime']['meissa_image'] = {'protocol': 'live_image_v1'}
    task['study_runtime']['recorded_observations'] = []
    for image in task['study_runtime']['initial_images']:
        path = ROOT / image['path']
        if path.is_file():
            image['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    task['available_tools'] = [dict(TOOLS[name]) for name in (
        'ReadCase', 'ReadImage', 'ZoomInSubfigure', 'SegmentRegionAroundPoint', 'BioMedParseTextSeg')]
    for definition in task['available_tools']:
        if definition['name'] in ('SegmentRegionAroundPoint', 'BioMedParseTextSeg'):
            definition['description'] = 'Run the configured live segmentation worker on this available image; unavailable without a backend.'
            if definition['name'] == 'SegmentRegionAroundPoint':
                definition['description'] += ' param must be a string such as "x=500 y=500" (0–1000 image coordinates), not a JSON coordinate array.'
            else:
                definition['description'] += ' param is a text region description, such as "tumor"; multiple descriptions may be separated with semicolons.'
        elif definition['name'] == 'ZoomInSubfigure':
            definition['description'] = ZOOM_DESCRIPTION
    task['study_metadata'].update(protocol='meissa_live_image',
        deviations=['BBox-only local zoom with upstream padding/JPEG semantics; masks and absolute coordinates unsupported.',
                    'Live segmentation requires separately deployed SAM2/BioMedParse workers.',
                    'Tool use is optional; native harness actions replace upstream textual action parsing.',
                    'GT semantic Judge and seven process metrics; not the published PathVQA matching metric.'])
    return task


def meissa_tasks(source=SOURCES / 'Meissa-SFT', image_mode='replay'):
    with (source / 'data/train.jsonl').open() as stream:
        for index, line in enumerate(stream):
            task = meissa_task(json.loads(line), index, source)
            if image_mode == 'live' and task['study_metadata']['source_meta'].get('framework') == 'interleaved_thinking_images':
                task = enable_live_image(task)
            yield task


def meissa_evaluation_tasks(manifest):
    """Import a locally sourced PathVQA test manifest, separately from SFT.

    Split/source attestations come from the manifest; no contamination-free claim.
    Source answers stay private, and outcome assessment still requires expert review.
    """
    manifest = Path(manifest)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    seen = set()
    with manifest.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ('id', 'question', 'answer', 'image', 'source_url'):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError('Missing PathVQA manifest field: ' + field)
            if row.get('dataset') != 'PathVQA' or row.get('split') != 'test':
                raise ValueError('MeissaEval currently accepts only explicitly sourced PathVQA test records')
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', row['id']) or row['id'] in seen:
                raise ValueError('Invalid or duplicate evaluation ID')
            seen.add(row['id'])
            image_path = (SOURCES / 'Meissa-Evaluation/images' / row['image']).resolve()
            image_path.relative_to((SOURCES / 'Meissa-Evaluation/images').resolve())
            if not image_path.is_file():
                raise ValueError('Missing evaluation image')
            data_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
            images = [{'asset_id': 'img_original', 'path': str(image_path.relative_to(ROOT)), 'sha256': data_sha}]
            task = task_base(EVALUATION_BENCHMARK, 'MEISSAEVAL-pathvqa-' + row['id'],
                             row['question'], row['answer'], kind='pathvqa',
                             group='PathVQA:image:' + data_sha, images=images)
            task['study_metadata'].update(data_role='evaluation_source', independent_evaluation=None,
                held_out_test=False, declared_source_split='test',
                split_verification='manifest_declared; contamination_not_verified',
                source_url=row['source_url'], source_id=row['id'], source_manifest_sha256=digest,
                source_image_sha256=data_sha, reference_kind='dataset_reference_answer')
            yield enable_live_image(task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all', 'meissa-eval', 'medcua'] + list(BENCHMARKS), default='all')
    parser.add_argument('--meissa-image-mode', choices=['replay', 'live'], default='replay',
                        help='For explicitly selected SFT data only; live does not make training rows independent tests')
    parser.add_argument('--meissa-eval-manifest', type=Path,
                        help='PathVQA test JSONL with id/question/answer/image/dataset/split/source_url')
    parser.add_argument('--medr-protocol', choices=['all', 'oracle', 'one_turn', 'free_turn'], default='all',
                        help='MedR task variants to prepare; all preserves Oracle and adds both interactive modes')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'benchmark_dataprocess')
    args = parser.parse_args()
    if args.dataset == 'meissa-eval':
        if not args.meissa_eval_manifest:
            parser.error('--dataset meissa-eval requires --meissa-eval-manifest')
        # Validate completely before replacing a previously prepared evaluation set.
        tasks = list(meissa_evaluation_tasks(args.meissa_eval_manifest))
        out = args.output_root / EVALUATION_BENCHMARK
        count = dump_jsonl(out / 'tasks_unified.jsonl', tasks)
        report = {'benchmark': EVALUATION_BENCHMARK, 'tasks': count, 'data_role': 'evaluation_source',
                  'protocol': 'meissa_live_image', 'split_verification': 'manifest_declared; contamination_not_verified',
                  'evaluation': 'gt_semantic_judge; expert_review_supplemental; original PathVQA scorer not invoked'}
        (out / 'conversion_report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
        return {EVALUATION_BENCHMARK: report}
    summary = {}
    if args.dataset in ('all', 'medcua'):
        from adapters.medcua.conversion import prepare
        summary['MedCUABench'] = prepare(args.output_root)
    for key, bench in BENCHMARKS.items():
        if args.dataset == 'all' and key == 'meissa':
            print('Meissa-SFT is training data; preserved separately. Use --dataset meissa explicitly for replay/development.')
            continue
        if args.dataset not in ('all', key):
            continue
        out = args.output_root / bench
        tasks = medr_tasks(protocol=args.medr_protocol) if key == 'medr' else meissa_tasks(image_mode=args.meissa_image_mode)
        groups, kinds, protocols = set(), Counter(), Counter()
        def tracked(rows):
            for task in rows:
                groups.add(task['study_metadata']['source_group'])
                kinds[task['study_metadata']['task_kind']] += 1
                protocols[task['study_metadata']['protocol']] += 1
                yield task
        count = dump_jsonl(out / 'tasks_unified.jsonl', tracked(tasks))
        report = {'benchmark': bench, 'tasks': count, 'unique_source_groups': len(groups), 'task_kinds': dict(kinds),
                  'protocols': dict(protocols),
                  'evaluation': 'human_review_required; reference agreement is diagnostic only', 'held_out_test': False}
        if key == 'meissa':
            report.update(data_role='training_trajectory', independent_evaluation=False)
        (out / 'conversion_report.json').write_text(json.dumps(report, indent=2) + '\n')
        summary[bench] = report
        print(json.dumps(report), flush=True)
    return summary


if __name__ == '__main__':
    main()
