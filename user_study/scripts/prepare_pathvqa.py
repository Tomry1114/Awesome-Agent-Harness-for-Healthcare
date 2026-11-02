"""Extract pinned PathVQA records and audit image/question overlap before selection."""
import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import sys
import unicodedata

from PIL import Image, ImageOps
import pyarrow.parquet as pq
from hub_download import sha256

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'user_study/Meissa-Evaluation'
SFT = ROOT / 'user_study/Meissa-SFT'


def normalized(text):
    text = unicodedata.normalize('NFKC', text.replace('<image>', ''))
    return re.sub(r'\W+', ' ', text.casefold()).strip()


def image_identity(data):
    with Image.open(io.BytesIO(data)) as image:
        fmt = image.format
        image = ImageOps.exif_transpose(image).convert('RGB')
        width, height = image.size
        pixels = hashlib.sha256(('%d,%d:' % image.size).encode() + image.tobytes()).hexdigest()
        gray = image.convert('L')
        horizontal = list(gray.resize((9, 8), Image.Resampling.LANCZOS).getdata())
        vertical = list(gray.resize((8, 9), Image.Resampling.LANCZOS).getdata())
        bits = [horizontal[y * 9 + x] > horizontal[y * 9 + x + 1] for y in range(8) for x in range(8)]
        bits += [vertical[y * 8 + x] > vertical[(y + 1) * 8 + x] for y in range(8) for x in range(8)]
        dhash = sum(int(bit) << i for i, bit in enumerate(bits))
    return {'sha256': hashlib.sha256(data).hexdigest(), 'rgb_sha256': pixels,
            'dhash128': '%032x' % dhash, 'width': width, 'height': height, 'format': fmt}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def main():
    manifest = json.loads((DEST / 'source/download_manifest.json').read_text())
    assert manifest['repo'] == 'flaviagiammarino/path-vqa' and manifest['kind'] == 'dataset'
    revision = manifest['revision']
    source_counts = Counter()
    identities, other_pixels, other_raw = {}, set(), set()
    test_rows = []
    for entry in sorted(manifest['files'], key=lambda e: e['path']):
        path = DEST / 'source' / entry['path']
        if sha256(path) != entry['sha256']:
            raise ValueError('Source parquet changed: ' + entry['path'])
        split = path.name.split('-', 1)[0]
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Unexpected source split')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                index = source_counts[split]
                source_counts[split] += 1
                data = row['image']['bytes']
                raw_hash = hashlib.sha256(data).hexdigest()
                if raw_hash not in identities:
                    identities[raw_hash] = image_identity(data)
                identity = identities[raw_hash]
                if split != 'test':
                    other_raw.add(raw_hash)
                    other_pixels.add(identity['rgb_sha256'])
                    continue
                if not isinstance(row['question'], str) or not row['question'].strip() or not row['answer'].strip():
                    raise ValueError('Empty question/answer')
                if identity['format'] not in ('PNG', 'JPEG'):
                    raise ValueError('Unexpected image format: ' + identity['format'])
                filename = raw_hash + ('.png' if identity['format'] == 'PNG' else '.jpg')
                image_path = DEST / 'images/pathvqa' / filename
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if not image_path.exists():
                    image_path.write_bytes(data)
                elif sha256(image_path) != raw_hash:
                    raise ValueError('Existing exported image mismatch')
                test_rows.append({'id': 'test-%05d' % index, 'dataset': 'PathVQA', 'split': 'test',
                    'question': row['question'], 'answer': row['answer'], 'image': 'pathvqa/' + filename,
                    'source_url': 'https://huggingface.co/datasets/flaviagiammarino/path-vqa/blob/' + revision + '/' + entry['path'],
                    'source_revision': revision, 'source_parquet': entry['path'],
                    'original_image_path': row['image'].get('path'), 'image_identity': identity})
        print('extracted', split, source_counts[split], flush=True)
    write_jsonl(DEST / 'pathvqa_test.all.jsonl', test_rows)

    image_rows = defaultdict(list)
    questions = {}
    with (SFT / 'data/train.jsonl').open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            questions[index] = normalized(row['conversations'][0]['value'])
            for path in row.get('images', []):
                image_rows[path].append(index)
    def sft_identity(item):
        path, indices = item
        result = image_identity((SFT / path).read_bytes())
        return dict(result, path=path, rows=indices)
    with ThreadPoolExecutor(max_workers=8) as pool:
        sft_images = list(pool.map(sft_identity, sorted(image_rows.items())))
    write_jsonl(DEST / 'sft_image_index.jsonl', sft_images)
    print('indexed SFT images', len(sft_images), flush=True)
    raw_index, pixel_index = defaultdict(list), defaultdict(list)
    near_index = defaultdict(list)
    for item in sft_images:
        raw_index[item['sha256']].append(item)
        pixel_index[item['rgb_sha256']].append(item)
        near_index[int(item['dhash128'], 16)].append(item)
    near_cache = {}
    findings, clean = [], []
    counts = Counter()
    for row in test_rows:
        identity = row['image_identity']
        raw_matches = raw_index[identity['sha256']]
        pixel_matches = pixel_index[identity['rgb_sha256']]
        h = int(identity['dhash128'], 16)
        if h not in near_cache:
            near_cache[h] = [item for value, items in near_index.items() if (h ^ value).bit_count() <= 4
                             for item in items]
        near_matches = near_cache[h]
        same_question = [item for item in near_matches if any(questions[i] == normalized(row['question']) for i in item['rows'])]
        source_overlap = identity['sha256'] in other_raw or identity['rgb_sha256'] in other_pixels
        flags = {'sft_same_file': bool(raw_matches), 'sft_same_pixels': bool(pixel_matches),
                 'sft_near_image_candidate': bool(near_matches), 'sft_near_image_same_question': bool(same_question),
                 'source_train_validation_same_image': source_overlap}
        for name, present in flags.items():
            counts[name] += int(present)
        excluded = any(flags.values())
        findings.append({'id': row['id'], 'image': row['image'], **flags, 'excluded': excluded,
                         'candidate_sft_paths': sorted({m['path'] for m in raw_matches + pixel_matches + near_matches})})
        if not excluded:
            clean.append(row)
    write_jsonl(DEST / 'overlap_findings.jsonl', findings)
    write_jsonl(DEST / 'pathvqa_test.screened.jsonl', clean)
    # Deterministic 10-image diagnostic sample, selected without reference answers.
    selected, seen = [], set()
    for row in clean:
        key = row['image_identity']['rgb_sha256']
        if key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) == 10:
            break
    write_jsonl(DEST / 'pilot10.jsonl', selected)
    report = {'dataset': manifest['repo'], 'revision': revision, 'source_counts': dict(source_counts),
              'source_files_verified': len(manifest['files']), 'test_unique_images': len({r['image'] for r in test_rows}),
              'sft_rows': len(questions), 'sft_images': len(sft_images), 'overlap_counts_by_question': dict(counts),
              'screened_questions': len(clean), 'screened_unique_images': len({r['image'] for r in clean}),
              'pilot_questions': len(selected), 'reference_preserved_verbatim': True,
              'method': 'SHA256 bytes; SHA256 decoded RGB with dimensions; 128-bit horizontal/vertical dHash distance<=4; normalized image-question pairs',
              'selection': 'Exclude all exact and near-image candidates, including repeated images with different questions; first ten distinct eligible images in source order.',
              'limits': ['Near-image candidates can be false positives.', 'Crops, large transforms and semantic paraphrases may evade matching.',
                         'Cannot establish absence from agent pretraining or unpublished SFT data.',
                         'Ten cases are integration diagnostics, not a statistically representative medical benchmark result.']}
    (DEST / 'overlap_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
