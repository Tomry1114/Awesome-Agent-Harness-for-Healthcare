"""Pinned Hugging Face downloads, verified against published LFS SHA-256 hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import requests


def request_json(url, payload=None):
    token_path = Path(os.environ.get('HF_TOKEN_PATH') or
                      str(Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))) / 'token'))
    token = os.environ.get('HF_TOKEN') or (token_path.read_text().strip() if token_path.is_file() else None)
    headers = {'Authorization': 'Bearer ' + token} if token else {}
    if payload is not None:
        headers['Accept'] = 'application/vnd.git-lfs+json'
    for attempt in range(5):
        try:
            response = requests.get(url, headers=headers, timeout=(20, 60)) if payload is None else requests.post(
                url, json=payload, timeout=(20, 60), headers=headers)
            if response.status_code in (401, 403):
                raise PermissionError('Hub access requires an authorized login for this repository')
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(repo, kind, revision, files, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    prefix = 'datasets/' if kind == 'dataset' else ''
    api = 'https://huggingface.co/' + prefix + repo + '.git/info/lfs/objects/batch'
    def one(file):
        path = destination / file['rfilename']
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = file['lfs']['sha256']
        if path.is_file() and path.stat().st_size == file['size'] and sha256(path) == expected:
            return {'path': file['rfilename'], 'sha256': expected, 'size': file['size']}
        part = path.with_name(path.name + '.part')
        for attempt in range(5):
            try:
                obj = request_json(api, {'operation': 'download', 'transfers': ['basic'],
                    'objects': [{'oid': expected, 'size': file['size']}]})['objects'][0]
                action = obj['actions']['download']
                session = requests.Session()
                session.trust_env = False  # public CDN works directly on this cluster
                offset = part.stat().st_size if part.exists() else 0
                headers = dict(action.get('header', {}))
                if offset:
                    headers['Range'] = 'bytes=%d-' % offset
                with session.get(action['href'], headers=headers, stream=True, timeout=(20, 90)) as response:
                    if response.status_code == 416:
                        part.unlink(missing_ok=True)
                    response.raise_for_status()
                    append = offset and response.status_code == 206
                    if append and not response.headers.get('Content-Range', '').startswith('bytes %d-' % offset):
                        raise ValueError('Unexpected range')
                    with part.open('ab' if append else 'wb') as stream:
                        for block in response.iter_content(1024 * 1024):
                            stream.write(block)
                if part.stat().st_size != file['size'] or sha256(part) != expected:
                    part.unlink(missing_ok=True)
                    raise ValueError('Download hash/size mismatch')
                part.replace(path)
                print('verified ' + file['rfilename'], flush=True)
                return {'path': file['rfilename'], 'sha256': expected, 'size': file['size']}
            except (requests.RequestException, ValueError, KeyError):
                if attempt == 4:
                    raise RuntimeError('Could not download verified file: ' + file['rfilename']) from None
                time.sleep(2 * (attempt + 1))
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(one, files))
    report = {'repo': repo, 'kind': kind, 'revision': revision, 'files': verified}
    (destination / 'download_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--kind', choices=['model', 'dataset'], required=True)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--pattern', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import fnmatch
    info = request_json('https://huggingface.co/api/%ss/%s/revision/%s?blobs=true' % (
        args.kind, args.repo, args.revision))
    files = [f for f in info['siblings'] if fnmatch.fnmatch(f['rfilename'], args.pattern)]
    if not files or any('lfs' not in f for f in files):
        raise ValueError('Pattern must select one or more LFS files')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'upstream_info.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps({'repo': args.repo, 'revision': info['sha'], 'files': len(files),
                      'bytes': sum(f['size'] for f in files)}), flush=True)
    download(args.repo, args.kind, info['sha'], files, args.out)


if __name__ == '__main__':
    main()
