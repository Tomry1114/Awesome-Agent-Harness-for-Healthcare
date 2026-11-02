"""Pinned source identity and isolated BrowserGym interpreter."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'user_study/MedCUA-BenchMark'
PIN = ROOT / 'user_study/config/medcua_source.json'
WORKER = Path(__file__).with_name('worker.py')


def python():
    return os.environ.get('MH_MEDCUA_PYTHON', str(ROOT / 'user_study/runtime/medcua_env/bin/python'))


def source_identity():
    names = subprocess.check_output(['git', '-C', str(SOURCE), 'ls-files'], text=True).splitlines()
    files = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in names}
    return {'repository': 'https://github.com/Jia7878/MedCUA-BenchMark',
            'commit': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
            'files': files, 'tree_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def verify_source(expected=None):
    pin = json.loads(PIN.read_text())
    actual = source_identity()
    if actual != pin or (expected and expected != actual['tree_sha256']):
        raise ValueError('MedCUA source changed; review and explicitly repin before preparing/running tasks')
    return actual


def runtime_env():
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    # The browser worker does not need model credentials.
    for key in list(env):
        if any(word in key.upper() for word in ('API_KEY', 'OPENAI_KEY', 'AGENT_KEY', 'JUDGE_KEY', 'HF_TOKEN')):
            env.pop(key)
    return env
