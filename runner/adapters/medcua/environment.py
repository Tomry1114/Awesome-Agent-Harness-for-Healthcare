"""Private native evaluator plus a screenshot-only agent observation boundary."""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import tempfile

from environment_base import EnvironmentAdapter
from .source import WORKER, python, runtime_env, verify_source


class MedCUAEnv(EnvironmentAdapter):
    type = 'medcua_native'
    IMPLEMENTED_TOOLS = ('BrowserAction',)
    BACKEND_DESCRIPTION = 'Official MedCUA/BrowserGym Chromium environment and deterministic safety checkers'

    def __init__(self, task, workspace=None, **kwargs):
        super().__init__(task)
        self.runtime = task.get('native_runtime') or {}
        self.workspace = Path(workspace) if workspace else Path(tempfile.mkdtemp(prefix='medcua_'))
        self.process = None
        self.log = None
        self.last = None
        self.failure = None

    def _rpc(self, request):
        if not self.process or self.process.poll() is not None:
            raise RuntimeError('MedCUA worker unavailable')
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 120)[0]:
            raise TimeoutError('MedCUA worker exceeded 120 seconds')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('MedCUA worker exited without a response')
        return json.loads(line)

    def _save(self, result):
        # Preserve actual initial and subsequent pixels for post-run scoring.
        # Request audits intentionally store hashes instead of repeating image bytes.
        import base64
        import hashlib
        for image in (result.get('output') or {}).get('images', []):
            raw = base64.b64decode(image['url'].split(',', 1)[1], validate=True)
            digest = hashlib.sha256(raw).hexdigest()
            if digest != image['sha256']:
                raise ValueError('Screenshot hash mismatch')
            frames = self.workspace / 'medcua_frames'
            frames.mkdir(parents=True, exist_ok=True)
            path = frames / (digest + '.png')
            if not path.exists():
                path.write_bytes(raw)
        private = result.pop('private', None)
        if private is not None:
            record = dict(private, task_id=self.task['task_id'], **self.runtime)
            path = self.workspace / 'medcua_native.json'
            path.write_text(json.dumps(record, indent=2) + '\n')
            with (self.workspace / 'medcua_native_history.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        self.last = result
        return result

    def reset(self):
        self.teardown()
        self.failure = None
        self.last = None
        verify_source(self.runtime.get('source_tree_sha256'))
        self.workspace.mkdir(parents=True, exist_ok=True)
        for name in ('medcua_native.json', 'medcua_native_history.jsonl', 'medcua_native_error.json'):
            (self.workspace / name).unlink(missing_ok=True)
        self.log = (self.workspace / 'medcua_worker.log').open('w')
        self.process = subprocess.Popen([python(), '-u', str(WORKER)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, text=True,
                                        env=runtime_env(), start_new_session=True, bufsize=1)
        response = self._rpc(dict(self.runtime, op='reset'))
        if not response.get('ok'):
            raise RuntimeError('MedCUA reset failed: ' + response.get('error', 'unknown error'))
        self._save(response)

    def initial_observation(self):
        return self.last

    def call_tool(self, name, args):
        if name != 'BrowserAction' or set(args) != {'action'} or not isinstance(args['action'], str):
            return {'error': 'Expected BrowserAction with one action string', 'error_type': 'invalid_arguments'}
        try:
            result = self._rpc({'op': 'step', 'action': args['action']})
            if not result.get('ok'):
                if result.get('error_type') in ('ValueError', 'SyntaxError'):
                    return {'error': result['error'], 'error_type': 'invalid_arguments', 'environment_failure': False}
                raise RuntimeError(result.get('error', 'Native browser failure'))
            result = self._save(result)
            if result.get('action_error'):
                result.update(error=result['action_error'], error_type='browser_action_error', environment_failure=False)
            return result
        except Exception as error:
            self.failure = str(error)
            (self.workspace / 'medcua_native_error.json').write_text(json.dumps({'error': self.failure}) + '\n')
            return {'error': self.failure, 'error_type': 'environment_error', 'environment_failure': True}

    def state_summary(self):
        output = (self.last or {}).get('output') or {}
        return {'screenshot_sha256': [im['sha256'] for im in output.get('images', [])],
                'terminated': output.get('terminated', False), 'truncated': output.get('truncated', False)}

    def _healthy(self):
        return self.failure is None and (self.process is None or self.process.poll() is None)

    def teardown(self):
        if self.process:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write('{"op":"close"}\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(self.process.pid, signal.SIGTERM)
                        self.process.wait(timeout=5)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        try:
                            os.killpg(self.process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            for stream in (self.process.stdin, self.process.stdout):
                stream.close()
            self.process = None
        if self.log:
            self.log.close()
            self.log = None
