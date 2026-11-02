"""Isolated upstream BrowserGym process; stdout is a private JSON protocol."""
import ast
import base64
import contextlib
import hashlib
import io
import json
import sys
from urllib.parse import urlsplit
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def validate_action(action, allowed):
    tree = ast.parse(action.strip(), mode='eval').body
    if not isinstance(tree, ast.Call) or not isinstance(tree.func, ast.Name) or tree.func.id not in allowed:
        raise ValueError('Use exactly one official coordinate/keyboard/browser action')
    if any(k.arg is None for k in tree.keywords):
        raise ValueError('Expanded keyword arguments are not allowed')
    for arg in tree.args + [k.value for k in tree.keywords]:
        ast.literal_eval(arg)  # no nested calls, attributes, comprehensions or executable expressions
    # Local checker/credential files are not task assets and must not be navigated to/uploaded.
    if tree.func.id == 'mouse_upload_file':
        raise ValueError('This adapter has no public upload-asset manifest; file upload is unavailable')
    if tree.func.id == 'goto':
        values = [ast.literal_eval(a) for a in tree.args] + [ast.literal_eval(k.value) for k in tree.keywords]
        if not values or not isinstance(values[0], str) or urlsplit(values[0]).scheme not in ('http', 'https'):
            raise ValueError('Navigation accepts HTTP(S); local source/checker files are private')
    return tree.func.id


def catalog():
    import browsergym.medgym as medgym
    import gymnasium as gym
    from browsergym.medgym.protocol import PIXEL_ACTION_SET
    entries = []
    for env_id in medgym.ALL_MEDGYM_TASK_IDS:
        spec = gym.spec('browsergym/' + env_id)
        shell = spec.entry_point(**spec.kwargs)
        task = shell.task_entrypoint(seed=42, **shell.task_kwargs)
        scenario = env_id.split('.')[1]
        backend = 'openemr' if scenario == 'openemr' else ('ohif' if scenario in ('pacs_radiology', 'pathology_viewer') else 'html')
        entries.append({'env_id': env_id, 'base_task_id': task.task_id, 'setting': task.setting,
                        'scenario': scenario, 'goal': task.goal, 'checker': task.checker_name,
                        'expected_values': task._expected, 'difficulty': task._difficulty,
                        'scored': task._scored, 'backend': backend})
    return {'tasks': entries, 'action_names': list(PIXEL_ACTION_SET.action_set),
            'action_description': PIXEL_ACTION_SET.describe()}


class Session:
    def __init__(self):
        self.env = None
        self.serial = 0
        self.done = False
        self.steps = 0
        self.env_id = None
        self.action_target = None

    def pack(self, obs, info, reward=0, terminated=False, truncated=False):
        from PIL import Image
        self.serial += 1
        stream = io.BytesIO()
        Image.fromarray(obs['screenshot']).save(stream, format='PNG')
        raw = stream.getvalue()
        # Checker-generated user chat contains score/target hints in upstream.
        # Keep it private, alongside reward and task_info. The agent already has
        # its own message history; only goal/pixels/action errors cross this seam.
        public = {'text': json.dumps({'goal': obs['goal'], 'last_action': obs['last_action'],
                                     'last_action_error': obs['last_action_error']}, ensure_ascii=False),
                  'images': [{'image_id': 'screen_%d' % self.serial,
                              'sha256': hashlib.sha256(raw).hexdigest(),
                              'url': 'data:image/png;base64,' + base64.b64encode(raw).decode()}],
                  'terminated': bool(terminated), 'truncated': bool(truncated)}
        telemetry = None
        try:
            # This worker is also executed as a script in an isolated env.
            from process_capture import enabled, STATE_JS, VERSION
            if enabled(self.env_id):
                telemetry = {'schema_version': VERSION, 'action_index': self.steps,
                             'screenshot_sha256': hashlib.sha256(raw).hexdigest(),
                             'action': obs['last_action'], 'action_target': self.action_target,
                             'state': self.env.unwrapped.page.evaluate(STATE_JS)}
        except Exception as error:
            telemetry = {'capture_error': type(error).__name__}
        return {'output': public, 'action_error': obs['last_action_error'],
                'private': {'native_info': info.get('task_info') or {}, 'emitted_reward': reward,
                            'terminated': bool(terminated), 'truncated': bool(truncated),
                            'actions_executed': self.steps, 'process_state': telemetry}}

    def handle(self, request):
        if request['op'] == 'reset':
            from browsergym.medgym import make_env
            if self.env:
                self.env.close()
            self.steps = self.serial = 0
            self.done = False
            self.max_steps = int(request.get('max_steps', 30))
            self.env_id = request['env_id']
            self.action_target = None
            self.env = make_env(request['env_id'], headless=True)
            obs, info = self.env.reset(seed=request['seed'])
            # Initial-state measurement invokes the same native checker; it is
            # not an agent action and does not reveal its diagnostic chat.
            reward, done, message, native = self.env.unwrapped._task_validate()
            info['task_info'] = native
            self.done = bool(done)
            return self.pack(obs, info, reward, terminated=done)
        if request['op'] == 'step':
            from browsergym.medgym.protocol import PIXEL_ACTION_SET
            if self.done:
                raise ValueError('Native episode has ended')
            validate_action(request['action'], PIXEL_ACTION_SET.action_set)
            self.action_target = None
            try:
                from process_capture import enabled, target
                if enabled(self.env_id):
                    self.action_target = target(self.env.unwrapped.page, request['action'])
            except Exception as error:
                self.action_target = {'capture_error': type(error).__name__}
            obs, reward, terminated, truncated, info = self.env.step(request['action'])
            self.steps += 1
            truncated = truncated or (self.steps >= self.max_steps and not terminated)
            self.done = bool(terminated or truncated)
            return self.pack(obs, info, reward, terminated, truncated)
        if request['op'] == 'close':
            if self.env:
                self.env.close()
                self.env = None
            return {'closed': True}
        raise ValueError('Unknown worker operation')


def main():
    wire = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        if '--catalog' in sys.argv:
            result = catalog()
            wire.write(json.dumps(result) + '\n')
            return
        session = Session()
        try:
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    result = {'ok': True, **session.handle(request)}
                except Exception as error:
                    result = {'ok': False, 'error_type': type(error).__name__, 'error': str(error)[:1600]}
                wire.write(json.dumps(result, ensure_ascii=False) + '\n')
                wire.flush()
                if result.get('closed'):
                    break
        finally:
            if session.env:
                session.env.close()


if __name__ == '__main__':
    main()
