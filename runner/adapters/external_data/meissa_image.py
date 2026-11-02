"""Live image operations using Meissa's worker or /call HTTP contracts.

Source contract: Schuture/Meissa @ eb119115d2f5dfcfae772ba5f2466b4c603fac90.
No teacher observations are consulted. Model installation is external to this client.
"""
import base64
import hashlib
import io
import ipaddress
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, ProxyHandler, build_opener

UPSTREAM_COMMIT = 'eb119115d2f5dfcfae772ba5f2466b4c603fac90'
SEGMENTATION = ('SegmentRegionAroundPoint', 'BioMedParseTextSeg')
MAX_RESPONSE = 32 * 1024 * 1024
RUNTIME_CONFIG = Path(__file__).resolve().parents[3] / 'external_data/config/meissa_runtime.json'


def settings():
    config_path = Path(os.environ.get('MH_MEISSA_RUNTIME_CONFIG') or RUNTIME_CONFIG)
    configured = json.loads(config_path.read_text())
    proxy_mode = os.environ.get('MH_MEISSA_TOOL_PROXY') or configured['tool_proxy']
    if proxy_mode not in ('direct', 'environment'):
        raise ValueError('Meissa tool_proxy must be direct or environment')
    result = {'timeout': float(os.environ.get('MH_MEISSA_TOOL_TIMEOUT', '60')),
              'tool_proxy': proxy_mode,
              'server': os.environ.get('MH_MEISSA_TOOL_SERVER', '').rstrip('/'),
              'workers': {}, 'revisions': {}}
    if not math.isfinite(result['timeout']) or not 0 < result['timeout'] <= 600:
        raise ValueError('MH_MEISSA_TOOL_TIMEOUT must be in (0,600]')
    for name, key in zip(SEGMENTATION, ('SAM', 'BIOMEDPARSE')):
        result['workers'][name] = os.environ.get('MH_MEISSA_' + key + '_URL', '').rstrip('/')
        result['revisions'][name] = os.environ.get('MH_MEISSA_' + key + '_REVISION') or 'unverified'
    for url in [result['server']] + list(result['workers'].values()):
        if url:
            parsed = urlsplit(url)
            if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username
                    or parsed.password or parsed.query or parsed.fragment):
                raise ValueError('Meissa endpoint must be an HTTP(S) base URL without credentials or query')
    return result


def validate_image(data):
    from PIL import Image
    if len(data) > MAX_RESPONSE:
        raise ValueError('Oversized image')
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in ('PNG', 'JPEG') or image.width * image.height > 40_000_000:
            raise ValueError('Expected a bounded PNG/JPEG image')
        mime = 'image/png' if image.format == 'PNG' else 'image/jpeg'
        image.verify()
    return mime


class InvalidZoomArguments(ValueError):
    pass


def zoom(data, param):
    """BBox subset of upstream zoom: 0–1000 coordinates, 10% padding, JPEG.

    Deliberately rejects masks, absolute coordinates and unordered/empty boxes.
    This implements the documented operation without importing GPU/server packages.
    """
    from PIL import Image
    try:
        box = json.loads(param)
    except (ValueError, TypeError) as error:
        raise InvalidZoomArguments('ZoomInSubfigure.param must be a JSON bbox string, e.g. "[250,250,750,750]" (0–1000). '
                                   'Point syntax "x=500 y=500" belongs to SegmentRegionAroundPoint, not ZoomInSubfigure.') from error
    if (not isinstance(box, list) or len(box) != 4
            or any(type(x) not in (int, float) or not 0 <= x <= 1000 for x in box)
            or box[0] >= box[2] or box[1] >= box[3]):
        raise InvalidZoomArguments('Expected an ordered bounding box in 0–1000 coordinates')
    with Image.open(io.BytesIO(data)) as source:
        w, h = source.size
        x1, y1, x2, y2 = [int(v * size / 1000) for v, size in zip(box, (w, h, w, h))]
        if x1 == x2 or y1 == y2:
            raise InvalidZoomArguments('Empty pixel crop; use 0–1000 coordinates')
        px, py = int((x2 - x1) * .1), int((y2 - y1) * .1)
        bounds = [max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)]
        output = io.BytesIO()
        source.crop(bounds).convert('RGB').save(output, format='JPEG')
    return 'Zoomed in successfully to region %s.' % bounds, output.getvalue(), 'image/jpeg'


class MeissaImageBackend:
    def __init__(self):
        self.config = settings()
        self.reset()

    def reset(self):
        self.events = []
        self.errors = {}

    def available(self, name):
        return name == 'ZoomInSubfigure' or bool(self.config['workers'].get(name) or self.config['server'])

    def execute(self, name, data, param):
        event = {'tool': name, 'input_sha256': hashlib.sha256(data).hexdigest(),
                 'param': param, 'status': 'failed'}
        self.events.append(event)
        try:
            if name == 'ZoomInSubfigure':
                text, edited, mime = zoom(data, param)
                event['transport'] = 'local_bbox_crop'
            else:
                worker = self.config['workers'].get(name)
                if not self.available(name):
                    raise RuntimeError('backend_not_configured')
                arguments = {'image': base64.b64encode(data).decode('ascii'), 'param': param}
                url = worker + '/worker_generate' if worker else self.config['server'] + '/call'
                payload = arguments if worker else {'tool': name, 'arguments': arguments}
                event['transport'] = 'worker_generate' if worker else 'call'
                request = Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
                host = urlsplit(url).hostname
                try:
                    loopback = ipaddress.ip_address(host).is_loopback
                except ValueError:
                    loopback = host == 'localhost'
                # Persistent runtime policy covers Slurm hostnames too, not just loopback.
                # This affects only image tools; agent/judge gateway routing is unchanged.
                opener = build_opener(ProxyHandler({})) if loopback or self.config['tool_proxy'] == 'direct' else build_opener()
                with opener.open(request, timeout=self.config['timeout']) as response:
                    raw = response.read(MAX_RESPONSE + 1)
                if len(raw) > MAX_RESPONSE:
                    raise ValueError('Oversized tool response')
                result = json.loads(raw)
                if (not isinstance(result, dict) or result.get('error') or result.get('error_code', 0) != 0
                        or not isinstance(result.get('edited_image'), str) or not result['edited_image']):
                    raise ValueError('Tool did not return a successful edited image')
                edited = base64.b64decode(result['edited_image'], validate=True)
                mime = validate_image(edited)
                text = result.get('text', result.get('result', ''))
                if not isinstance(text, str):
                    raise ValueError('Tool text must be a string')
            event.update(status='success', output_sha256=hashlib.sha256(edited).hexdigest())
            self.errors.pop(name, None)
            return {'text': text, 'data': edited, 'mime': mime, 'execution': dict(event)}
        except InvalidZoomArguments as error:
            event['error_kind'] = 'invalid_arguments'
            return {'error': str(error), 'error_type': 'invalid_arguments',
                    'environment_failure': False, 'execution': dict(event)}
        except Exception as error:
            # Do not expose endpoint credentials, server traces or payloads in errors.
            kind = 'backend_not_configured' if not self.available(name) else type(error).__name__
            event['error_kind'] = kind
            self.errors[name] = kind
            return {'error': 'environment_error: Meissa image tool failed (%s)' % kind,
                    'environment_failure': True, 'execution': dict(event)}

    def provenance(self):
        return {'environment_fidelity': 'meissa_live_image_tools', 'upstream_commit': UPSTREAM_COMMIT,
                'settings': self.config, 'model_revisions_verified': False,
                'execution': self.events, 'teacher_replay': False,
                'deviations': ['BBox-only local zoom; masks/absolute coordinates unsupported.',
                               'No forced initial zoom; tool use remains optional.',
                               'All remote image references resolved to current image bytes.',
                               'Worker error codes and missing edited images fail closed.']}
