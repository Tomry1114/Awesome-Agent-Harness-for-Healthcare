"""Case evidence, explicitly selected live services, and separate frozen replay."""
import base64
import hashlib
import io
import json
from pathlib import Path

from environment_base import EnvironmentAdapter
from .conversion import ROOT, TOOLS


class UserStudyEnv(EnvironmentAdapter):
    type = 'study_case'
    IMPLEMENTED_TOOLS = tuple(TOOLS)
    BACKEND_DESCRIPTION = 'source_case; live_pixel_crop; frozen_exact_request_observations'

    def __init__(self, task, **kwargs):
        super().__init__(task)
        self.runtime = task.get('study_runtime') or {}
        self.images = {}
        self.last_image = None
        self.serial = 0
        self.patient = None
        self.image_backend = None
        if self.runtime.get('meissa_image'):
            from .meissa_image import MeissaImageBackend
            self.image_backend = MeissaImageBackend()
            self.BACKEND_DESCRIPTION = 'Meissa live bbox zoom and remote image workers; no teacher replay'
        if self.runtime.get('medr_interaction'):
            from .medr_patient import MedRPatient
            self.patient = MedRPatient(self.runtime['medr_interaction'])
            self.BACKEND_DESCRIPTION = 'MedRBench model-mediated ancillary information service'
        self.backend_model = self.patient.settings['model'] if self.patient else None

    @staticmethod
    def _asset(path):
        path = (ROOT / path).resolve()
        roots = [ROOT / 'user_study/Meissa-SFT/images', ROOT / 'user_study/Meissa-Evaluation/images']
        if not any(root.resolve() in path.parents for root in roots):
            raise ValueError('Study image is outside the permitted asset directories')
        if not path.is_file():
            raise ValueError('Missing study image')
        return path.read_bytes(), 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'

    def _load_images(self, items):
        delivered = []
        for item in items:
            data, mime = self._asset(item['path'])
            if item.get('sha256') and hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Study image changed after task preparation')
            image_id = item['asset_id']
            self.images[image_id] = (data, mime)
            self.last_image = image_id
            delivered.append(self._image(image_id))
        return delivered

    def _image(self, image_id):
        if image_id == 'img_last':
            image_id = self.last_image
        if image_id not in self.images:
            raise ValueError('Image is not available in this episode')
        data, mime = self.images[image_id]
        return {'image_id': image_id, 'sha256': hashlib.sha256(data).hexdigest(),
                'url': 'data:%s;base64,%s' % (mime, base64.b64encode(data).decode('ascii'))}

    def reset(self):
        self.images.clear()
        self.last_image, self.serial = None, 0
        self._initial_images = self._load_images(self.runtime.get('initial_images') or [])
        if self.patient:
            self.patient.reset()
        if self.image_backend:
            self.image_backend.reset()

    def initial_observation(self):
        observation = {'output': {'case_text': (self.task.get('context') or {}).get('text', ''),
                           'available_recorded_requests': (self.task.get('context') or {}).get('available_recorded_requests', {}),
                           'images': self._initial_images}, 'mode': 'source_case'}
        if self.patient:
            observation['output']['interaction'] = self.patient.state()
        if self.image_backend:
            observation['output']['image_backend'] = {
                'mode': 'live', 'segmentation': {
                    name: 'configured_not_verified' if self.image_backend.available(name) else 'unavailable'
                    for name in ('SegmentRegionAroundPoint', 'BioMedParseTextSeg')}}
        return observation

    def study_provenance(self):
        if self.image_backend:
            return self.image_backend.provenance()
        return self.patient.provenance() if self.patient else None

    def capabilities(self):
        caps = super().capabilities()
        if 'RequestInformation' in caps:
            caps['RequestInformation']['available'] = bool(self.patient) and 'RequestInformation' in self.available_tools()
            caps['RequestInformation']['healthy'] = self.patient.last_error is None if self.patient else True
        if self.image_backend:
            for name in ('ZoomInSubfigure', 'SegmentRegionAroundPoint', 'BioMedParseTextSeg'):
                if name in caps:
                    caps[name]['available'] = name in self.available_tools() and self.image_backend.available(name)
                    caps[name]['healthy'] = name not in self.image_backend.errors
        return caps

    def _frozen(self, name, args):
        matches = []
        for record in self.runtime.get('recorded_observations') or []:
            if name != record['tool']:
                continue
            expected = dict(record['args'])
            actual = dict(args)
            if record.get('input_image_path'):
                expected.pop('image', None)
                image_id = actual.pop('image', 'img_last')
                image = self._image(image_id)
                data, _ = self._asset(record['input_image_path'])
                if image['sha256'] != hashlib.sha256(data).hexdigest():
                    continue
            if actual == expected:
                matches.append(record)
        if not matches:
            return {'error': 'missing_capability: this request has no recorded observation',
                    'mode': 'frozen_exact_request', 'unavailable_request': True}
        # Different outputs for the same request cannot be resolved from this release.
        identities = {json.dumps([r['text'], r['images']], sort_keys=True) for r in matches}
        if len(identities) != 1:
            return {'error': 'missing_capability: ambiguous recorded observations', 'mode': 'frozen_exact_request'}
        record = matches[0]
        return {'output': {'text': record['text'], 'images': self._load_images(record['images'])},
                'mode': 'frozen_exact_request'}

    def call_tool(self, name, args):
        if name not in self.available_tools():
            return {'error': 'missing_capability: tool is not offered for this task'}
        if not isinstance(args, dict):
            return {'error': 'invalid arguments: expected an object'}
        try:
            from jsonschema import validate, ValidationError
            try:
                validate(args, TOOLS[name]['parameters'])
            except ValidationError:
                if name == 'ZoomInSubfigure':
                    from .tool_contracts import ZOOM_DESCRIPTION
                    return {'error': 'invalid tool request: ' + ZOOM_DESCRIPTION,
                            'error_type': 'invalid_arguments', 'environment_failure': False}
                raise
            if name == 'RequestInformation':
                if not self.patient:
                    return {'error': 'missing_capability: this task has no MedR patient service'}
                return self.patient.request(args['request'])
            if name == 'ReadCase':
                out = {'text': (self.task.get('context') or {}).get('text', '')}
                if self.patient:
                    out['interaction'] = self.patient.state()
                return {'output': out, 'mode': 'source_case'}
            if name == 'ReadImage':
                return {'output': {'images': [self._image(args['image'])]}, 'mode': 'source_image'}
            if self.image_backend and name in ('ZoomInSubfigure', 'SegmentRegionAroundPoint', 'BioMedParseTextSeg'):
                image_id = self._image(args['image'])['image_id']
                result = self.image_backend.execute(name, self.images[image_id][0], args['param'])
                if 'error' in result:
                    return dict(result, mode='meissa_live_image')
                self.serial += 1
                image_id = 'live_image_%d' % self.serial
                self.images[image_id] = (result['data'], result['mime'])
                self.last_image = image_id
                return {'output': {'text': result['text'], 'images': [self._image(image_id)]},
                        'mode': 'meissa_live_image', 'execution': result['execution']}
            if name == 'ZoomInSubfigure':
                from PIL import Image
                image_id = self.last_image if args['image'] == 'img_last' else args['image']
                if image_id not in self.images:
                    raise ValueError('Image is not available in this episode')
                box = json.loads(args['param'])
                if (not isinstance(box, list) or len(box) != 4
                        or any(type(x) not in (int, float) or not 0 <= x <= 1000 for x in box)
                        or box[0] >= box[2] or box[1] >= box[3]):
                    raise ValueError('Crop requires ordered coordinates in [0,1000]')
                with Image.open(io.BytesIO(self.images[image_id][0])) as source:
                    w, h = source.size
                    pixel_box = (int(box[0] * w / 1000), int(box[1] * h / 1000),
                                 int(box[2] * w / 1000), int(box[3] * h / 1000))
                    if pixel_box[0] == pixel_box[2] or pixel_box[1] == pixel_box[3]:
                        raise ValueError('Crop has zero pixel area')
                    output = io.BytesIO()
                    source.crop(pixel_box).save(output, format='PNG')
                self.serial += 1
                image_id = 'live_crop_%d' % self.serial
                self.images[image_id] = (output.getvalue(), 'image/png')
                self.last_image = image_id
                return {'output': {'text': 'Cropped image: ' + image_id, 'images': [self._image(image_id)]},
                        'mode': 'deterministic_pixel_crop'}
            return self._frozen(name, args)
        except Exception as error:
            return {'error': 'invalid tool request: %s' % str(error)[:250]}

    def teardown(self):
        self.images.clear()
