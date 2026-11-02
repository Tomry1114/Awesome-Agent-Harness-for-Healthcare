"""MedR's model-mediated ancillary information service, separate from the diagnostic agent."""
import hashlib
import os

UPSTREAM_MODEL = 'gpt-4o-2024-11-20'


def patient_settings():
    """Public configuration only: safe for provenance and resume keys, never includes a credential."""
    import gateway
    base = (os.environ.get('MH_MEDR_PATIENT_BASE') or gateway.base_url()).strip().rstrip('/')
    if base.endswith('/v1'):
        base = base[:-3]
    model = os.environ.get('MH_MEDR_PATIENT_MODEL', UPSTREAM_MODEL).strip()
    max_tokens = int(os.environ.get('MH_MEDR_PATIENT_MAX_TOKENS', '2048'))
    timeout = int(os.environ.get('MH_MEDR_PATIENT_TIMEOUT', '120'))
    if not model or max_tokens < 1 or timeout < 1:
        raise ValueError('Invalid MedR patient service model or budget')
    return {'model': model, 'base': base, 'max_tokens': max_tokens, 'timeout': timeout}


class MedRPatient:
    def __init__(self, config):
        self.protocol = config['protocol']
        expected_limit = {'one_turn': 1, 'free_turn': 4}.get(self.protocol)
        if expected_limit is None or config['max_information_requests'] != expected_limit:
            raise ValueError('MedR protocol has an invalid information-request limit')
        self.limit = expected_limit
        self.template = config['patient_prompt_template']
        # The service receives only case facts and ancillary results, never reference diagnoses or teacher reasoning.
        self.system_prompt = self.template.format(case=config['initial_case'],
                                                  ancillary_test_results=config['ancillary_tests'])
        self.settings = patient_settings()
        self.reset()

    def reset(self):
        self.requests_used = 0
        self.calls_attempted = 0
        self.last_error = None

    def state(self):
        return {'protocol': self.protocol, 'max_information_requests': self.limit,
                'information_requests_used': self.requests_used,
                'information_requests_remaining': self.limit - self.requests_used,
                'must_finalize': self.requests_used >= self.limit}

    def provenance(self):
        return dict(self.state(), patient_service=dict(self.settings),
                    upstream_patient_model=UPSTREAM_MODEL,
                    model_matches_upstream=self.settings['model'] == UPSTREAM_MODEL,
                    patient_prompt_sha256=hashlib.sha256(self.template.encode()).hexdigest(),
                    calls_attempted=self.calls_attempted, last_error=self.last_error,
                    response_mode='live_model_on_source_ancillary_results',
                    deviations=['Gateway retries and a configured output-token cap replace upstream transport defaults.'])

    def request(self, request):
        if not isinstance(request, str) or not request.strip():
            return {'error': 'invalid arguments: request must contain diagnostic information needs'}
        if self.requests_used >= self.limit:
            return {'error': 'interaction_limit: no information requests remain; provide a final diagnosis',
                    'error_type': 'interaction_limit', 'interaction': self.state()}
        import gateway
        self.calls_attempted += 1
        try:
            response = gateway.chat(
                [{'role': 'system', 'content': self.system_prompt},
                 {'role': 'user', 'content': 'The junior physician wants the following information:\n' + request}],
                model=self.settings['model'], base=self.settings['base'],
                key=os.environ.get('MH_MEDR_PATIENT_KEY'), judge=False,
                max_tokens=self.settings['max_tokens'], timeout=self.settings['timeout'])
        except Exception:
            # Do not leak an exception body that could echo the private source prompt or a credential.
            response = {'ok': False, 'error_type': 'exception'}
        content = response.get('content') if isinstance(response, dict) else None
        if not isinstance(response, dict) or not response.get('ok') or not isinstance(content, str) or not content.strip():
            kind = response.get('error_type') if isinstance(response, dict) else None
            kind = kind if kind in ('billing', 'auth', 'http_4xx', 'http_5xx', 'empty', 'timeout', 'exception') else 'empty'
            self.last_error = kind
            return {'error': 'environment_error: MedR patient service failed (%s); no clinical result was obtained' % kind,
                    'error_type': kind, 'environment_failure': True, 'interaction': self.state(),
                    'backend_model': self.settings['model']}
        self.requests_used += 1
        self.last_error = None
        return {'output': {'text': content, 'interaction': self.state(),
                           'next_action': 'Provide the final diagnosis.' if self.requests_used >= self.limit
                           else 'Request more information if needed, or provide the final diagnosis.'},
                'mode': 'medr_live_patient_service', 'backend_model': self.settings['model']}
