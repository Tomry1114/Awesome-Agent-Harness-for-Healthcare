"""Shared process semantics. No teacher action is a required clinical workflow."""
import substrate as S

TOOLS = ('ReadCase', 'ReadImage', 'ZoomInSubfigure', 'RequestPhysicalExam', 'RequestTest', 'RequestInformation',
         'BioMedParseTextSeg', 'SegmentRegionAroundPoint')


def evidence(trace, benchmark=None):
    units = []
    for i, event in enumerate(trace):
        if event.get('event_type') != 'tool_call':
            continue
        delivery = S._real_delivery(event)
        out = S._result_output(event)
        images = out.get('images', []) if isinstance(out, dict) else []
        payload = out.get('text', '') if isinstance(out, dict) else str(out or '')
        if images:
            payload += ' Image bytes delivered: ' + ', '.join(x.get('sha256', '') for x in images)
        case_id = out.get('source_case_id') if isinstance(out, dict) else None
        # A mask/operation description repeats the requested label. It is not
        # independent medical evidence that the structure was correctly found.
        image_metadata = event.get('tool') in ('ReadImage', 'ZoomInSubfigure',
                                               'BioMedParseTextSeg', 'SegmentRegionAroundPoint')
        image_metadata = image_metadata or (benchmark == 'MeissaEval' and event.get('tool') == 'ReadCase')
        units.append({'id': '%s#%d' % (event.get('tool'), i),
                      'delivered_to_agent': delivery['delivered'], 'delivery_fidelity': delivery['fidelity'],
                      'error_visible': delivery['error_visible'], 'payload': payload,
                      'claim_support_eligible': not image_metadata,
                      'clinical_quality_status': 'unassessed',
                      'claim_support_limitation': 'Image pixels require a visual or human assessor; question/tool metadata is not medical support.' if image_metadata else None,
                      'usable_for_context': bool(delivery['delivered'] and payload and not S._errored(event)),
                      'context_type': 'case_evidence', 'source_channel': 'case_image' if images else 'case_record',
                      # Unknown binding is missing evidence, not an invented case
                      # ID that makes every expected_subject comparison fail.
                      'source_instance_id': case_id,
                      'subject_token': 'subject:Subject/' + case_id if case_id else None,
                      'extractor': event.get('tool')})
    return units


def make_plugin(benchmark):
    return {'benchmark': benchmark, 'default_tool_role': 'acquire',
            # These tools are confined to the assigned case. The agent cannot
            # select another patient/case, so binding is not an assessed decision.
            'expected_subject_resolver': lambda task: {'type': None, 'id': None},
            'tool_semantics': {name: {'role': 'acquire', 'success_milestones': ['case_evidence_observed']}
                               for name in TOOLS},
            'evidence_extractor': lambda trace: evidence(trace, benchmark),
            'dimension_policy': {'required_milestones': [], 'required_context_units': [],
                                 'visual_claim_support_unassessed': benchmark == 'MeissaEval',
                                 'governance_policy_id': benchmark}}
