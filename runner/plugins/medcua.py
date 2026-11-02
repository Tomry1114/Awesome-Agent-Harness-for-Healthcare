"""Process telemetry is separate from the upstream five safety dimensions."""
import substrate as S


def evidence(trace):
    result = []
    for index, event in enumerate(trace):
        if event.get('event_type') != 'tool_call':
            continue
        delivery = S._real_delivery(event)
        output = S._result_output(event)
        output = output if isinstance(output, dict) else {}
        images = output.get('images') or []
        result.append({'id': 'BrowserAction#%d' % index,
                       'delivered_to_agent': delivery['delivered'], 'delivery_fidelity': delivery['fidelity'],
                       'error_visible': delivery['error_visible'],
                       'payload': output.get('text', ''), 'context_type': 'browser_screenshot',
                       'source_channel': 'browser_screenshot',
                       'source_instance_id': None,
                       'frame_id': images[-1]['sha256'] if images else None,
                       'subject_identity_available': False,
                       'usable_for_context': bool(images and delivery['delivered'] and not S._errored(event)),
                       'claim_support_eligible': False, 'clinical_quality_status': 'unassessed'})
    return result


PLUGIN = {'benchmark': 'MedCUABench', 'default_tool_role': 'act',
          'tool_semantics': {'BrowserAction': {'role': 'act'}}, 'evidence_extractor': evidence,
          'dimension_policy': {'required_milestones': [], 'required_context_units': [],
                               'visual_claim_support_unassessed': True}}
S.register_plugin(PLUGIN)
