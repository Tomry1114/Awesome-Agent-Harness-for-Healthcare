"""Regressions found while exercising the real study stack."""
import importlib.util
import io
from pathlib import Path
import sys

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'runner'), str(ROOT / 'runner/scorers')]
import governance
import substrate
import dim_context
import scoring
from adapters.user_study.conversion import enable_live_image, task_base


@pytest.mark.parametrize('bench', ['MedRBench', 'MeissaEval'])
def test_study_governance_allows_visible_images_but_rejects_invented_tools(bench):
    policy = governance.resolve_policy_strict(bench)
    assert policy['benchmark'] == bench
    evidence = governance.g2_provenance_honesty('I used SegmentRegionAroundPoint to obtain a segmentation.', [], policy)
    assert 'claimed_unperformed_tool:SegmentRegionAroundPoint' in evidence['violations']
    access = governance.g1_information_access('Looking at the image I can see the finding.', [], policy)
    assert not access['violations']


def test_live_sam_definition_uses_real_worker_point_syntax():
    task = enable_live_image(task_base('MeissaEval', 'fixture', 'Question', 'Private answer',
                                       kind='fixture', group='fixture'))
    definition = next(x for x in task['available_tools'] if x['name'] == 'SegmentRegionAroundPoint')
    assert 'x=500 y=500' in definition['description']


def test_case_local_study_does_not_invent_a_failed_subject_selection():
    task = task_base('MedRBench', 'MEDR-test', 'Question', 'Private answer', kind='fixture', group='fixture')
    plugin, _ = substrate.require_plugin('MedRBench')
    policy = substrate.dimension_policy(task, plugin)
    assert policy['expected_subject']['id'] is None
    result = dim_context.context([], [], policy, judge_model=False)
    assert result['score'] is None
    assert result['submetrics']['expected_subject_match']['status'] == 'not_applicable'
    assert scoring._assigned_subject_id(policy, {'goal': 'Review DEN-001'}) == (None, 'explicit_no_subject')
    # A real explicit subject constraint still wins over the plugin's default.
    task['expected_subject'] = {'type': 'Patient', 'id': 'explicit'}
    assert substrate.dimension_policy(task, plugin)['expected_subject']['id'] == 'explicit'


def test_zoom_bad_coordinates_are_argument_errors_not_environment_failures():
    from adapters.user_study.meissa_image import MeissaImageBackend
    image = io.BytesIO()
    Image.new('RGB', (100, 100)).save(image, format='PNG')
    backend = MeissaImageBackend()
    bad = backend.execute('ZoomInSubfigure', image.getvalue(), '[0.5,0,1,1]')
    assert bad['error_type'] == 'invalid_arguments'
    assert bad['environment_failure'] is False
    good = backend.execute('ZoomInSubfigure', image.getvalue(), '[500,0,1000,1000]')
    assert good['execution']['status'] == 'success'
    task = enable_live_image(task_base('MeissaEval', 'fixture', 'Question', 'Private answer', kind='fixture', group='fixture'))
    definition = next(x for x in task['available_tools'] if x['name'] == 'ZoomInSubfigure')
    assert '0–1000' in definition['description']


def test_image_overlap_detects_reencoding_and_preserves_spatial_structure():
    # The audit's perceptual hash catches images that differ as files/pixels.
    pytest.importorskip('pyarrow')
    sys.path.insert(0, str(ROOT / 'user_study/scripts'))
    spec = importlib.util.spec_from_file_location('pathvqa_audit', ROOT / 'user_study/scripts/prepare_pathvqa.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = Image.new('RGB', (128, 96))
    image.putdata([(x * 2, y * 2, (x + y) % 256) for y in range(96) for x in range(128)])
    a, b = io.BytesIO(), io.BytesIO()
    image.save(a, format='PNG')
    image.save(b, format='JPEG', quality=90)
    first, second = module.image_identity(a.getvalue()), module.image_identity(b.getvalue())
    assert first['sha256'] != second['sha256'] and first['rgb_sha256'] != second['rgb_sha256']
    distance = (int(first['dhash128'], 16) ^ int(second['dhash128'], 16)).bit_count()
    assert distance <= 4


def test_biomed_worker_preserves_each_prompt_mask():
    import numpy as np
    spec = importlib.util.spec_from_file_location('study_worker', ROOT / 'user_study/scripts/serve_meissa_worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    masks = np.stack([np.full((7, 9), .2), np.full((7, 9), .8)])
    result = module.split_biomed_masks(masks, 2)
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], masks[0])
    np.testing.assert_array_equal(result[1], masks[1])
    with pytest.raises(ValueError, match='count/shape'):
        module.split_biomed_masks(masks, 1)
    with pytest.raises(ValueError, match='invalid mask'):
        module.split_biomed_masks(np.full((1, 7, 9), np.nan), 1)
