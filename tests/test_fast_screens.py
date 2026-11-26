from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from gameagent.server.fast_screens import VerifiedScreens, cv2, thumbnail


pytestmark = pytest.mark.skipif(cv2 is None, reason='opencv is not installed')


@pytest.fixture
def orb_references():
    return Path(__file__).resolve().parents[1] / 'assets/router_references/candy_crush_soda'


@pytest.mark.parametrize('name, expected', [
    ('stage', 'S'),
    ('board', 'B'),
    ('popup', 'P'),
    ('tutorial', 'T'),
    ('clear', 'C'),
])
def test_orb_classifies_reference_screens(orb_references, name, expected):
    path = orb_references / f'{name}.png'
    with Image.open(path) as image:
        screen = SimpleNamespace(width=image.width, height=image.height)

    matcher = VerifiedScreens(orb_references=orb_references)
    result = matcher.classify(thumbnail(path), screen, image_path=path)

    assert result['state'] == expected
    assert result['_classification_source'] == 'orb'
    assert result['orb']['reference'] == name
    assert result['orb']['inliers'] >= 8


def test_orb_returns_none_for_unrelated_blank_screen(tmp_path, orb_references):
    path = tmp_path / 'blank.png'
    Image.new('RGB', (1920, 1080), (0, 0, 0)).save(path)
    screen = SimpleNamespace(width=1920, height=1080)
    matcher = VerifiedScreens(orb_references=orb_references)

    assert matcher.classify(thumbnail(path), screen, image_path=path) is None


def test_orb_variant_bank_handles_different_level_numbers(orb_references):
    matcher = VerifiedScreens(orb_references=orb_references)
    for name in ('level_1', 'level_2', 'level_3'):
        path = orb_references / 'variants' / 'stage' / f'{name}.png'
        with Image.open(path) as image:
            screen = SimpleNamespace(width=image.width, height=image.height)
        result = matcher.classify(thumbnail(path), screen, image_path=path)
        assert result['state'] == 'S'
        assert result['orb']['reference'] == f'stage/{name}'
