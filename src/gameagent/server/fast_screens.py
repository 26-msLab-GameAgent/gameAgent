"""Conservative visual matching. Unknown images always fall through to semantic inference.

Templates are scene-specific, including the level title, not generic green-button detectors.
They never transfer a stage identity to an unseen title. Images are compared without the
Android pointer debug strip. Button/title anchors have a stricter local error check.
"""
from __future__ import annotations

import base64
import io
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal installs
    cv2 = None


# These references are intentionally conservative.  The ORB matcher only supplies
# a screen state and coarse safe geometry; semantic grounding remains available when
# a reference is ambiguous or the screen needs a precise/stage-specific target.
ORB_REFERENCE_SPECS = {
    'stage': {
        'state': 'S',
        'controls': [],
    },
    'board': {
        'state': 'B',
        # Safe central envelope covering the observed 0916 board layouts.
        # The solver still supplies the actual cell coordinates; this box only
        # prevents actions from being sent to the side HUD/navigation areas.
        'board_bbox': [0.230, 0.030, 0.770, 0.880],
    },
    'popup': {
        'state': 'P',
        'controls': [
            {
                'id': 'orb_popup_close', 'role': 'close', 'label': 'close X',
                'bbox': [0.935, 0.000, 0.998, 0.090],
            },
        ],
    },
    'tutorial': {
        'state': 'T',
        'controls': [
            {
                'id': 'orb_tutorial_start', 'role': 'start', 'label': '게임 시작!',
                'bbox': [0.360, 0.780, 0.660, 0.950],
            },
        ],
    },
    'clear': {
        'state': 'C',
        'controls': [
            {
                'id': 'orb_clear_next', 'role': 'next', 'label': '다음',
                'bbox': [0.250, 0.780, 0.530, 0.950],
            },
        ],
    },
}


def thumbnail(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        # Keep dimensions fixed; ignore the pointer-debug strip only.
        im.paste(im.crop((0, 30, im.width, 31)).resize((im.width, 30)), (0, 0))
        return np.asarray(im.resize((320, 192)), dtype=np.float32)


def local_error(a, b):
    """Maximum block error prevents a changed title digit hiding in global averages."""
    delta = np.abs(a - b).mean(axis=2)
    h, w = delta.shape
    h, w = h // 4 * 4, w // 4 * 4
    return float(delta[:h, :w].reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3)).max())


def crop_relative(array, rect):
    l, t, r, b = rect
    h, w = array.shape[:2]
    return array[int(t*h):max(int(t*h)+4, int(b*h)),
                 int(l*w):max(int(l*w)+4, int(r*w))]


def _orb_image(path=None, pixels=None):
    """Prepare a screenshot/reference at one stable size for feature matching."""
    if cv2 is None:
        return None
    if path is not None:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
    elif pixels is not None:
        array = np.asarray(pixels)
        if array.ndim == 3:
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        else:
            image = np.clip(array, 0, 255).astype(np.uint8)
    else:
        return None
    # Captures contain a 30-pixel Android pointer/debug strip.  Match the
    # existing thumbnail behavior without changing the image geometry.
    if image.shape[0] > 30:
        image[:30] = image[30:31]
    return cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)


class OrbScreenMatcher:
    """Classify known screens using ORB features and a RANSAC homography.

    Whole-screen pixel averages are brittle to scale, animation and small UI
    changes.  ORB keeps the fast local path offline while the homography and
    best-vs-second margin prevent a few generic icons from becoming a screen
    classification.  A failed/ambiguous match returns ``None`` so callers can
    use the existing semantic classifier.
    """

    def __init__(self, reference_dir=None, *, min_good_matches=12,
                 min_inliers=8, min_margin=1.25):
        self.entries = []
        self.min_good_matches = int(min_good_matches)
        self.min_inliers = int(min_inliers)
        self.min_margin = float(min_margin)
        self._orb = None
        self._matcher = None
        if cv2 is None or not reference_dir:
            return
        directory = Path(reference_dir)
        if not directory.is_dir():
            return
        self._orb = cv2.ORB_create(
            nfeatures=2500, scaleFactor=1.2, nlevels=8,
            edgeThreshold=15, fastThreshold=10,
        )
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        for name, spec in ORB_REFERENCE_SPECS.items():
            paths = [directory / f'{name}.png']
            # A single screenshot should not define a whole UI state.  Optional
            # per-state variants let the same ORB state survive level numbers,
            # board contents, scores and other expected visual changes.
            paths.extend(sorted((directory / 'variants' / name).glob('*.png')))
            for path in paths:
                if not path.is_file():
                    continue
                image = _orb_image(path=path)
                if image is None:
                    continue
                keypoints, descriptors = self._orb.detectAndCompute(image, None)
                if descriptors is not None and len(keypoints) >= 4:
                    variant_name = name if path.parent.name == directory.name else f'{name}/{path.stem}'
                    self.entries.append((variant_name, spec, keypoints, descriptors, path))

    @property
    def enabled(self):
        return bool(self.entries and self._orb and self._matcher)

    @staticmethod
    def _confidence(inliers):
        return min(0.99, 0.80 + 0.01 * min(int(inliers), 19))

    @staticmethod
    def _scaled_bbox(rect, screen):
        return [rect[0] * screen.width, rect[1] * screen.height,
                rect[2] * screen.width, rect[3] * screen.height]

    def classify(self, pixels, screen, image_path=None):
        if not self.enabled:
            return None
        image = _orb_image(path=image_path, pixels=pixels)
        if image is None:
            return None
        keypoints, descriptors = self._orb.detectAndCompute(image, None)
        if descriptors is None or len(keypoints) < 4:
            return None

        candidates = []
        for name, spec, ref_keypoints, ref_descriptors, _path in self.entries:
            pairs = self._matcher.knnMatch(descriptors, ref_descriptors, k=2)
            good = [pair[0] for pair in pairs
                    if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
            inliers = 0
            if len(good) >= 4:
                source = np.float32([keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                target = np.float32([ref_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                try:
                    _homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
                    if mask is not None:
                        inliers = int(mask.ravel().sum())
                except cv2.error:
                    # Degenerate feature geometry is an ordinary false-match
                    # outcome; leave this candidate below the acceptance bar.
                    inliers = 0
            candidates.append({
                'name': name,
                'spec': spec,
                'good_matches': len(good),
                'inliers': inliers,
            })

        candidates.sort(key=lambda item: (item['inliers'], item['good_matches']), reverse=True)
        best = candidates[0]
        # Compare against the strongest *different state*, not another variant
        # of the winning state.  Otherwise adding useful examples would make a
        # correct state look ambiguous merely because its bank got larger.
        second = next((item for item in candidates[1:]
                       if item['spec']['state'] != best['spec']['state']), None)
        if best['good_matches'] < self.min_good_matches or best['inliers'] < self.min_inliers:
            return None
        if best['inliers'] / max(best['good_matches'], 1) < 0.30:
            return None
        if second:
            margin = best['inliers'] / max(second['inliers'], 1)
            if margin < self.min_margin and best['inliers'] - second['inliers'] < 4:
                return None

        confidence = self._confidence(best['inliers'])
        spec = best['spec']
        result = {
            'state': spec['state'], 'confidence': confidence,
            'evidence': (
                f"ORB reference '{best['name']}' matched with "
                f"{best['inliers']} geometric inliers/{best['good_matches']} good matches"
            ),
            'stage': None, 'board_bbox': None, 'board_confidence': 0.0,
            'controls': [], 'visible_text': '',
            'orb': {
                'reference': best['name'], 'inliers': best['inliers'],
                'good_matches': best['good_matches'],
                'second_inliers': second['inliers'] if second else 0,
            },
            '_classification_source': 'orb',
        }
        if spec.get('board_bbox'):
            result['board_bbox'] = self._scaled_bbox(spec['board_bbox'], screen)
            result['board_confidence'] = confidence
        for control in spec.get('controls', []):
            result['controls'].append({
                **control,
                'bbox': self._scaled_bbox(control['bbox'], screen),
                'confidence': confidence,
                'enabled': True,
                'free': True,
            })
        return result


class VerifiedScreens:
    def __init__(self, manifest=None, orb_references=None, *,
                 orb_min_good_matches=12, orb_min_inliers=8, orb_min_margin=1.25):
        self.entries = []
        self.orb = OrbScreenMatcher(
            orb_references,
            min_good_matches=orb_min_good_matches,
            min_inliers=orb_min_inliers,
            min_margin=orb_min_margin,
        )
        if manifest and Path(manifest).is_file():
            doc = json.loads(Path(manifest).read_text())
            for entry in doc['templates']:
                with Image.open(io.BytesIO(base64.b64decode(entry['image']))) as im:
                    pixels = np.asarray(im.convert('RGB'), dtype=np.float32)
                if pixels.shape != (192, 320, 3):
                    raise ValueError('invalid UI template dimensions')
                self.entries.append((entry, pixels))

    def classify(self, pixels, screen, image_path=None):
        # ORB references cover the broad screen taxonomy before the older exact
        # templates.  If the match is uncertain, retain the conservative fallback.
        result = self.orb.classify(pixels, screen, image_path=image_path)
        if result is not None:
            return result
        for entry, reference in self.entries:
            if entry['screen'] != [screen.width, screen.height]:
                continue
            if np.abs(pixels-reference).mean() > 3.0:
                continue
            if any(local_error(crop_relative(pixels, box), crop_relative(reference, box)) > 8.0
                   for box in entry['anchors']):
                continue
            result = deepcopy(entry['classification'])
            result['evidence'] = 'Verified local UI template: ' + entry['name']
            return result
        return None


def stationary(a, b):
    return a is not None and float(np.abs(a-b).mean()) < 1.5 and local_error(a, b) < 14


def board_stationary(a, b, bbox, screen):
    """Small board-only pixel change; callers also require identical grid labels.

    Ignore animated HUD/characters and tolerate small candy highlights, but not
    a newly appearing overlay or a broadly changing/cascading board.
    """
    if a is None or not bbox:
        return False
    l, t, r, bottom = bbox
    rect = (l / screen.width, t / screen.height,
            r / screen.width, bottom / screen.height)
    return float(np.abs(crop_relative(a, rect)-crop_relative(b, rect)).mean()) < 4.0


def same_board_chrome(a, b, bbox, screen):
    """Only reuse a semantically confirmed board when its HUD and frame still match."""
    if a is None or not bbox:
        return False
    mask = np.ones(a.shape[:2], dtype=bool)
    l, t, r, bot = bbox
    h, w = mask.shape
    mask[max(0, int(t/screen.height*h)-2):min(h, int(bot/screen.height*h)+3),
         max(0, int(l/screen.width*w)-2):min(w, int(r/screen.width*w)+3)] = False
    # Move/goal counters change normally; retain outer margins and top-left menu.
    mask[int(.10*h):int(.91*h), :int(.20*w)] = False
    if mask.mean() < .12:
        return False
    return float(np.abs(a-b).mean(axis=2)[mask].mean()) < 2.5
