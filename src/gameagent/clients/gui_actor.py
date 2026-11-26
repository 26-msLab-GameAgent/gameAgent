"""GUI Actor HTTP grounding; scores are ranking signals, not probabilities."""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4


class GroundingError(ValueError):
    pass


class GuiActorClient:
    def __init__(self, url: str, timeout_s: float = 20, top_k: int = 5):
        self.url = url.rstrip('/')
        self.timeout_s = timeout_s
        self.top_k = top_k

    def ground(self, image_path: Path, instruction: str, width: int, height: int,
               target_bbox: list | None = None) -> tuple[dict, dict]:
        image = image_path.read_bytes()
        request_id = uuid4().hex
        payload = {'screenshot_base64': base64.b64encode(image).decode('ascii'),
                   'instruction': instruction, 'top_k': self.top_k, 'request_id': request_id}
        request = Request(self.url + '/v1/ground',
                          data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                          headers={'Content-Type': 'application/json'}, method='POST')
        with urlopen(request, timeout=self.timeout_s) as response:
            body = response.read(2 * 1024 * 1024 + 1)
        if len(body) > 2 * 1024 * 1024:
            raise GroundingError('GUI Actor response too large')
        result = json.loads(body)
        if not isinstance(result, dict):
            raise GroundingError('GUI Actor response must be an object')
        if (result.get('image_width'), result.get('image_height')) != (width, height):
            raise GroundingError('GUI Actor image dimensions differ from captured screen')
        if result.get('image_sha256') != hashlib.sha256(image).hexdigest():
            raise GroundingError('GUI Actor response belongs to a different image')
        if result.get('request_id') != request_id or result.get('coord_mode') != 'norm1':
            raise GroundingError('GUI Actor request/coordinate mode mismatch')
        candidates = result.get('candidates')
        if not isinstance(candidates, list):
            raise GroundingError('GUI Actor candidates must be a list')
        valid = []
        for c in candidates:
            if not isinstance(c, dict) or type(c.get('rank')) is not int or c['rank'] < 1:
                continue
            x, y = c.get('pixel_x'), c.get('pixel_y')
            nx, ny = c.get('x'), c.get('y')
            if type(x) is not int or type(y) is not int or not (0 <= x < width and 0 <= y < height):
                continue
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0 <= v <= 1 for v in (nx, ny)):
                continue
            if abs(nx * width - x) > 1.5 or abs(ny * height - y) > 1.5:
                continue
            if target_bbox is not None:
                l, t, r, b = target_bbox
                if not (l <= x <= r and t <= y <= b):
                    continue
            valid.append(c)
        if not valid:
            raise GroundingError('GUI Actor returned no valid candidate for the allowed target')
        chosen = min(valid, key=lambda c: c['rank'])
        action = {'type': 'tap', 'x': chosen['pixel_x'], 'y': chosen['pixel_y'],
                  'duration_ms': 80, 'reason': instruction}
        return action, {'request_id': request_id, 'instruction': instruction,
                        'selected_rank': chosen['rank'], 'response': result}
