import numpy as np
import cv2

ROWS, COLS = 9, 7
NUMS = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"}


def _hsv(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return hsv[:, :, 0].astype(np.int32), hsv[:, :, 1].astype(np.int32), hsv[:, :, 2].astype(np.int32)


def _pink(H, S, V):
    return (((H <= 22) | (H >= 155)) & (S > 15) & (S < 175) & (V > 130))


def _blue(H, S, V):
    return ((H >= 88) & (H <= 128) & (S > 12) & (S < 140) & (V > 145))


def _board_bbox(rgb):
    h, w = rgb.shape[:2]
    H, S, V = _hsv(rgb)
    m = _pink(H, S, V).astype(np.uint8)
    k = max(3, int(min(h, w) * 0.006)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k * 3, k * 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for i in range(1, n):
        x, y, cw, ch, a = stats[i]
        if cw < 0.35 * w or ch < 0.25 * h:
            continue
        if a < 0.05 * w * h:
            continue
        if best is None or a > best[4]:
            best = (int(x), int(y), int(cw), int(ch), int(a))
    if best is None:
        return None
    x, y, cw, ch, _ = best
    return [x, y, x + cw, y + ch]


def _glyph(sub):
    sub = cv2.copyMakeBorder(sub, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    h, w = sub.shape
    cnts, hier = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    if hier is not None and len(hier) > 0:
        hh = hier[0]
        for i in range(len(cnts)):
            if hh[i][3] >= 0 and cv2.contourArea(cnts[i]) > 0.015 * float(w * h):
                holes.append((cv2.contourArea(cnts[i]), cnts[i]))
    ink = sub > 0
    hh_ = float(h - 4)
    ww_ = float(w - 4)
    if hh_ <= 0 or ww_ <= 0:
        return "?"
    if ww_ / hh_ < 0.46:
        return "1"
    if len(holes) >= 2:
        return "8"
    if len(holes) == 1:
        a, c = holes[0]
        mm = cv2.moments(c)
        cy = (mm["m01"] / mm["m00"]) if mm["m00"] > 0 else h * 0.5
        rel = (cy - 2.0) / hh_
        ratio = a / float(ww_ * hh_)
        if ratio > 0.20:
            return "0"
        if rel < 0.45:
            return "4" if ratio < 0.10 else "9"
        return "6"
    top = ink[:int(h * 0.25), :]
    bot = ink[int(h * 0.80):, :]
    bl = ink[int(h * 0.60):, :int(w * 0.45)]
    tf = float(top.mean()) if top.size else 0.0
    bf = float(bot.mean()) if bot.size else 0.0
    blf = float(bl.mean()) if bl.size else 0.0
    if blf < 0.10 and tf > 0.30:
        return "7"
    if bf > 0.45:
        return "2"
    return "3"


def _digits(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    Hm = mask.shape[0]
    comps = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if h < Hm * 0.32 or a < 8:
            continue
        comps.append((int(x), int(y), int(w), int(h), i))
    if not comps or len(comps) > 2:
        return None
    comps.sort()
    out = ""
    for (x, y, w, h, i) in comps:
        sub = ((lab[y:y + h, x:x + w] == i).astype(np.uint8)) * 255
        out += _glyph(sub)
    return out


def _color_name(patch, objmask):
    if objmask.sum() < 12:
        return "unknown"
    H, S, V = _hsv(patch)
    hs = H[objmask]
    ss = S[objmask]
    vs = V[objmask]
    sat = ss > 60
    if sat.mean() < 0.22:
        mv = float(np.median(vs))
        if mv > 190:
            return "white"
        if mv < 80:
            return "black"
        return "white"
    hh = hs[sat]
    med = float(np.median(hh))
    mv = float(np.median(vs[sat]))
    if med < 8 or med >= 168:
        return "pink" if mv > 190 else "red"
    if med < 18:
        return "orange" if mv > 120 else "red"
    if med < 34:
        return "yellow"
    if med < 85:
        return "green"
    if med < 130:
        return "blue"
    if med < 150:
        return "purple"
    return "pink"


def _cell(rgb, cx, cy, r):
    h, w = rgb.shape[:2]
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, None, None
    patch = rgb[y0:y1, x0:x1]
    H, S, V = _hsv(patch)
    bg = _pink(H, S, V) | _blue(H, S, V)
    obj = ~bg
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    R = patch[:, :, 0].astype(np.int32)
    G = patch[:, :, 1].astype(np.int32)
    B = patch[:, :, 2].astype(np.int32)
    med = float(np.median(gray))
    gm = ((gray < med - 24) & (R > B + 10) & (R > G + 10)).astype(np.uint8) * 255
    gf = float((gm > 0).mean())
    d = _digits(gm) if gf > 0.012 else None
    objf = float(obj.mean())
    if d is not None and d in NUMS and objf < 0.22:
        return d, patch, obj
    if objf > 0.12:
        return _color_name(patch, obj), patch, obj
    if gf > 0.03 and float(gray.std()) > 24.0:
        return "blocker", patch, obj
    return "empty", patch, obj


def _desc(patch, obj):
    H, S, V = _hsv(patch)
    m = obj & (S > 40) & (V > 40)
    hist = np.zeros(12, np.float32)
    tot = float(m.sum())
    if tot > 0:
        bins = np.clip((H[m] / 15).astype(np.int32), 0, 11)
        for b in bins:
            hist[int(b)] += 1.0
        hist /= tot
    dark = float((V[obj] < 90).mean()) if obj.sum() > 0 else 0.0
    bright = float(((V > 220) & (S < 40) & obj).mean())
    vec = list(hist) + [dark, bright, float(obj.mean())]
    return [int(round(float(x) * 8)) for x in vec]


def observe(image, context):
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    if w < 200 or h < 200:
        return None
    bbox = _board_bbox(rgb)
    if bbox is None:
        return None
    l, t, r, b = bbox
    bw = r - l
    bh = b - t
    cw = bw / float(COLS)
    ch = bh / float(ROWS)
    if cw < 12 or ch < 12:
        return None
    if abs(cw - ch) > 0.13 * max(cw, ch):
        return None
    board = rgb[t:b, l:r]
    bh_, bs_, bv_ = _hsv(board)
    if float(np.median(bv_)) < 110.0:
        return None
    pr = int(max(4, min(cw, ch) * 0.32))
    centers, labels, items = [], [], []
    for i in range(ROWS):
        rowc, rowl = [], []
        for j in range(COLS):
            cx = int(round(l + (j + 0.5) * cw))
            cy = int(round(t + (i + 0.5) * ch))
            cx = max(l + 1, min(r - 2, cx))
            cy = max(t + 1, min(b - 2, cy))
            rowc.append([int(cx), int(cy)])
            lab, patch, obj = _cell(rgb, cx, cy, pr)
            if lab is None:
                return None
            rowl.append(lab)
            if lab not in NUMS and lab not in ("empty", "blocker"):
                items.append({"r": int(i), "c": int(j), "label": lab, "d": _desc(patch, obj)})
        centers.append(rowc)
        labels.append(rowl)
    n_num = sum(1 for row in labels for v in row if v in NUMS)
    if n_num < 6:
        return None
    sig = "tt:%dx%d#" % (ROWS, COLS)
    sig += "|".join(",".join(row) for row in labels)
    sig += "#" + ";".join("%d,%d:%s" % (d["r"], d["c"], "-".join(str(x) for x in d["d"])) for d in items)
    return {
        "signature": sig,
        "confidence": 0.88,
        "board_bbox": [int(l), int(t), int(r), int(b)],
        "state": {"items": items, "n_items": int(len(items))},
        "grid": {"rows": ROWS, "cols": COLS, "centers": centers, "labels": labels},
    }


def _dist(a, b):
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b))


def choose_action(observation, context):
    if not observation:
        return None
    grid = observation.get("grid", None)
    if not grid:
        return None
    centers = grid["centers"]
    bbox = observation["board_bbox"]
    sig = observation.get("signature", "")
    failed = []
    fa = context.failed_actions if hasattr(context, "failed_actions") else None
    if fa:
        for e in fa:
            if isinstance(e, dict) and e.get("signature", None) == sig:
                a = e.get("action", None)
                if isinstance(a, dict):
                    failed.append(a)

    def blocked(act):
        for f in failed:
            if f.get("type", "") != act["type"]:
                continue
            if abs(int(f.get("x", -99)) - act["x"]) <= 6 and abs(int(f.get("y", -99)) - act["y"]) <= 6:
                if act["type"] != "swipe":
                    return True
                if abs(int(f.get("x2", -99)) - act["x2"]) <= 6 and abs(int(f.get("y2", -99)) - act["y2"]) <= 6:
                    return True
        return False

    def inside(x, y):
        return bbox[0] + 2 <= x <= bbox[2] - 2 and bbox[1] + 2 <= y <= bbox[3] - 2

    items = observation.get("state", {}).get("items", [])
    items = [it for it in items if it.get("label", "unknown") != "unknown"]
    pairs = []
    for a in range(len(items)):
        for c in range(a + 1, len(items)):
            ia, ib = items[a], items[c]
            if ia["label"] != ib["label"]:
                continue
            d = _dist(ia["d"], ib["d"])
            if d <= 2.0:
                pairs.append((d, abs(ia["r"] - ib["r"]) + abs(ia["c"] - ib["c"]), ia, ib))
    pairs.sort(key=lambda p: (p[0], p[1]))
    for d, man, ia, ib in pairs:
        x1, y1 = centers[ia["r"]][ia["c"]]
        x2, y2 = centers[ib["r"]][ib["c"]]
        if not (inside(x1, y1) and inside(x2, y2)):
            continue
        act = {"type": "swipe", "x": int(x1), "y": int(y1), "x2": int(x2), "y2": int(y2),
               "duration_ms": 320, "reason": "drag item onto matching item to merge"}
        if not blocked(act):
            return act
        act2 = dict(act)
        act2["x"], act2["y"], act2["x2"], act2["y2"] = int(x2), int(y2), int(x1), int(y1)
        act2["reason"] = "drag matching item (reverse)"
        if not blocked(act2):
            return act2
    return None
