import math
import numpy as np
import cv2

COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink")


def _prep(image):
    arr = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    return arr, hsv


def _candy_mask(hsv):
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)
    nonblue = ~((h >= 98) & (h <= 132))
    return ((s > 95) & (v > 80) & nonblue).astype(np.uint8)


def _blobs(mask, w, h):
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    out = []
    lo = max(10, int(min(w, h) * 0.02))
    hi = int(min(w, h) * 0.22)
    for i in range(1, n):
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        a = int(stats[i, cv2.CC_STAT_AREA])
        if bw < lo or bh < lo or bw > hi or bh > hi:
            continue
        ar = bw / float(bh)
        if ar < 0.5 or ar > 2.0:
            continue
        if a < 0.3 * bw * bh:
            continue
        out.append((float(cent[i][0]), float(cent[i][1]), bw, bh))
    return out


def _pitch(pts):
    if len(pts) < 8:
        return None
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    d = []
    for i in range(len(pts)):
        dd = np.hypot(xs - xs[i], ys - ys[i])
        dd[i] = 1e9
        d.append(float(dd.min()))
    d = [v for v in d if v > 8]
    if len(d) < 8:
        return None
    d.sort()
    return float(d[len(d) // 2])


def _clusters(vals, tol):
    vals = sorted(vals)
    groups = []
    cur = [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)
    return [(sum(g) / len(g), len(g)) for g in groups]


def _chain(clusters, pitch):
    """longest chain of cluster centers spaced ~pitch (allow single gaps)."""
    best = []
    n = len(clusters)
    for i in range(n):
        chain = [clusters[i][0]]
        last = clusters[i][0]
        for j in range(i + 1, n):
            gap = clusters[j][0] - last
            k = int(round(gap / pitch))
            if k >= 1 and k <= 2 and abs(gap - k * pitch) <= 0.28 * pitch:
                chain.append(clusters[j][0])
                last = clusters[j][0]
        if len(chain) > len(best):
            best = chain
    return best


def _axis(chain, pitch):
    if len(chain) < 4:
        return None
    first, last = chain[0], chain[-1]
    steps = int(round((last - first) / pitch))
    if steps < 3:
        return None
    step = (last - first) / float(steps)
    if step <= 0:
        return None
    return first, step, steps + 1


def _label(hsv, cx, cy, r):
    hh, ww = hsv.shape[:2]
    x0, x1 = max(0, cx - r), min(ww, cx + r + 1)
    y0, y1 = max(0, cy - r), min(hh, cy + r + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return "unknown"
    patch = hsv[y0:y1, x0:x1].reshape(-1, 3).astype(np.int16)
    s = patch[:, 1]
    v = patch[:, 2]
    sel = (s > 95) & (v > 80)
    frac = float(sel.mean())
    if frac < 0.22:
        bright = float(((v > 165) & (s < 85)).mean())
        if bright > 0.30:
            return "blocker"
        return "empty"
    hue = patch[sel, 0]
    hist = np.bincount(hue, minlength=180).astype(np.float32)
    hist = np.convolve(hist, np.ones(7, np.float32) / 7.0, mode="same")
    top = int(hist.argmax())
    dom = float(hist[max(0, top - 6):top + 7].sum() / max(1.0, hist.sum()))
    if dom < 0.45:
        return "unknown"
    if top < 10 or top >= 170:
        return "red"
    if top < 22:
        return "orange"
    if top < 40:
        return "yellow"
    if top < 82:
        return "green"
    if top < 100:
        return "blocker"
    if top < 132:
        return "blue"
    if top < 156:
        return "purple"
    return "pink"


def observe(image, context):
    w, h = image.width, image.height
    arr, hsv = _prep(image)
    mask = _candy_mask(hsv)
    pts = _blobs(mask, w, h)
    pitch = _pitch(pts)
    if pitch is None or pitch < 24 or pitch > 0.3 * min(w, h):
        return None
    colc = _clusters([p[0] for p in pts], 0.42 * pitch)
    cx_chain = _chain([c for c in colc if c[1] >= 2], pitch)
    ax = _axis(cx_chain, pitch)
    if ax is None:
        return None
    x0, sx, cols = ax
    inside = [p for p in pts if x0 - 0.6 * sx <= p[0] <= x0 + (cols - 0.4) * sx]
    if len(inside) < 12:
        return None
    rowc = _clusters([p[1] for p in inside], 0.42 * pitch)
    cy_chain = _chain([c for c in rowc if c[1] >= 2], pitch)
    ay = _axis(cy_chain, pitch)
    if ay is None:
        return None
    y0, sy, rows = ay
    if not (4 <= rows <= 12 and 4 <= cols <= 12):
        return None
    if abs(sx - sy) > 0.25 * max(sx, sy):
        return None
    r_s = int(max(5, min(sx, sy) * 0.28))
    centers = []
    labels = []
    for r in range(rows):
        crow = []
        lrow = []
        for c in range(cols):
            cx = int(round(x0 + c * sx))
            cy = int(round(y0 + r * sy))
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            crow.append([cx, cy])
            lrow.append(_label(hsv, cx, cy, r_s))
        centers.append(crow)
        labels.append(lrow)
    total = rows * cols
    known = sum(1 for row in labels for v in row if v in COLORS)
    unk = sum(1 for row in labels for v in row if v == "unknown")
    if known < 0.5 * total or unk > 0.22 * total:
        return None
    left = max(0, int(round(x0 - 0.6 * sx)))
    top = max(0, int(round(y0 - 0.6 * sy)))
    right = min(w - 1, int(round(x0 + (cols - 1) * sx + 0.6 * sx)))
    bottom = min(h - 1, int(round(y0 + (rows - 1) * sy + 0.6 * sy)))
    if not (0 <= left < right < w and 0 <= top < bottom < h):
        return None
    sig = "%dx%d|%d|" % (rows, cols, int(round(sx))) + ";".join(
        "".join(v[0] for v in row) for row in labels)
    return {
        "signature": sig,
        "confidence": 0.86,
        "board_bbox": [left, top, right, bottom],
        "state": {"rows": rows, "cols": cols, "labels": labels, "centers": centers,
                  "cell": [int(round(sx)), int(round(sy))]},
        "grid": {"rows": rows, "cols": cols, "centers": centers, "labels": labels},
    }


def _best_run(grid, rows, cols):
    best = 0
    for r in range(rows):
        for c in range(cols):
            lab = grid[r][c]
            if lab not in COLORS:
                continue
            run = 1
            cc = c + 1
            while cc < cols and grid[r][cc] == lab:
                run += 1
                cc += 1
            if run >= 3:
                best = max(best, run)
            run = 1
            rr = r + 1
            while rr < rows and grid[rr][c] == lab:
                run += 1
                rr += 1
            if run >= 3:
                best = max(best, run)
    return best


def _swap_score(labels, rows, cols, a, b):
    (r1, c1), (r2, c2) = a, b
    l1, l2 = labels[r1][c1], labels[r2][c2]
    if l1 not in COLORS or l2 not in COLORS or l1 == l2:
        return 0
    grid = [row[:] for row in labels]
    grid[r1][c1], grid[r2][c2] = l2, l1
    return _best_run(grid, rows, cols)


def choose_action(observation, context):
    if not observation:
        return None
    st = observation.get("state", {})
    rows = st.get("rows", 0)
    cols = st.get("cols", 0)
    labels = st.get("labels")
    centers = st.get("centers")
    if not labels or not centers or rows < 4 or cols < 4:
        return None
    sig = observation.get("signature")
    banned = set()
    fails = context.failed_actions if hasattr(context, "failed_actions") else None
    for entry in (fails or []):
        if entry.get("signature") == sig:
            act = entry.get("action") or {}
            banned.add((act.get("x"), act.get("y"), act.get("x2"), act.get("y2")))
    bbox = observation["board_bbox"]

    def ok(x, y):
        return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]

    best = None
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0)):
                r2, c2 = r + dr, c + dc
                if r2 >= rows or c2 >= cols:
                    continue
                sc = _swap_score(labels, rows, cols, (r, c), (r2, c2))
                if sc < 3:
                    continue
                x1, y1 = centers[r][c]
                x2, y2 = centers[r2][c2]
                if (x1, y1, x2, y2) in banned:
                    continue
                if not (ok(x1, y1) and ok(x2, y2)):
                    continue
                key = (sc, -(r + c))
                if best is None or key > best[0]:
                    best = (key, x1, y1, x2, y2, sc)
    if best is None:
        return None
    _, x1, y1, x2, y2, sc = best
    return {"type": "swipe", "x": int(x1), "y": int(y1), "x2": int(x2), "y2": int(y2),
            "duration_ms": 160, "reason": "swap creates match of %d" % sc}
