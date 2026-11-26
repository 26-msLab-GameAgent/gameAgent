import math, collections
import numpy as np
import cv2

COLOR_BANDS = [("red", 0, 8), ("orange", 9, 21), ("yellow", 22, 36), ("green", 40, 85),
               ("purple", 125, 162), ("pink", 163, 172), ("red", 173, 179)]
COLORSET = ("red", "orange", "yellow", "green", "purple", "pink")


def _hsv(image):
    arr = np.asarray(image.convert("RGB"))
    return arr, cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)


def _candy_mask(hsv):
    h = hsv[:, :, 0].astype(np.int32)
    s = hsv[:, :, 1].astype(np.int32)
    v = hsv[:, :, 2].astype(np.int32)
    strong = (s > 110) & (v > 110)
    notblue = ~((h > 86) & (h < 124))
    return (strong & notblue).astype(np.uint8)


def _components(mask, min_area, max_area):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]
        if a < min_area or a > max_area:
            continue
        if ww == 0 or hh == 0:
            continue
        ar = float(ww) / float(hh)
        if ar < 0.45 or ar > 2.2:
            continue
        out.append((float(cent[i][0]), float(cent[i][1])))
    return out


def _pitch(comps):
    if len(comps) < 10:
        return None
    d = []
    for i, a in enumerate(comps):
        best = None
        for j, b in enumerate(comps):
            if i == j:
                continue
            dd = math.hypot(a[0] - b[0], a[1] - b[1])
            if best is None or dd < best:
                best = dd
        if best:
            d.append(best)
    if not d:
        return None
    d.sort()
    return d[len(d) // 2]


def _cluster(vals, tol):
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


def _axis(vals, pitch):
    if len(vals) < 8:
        return None
    groups = _cluster(vals, pitch * 0.42)
    if not groups:
        return None
    mx = max(g[1] for g in groups)
    keep = [g[0] for g in groups if g[1] >= max(3, 0.45 * mx)]
    if len(keep) < 4:
        return None
    best = None
    i = 0
    while i < len(keep):
        chain = [keep[i]]
        j = i + 1
        while j < len(keep) and abs((keep[j] - chain[-1]) - pitch) <= 0.3 * pitch:
            chain.append(keep[j])
            j += 1
        if best is None or len(chain) > len(best):
            best = chain
        i = j if j > i + 1 else i + 1
    if best is None or len(best) < 4:
        return None
    return best


def _classify(patch):
    h = patch[:, :, 0].astype(np.int32)
    s = patch[:, :, 1].astype(np.int32)
    v = patch[:, :, 2].astype(np.int32)
    sel = (s > 110) & (v > 110) & ~((h > 86) & (h < 124))
    tot = h.size
    cnt = int(sel.sum())
    if cnt < 0.12 * tot:
        bright = (v > 200) & (s < 60)
        if bright.sum() > 0.45 * tot:
            return "blocker", 0.7
        return "empty", 0.55
    hh = h[sel]
    counts = collections.Counter()
    for name, lo, hi in COLOR_BANDS:
        counts[name] += int(((hh >= lo) & (hh <= hi)).sum())
    name, c = counts.most_common(1)[0]
    if c < 0.45 * cnt:
        return "unknown", 0.4
    return name, min(0.97, 0.6 + 0.4 * c / max(1, cnt))


def observe(image, context):
    w, h = image.width, image.height
    arr, hsv = _hsv(image)
    mask = _candy_mask(hsv)
    m2 = np.zeros_like(mask)
    x0, x1 = int(0.26 * w), int(0.77 * w)
    m2[:, x0:x1] = mask[:, x0:x1]
    comps = _components(m2, max(120, int(0.0004 * w * h)), int(0.02 * w * h))
    if len(comps) < 20:
        return None
    pitch = _pitch(comps)
    if not pitch or pitch < 25 or pitch > 0.25 * min(w, h):
        return None
    xs = _axis([c[0] for c in comps], pitch)
    ys = _axis([c[1] for c in comps], pitch)
    if xs is None or ys is None:
        return None
    cols, rows = len(xs), len(ys)
    if not (4 <= cols <= 10 and 4 <= rows <= 10):
        return None
    cw = (xs[-1] - xs[0]) / max(1, cols - 1)
    ch = (ys[-1] - ys[0]) / max(1, rows - 1)
    if abs(cw - ch) > 0.15 * max(cw, ch):
        return None
    L = max(0, int(round(xs[0] - cw / 2.0)))
    T = max(0, int(round(ys[0] - ch / 2.0)))
    R = min(w - 1, int(round(xs[-1] + cw / 2.0)))
    B = min(h - 1, int(round(ys[-1] + ch / 2.0)))
    if R - L < 60 or B - T < 60:
        return None
    rad = int(max(4, 0.28 * min(cw, ch)))
    centers, labels, confs = [], [], []
    for r in range(rows):
        cro, lro = [], []
        for c in range(cols):
            cx = int(max(L, min(R, round(xs[c]))))
            cy = int(max(T, min(B, round(ys[r]))))
            p = hsv[max(0, cy - rad):cy + rad, max(0, cx - rad):cx + rad]
            if p.size == 0:
                return None
            lab, cf = _classify(p)
            cro.append([cx, cy])
            lro.append(lab)
            confs.append(cf)
        centers.append(cro)
        labels.append(lro)
    flat = [l for row in labels for l in row]
    n = len(flat)
    unknown = sum(1 for l in flat if l == "unknown")
    colored = sum(1 for l in flat if l in COLORSET)
    if unknown > 0.08 * n or colored < 0.75 * n:
        return None
    conf = float(min(0.95, 0.5 + 0.5 * (sum(confs) / max(1, len(confs)))))
    if conf < 0.65:
        return None
    sig = "%dx%d|" % (rows, cols) + "/".join("".join(l[0] for l in row) for row in labels)
    return {"signature": sig, "confidence": conf,
            "board_bbox": [L, T, R, B],
            "state": {"rows": rows, "cols": cols, "labels": labels, "centers": centers},
            "grid": {"rows": rows, "cols": cols, "centers": centers, "labels": labels}}


def _match_count(g, rows, cols):
    total = 0
    for r in range(rows):
        for c in range(cols):
            v = g[r][c]
            if v not in COLORSET:
                continue
            if c + 2 < cols and g[r][c + 1] == v and g[r][c + 2] == v:
                total += 1
            if r + 2 < rows and g[r + 1][c] == v and g[r + 2][c] == v:
                total += 1
    return total


def choose_action(observation, context):
    if not observation:
        return None
    st = observation["state"]
    rows, cols = st["rows"], st["cols"]
    labels = [list(r) for r in st["labels"]]
    centers = st["centers"]
    sig = observation["signature"]
    L, T, R, B = observation["board_bbox"]
    pitch = 0.0
    if cols > 1:
        pitch = abs(centers[0][1][0] - centers[0][0][0])
    if pitch <= 0 and rows > 1:
        pitch = abs(centers[1][0][1] - centers[0][0][1])
    if pitch <= 0:
        pitch = 40.0

    def cl(v, lo, hi):
        return int(max(lo, min(hi, v)))

    def cell_of(x, y):
        if x is None or y is None:
            return None
        best = None
        for r in range(rows):
            for c in range(cols):
                cx, cy = centers[r][c]
                d = abs(cx - x) + abs(cy - y)
                if best is None or d < best[0]:
                    best = (d, r, c)
        if best is None or best[0] > pitch:
            return None
        return (best[1], best[2])

    tried = set()
    failed = context.failed_actions if hasattr(context, "failed_actions") else None
    for e in (failed or []):
        if not isinstance(e, dict) or e.get("signature") != sig:
            continue
        a = e.get("action") or {}
        if not isinstance(a, dict):
            continue
        p = cell_of(a.get("x"), a.get("y"))
        q = cell_of(a.get("x2"), a.get("y2"))
        if p and q:
            tried.add((p, q, int(a.get("duration_ms") or 0)))

    def run_at(g, r, c):
        v = g[r][c]
        if v not in COLORSET:
            return 0, 0
        total, longest = 0, 0
        for dr, dc in ((0, 1), (1, 0)):
            k = 1
            rr, cc = r - dr, c - dc
            while 0 <= rr < rows and 0 <= cc < cols and g[rr][cc] == v:
                k += 1
                rr -= dr
                cc -= dc
            rr, cc = r + dr, c + dc
            while 0 <= rr < rows and 0 <= cc < cols and g[rr][cc] == v:
                k += 1
                rr += dr
                cc += dc
            if k >= 3:
                total += k
                longest = max(longest, k)
        return total, longest

    def near_blocker(r, c):
        n = 0
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and labels[rr][cc] in ("blocker", "empty", "unknown"):
                n += 1
        return n

    cands = []
    for r in range(rows):
        for c in range(cols):
            a = labels[r][c]
            if a not in COLORSET:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                r2, c2 = r + dr, c + dc
                if r2 >= rows or c2 >= cols:
                    continue
                b = labels[r2][c2]
                if b not in COLORSET or b == a:
                    continue
                labels[r][c], labels[r2][c2] = b, a
                t1, l1 = run_at(labels, r, c)
                t2, l2 = run_at(labels, r2, c2)
                labels[r][c], labels[r2][c2] = a, b
                longest = max(l1, l2)
                if longest < 3:
                    continue
                score = longest * 10 + (t1 + t2) * 2
                score += 3.0 * (near_blocker(r, c) + near_blocker(r2, c2))
                score += 0.4 * (r + r2)
                cands.append((-score, r, c, r2, c2))
    if not cands:
        return None
    cands.sort()

    # try each candidate in score order, with alternate swipe direction and duration
    variants = [((0, 1), 90), ((1, 0), 90), ((0, 1), 250), ((1, 0), 250)]
    for negscore, r, c, r2, c2 in cands:
        for order, dur in variants:
            if order[0] == 0:
                pa, pb = (r, c), (r2, c2)
            else:
                pa, pb = (r2, c2), (r, c)
            if (pa, pb, dur) in tried:
                continue
            xa, ya = centers[pa[0]][pa[1]]
            xb, yb = centers[pb[0]][pb[1]]
            return {"type": "swipe",
                    "x": cl(xa, L, R), "y": cl(ya, T, B),
                    "x2": cl(xb, L, R), "y2": cl(yb, T, B),
                    "duration_ms": dur,
                    "reason": "swap (%d,%d)-(%d,%d) score %.1f" % (pa[0], pa[1], pb[0], pb[1], -negscore)}
    return None
