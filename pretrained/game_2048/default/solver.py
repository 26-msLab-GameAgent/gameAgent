import math
import numpy as np
import cv2

BOARD_BG = (187, 173, 160)
EMPTY = (205, 193, 180)
PALETTE = {
    "2": (238, 228, 218),
    "4": (237, 224, 200),
    "8": (242, 177, 121),
    "16": (245, 149, 99),
    "32": (246, 124, 95),
    "64": (246, 94, 59),
    "128": (237, 207, 114),
    "256": (237, 204, 97),
    "512": (237, 200, 80),
    "1024": (237, 197, 63),
    "2048": (237, 194, 46),
}
PAD_R = 0.0217
CELL_R = 0.2231
PITCH_R = 0.2441


def _dist(a, b):
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def _bands(mask_1d, min_len):
    out = []
    start = None
    for i, v in enumerate(mask_1d):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    if start is not None and len(mask_1d) - start >= min_len:
        out.append((start, len(mask_1d) - 1))
    return out


def _find_board(arr):
    h, w = arr.shape[:2]
    diff = np.abs(arr.astype(np.int16) - np.array(BOARD_BG, dtype=np.int16))
    m = (diff.max(axis=2) < 30).astype(np.uint8)
    if int(m.sum()) < 0.005 * h * w:
        return None
    k = max(3, int(0.012 * min(h, w)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if cw < 0.35 * w or ch < 0.12 * h:
            continue
        if abs(cw - ch) > 0.18 * max(cw, ch):
            continue
        if best is None or area > best[4]:
            best = (x, y, cw, ch, area)
    if best is None:
        return None
    x, y, cw, ch, _ = best
    return int(x), int(y), int(x + cw - 1), int(y + ch - 1)


def _frame_color(arr, box):
    l, t, r, b = box
    sz = max(r - l, b - t)
    ring = max(2, int(0.012 * sz))
    sub = arr[t:b + 1, l:r + 1]
    parts = [sub[:ring, :].reshape(-1, 3), sub[-ring:, :].reshape(-1, 3),
             sub[:, :ring].reshape(-1, 3), sub[:, -ring:].reshape(-1, 3)]
    pix = np.concatenate(parts, axis=0).astype(np.float32)
    med = np.median(pix, axis=0)
    return (float(med[0]), float(med[1]), float(med[2]))


def _scale(frame):
    s = []
    for i in range(3):
        v = frame[i] / float(BOARD_BG[i]) if BOARD_BG[i] else 1.0
        s.append(min(1.25, max(0.8, v)))
    return s


def _axis_bands(notbg, axis, length, horiz_kernel):
    if horiz_kernel:
        er = cv2.erode(notbg, np.ones((1, 9), np.uint8))
    else:
        er = cv2.erode(notbg, np.ones((9, 1), np.uint8))
    frac = er.mean(axis=axis)
    return _bands(frac > 0.5, max(2, int(length * 0.08)))


def _geometry(arr, box, frame):
    l, t, r, b = box
    sub = arr[t:b + 1, l:r + 1].astype(np.int16)
    d = np.abs(sub - np.array(frame, dtype=np.int16)).max(axis=2)
    notbg = (d > 20).astype(np.uint8)
    H, W = notbg.shape
    cb = _axis_bands(notbg, 0, W, True)
    rb = _axis_bands(notbg, 1, H, False)
    exact = (len(cb) == 4 and len(rb) == 4)
    if not exact:
        cb = _uniform(W)
        rb = _uniform(H)
    return ([(l + a, l + c) for a, c in cb],
            [(t + a, t + c) for a, c in rb], exact)


def _uniform(size):
    pad = PAD_R * size
    cell = CELL_R * size
    pitch = PITCH_R * size
    out = []
    for i in range(4):
        a = pad + i * pitch
        out.append((int(round(a)), int(round(a + cell - 1))))
    return out


def _label_tile(tile, s):
    hh, ww = tile.shape[:2]
    if hh < 8 or ww < 8:
        return "unknown"
    c = tile[int(hh * 0.18):int(hh * 0.82), int(ww * 0.18):int(ww * 0.82)]
    if c.size == 0:
        return "unknown"
    flat = c.reshape(-1, 3).astype(np.float32)
    med = np.median(flat, axis=0)
    bg = (float(med[0]), float(med[1]), float(med[2]))
    cands = [("empty", EMPTY), ("frame", BOARD_BG)] + list(PALETTE.items())
    best = None
    for name, col in cands:
        pred = (col[0] * s[0], col[1] * s[1], col[2] * s[2])
        dd = _dist(bg, pred)
        if best is None or dd < best[1]:
            best = (name, dd)
    if best[1] > 34:
        return "unknown"
    if best[0] == "frame":
        return "empty"
    return best[0]


def observe(image, context):
    arr = np.asarray(image.convert("RGB"))
    box = _find_board(arr)
    if box is None:
        return None
    if box[2] - box[0] < 60 or box[3] - box[1] < 60:
        return None
    frame = _frame_color(arr, box)
    if _dist(frame, BOARD_BG) > 60:
        l0, t0, r0, b0 = box
        sub = arr[t0:b0 + 1, l0:r0 + 1].astype(np.int16)
        dd = np.abs(sub - np.array(BOARD_BG, dtype=np.int16)).max(axis=2)
        mm = (dd < 30).astype(np.uint8)
        if int(mm.sum()) < 400:
            return None
        rows = np.where(mm.mean(axis=1) > 0.35)[0]
        cols = np.where(mm.mean(axis=0) > 0.35)[0]
        if rows.size < 40 or cols.size < 40:
            return None
        nb = (int(l0 + cols[0]), int(t0 + rows[0]),
              int(l0 + cols[-1]), int(t0 + rows[-1]))
        if nb[2] - nb[0] < 60 or nb[3] - nb[1] < 60:
            return None
        if abs((nb[2] - nb[0]) - (nb[3] - nb[1])) > 0.18 * max(nb[2] - nb[0], nb[3] - nb[1]):
            return None
        nf = _frame_color(arr, nb)
        if _dist(nf, BOARD_BG) > 60:
            return None
        box = nb
        frame = nf
    s = _scale(frame)
    cb, rb, exact = _geometry(arr, box, frame)
    centers = []
    labels = []
    for (t0, t1) in rb:
        crow = []
        lrow = []
        for (l0, l1) in cb:
            crow.append([int((l0 + l1) // 2), int((t0 + t1) // 2)])
            tile = arr[max(0, t0):t1 + 1, max(0, l0):l1 + 1]
            lrow.append(_label_tile(tile, s))
        centers.append(crow)
        labels.append(lrow)
    unk = sum(1 for row in labels for v in row if v == "unknown")
    if unk > 3:
        return None
    conf = 0.93 if exact else 0.85
    if unk:
        conf -= 0.1 * unk
    sig = "2048|" + "/".join(",".join(r) for r in labels)
    return {
        "signature": sig,
        "confidence": round(float(conf), 3),
        "board_bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
        "state": {"labels": labels, "unknown": int(unk)},
        "grid": {"rows": 4, "cols": 4, "centers": centers, "labels": labels},
    }


def _to_nums(labels):
    out = []
    for row in labels:
        r = []
        for v in row:
            if v in ("empty", "unknown"):
                r.append(0)
            else:
                r.append(int(v))
        out.append(r)
    return out


def _slide(line):
    vals = [v for v in line if v != 0]
    res = []
    i = 0
    gain = 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i + 1]:
            res.append(vals[i] * 2)
            gain += vals[i] * 2
            i += 2
        else:
            res.append(vals[i])
            i += 1
    while len(res) < len(line):
        res.append(0)
    return res, gain


def _move(board, d):
    b = [row[:] for row in board]
    gain = 0
    if d in ("left", "right"):
        for i in range(4):
            line = b[i][:]
            if d == "right":
                line = line[::-1]
            nl, g = _slide(line)
            if d == "right":
                nl = nl[::-1]
            b[i] = nl
            gain += g
    else:
        for j in range(4):
            line = [b[i][j] for i in range(4)]
            if d == "down":
                line = line[::-1]
            nl, g = _slide(line)
            if d == "down":
                nl = nl[::-1]
            for i in range(4):
                b[i][j] = nl[i]
            gain += g
    return b, gain, b != board


def _score(board):
    empt = sum(1 for r in board for v in r if v == 0)
    w = [[12, 8, 5, 3], [8, 5, 3, 2], [5, 3, 2, 1], [3, 2, 1, 0]]
    pos = 0.0
    smooth = 0.0
    for i in range(4):
        for j in range(4):
            v = board[i][j]
            if v <= 0:
                continue
            pos += math.log(v, 2) * w[i][j]
            if j + 1 < 4 and board[i][j + 1] > 0:
                smooth -= abs(math.log(v, 2) - math.log(board[i][j + 1], 2))
            if i + 1 < 4 and board[i + 1][j] > 0:
                smooth -= abs(math.log(v, 2) - math.log(board[i + 1][j], 2))
    return pos * 2.0 + empt * 6.0 + smooth * 1.5


def _best(board, depth):
    best_v = None
    for d in ("left", "up", "right", "down"):
        nb, gain, ch = _move(board, d)
        if not ch:
            continue
        v = gain * 0.5 + _score(nb)
        if depth > 0:
            sv = _best(nb, depth - 1)
            if sv is not None:
                v += 0.6 * sv
        if best_v is None or v > best_v:
            best_v = v
    return best_v


def choose_action(observation, context):
    if not observation:
        return None
    board = _to_nums(observation["state"]["labels"])
    sig = observation["signature"]
    banned = set()
    fails = context.failed_actions if hasattr(context, "failed_actions") else None
    for fa in (fails or []):
        if fa.get("signature") == sig:
            rsn = (fa.get("action") or {}).get("reason", "")
            for d in ("left", "right", "up", "down"):
                if d in rsn:
                    banned.add(d)
    options = []
    for d in ("left", "up", "right", "down"):
        nb, gain, ch = _move(board, d)
        if not ch or d in banned:
            continue
        sv = _best(nb, 1)
        v = gain * 0.5 + _score(nb) + (0.6 * sv if sv is not None else 0.0)
        options.append((v, d))
    if not options:
        return None
    options.sort(reverse=True)
    d = options[0][1]
    l, t, r, b = observation["board_bbox"]
    cx = (l + r) // 2
    cy = (t + b) // 2
    dx = int((r - l) * 0.32)
    dy = int((b - t) * 0.32)
    if d == "left":
        x1, y1, x2, y2 = cx + dx, cy, cx - dx, cy
    elif d == "right":
        x1, y1, x2, y2 = cx - dx, cy, cx + dx, cy
    elif d == "up":
        x1, y1, x2, y2 = cx, cy + dy, cx, cy - dy
    else:
        x1, y1, x2, y2 = cx, cy - dy, cx, cy + dy
    cl = lambda v, lo, hi: int(min(max(v, lo + 2), hi - 2))
    return {"type": "swipe", "x": cl(x1, l, r), "y": cl(y1, t, b),
            "x2": cl(x2, l, r), "y2": cl(y2, t, b),
            "duration_ms": 160, "reason": "swipe " + d + " to merge tiles"}
