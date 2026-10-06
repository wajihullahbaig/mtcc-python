"""Minutiae extraction from the binary ridge map: thinning, crossing number, ridge tracing, cleaning.

Each minutia is a row (x, y, theta, type, quality) with theta in [0, 2*pi) measured in
image axes (x right, y down) and type 1 = termination, 3 = bifurcation.
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

# 8-neighbourhood in circular order as (dy, dx); odd indices are 4-neighbours.
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _crossing_number(skel):
    """Crossing number of every skeleton pixel (1 = ending, 3 = bifurcation)."""
    s = np.pad(skel, 1).astype(np.int8)
    h, w = skel.shape
    ring = [s[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in _NB]
    return (sum(np.abs(ring[i] - ring[(i + 1) % 8]) for i in range(8)) // 2) * skel


def _branch_starts(skel, y, x):
    """First pixel of each ridge leaving (y, x): one per run of skeleton neighbours, 4-neighbours preferred."""
    on = [bool(skel[y + dy, x + dx]) for dy, dx in _NB]
    k0 = next((k for k in range(8) if not on[k]), None)
    if k0 is None:
        return []
    starts, run = [], []
    for k in list(range(k0 + 1, 8)) + list(range(k0 + 1)):
        if on[k]:
            run.append(k)
        elif run:
            best = next((r for r in run if r % 2), run[0])
            starts.append((y + _NB[best][0], x + _NB[best][1]))
            run = []
    return starts


def _trace(skel, start, visited, length):
    """Follow a ridge from `start` for up to `length` pixels; return (end point, steps taken)."""
    cur, steps = start, 1
    while steps < length:
        cy, cx = cur
        nxt = [(cy + dy, cx + dx) for dy, dx in _NB if skel[cy + dy, cx + dx] and (cy + dy, cx + dx) not in visited]
        if len(nxt) > 1:
            nxt = [q for q in nxt if abs(q[0] - cy) + abs(q[1] - cx) == 1]
        if len(nxt) != 1:
            break
        cur = nxt[0]
        visited.add(cur)
        steps += 1
    return cur, steps


def _angle_diff(a, b):
    """Absolute angular difference in [0, pi]."""
    return np.abs(np.mod(a - b + np.pi, 2 * np.pi) - np.pi)


def extract_minutiae(ridges, mask, orientation, coherence, p):
    """Minutiae (x, y, theta, type, quality) from the binary ridge map."""
    skel = np.pad(skeletonize(ridges), 1)
    cn = _crossing_number(skel)
    dist = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)
    found = []
    for y, x in zip(*np.nonzero((cn == 1) | (cn == 3))):
        if dist[y, x] < p.border_dist:
            continue
        starts = _branch_starts(skel, y, x)
        if len(starts) != cn[y, x]:
            continue
        visited = {(y, x)} | {(y + dy, x + dx) for dy, dx in _NB if skel[y + dy, x + dx]}
        traces = [_trace(skel, s, visited, p.trace_len) for s in starts]
        if min(steps for _, steps in traces) < p.trace_len // 2:
            continue                                        # short spur, island or broken ridge
        out = np.array([np.arctan2(ey - y, ex - x) for (ey, ex), _ in traces])
        if cn[y, x] == 1:
            direction = out[0] + np.pi                      # from the ridge towards its ending
        else:
            stem = np.argmax([_angle_diff(a, out).sum() for a in out])
            direction = out[stem] + np.pi                   # from the stem into the fork
        o = orientation[y - 1, x - 1]                       # snap to the (smoother) STFT orientation
        theta = o if _angle_diff(direction, o) <= np.pi / 2 else o + np.pi
        found.append((x - 1, y - 1, np.mod(theta, 2 * np.pi), cn[y, x], coherence[y - 1, x - 1]))

    m = np.array(found, np.float64).reshape(-1, 5)
    if len(m) > 1:                                          # drop clustered minutiae (bridges, breaks, spurs)
        d = np.hypot(m[:, None, 0] - m[None, :, 0], m[:, None, 1] - m[None, :, 1])
        np.fill_diagonal(d, np.inf)
        m = m[d.min(1) >= p.min_minutia_dist]
    return m
