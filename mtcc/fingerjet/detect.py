"""Minutia detection on the phasemap: structure-tensor candidates, filter-bank confirmation, angle tracing.

Port of extract_minutia.h and top_n.h (libFRFXLL/src/algorithm) from
https://github.com/wajihullahbaig/FingerJetFXOSE_VS. The C++ code is one streaming pass; here the candidate
maps are computed with numpy and only the (few hundred) candidates are visited in stream order.
"""
import numpy as np

from .biffilt import TYPE_THRESHOLD, biffilt
from .intmath import atan2, cos, delay, i32, oct_sign, sin
from .phasemap import FILLER

TB = 1328                          # SmmeParameters::Tb, eigenvalue threshold of the phase-gradient tensor
ORI_FILT = 13                      # orientation smoothing window (in 2-pixel steps)
_OFFS = (0, 1, 1, 1, 0, -1, -1, -1)


def _tensor(pm):
    """Smoothed phase-gradient tensor of the phasemap stream; returns (gxx, gxy, gyy, outside) as (rows, w) maps."""
    h, w = pm.shape
    flat = pm.ravel().astype(np.int64)
    n = (h + ORI_FILT) * w
    q = 3 * w + np.arange(n)
    ins = q < h * w - w
    qi = q[ins]
    gx, gy = np.zeros(n, np.int64), np.zeros(n, np.int64)
    gx[ins], gy[ins] = flat[qi + 1] - flat[qi - 1], flat[qi + w] - flat[qi - w]
    outside = ~ins
    outside[ins] = ((flat[qi + 1] == 255) | (flat[qi - 3] == 255) | (flat[qi + w] == 255)
                    | (flat[qi - 3 * w] == 255))

    def conv3(v):
        """[1 2 1] x [1 2 1] streaming filter with rounding (conv2d3)."""
        v = 2 * delay(v, w) + delay(v, 2 * w) + v
        v = 2 * delay(v, 1) + delay(v, 2) + v
        return i32(v + 16) >> 5

    g = [conv3(a).reshape(-1, w) for a in (gx * gx, gx * gy, gy * gy)]
    return g[0], g[1], g[2], outside.reshape(-1, w)


def _directions(gxx, gxy, gyy):
    """Ridge-normal direction (1/256 turns, 0..127) seen by each stream row, from 13x13 averaged tensor samples."""
    rows, w = gxx.shape
    re, im = oct_sign(gxx[0::2, 0::2] - gyy[0::2, 0::2], 2 * gxy[0::2, 0::2])
    z = np.stack([re, im], -1)
    c = np.cumsum(np.pad(z, ((ORI_FILT, 0), (0, 0), (0, 0))), 0)
    s1 = c[ORI_FILT:] - c[:-ORI_FILT]                        # vertical window of 13 even rows
    c = np.cumsum(np.pad(s1, ((0, 0), (ORI_FILT // 2 + 1, ORI_FILT // 2), (0, 0))), 1)
    s2 = c[:, ORI_FILT:] - c[:, :-ORI_FILT]                 # horizontal centred window of 13
    d = atan2(s2[..., 0], s2[..., 1]) // 2
    # the direction is updated after each even row, so row y uses the last even row before it
    used = np.zeros((rows, w // 2), np.int64)
    used[1:] = d[(np.arange(1, rows) - 1) // 2]
    return used


def _local_maxima(v):
    """Stream positions reported by max2d5fast: maxima of 3x3 blocks that also dominate their 5x5 neighbourhood."""
    rows, w = v.shape
    ii, jj = np.arange(2, rows - 4, 3), np.arange(2, w - 4, 3)
    out = []
    for i in ii:
        blk = np.lib.stride_tricks.sliding_window_view(v[i:i + 3], (3, 3))[0, jj]   # (nb, 3, 3)
        flat = blk.reshape(len(jj), 9)
        k = flat.argmax(1)
        best = flat[np.arange(len(jj)), k]
        ok = (flat.min(1) >= 0) & (best > 0)
        for b in np.nonzero(ok)[0]:
            mi, mj = i + k[b] // 3, jj[b] + k[b] % 3
            nb = v[mi - 2:mi + 3, mj - 2:mj + 3].copy()
            nb[i - mi + 2:i - mi + 5, jj[b] - mj + 2:jj[b] - mj + 5] = 0   # block cells are not re-checked
            if nb.min() >= 0 and nb.max() <= best[b] and mj + 4 < w and mi + 4 < rows:
                out.append((mi, mj))
    return sorted(out)


def _next(pm, w, x, y, a, mn, relative):
    """Step to the darkest (mn) or brightest of the three pixels ahead in octant a; returns (x, y, a, v)."""
    best, bv = None, 255 if mn else 0
    for i in (-1, 0, 1):
        nx, ny = x + _OFFS[(a + i) & 7], y + _OFFS[(a + i - 2) & 7]
        v = int(pm[nx + ny * w])
        if (v <= bv) if mn else (v >= bv):
            best, bv = (nx, ny, ((a + i) & 255) if relative else a, v), v
    return best


def in_footprint(pm, w, size, x, y):
    """True when the four diagonal neighbours of (x, y) lie inside the phasemap footprint."""
    x, y = x + 1, y + 1
    o = x + y * w
    return (x >= 2 and y >= 2 and x < w and o < size and pm[o - 2] != FILLER and pm[o] != FILLER
            and pm[o - 2 * w - 2] != FILLER and pm[o - 2 * w] != FILLER)


def adjust_angle(pm, w, size, a, xp, yp, relative):
    """Refine the minutia angle by tracing the ridge/valley leaving (xp, yp); returns (ok, angle)."""
    cur = _next(pm, w, xp, yp, a >> 5, False, relative)
    p2 = _next(pm, w, xp, yp, a >> 5, True, relative)
    mn = 255 - p2[3] > cur[3]
    if mn:
        cur = p2
    d0 = 0
    for i in range(20):
        cur = _next(pm, w, *cur[:3], mn, relative)
        if mn == (cur[3] >= 128) or not in_footprint(pm, w, size, cur[0], cur[1]):
            break
        d = (cur[0] - xp) ** 2 + (cur[1] - yp) ** 2
        if d <= d0:
            break
        d0 = d
        if d > 400:
            break
        if i == 0:
            p2 = cur
    if d0 < 14 * 14:
        return False, a
    return True, int(atan2(-(cur[1] - p2[1]), cur[0] - p2[0]))


def extract_minutia(pm, ori, max_minutiae=68):
    """Minutiae (x, y, theta, conf, type) in phasemap pixels, theta in 1/256 turns clockwise from 'up'."""
    h, w = pm.shape
    gxx, gxy, gyy, outside = _tensor(pm)
    e = (gxx + gyy > 2 * TB)
    b2 = i32((TB - gxx) * (TB - gyy) - gxy * gxy)
    v = np.where(outside, -1, np.where(e & (b2 > 0), b2, 0))
    used = _directions(gxx, gxy, gyy)
    # pad like the C++ buffer: the orientation bytes follow the phasemap in memory
    flat = np.concatenate([pm.ravel(), ori.astype(np.int8).view(np.uint8).ravel(),
                           np.zeros(2 * w + 2, np.uint8)]).astype(np.int64)
    found, rot180 = [], False
    for mi, mj in _local_maxima(v):                  # candidate emerges at stream (mi + 13, mj)
        if mi >= h:
            continue
        xp, yp, dirv = mj - 1, mi + 2, int(used[mi + ORI_FILT, mj // 2])
        if not in_footprint(flat, w, h * w, xp, yp):
            continue
        conf, a, typ, confirmed = 0, 0, False, False
        for i in (-2, 2):
            a0 = (dirv + i) & 255
            ok, c, t, r = biffilt(flat, w, h, xp, yp, int(cos(a0)), int(sin(a0)))
            if ok:
                confirmed, rot180 = True, r          # rotate180 is sticky state in the C++ filter object
                if c > conf:
                    conf, a, typ = c, a0, t
        if not confirmed:
            continue
        if rot180:
            a = (a + 128) & 255
        ok, a2 = adjust_angle(flat, w, h * w, a, xp, yp, False)
        if not ok:
            ok, a2 = adjust_angle(flat, w, h * w, a, xp, yp, True)
        kind = (1 if typ else 2) if conf > TYPE_THRESHOLD else 0
        found.append((xp, yp, a2 if ok else a, conf & 255, kind))
    found.sort(key=lambda m: (-m[3], m[1], m[0], m[2]))    # top_n order: confidence, then y, x, theta
    return np.array(found[:max_minutiae], np.int64).reshape(-1, 5)
