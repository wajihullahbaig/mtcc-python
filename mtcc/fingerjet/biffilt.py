"""Rotated bifurcation filter bank that confirms a minutia candidate on the phasemap.

Port of biffilt.h (libFRFXLL/src/algorithm) from https://github.com/wajihullahbaig/FingerJetFXOSE_VS.
A 19x19 patch aligned with the local orientation is filtered by two separable 17x17 kernels at 5 ridge
periods; the strongest of the symmetric / antisymmetric responses gives confidence, type and polarity.
"""
import numpy as np

from .intmath import reduce
from .phasemap import FILLER

THRESHOLD, RATIO, TYPE_THRESHOLD = 80, 102, 105     # biffilt::parameters defaults
_N, _M, _PW = 17, 3, 19                              # filter size, response size, patch width
_CV1 = (50, 48, 41, 32, 23, 14, 8, 4, 2)             # vertical, symmetric (Gxr)
_CV2 = (0, 48, 82, 96, 90, 72, 50, 30, 16)           # vertical, antisymmetric (Gyr)
_CH1 = [(0, 55, -53, -61, 74, 3, -28, 7, 3), (0, 42, 0, -83, 3, 63, -1, -25, 1), (0, 29, 25, -50, -58, 22, 51, 5, -24),
        (-2, 16, 28, -14, -60, -41, 18, 41, 13), (0, 13, 28, 13, -27, -50, -29, 14, 38)]
_CH2 = [(52, -14, -32, 22, 6, -9, 1, 1, 0), (44, 0, -36, 1, 19, 0, -7, 0, 2), (37, 9, -28, -18, 12, 14, -1, -6, -1),
        (31, 13, -17, -24, -5, 13, 11, 0, -5), (26, 15, -7, -21, -16, 0, 11, 10, 3)]
_WEIGHT = np.repeat([16, 15, 14, 13, 12], 9)          # confweight: decreasing with the response row


def _full(t, sign=1):
    """17 taps for offsets -8..8 from the 9 half taps."""
    t = np.array(t, np.int64)
    return np.concatenate([sign * t[:0:-1], t])


_V1, _V2 = _full(_CV1), _full(_CV2, -1)
_H1, _H2 = np.stack([_full(t) for t in _CH1], 1), np.stack([_full(t) for t in _CH2], 1)


def _patch(flat, w, h, x, y, c, s):
    """19x19 bilinear patch around (x, y) rotated by (c, s); positions in 1/128 pixel."""
    r, k = np.arange(_PW)[:, None], np.arange(_PW)[None, :]
    X = (x << 7) - 9 * (c - s) - r * s + k * c
    Y = (y << 7) - 9 * (s + c) + r * c + k * s
    x0, y0 = X >> 7, Y >> 7
    out = (X < 0) | (Y < 0) | (x0 > w - 1) | (y0 > h - 1)
    i = np.where(out, 0, x0 + y0 * w)
    wx1, wy1 = X & 127, Y & 127
    wx0, wy0 = 128 - wx1, 128 - wy1
    v = flat[i] * wx0 * wy0 + flat[i + 1] * wx1 * wy0 + flat[i + w] * wx0 * wy1 + flat[i + w + 1] * wx1 * wy1
    return np.where(out, FILLER, reduce(v, 14))


def _responses(patch, v, hs):
    """3x3 positions x 5 periods of the separable filter (v vertical, hs horizontal), flattened [row, col, period]."""
    vert = np.stack([v @ patch[r - 8:r + 9] for r in range(8, 11)])
    win = np.lib.stride_tricks.sliding_window_view(vert, _N, axis=1)
    return (win @ hs).ravel()


def biffilt(flat, w, h, x, y, c, s):
    """Filter response at candidate (x, y) along direction (c, s); returns (passed, confidence, type, rotate180)."""
    p = _patch(flat, w, h, x, y, c, s)
    r1, r2 = _responses(p, _V1, _H1), _responses(p, _V2, _H2)
    sm, df = r1 + r2, r1 - r2
    a = np.stack([reduce(np.abs(sm) * _WEIGHT, 4), reduce(np.abs(df) * _WEIGHT, 4)], 1).ravel()
    k = int(np.argmax(a))                              # first strict maximum, scanning sum/diff alternately
    if a[k] <= 0:
        return False, 0, False, False
    conf, mirr = int(reduce(a[k], 17)), int(reduce(a[k ^ 1], 17))
    if conf < THRESHOLD or conf * RATIO < 256 * mirr:
        return False, conf, False, False
    peak = (sm if k % 2 == 0 else df)[k // 2]
    return True, conf, bool(peak > 0), k % 2 == 1
