"""Singular points (Poincare index) and the singular-point quality weights of the _SP matchers."""
import numpy as np
from scipy import ndimage

_RING = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))   # closed loop (dy, dx)
_SECTORS = ((np.deg2rad(135), np.deg2rad(180), 0.75), (np.deg2rad(180), np.deg2rad(285), 0.5))


def singular_points(orientation, mask, p):
    """Cores and deltas (x, y) where the Poincare index of the smoothed block orientation field is +-1/2."""
    s = p.sp_block
    o = orientation[s // 2::s, s // 2::s]
    inside = ndimage.binary_erosion(mask[s // 2::s, s // 2::s], iterations=2)
    c = ndimage.gaussian_filter(np.cos(2 * o), p.sp_sigma)
    d = ndimage.gaussian_filter(np.sin(2 * o), p.sp_sigma)
    o = np.arctan2(d, c) / 2
    loop = [np.roll(o, (-dy, -dx), (0, 1)) for dy, dx in _RING]
    turn = sum(np.mod(b - a + np.pi / 2, np.pi) - np.pi / 2 for a, b in zip(loop, loop[1:] + loop[:1]))
    out = {}
    for name, sign in (('core', 1), ('delta', -1)):
        labels, n = ndimage.label((sign * turn > np.pi / 2) & inside)
        yx = np.array(ndimage.center_of_mass(labels > 0, labels, range(1, n + 1))).reshape(-1, 2)
        out[name] = (yx[:, ::-1] * s + s // 2).astype(np.float32)
    return out


def sp_weights(minutiae, sp, p):
    """Quality weights: 0.75 / 0.5 for minutiae beyond R + 3 sigma_S in the 135-180 / 180-285 degree sectors
    around the first core, else the first delta, else the minutiae centroid (angles counter-clockwise, y up)."""
    ref = next((v[0] for v in (sp['core'], sp['delta']) if len(v)), minutiae[:, :2].mean(0))
    dx, dy = minutiae[:, 0] - ref[0], minutiae[:, 1] - ref[1]
    a, far = np.mod(np.arctan2(-dy, dx), 2 * np.pi), np.hypot(dx, dy) >= p.R + 3 * p.sigma_s
    w = np.ones(len(minutiae), np.float32)
    for lo, hi, k in _SECTORS:
        w[far & (a > lo) & (a <= hi)] = k
    return w
