"""Freeman steerable-filter phasemap: ridge phase quantised to 16 levels, 127 outside the footprint.

Port of freeman.h and conv2.h (libFRFXLL/src/algorithm) from https://github.com/wajihullahbaig/FingerJetFXOSE_VS.
Second and third order separable 9-tap filters are steered by the block orientation.
"""
import numpy as np

from .intmath import i32, oct_sign

FILLER = 127                       # marks pixels outside the footprint
_SYM = {'x20': (-112, -7, 48, 14, 1), 'x22': (122, 78, 20, 2, 0), 'x31': (-90, -23, 21, 7, 1)}
_ANTI = {'x21': (0, 71, 37, 6, 0), 'x30': (0, -92, -12, 8, 1), 'x32': (0, 52, 27, 4, 0)}


def _taps(name):
    """9 filter taps for offsets -4..4 (symmetric or antisymmetric, as in conv9)."""
    t = np.array(_SYM.get(name) or _ANTI[name], np.int64)
    sign = 1 if name in _SYM else -1
    return np.concatenate([sign * t[:0:-1], t])


def _vert(flat, s, w, name):
    """Vertical 9-tap filter at flat stream positions s."""
    return sum(k * flat[s + d * w] for d, k in zip(range(-4, 5), _taps(name)))


def _horz(v, name):
    """Horizontal 9-tap filter of a stream, output lagging 4 samples (zero initial state)."""
    p = np.concatenate([np.zeros(8, np.int64), v])
    return sum(k * p[4 + d:4 + d + len(v)] for d, k in zip(range(-4, 5), _taps(name)))


def phasemap(img, ori):
    """Phase image (uint8) of the enhanced image steered by the block orientation `ori` (oh, ow, 2)."""
    h, w = img.shape
    oh, ow = ori.shape[:2]
    # the C++ filter reads 4 bytes past the image, i.e. into the orientation buffer stored after it
    flat = np.concatenate([img.ravel(), ori.astype(np.int8).view(np.uint8).ravel()]).astype(np.int64)
    n = (oh - 2) * 4 * w
    s = 4 * (w + 1) + np.arange(n)
    v = {k: _vert(flat, s, w, k) for k in ('x20', 'x21', 'x22', 'x30', 'x31', 'x32')}
    v20, v21, v22 = _horz(v['x22'], 'x20'), _horz(v['x21'], 'x21'), _horz(v['x20'], 'x22')
    v30, v31, v32, v33 = _horz(v['x22'], 'x30'), _horz(v['x32'], 'x31'), _horz(v['x31'], 'x32'), _horz(v['x30'], 'x22')
    k = np.arange(n)
    blk = ori[1 + k // (4 * w), (k % w) // 4]
    a10, a11 = blk[:, 0], blk[:, 1]
    sc = lambda x: (x + 64) >> 7
    a20, a21, a22 = sc(a10 * a10), sc(2 * a10 * a11), sc(a11 * a11)
    a30, a31, a32, a33 = sc(a10 * a20), sc(3 * a20 * a11), sc(3 * a22 * a10), sc(a11 * a22)
    x2 = i32(i32(a20 * v20 + a21 * v21 + a22 * v22) + 1024) >> 11
    x3 = i32(i32(a30 * v30 + a31 * v31 + a32 * v32 + a33 * v33) + 1024) >> 11
    re, _ = oct_sign(x2, x3)
    out = np.full(h * w, FILLER, np.int64)
    out[:n] = np.where(x2 != 0, (127 - re) & 0xF0, FILLER)
    return out.reshape(h, w).astype(np.uint8)
