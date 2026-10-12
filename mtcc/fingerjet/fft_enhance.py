"""Contextual enhancement with overlapping 32x32 integer FFT blocks.

Port of block_fft.h and fft_enhance.h (libFRFXLL/src/algorithm) from
https://github.com/wajihullahbaig/FingerJetFXOSE_VS. All blocks are processed as one numpy batch; the
fixed-point butterflies, twiddle table and rounding follow the C++ code exactly.
"""
import numpy as np

from .intmath import i32, oct_abs, reduce

BLOCK, SPACING = 32, 17
_SIN = np.array([0, 799, 1567, 2276, 2896, 3406, 3784, 4017, 4096, 4017, 3784, 3406, 2896, 2276, 1567, 799],
                np.int64)


def _sin(a):
    """Twiddle sine, angle in 1/32 turns, amplitude 4096."""
    a = np.asarray(a, np.int64)
    return np.where(a & 16, -1, 1) * _SIN[a & 15]


def _cos(a):
    """Twiddle cosine, angle in 1/32 turns, amplitude 4096."""
    return _sin(np.asarray(a, np.int64) + 8)


def _fft(re, im, inverse):
    """In-place radix-2 fixed-point FFT along the last axis."""
    n = re.shape[-1]
    bits = n.bit_length() - 1
    rev = np.array([int(f'{i:0{bits}b}'[::-1], 2) for i in range(n)])
    re[:], im[:] = re[..., rev], im[..., rev]
    dt, m = 32 if inverse else -32, 1
    while m < n:
        dt >>= 1
        k = np.arange(m)
        wr, wi = np.tile(_cos(k * dt), n // (2 * m)), np.tile(_sin(k * dt), n // (2 * m))
        i = (np.arange(0, n, 2 * m)[:, None] + k).ravel()
        j = i + m
        tr = reduce(i32(wr * re[..., j] - wi * im[..., j]), 12)
        ti = reduce(i32(wr * im[..., j] + wi * re[..., j]), 12)
        pr, pi = re[..., i], im[..., i]
        re[..., j], im[..., j] = i32(pr - tr), i32(pi - ti)
        re[..., i], im[..., i] = i32(pr + tr), i32(pi + ti)
        m *= 2


def _real_fixup(re, im, inverse):
    """Split/merge step that turns a half-length complex FFT into a real FFT (fft1)."""
    n = re.shape[-1]
    k = np.arange(1, n // 2 + 1)
    t = k + n if inverse else -k
    wr, wi = _cos(t), _sin(t)
    shift = 12 if inverse else 13
    p1r, p1i, p2r, p2i = re[..., k], im[..., k], re[..., n - k], im[..., n - k]
    h1r, h1i = (p2r + p1r) << 12, (p1i - p2i) << 12
    ar, ai = p1i + p2i, p2r - p1r
    h2r, h2i = i32(wr * ar - wi * ai), i32(wr * ai + wi * ar)
    re[..., k], im[..., k] = reduce(h1r + h2r, shift), reduce(h1i + h2i, shift)
    re[..., n - k], im[..., n - k] = reduce(h1r - h2r, shift), reduce(-(h1i - h2i), shift)
    tr, ti = re[..., 0].copy(), im[..., 0].copy()
    re[..., 0], im[..., 0] = tr + ti, (tr - ti) if inverse else 0


def _fft2(d, inverse):
    """2-D real FFT of (B, 32, 32) int blocks, packed as 32 x 16 complex values."""
    re, im = d[..., 0::2].copy(), d[..., 1::2].copy()
    if not inverse:
        _fft(re, im, False)
        _real_fixup(re, im, False)
    rt, it = re.swapaxes(-1, -2).copy(), im.swapaxes(-1, -2).copy()
    _fft(rt, it, inverse)
    re, im = rt.swapaxes(-1, -2).copy(), it.swapaxes(-1, -2).copy()
    if inverse:
        _real_fixup(re, im, True)
        _fft(re, im, True)
    d[..., 0::2], d[..., 1::2] = re, im


def _enhance_spectrum(d):
    """Band-pass the spectrum and raise it to the power ~2 (enhance)."""
    re, im = d[..., 0::2], d[..., 1::2]
    fy = np.arange(BLOCK)
    fy = np.where(fy < BLOCK // 2, fy, fy - BLOCK)[:, None]
    fx = np.arange(BLOCK // 2)[None, :]
    band = ((fx ** 2 + fy ** 2) > 6) & ((fx ** 2 + fy ** 2) < 169)
    a = reduce(oct_abs(re, im), 5)
    re2, im2 = reduce(re + reduce(i32(re * a), 7), 3), reduce(im + reduce(i32(im * a), 7), 3)
    d[..., 0::2], d[..., 1::2] = np.where(band, re2, 0), np.where(band, im2, 0)


def _normalize(d):
    """Stretch each block to [0, 251] and apply the overlap-add envelope."""
    mn, mx = d.min((1, 2)), d.max((1, 2))
    rng = mx - mn
    div = np.maximum(rng, 16)
    mn = np.where(rng < 16, mn - (16 - rng) // 2, mn)
    norm = BLOCK + 1 - SPACING
    div = (div * norm * norm)[:, None, None]
    i = np.arange(BLOCK)
    env = np.minimum(np.minimum(i + 1, BLOCK - i), min(norm, SPACING))
    x = i32((d - mn[:, None, None]) * 251 * env[None, :] * env[:, None])
    return np.sign(x + 0.5).astype(np.int64) * ((np.abs(x) + (div >> 1)) // div)   # rounded divide()


def fft_enhance(img):
    """Enhanced copy of a uint8 image (overlap-add of enhanced 32x32 blocks spaced 17 pixels apart)."""
    h, w = img.shape
    ys, xs = np.arange(SPACING - BLOCK, h, SPACING), np.arange(SPACING - BLOCK, w, SPACING)
    pad = BLOCK
    inv = np.zeros((h + 2 * pad, w + 2 * pad), np.int64)
    inv[pad:pad + h, pad:pad + w] = 255 - img.astype(np.int64)
    win = np.lib.stride_tricks.sliding_window_view(inv, (BLOCK, BLOCK))
    d = win[(ys + pad)[:, None], (xs + pad)[None, :]].reshape(-1, BLOCK, BLOCK).copy()
    _fft2(d, False)
    _enhance_spectrum(d)
    _fft2(d, True)
    d = _normalize(reduce(d, 10))
    out = np.zeros_like(inv)
    for b, (y, x) in enumerate((y, x) for y in ys for x in xs):
        out[y + pad:y + pad + BLOCK, x + pad:x + pad + BLOCK] += d[b] & 255
    return (255 - (out[pad:pad + h, pad:pad + w] & 255)).astype(np.uint8)
