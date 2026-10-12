"""Resampling to 333 dpi, block orientation map and footprint.

Port of imresize.h, orimap.h and the resize step of FeatureExtraction.h (libFRFXLL/src/algorithm) from
https://github.com/wajihullahbaig/FingerJetFXOSE_VS. The streaming delay-line filters of the C++ code are
evaluated as shifts of the flattened image, which keeps the integer results identical.
"""
import numpy as np

from .intmath import atan2, cos, delay, i16, i32, oct_sign, sin

ORI_SCALE = 4                      # orientation block size (pixels)
MAX_W, MAX_H = 256, 400            # C++ working-buffer limits (used only when crop=True)


def to_333dpi(img, dpi, crop=False):
    """Resample to ~333 dpi as the C++ library does; returns (image, scale, x0, y0) with x_in = x * scale + x0."""
    h_in, w_in = img.shape
    near500 = 450 <= dpi <= 550
    if near500:
        w, h = w_in // 6 * 4, h_in // 6 * 4
        in_px = lambda n: n * 3 // 2
    else:
        w, h = (w_in * 333 // dpi) & ~3, (h_in * 333 // dpi) & ~3
        in_px = lambda n: n * dpi // 333
    x0 = y0 = 0
    if crop and h > MAX_H:                                  # centre the image vertically
        y0, h = (h_in - in_px(MAX_H)) // 2, MAX_H
    if crop and w > MAX_W:                                  # centre the image horizontally
        x0, w = (w_in - in_px(MAX_W)) // 2, MAX_W
    flat = np.concatenate([img.ravel(), np.zeros(2 * w_in + 2, np.uint8)]).astype(np.int64)
    base = y0 * w_in + x0
    if near500:                                             # imresize23: 3x3 input -> 2x2 output
        r, c = np.meshgrid(np.arange(h // 2) * 3, np.arange(w // 2) * 3, indexing='ij')
        p = lambda dy, dx: flat[base + (r + dy) * w_in + c + dx]
        o = np.empty((h, w), np.int64)
        o[0::2, 0::2] = 4 * p(0, 0) + 2 * p(0, 1) + 2 * p(1, 0) + p(1, 1)
        o[0::2, 1::2] = 4 * p(0, 2) + 2 * p(0, 1) + 2 * p(1, 2) + p(1, 1)
        o[1::2, 0::2] = 4 * p(2, 0) + 2 * p(2, 1) + 2 * p(1, 0) + p(1, 1)
        o[1::2, 1::2] = 4 * p(2, 2) + 2 * p(2, 1) + 2 * p(1, 2) + p(1, 1)
        return (o * 28 >> 8).astype(np.uint8), 1.5, x0, y0
    if 300 <= dpi <= 366 and w_in % 4 == 0:                 # already ~333 dpi: plain copy
        return flat[base:base + w * h].reshape(h, w).astype(np.uint8), 1.0, x0, y0
    s256 = dpi * 256 // 333                                 # bilinear resampling
    ys, xs = np.arange(h) * s256, np.arange(w) * s256
    dy, dx = (ys & 255)[:, None], (xs & 255)[None, :]
    i = base + (ys >> 8)[:, None] * w_in + (xs >> 8)[None, :]
    v = ((flat[i] * (256 - dx) + flat[i + 1] * dx) * (256 - dy)
         + (flat[i + w_in] * (256 - dx) + flat[i + w_in + 1] * dx) * dy) >> 16
    return v.astype(np.uint8), s256 / 256, x0, y0


def raw_orimap(img):
    """Unsmoothed block orientation (complex int8, doubled angle) and raw footprint."""
    h, w = img.shape
    ow, oh = w // ORI_SCALE, h // ORI_SCALE
    flat = img.ravel().astype(np.int64)
    n = (oh - 1) * ORI_SCALE * w
    p0 = flat[2 * w + 6:2 * w + 6 + n]
    p1 = flat[3 * w + 5:3 * w + 5 + n]
    cur = p0 + p1                                           # 5-pixel cross, then two 2x2 sums
    v1 = i16(delay(cur, w + 1, 510) + cur + delay(p0, 1))
    v2 = i16(delay(v1, w + 1, 1275) + v1)
    v1 = i16(delay(v2, w - 1, 2550) + v2 + delay(v1, w, 1275))
    v1x = delay(v1, w, 6375)                                # first derivatives
    v10 = i16(v1x + v1)
    v10 = i16(delay(v10, 1, 12750) - v10)
    v11 = i16(v1x - v1)
    v11 = i16(delay(v11, 1) + v11)
    v10x = delay(v10, w)                                    # second derivatives
    v20 = i16(v10x + v10)
    v20 = i16(delay(v20, 1) - v20)
    v21 = i16(v10x - v10)
    v21 = i16(delay(v21, 1) + v21)
    v22 = i16(delay(v11, w) - v11)
    v22 = i16(delay(v22, 1) + v22)
    g20, g21, g22 = v20 + v22, v20 - v22, 2 * v21
    z = np.stack([i32(g20 * g21), i32(g20 * g22)], -1)      # second-order orientation tensor
    mag = i32(z.reshape(oh - 1, ORI_SCALE, ow, ORI_SCALE, 2).sum((1, 3))).reshape(-1, 2)
    mag = delay(mag, 1)                                     # the C++ output lags one block
    ori = np.zeros((oh, ow, 2), np.int64)
    ori[:-1] = np.stack(oct_sign(mag[:, 0], mag[:, 1], 50000), -1).reshape(oh - 1, ow, 2)
    return ori, (ori != 0).any(-1).astype(np.int64)


def _boxsum(a, n):
    """Centred n x n box sum with zero padding."""
    v = np.lib.stride_tricks.sliding_window_view(np.pad(a, n // 2), (n, n), axis=(0, 1))
    return v.sum((-2, -1))


def _fill_holes(fp):
    """Fill each row between its first and last foreground block."""
    on = fp.astype(bool)
    idx = np.arange(fp.shape[1])
    first = np.where(on.any(1), on.argmax(1), fp.shape[1])
    last = np.where(on.any(1), fp.shape[1] - 1 - on[:, ::-1].argmax(1), -1)
    return (on | ((idx > first[:, None]) & (idx < last[:, None]))).astype(np.int64)


def footprint(raw_fp):
    """Clean the raw footprint: majority box filters and hole filling."""
    fp = (_boxsum(raw_fp, 3) > 3).astype(np.int64)
    for _ in range(2):
        fp = (_boxsum(fp, 5) > 11).astype(np.int64)
    fp = _fill_holes(_fill_holes(fp).T).T
    for _ in range(2):
        fp = (_boxsum(fp, 5) > 14).astype(np.int64)
    return fp


def _smooth(ori, fp):
    """9x9 triangular smoothing of the orientation field (two cascaded 5x5 boxes), zero outside footprint."""
    t = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], np.int64)
    p = np.pad(ori, ((4, 4), (4, 4), (0, 0)))
    v = np.lib.stride_tricks.sliding_window_view(p, 9, axis=0) @ t
    v = np.lib.stride_tricks.sliding_window_view(v, 9, axis=1) @ t
    return np.where(fp[..., None] > 0, v, 0)


def orientation_map(img, fp):
    """Smoothed block orientation as unit complex int8 (single angle in the doubled-angle field)."""
    ori, _ = raw_orimap(img)
    s = _smooth(ori, fp)
    ori = np.stack(oct_sign(s[..., 0], s[..., 1], 128), -1)
    s = _smooth(ori, fp)
    a = atan2(s[..., 0], s[..., 1]) // 2
    return np.where(fp[..., None] > 0, np.stack([cos(a), sin(a)], -1), 0)
