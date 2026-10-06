"""Fingerprint enhancement: segmentation -> SMQT -> STFT analysis/enhancement -> Gabor -> ridge map.

The STFT step also yields the texture images used by MTCC:
I_o (orientation), I_f (frequency) and I_e (log energy).
"""
import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage


def normalize(img, mask=None):
    """Zero-mean, unit-variance image, statistics taken over the mask."""
    v = img[mask] if mask is not None and mask.any() else img
    return ((img - v.mean()) / (v.std() + 1e-8)).astype(np.float32)


def segment(img, p):
    """Block-wise variance segmentation with morphological smoothing (bool mask)."""
    x = normalize(img.astype(np.float32))
    b = p.seg_block
    mean = cv2.blur(x, (b, b))
    std = np.sqrt(np.maximum(cv2.blur(x * x, (b, b)) - mean ** 2, 0))
    mask = (std > p.seg_thresh).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (b, b))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        mask = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return ndimage.binary_fill_holes(mask)


def smqt(img, mask, levels=8):
    """Successive Mean Quantization Transform (Nilsson et al.) over the foreground, scaled to [0, 1]."""
    x = img.astype(np.float64)[mask]
    code = np.zeros(x.shape, np.int64)
    for _ in range(levels):
        means = np.bincount(code, weights=x) / np.maximum(np.bincount(code), 1)
        code = 2 * code + (x > means[code])
    out = np.full(img.shape, code.mean() if code.size else 0, np.float32)
    out[mask] = code
    return out / (2 ** levels - 1)


def _window(p):
    """Separable window: flat centre, raised-cosine tapers; shifted copies sum to one."""
    o2 = 2 * p.stft_overlap
    w = np.ones(p.stft_block + o2)
    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(o2) + 0.5) / o2)
    w[:o2], w[-o2:] = ramp, ramp[::-1]
    return np.outer(w, w)


def _blocks(img, p):
    """Overlapping windows, one per stft_block x stft_block block: (nby, nbx, n, n)."""
    b, o = p.stft_block, p.stft_overlap
    h, w = img.shape
    nby, nbx = -(-h // b), -(-w // b)
    pad = np.pad(img, ((o, nby * b - h + o), (o, nbx * b - w + o)))
    return sliding_window_view(pad, (b + 2 * o, b + 2 * o))[::b, ::b][:nby, :nbx]


def _to_pixels(block_map, shape, b):
    """Bilinearly upsample a block map to pixel resolution."""
    big = cv2.resize(block_map.astype(np.float32), (block_map.shape[1] * b, block_map.shape[0] * b),
                     interpolation=cv2.INTER_LINEAR)
    return big[:shape[0], :shape[1]]


def _smooth(x, m, sigma=1.0):
    """Gaussian smoothing of a block map restricted to foreground blocks."""
    return ndimage.gaussian_filter(x * m, sigma) / (ndimage.gaussian_filter(m, sigma) + 1e-8)


def stft(img, mask, p):
    """STFT analysis (orientation, frequency, energy, coherence) and contextual filtering (Chikkerur et al.)."""
    b, N = p.stft_block, p.stft_nfft
    n = b + 2 * p.stft_overlap
    win = _window(p)
    bmask = _blocks(mask.astype(np.float32), p).mean((-2, -1)) > 0.5

    F = np.fft.fftshift(np.fft.fft2(_blocks(img, p) * win, s=(N, N)), axes=(-2, -1))
    u = np.arange(N) - N // 2
    U, V = np.meshgrid(u, u)
    r, phi = np.hypot(U, V), np.arctan2(V, U)
    band = (r >= N / p.max_period) & (r <= N / p.min_period)
    P = np.abs(F) ** 2 * band
    tot = P.sum((-2, -1)) + 1e-12

    # Spectral moments: angular density -> orientation, radial density -> frequency.
    c2 = (P * np.cos(2 * phi)).sum((-2, -1)) / tot
    s2 = (P * np.sin(2 * phi)).sum((-2, -1)) / tot
    freq = (P * r).sum((-2, -1)) / tot / N
    energy = np.log(tot)

    m = bmask.astype(np.float64)
    c2, s2 = _smooth(c2, m), _smooth(s2, m)
    theta = 0.5 * np.arctan2(s2, c2)                       # angle of the frequency vector
    orient = np.mod(theta + np.pi / 2, np.pi)               # ridge orientation in [0, pi)
    unit = np.exp(2j * theta)
    coh = np.abs(ndimage.uniform_filter(unit.real * m, 3) + 1j * ndimage.uniform_filter(unit.imag * m, 3))
    coh = coh / (ndimage.uniform_filter(m, 3) + 1e-8)
    freq = _smooth(freq, m)

    # Contextual filter per block: raised-cosine angular x Butterworth radial band-pass.
    dphi = np.mod(phi - theta[..., None, None] + np.pi / 2, np.pi) - np.pi / 2
    bw_phi = (np.pi / 6 + np.pi / 3 * (1 - np.clip(coh, 0, 1)))[..., None, None]
    h_ang = np.where(np.abs(dphi) < bw_phi, np.cos(np.pi * dphi / (2 * bw_phi)) ** 2, 0)
    rc = (freq * N)[..., None, None]
    rbw = (r * 0.7 * rc) ** 4
    h_rad = np.sqrt(rbw / (rbw + (r ** 2 - rc ** 2) ** 4 + 1e-12))
    H = h_ang * h_rad * bmask[..., None, None]
    y = np.real(np.fft.ifft2(np.fft.ifftshift(F * H, axes=(-2, -1))))[..., :n, :n]

    h, w = img.shape
    nby, nbx = bmask.shape
    o = p.stft_overlap
    out = np.zeros((nby * b + 2 * o, nbx * b + 2 * o))
    for i in range(nby):
        for j in range(nbx):
            out[i * b:i * b + n, j * b:j * b + n] += y[i, j]
    enhanced = out[o:o + h, o:o + w].astype(np.float32)

    # Pixel-level maps (orientation interpolated through its doubled-angle vector).
    ori = 0.5 * np.arctan2(_to_pixels(np.sin(2 * orient), img.shape, b), _to_pixels(np.cos(2 * orient), img.shape, b))
    maps = {
        'orientation': np.mod(ori, np.pi),
        'frequency': _to_pixels(freq, img.shape, b),
        'energy': _to_pixels(energy, img.shape, b),
        'coherence': _to_pixels(coh, img.shape, b),
    }
    return maps, enhanced


def gabor(img, orient, freq, mask, p):
    """Orientation-adaptive even Gabor filtering at the median ridge frequency."""
    f = float(np.median(freq[mask])) if mask.any() else 0.1
    s = p.gabor_k / f
    half = int(np.ceil(3 * s))
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    n = p.gabor_orients
    idx = np.round(orient / np.pi * n).astype(int) % n
    out = np.zeros_like(img, np.float32)
    for i in range(n):
        normal = i * np.pi / n + np.pi / 2                  # filter oscillates across the ridges
        g = np.exp(-(x ** 2 + y ** 2) / s ** 2) * np.cos(2 * np.pi * f * (x * np.cos(normal) + y * np.sin(normal)))
        resp = cv2.filter2D(img, cv2.CV_32F, (g - g.mean()).astype(np.float32))
        sel = idx == i
        out[sel] = resp[sel]
    return out


def enhance(img, p):
    """Run the full enhancement chain on a grayscale uint8 image."""
    img = img.astype(np.float32)
    mask = segment(img, p)
    x = normalize(smqt(img, mask, p.smqt_levels), mask)
    x[~mask] = 0
    maps, x_stft = stft(x, mask, p)
    g = gabor(normalize(x_stft, mask), maps['orientation'], maps['frequency'], mask, p)
    return {'mask': mask, 'smqt': x, 'stft': x_stft, 'gabor': g, 'ridges': (g < 0) & mask, **maps}
