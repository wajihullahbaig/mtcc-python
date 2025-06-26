# -*- coding: utf-8 -*-
"""Fingerprint MTCC Pipeline
=====================================================
Minimal, modular implementation of the complete MTCC
(Minutiae Texture Cylinder Codes) fingerprint-matching
pipeline described in the referenced papers.

The file is organised as small, easily‑testable functions
that can be imported individually or run end‑to‑end via
``match_two_fingerprints``.

Dependencies
------------
Python ≥ 3.9 plus the following libraries:

- numpy
- scipy
- scikit‑image
- opencv‑python (cv2)
- matplotlib (visualisation only)

Install with:
```
pip install numpy scipy scikit-image opencv-python matplotlib
```

Author: ChatGPT (OpenAI) – June 2025
"""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np
import scipy.fft as fft
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu

# -----------------------------------------------------------------------------
# 0. Configuration ─────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------
@dataclass
class Params:
    """Centralised configuration and magic numbers."""

    # Normalisation
    target_mean: float = 0.0
    target_std: float = 1.0

    # Segmentation
    block_size: int = 16
    var_thresh: float = 0.01

    # Gabor bank
    gabor_ksize: int = 31
    gabor_freq: float = 0.12  # cycles/pixel
    gabor_sigma: float = 4.0
    gabor_orientations: int = 8

    # SMQT
    smqt_levels: int = 8

    # STFT
    stft_window: int = 32
    stft_step: int = 16

    # MTCC descriptor
    cyl_radius: int = 65
    ns: int = 18  # spatial subdivisions per axis
    nd: int = 5   # directional bins
    sigma_s: float = 6.0
    sigma_d: float = math.pi / 36.0

    # Matcher
    max_rotation: float = math.pi * 2 / 3  # ±120°
    min_valid_cell_ratio: float = 0.2
    top_r: int = 300  # LSS parameter (nr)
    top_p: int = 60   # LSS parameter (np)


P = Params()  # global params instance

# -----------------------------------------------------------------------------
# 1. IO ─────────────────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

def load_image(path: str | pathlib.Path) -> np.ndarray:
    """Load a greyscale image and return float32 array in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0

# -----------------------------------------------------------------------------
# 2. Normalisation ─────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

def normalize_image(img: np.ndarray, *, target_mean: float = P.target_mean,
                     target_std: float = P.target_std) -> np.ndarray:
    """Zero‑mean / unit‑variance normalisation with optional rescale."""
    mean, std = img.mean(), img.std()
    if std < 1e-5:
        return np.zeros_like(img)
    norm = (img - mean) / std
    return norm * target_std + target_mean

# -----------------------------------------------------------------------------
# 3. Segmentation (simple blockwise variance) ──────────────────────────────────
# -----------------------------------------------------------------------------

def segment_image(img: np.ndarray, block_size: int = P.block_size,
                  var_thresh: float = P.var_thresh) -> np.ndarray:
    h, w = img.shape
    mask = np.zeros_like(img, dtype=np.uint8)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            blk = img[y:y + block_size, x:x + block_size]
            if blk.size == 0:
                continue
            if blk.var() > var_thresh:
                mask[y:y + block_size, x:x + block_size] = 1
    # smooth mask (morph close + open)
    kernel = np.ones((block_size // 2, block_size // 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# -----------------------------------------------------------------------------
# 4. Gabor enhancement ─────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

def build_gabor_kernel(theta: float, frequency: float = P.gabor_freq,
                       sigma: float = P.gabor_sigma,
                       ksize: int = P.gabor_ksize) -> np.ndarray:
    lambd = 1.0 / frequency
    return cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)


def gabor_enhance(img: np.ndarray, orientations: int = P.gabor_orientations) -> np.ndarray:
    """Enhance ridges using a symmetric Gabor filter bank and max‑combine."""
    responses = []
    for i in range(orientations):
        theta = i * math.pi / orientations
        k = build_gabor_kernel(theta)
        responses.append(cv2.filter2D(img, cv2.CV_32F, k))
    return np.max(np.stack(responses, axis=0), axis=0)

# -----------------------------------------------------------------------------
# 5. SMQT ──────────────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

def smqt(img: np.ndarray, levels: int = P.smqt_levels) -> np.ndarray:
    """Successive Mean Quantisation Transform (recursive)."""
    def _smqt(x: np.ndarray, level: int) -> np.ndarray:
        if level == 0 or x.size == 0:
            return np.zeros_like(x)
        m = x.mean()
        low = _smqt(x[x < m], level - 1)
        high = _smqt(x[x >= m], level - 1)
        out = np.empty_like(x, dtype=np.float32)
        out[x < m] = low + 0
        out[x >= m] = high + (1 << (level - 1))
        return out
    # flatten / reconstruct trick to keep shape
    flat = img.flatten()
    transformed = _smqt(flat, levels)
    transformed = transformed.reshape(img.shape)
    return transformed / (1 << levels)

# -----------------------------------------------------------------------------
# 6. STFT analysis ─────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

def stft_analysis(img: np.ndarray, win: int = P.stft_window,
                  step: int = P.stft_step) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block‑wise FFT to compute orientation, frequency and log‑energy maps."""
    h, w = img.shape
    orient = np.zeros_like(img)
    freq = np.zeros_like(img)
    energy = np.zeros_like(img)
    hann = np.outer(np.hanning(win), np.hanning(win))

    for y in range(0, h - win, step):
        for x in range(0, w - win, step):
            blk = img[y:y + win, x:x + win] * hann
            F = np.abs(fft.fftshift(fft.fft2(blk)))
            Fy, Fx = np.unravel_index(np.argmax(F), F.shape)
            # orientation: angle of dominant frequency vector
            dy = Fy - win / 2
            dx = Fx - win / 2
            theta = 0.5 * math.atan2(dy, dx)  # divide by 2 due to orientation periodicity
            orient[y:y + win, x:x + win] = theta
            # dominant spatial frequency (cycles/pixel)
            radius = math.hypot(dx, dy)
            f = radius / win
            freq[y:y + win, x:x + win] = f * 2 * math.pi  # scale to [0, 2π]
            # energy
            energy_val = np.log1p(F.max())
            energy[y:y + win, x:x + win] = energy_val
    return orient, freq, energy

# -----------------------------------------------------------------------------
# 7. Binarisation & thinning ─────────────────────────────────────────────────--
# -----------------------------------------------------------------------------

def binarize_and_thin(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    thresh = threshold_otsu(img)
    binary = (img > thresh).astype(np.uint8)
    thin = skeletonize(binary).astype(np.uint8)
    return binary, thin

# -----------------------------------------------------------------------------
# 8. Minutiae extraction (very naive) ─────────────────────────────────────────-
# -----------------------------------------------------------------------------

def extract_minutiae(thin: np.ndarray) -> List[Tuple[int, int, float]]:
    """Skeleton minutiae by crossing‑number; returns (x, y, θ)."""
    h, w = thin.shape
    minutiae = []
    # 8‑neighbour offsets
    N = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
         (1, 1), (1, 0), (1, -1), (0, -1)]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if thin[y, x] == 0:
                continue
            neighbours = [thin[y + dy, x + dx] for dy, dx in N]
            cn = sum((neighbours[i] == 0 and neighbours[(i + 1) % 8] == 1) for i in range(8))
            if cn == 1:  # ridge ending
                theta = math.atan2(N[neighbours.index(1)][0], N[neighbours.index(1)][1])
                minutiae.append((x, y, theta))
            elif cn == 3:  # bifurcation
                theta = 0.0
                minutiae.append((x, y, theta))
    return minutiae

# -----------------------------------------------------------------------------
# 9. MTCC descriptor ─────────────────────────────────────────────────────────--
# -----------------------------------------------------------------------------
@dataclass
class Cylinder:
    x: int
    y: int
    theta: float
    cells: np.ndarray  # shape (ns, ns, nd)


def _angular_diff(a: float, b: float) -> float:
    d = a - b
    while d < -math.pi:
        d += 2 * math.pi
    while d > math.pi:
        d -= 2 * math.pi
    return d


def generate_mtcc_descriptor(minutiae: List[Tuple[int, int, float]],
                              orient_img: np.ndarray,
                              freq_img: np.ndarray,
                              energy_img: np.ndarray,
                              params: Params = P) -> List[Cylinder]:
    """Generate MTCC cylinders for each minutia (orientation centred)."""
    ns, nd = params.ns, params.nd
    delta_s = 2 * params.cyl_radius / ns
    delta_d = 2 * math.pi / nd

    cylinders: List[Cylinder] = []
    for (xm, ym, theta_m) in minutiae:
        cyl = np.zeros((ns, ns, nd), dtype=np.float32)
        for i in range(ns):
            for j in range(ns):
                # centre of cell in local coords -> global
                dx = (i - ns / 2 + 0.5) * delta_s
                dy = (j - ns / 2 + 0.5) * delta_s
                # rotate according to minutia angle
                gx = int(round(xm + dx * math.cos(theta_m) - dy * math.sin(theta_m)))
                gy = int(round(ym + dx * math.sin(theta_m) + dy * math.cos(theta_m)))
                if gx < 0 or gx >= orient_img.shape[1] or gy < 0 or gy >= orient_img.shape[0]:
                    continue  # outside image
                # orientation at cell centre
                theta_cell = orient_img[gy, gx]
                d_theta = _angular_diff(theta_cell, theta_m)
                k = int(((d_theta + math.pi) // delta_d) % nd)
                cyl[i, j, k] += 1.0  # simple presence contribution
        cylinders.append(Cylinder(xm, ym, theta_m, cyl))
    return cylinders

# -----------------------------------------------------------------------------
# 10. MTCC matching (LSS + cosine distance) ───────────────────────────────────
# -----------------------------------------------------------------------------

def _cyl_distance(a: Cylinder, b: Cylinder) -> float:
    # cosine distance between aligned valid cells
    mask = (a.cells + b.cells) > 0
    if mask.sum() == 0:
        return 1.0
    v1 = a.cells[mask].flatten()
    v2 = b.cells[mask].flatten()
    num = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 1 - num / (denom + 1e-6)


def match_mtcc(desc1: List[Cylinder], desc2: List[Cylinder],
               params: Params = P) -> float:
    """Return similarity score (higher is more similar, 0‑1)."""
    # build local similarity matrix
    n1, n2 = len(desc1), len(desc2)
    lsm = np.zeros((n1, n2))
    for i, c1 in enumerate(desc1):
        for j, c2 in enumerate(desc2):
            lsm[i, j] = 1 - _cyl_distance(c1, c2)
    # pick top‑r pairs
    idx = np.dstack(np.unravel_index(np.argsort(lsm.ravel())[::-1][:params.top_r], (n1, n2)))[0]
    selected = [(i, j, lsm[i, j]) for i, j in idx]
    # simple aggregate (skip relaxation for brevity)
    score = np.mean([s for *_rest, s in selected[:params.top_p]]) if selected else 0.0
    return float(score)

# -----------------------------------------------------------------------------
# 11. EER calculation ─────────────────────────────────────────────────────────-
# -----------------------------------------------------------------------------

def compute_eer(genuine: List[float], impostor: List[float]) -> float:
    from sklearn.metrics import roc_curve
    y_true = np.hstack([np.ones_like(genuine), np.zeros_like(impostor)])
    scores = np.hstack([genuine, impostor])
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2

# -----------------------------------------------------------------------------
# 12. End‑to‑end helper ────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

def match_two_fingerprints(path1: str, path2: str, params: Params = P,
                           visualise: bool = False) -> float:
    """Convenience wrapper. Returns similarity score ∈ [0, 1]."""
    img1 = load_image(path1)
    img2 = load_image(path2)

    # Pipeline for img1
    norm1 = normalize_image(img1)
    mask1 = segment_image(norm1)
    gabor1 = gabor_enhance(norm1)
    smqt1 = smqt(gabor1)
    o1, f1, e1 = stft_analysis(smqt1)
    bin1, thin1 = binarize_and_thin(smqt1)
    min1 = extract_minutiae(thin1 * mask1)
    desc1 = generate_mtcc_descriptor(min1, o1, f1, e1)

    # Pipeline for img2
    norm2 = normalize_image(img2)
    mask2 = segment_image(norm2)
    gabor2 = gabor_enhance(norm2)
    smqt2 = smqt(gabor2)
    o2, f2, e2 = stft_analysis(smqt2)
    bin2, thin2 = binarize_and_thin(smqt2)
    min2 = extract_minutiae(thin2 * mask2)
    desc2 = generate_mtcc_descriptor(min2, o2, f2, e2)

    score = match_mtcc(desc1, desc2, params)

    if visualise:
        _visualise_steps({
            "raw": img1,
            "norm": norm1,
            "mask": mask1,
            "gabor": gabor1,
            "smqt": smqt1,
            "orient": o1,
            "binary": bin1,
            "skeleton": thin1,
        })
    return score

# -----------------------------------------------------------------------------
# 13. Visualisation ───────────────────────────────────────────────────────────-
# -----------------------------------------------------------------------------

def _visualise_steps(steps: Dict[str, np.ndarray]):
    """Plot a dict of images in a single figure."""
    n = len(steps)
    cols = 4
    rows = math.ceil(n / cols)
    plt.figure(figsize=(4 * cols, 3 * rows))
    for i, (name, im) in enumerate(steps.items(), 1):
        plt.subplot(rows, cols, i)
        plt.title(name)
        plt.axis("off")
        plt.imshow(im, cmap="gray")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# 14. Main guard for CLI usage ────────────────────────────────────────────────
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    img1 = R'C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2000\FVC2000\Db1_a\1_1.tif'
    img2 = R'C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2000\FVC2000\Db1_a\1_2.tif'

    s = match_two_fingerprints(img1, img2, visualise=True)
    print(f"Similarity score: {s:.4f}")
