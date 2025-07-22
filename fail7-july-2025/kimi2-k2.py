#!/usr/bin/env python3
"""
fp_enhance.py
STFT-based fingerprint enhancement (0-1 normalized internally)
Usage:
    python fp_enhance.py [image_path]
"""

import os
import sys
import numpy as np
from scipy.signal import windows
from skimage import io, filters
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1.  ORIENTATION FIELD ESTIMATION
# ---------------------------------------------------------
def gradient_orientation(im: np.ndarray, blk: int = 32) -> np.ndarray:
    gX = filters.sobel_h(im)
    gY = filters.sobel_v(im)

    h, w = im.shape
    orient = np.zeros((h // blk, w // blk))
    for i in range(orient.shape[0]):
        for j in range(orient.shape[1]):
            y0, x0 = i * blk, j * blk
            gx = gX[y0:y0 + blk, x0:x0 + blk].ravel()
            gy = gY[y0:y0 + blk, x0:x0 + blk].ravel()

            Gxx = np.sum(gx ** 2)
            Gyy = np.sum(gy ** 2)
            Gxy = np.sum(gx * gy)
            orient[i, j] = 0.5 * np.arctan2(2 * Gxy, Gxx - Gyy) + np.pi / 2
    return orient


# ---------------------------------------------------------
# 2.  LOCAL RIDGE FREQUENCY
# ---------------------------------------------------------
def ridge_frequency(im: np.ndarray, orient: np.ndarray, blk: int = 32,
                    min_w: int = 3, max_w: int = 20) -> np.ndarray:
    h, w = im.shape
    freq = np.zeros_like(orient)

    for i in range(orient.shape[0]):
        for j in range(orient.shape[1]):
            y0, x0 = i * blk, j * blk
            block = im[y0:y0 + blk, x0:x0 + blk]

            angle = orient[i, j]
            c, s = np.cos(angle), np.sin(angle)
            sig = []
            for k in range(-blk // 2, blk // 2):
                for l in range(-blk // 2, blk // 2):
                    dx = int(round(l * c - k * s))
                    dy = int(round(l * s + k * c))
                    if 0 <= y0 + dy < h and 0 <= x0 + dx < w:
                        sig.append(im[y0 + dy, x0 + dx])
                    else:
                        sig.append(0.5)          # neutral value for 0-1 image
            sig = np.array(sig) - np.mean(sig)

            zc = ((sig[:-1] * sig[1:]) < 0).sum()
            wavelength = max(1, len(sig) / (zc + 1e-6))
            freq[i, j] = 1.0 / wavelength
    freq = np.clip(freq, 1.0 / max_w, 1.0 / min_w)
    return freq


# ---------------------------------------------------------
# 3.  STFT ENHANCEMENT WITH ORIENT + FREQ
# ---------------------------------------------------------
def stft_enhance(im: np.ndarray,
                 win_size: int = 32,
                 overlap: float = 0.5,
                 blk: int = 8) -> np.ndarray:
    im = im.astype(np.float32)
    h, w = im.shape
    hop = int(win_size * (1 - overlap))

    orient = gradient_orientation(im, blk=blk)
    freq = ridge_frequency(im, orient, blk=blk)

    win2d = windows.hann(win_size, sym=False)
    win2d = np.sqrt(np.outer(win2d, win2d))

    acc = np.zeros_like(im)
    wgt = np.zeros_like(im)

    for y in range(0, h - win_size + 1, hop):
        for x in range(0, w - win_size + 1, hop):
            patch = im[y:y + win_size, x:x + win_size]
            patch_win = patch * win2d

            by = min((y + win_size // 2) // blk, orient.shape[0] - 1)
            bx = min((x + win_size // 2) // blk, orient.shape[1] - 1)
            theta = orient[by, bx]
            f0 = freq[by, bx]

            f = np.fft.fft2(patch_win)
            fshift = np.fft.fftshift(f)

            mid = win_size // 2
            u, v = np.meshgrid(np.arange(-mid, mid), np.arange(-mid, mid))
            c, s = np.cos(theta), np.sin(theta)
            u_rot = u * c + v * s
            v_rot = -u * s + v * c
            dist = np.sqrt(u_rot ** 2 + (v_rot * 0.5) ** 2)
            sigma = 1.0 / f0 / 3
            mask = np.exp(-0.5 * ((dist - 1.0 / f0) / sigma) ** 2)

            filtered = fshift * mask
            inv = np.fft.ifftshift(filtered)
            patch_enh = np.real(np.fft.ifft2(inv)) * win2d

            acc[y:y + win_size, x:x + win_size] += patch_enh
            wgt[y:y + win_size, x:x + win_size] += win2d ** 2

    enhanced = acc / np.maximum(wgt, 1e-6)
    enhanced = np.clip(enhanced, 0, 1)      # stay in 0-1 range
    return enhanced


# ---------------------------------------------------------
# 4.  CLI + VISUALIZATION
# ---------------------------------------------------------
def load_gray(path: str) -> np.ndarray:
    """Load image, convert to 0-1 float grayscale."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    img = io.imread(path)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    return (img / 255.0).astype(np.float32)


def main():
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print("Loading:", img_path)
        img = load_gray(img_path)
    else:
        print("Using synthetic fingerprint for demo.")
        from skimage.data import binary_blobs
        img_path = "C:/Users/Precision/Onus/Data/FVC-DataSets/DataSets/FVC2002/Db1_a/1_1.tif"
        img = load_gray(img_path)


    enhanced = stft_enhance(img, win_size=32, overlap=0.5)
    enhanced = 1.0 - (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Enhanced")
    plt.imshow(enhanced, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
    