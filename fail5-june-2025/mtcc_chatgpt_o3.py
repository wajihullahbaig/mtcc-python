import os
from glob import glob
from typing import Tuple, List, Dict

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import stft
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

################################################################################
# 1. I/O
################################################################################

def load_image(path: str) -> np.ndarray:
    """Read grayscale image as float32."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read {path}")
    return img.astype(np.float32)

################################################################################
# 2. Pre‑processing utilities
################################################################################

def normalize(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Zero‑mean, unit‑variance normalization."""
    std = img.std() + eps
    return (img - img.mean()) / std

# --- Segmentation ----------------------------------------------------------------

def segment(img: np.ndarray, block_size: int = 16, var_thresh: float = 0.01) -> np.ndarray:
    """Return binary mask where fingerprint region == 1."""
    h, w = img.shape
    mask = np.zeros((h, w), np.uint8)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            blk = img[y:y + block_size, x:x + block_size]
            if blk.size == 0:
                continue
            if blk.var() > var_thresh:
                mask[y:y + block_size, x:x + block_size] = 1
    # morphology clean‑up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# --- Orientation field -------------------------------------------------------------

def _gradient_orientation(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    ori = 0.5 * np.arctan2(2 * gx * gy, gx ** 2 - gy ** 2)
    return ori

# --- Gabor enhancement -------------------------------------------------------------

def _build_gabor(theta: float, freq: float, ksize: int = 21, sigma: float = 4.5) -> np.ndarray:
    lambd = 1.0 / (freq + 1e-3)
    return cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)

def gabor_enhance(img: np.ndarray, orientation_map: np.ndarray, freq: float = 0.1) -> np.ndarray:
    """Simple Gabor enhancement using global freq but local orientation."""
    out = np.zeros_like(img)
    ksize = 21
    for angle in np.linspace(0, np.pi, 16, endpoint=False):
        mask = ((np.abs(np.angle(np.exp(1j * (orientation_map - angle))) ) < np.pi / 16)).astype(np.uint8)
        if not np.any(mask):
            continue
        kernel = _build_gabor(angle, freq, ksize)
        filtered = cv2.filter2D(img, -1, kernel)
        out += filtered * mask
    return out

# --- SMQT ----------------------------------------------------------------------------

def smqt(img: np.ndarray, levels: int = 8) -> np.ndarray:
    """Successive Mean Quantization Transform (fast scalar version)."""
    def _smqt(channel, level):
        if level == 0:
            return np.zeros_like(channel)
        m = np.mean(channel)
        high = channel >= m
        low = ~high
        return (high.astype(np.float32) * (1 << (level - 1)) +
                _smqt(channel[high], level - 1) + _smqt(channel[low], level - 1))
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return _smqt(img_norm, levels).astype(np.float32)

# --- STFT‑based texture maps ----------------------------------------------------------

def stft_features(img: np.ndarray, window: int = 32, overlap: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orientation, frequency, energy maps using block STFT."""
    h, w = img.shape
    orient = np.zeros((h, w), np.float32)
    freq = np.zeros((h, w), np.float32)
    energy = np.zeros((h, w), np.float32)
    step = window - overlap
    for y in range(0, h - window + 1, step):
        for x in range(0, w - window + 1, step):
            patch = img[y:y + window, x:x + window] * np.hanning(window)[:, None] * np.hanning(window)[None, :]
            f = np.fft.fftshift(np.abs(np.fft.fft2(patch)))
            # polar coordinates
            ys, xs = np.mgrid[-window // 2:window // 2, -window // 2:window // 2]
            r = np.hypot(xs, ys)
            theta = np.arctan2(ys, xs) % np.pi
            idx = np.argmax(f)
            fy, fx = np.unravel_index(idx, f.shape)
            dom_freq = r[fy, fx] / window
            dom_theta = theta[fy, fx]
            block_slice = np.s_[y:y + window, x:x + window]
            orient[block_slice] = dom_theta
            freq[block_slice] = dom_freq
            energy[block_slice] = f.max()
    return orient, freq, energy

# --- Binarization + Thinning ---------------------------------------------------------

def _thin(img_bin: np.ndarray) -> np.ndarray:
    """Zhang‑Suen thinning using OpenCV morphology."""
    prev = np.zeros_like(img_bin)
    skel = img_bin.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        open_ = cv2.morphologyEx(skel, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(skel, open_)
        eroded = cv2.erode(skel, element)
        skel = cv2.bitwise_or(temp, eroded)
        if cv2.countNonZero(cv2.absdiff(skel, prev)) == 0:
            break
        prev = skel.copy()
    return skel

def binarize_thin(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bin_ = cv2.adaptiveThreshold(img_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 17, 2)
    thin = _thin(bin_)
    return bin_, thin

# --- Minutiae extraction -------------------------------------------------------------

def extract_minutiae(skeleton: np.ndarray) -> List[Tuple[int, int, float]]:
    """Return minutiae list (x, y, angle) using Crossing Number."""
    skel = skeleton // 255
    h, w = skel.shape
    minutiae = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x] == 0:
                continue
            nb = [skel[y - 1, x], skel[y - 1, x + 1], skel[y, x + 1], skel[y + 1, x + 1],
                  skel[y + 1, x], skel[y + 1, x - 1], skel[y, x - 1], skel[y - 1, x - 1]]
            cn = sum((nb[i] == 0 and nb[(i + 1) % 8] == 1) for i in range(8))
            if cn == 1:
                minutiae.append((x, y, 0.0))  # termination
            elif cn == 3:
                minutiae.append((x, y, 0.0))  # bifurcation
    return minutiae

################################################################################
# 3. MTCC Descriptor + Matching
################################################################################

# Cylinder parameters
RADIUS = 70
NS = 16
ND = 5
CELL_SIZE = RADIUS * 2 / NS


def _cell_indices() -> np.ndarray:
    idx = np.indices((NS, NS)).reshape(2, -1).T  # (N^2, 2)
    return idx - NS // 2 + 0.5

CELL_IDX = _cell_indices()
ANG_BINS = np.linspace(-np.pi, np.pi, ND, endpoint=False) + np.pi / ND


def create_cylinders(minutiae: List[Tuple[int, int, float]],
                     orient_map: np.ndarray,
                     texture_maps: Tuple[np.ndarray, np.ndarray, np.ndarray],
                     radius: int = RADIUS) -> List[np.ndarray]:
    """Generate MTCC descriptor for each minutia (orientation replaced by STFT textures)."""
    ori_map, freq_map, ener_map = texture_maps
    cylinders = []
    for x, y, _ in minutiae:
        cells = np.zeros((NS * NS, ND), np.float32)
        cx, cy = x, y
        for c_idx, (dx, dy) in enumerate(CELL_IDX):
            px, py = int(cx + dx * CELL_SIZE), int(cy + dy * CELL_SIZE)
            if 0 <= px < orient_map.shape[1] and 0 <= py < orient_map.shape[0]:
                theta = orient_map[py, px]
                freq = freq_map[py, px]
                ener = ener_map[py, px]
                for k, ang in enumerate(ANG_BINS):
                    dist_ang = 1 - np.cos(2 * (ang - theta))
                    cells[c_idx, k] = np.exp(-dist_ang * 2) * (ener + 1e-3)
        cylinders.append(cells.flatten())
    return cylinders

# --- Matching -----------------------------------------------------------------------

def match(cyl1: List[np.ndarray], cyl2: List[np.ndarray]) -> float:
    """Return similarity score between two templates."""
    if not cyl1 or not cyl2:
        return 0.0
    A = np.vstack(cyl1)
    B = np.vstack(cyl2)
    D = cdist(A, B, "cosine")
    sim = 1.0 - D.min(axis=1).mean()
    return float(sim)

# --- Error rates --------------------------------------------------------------------

def calculate_eer(genuine: List[float], impostor: List[float]) -> Tuple[float, float]:
    scores = np.array(genuine + impostor)
    labels = np.array([1] * len(genuine) + [0] * len(impostor))
    thresholds = np.unique(scores)
    fars, frrs = [], []
    for t in thresholds:
        preds = scores >= t
        fa = np.mean((preds == 1) & (labels == 0))
        fr = np.mean((preds == 0) & (labels == 1))
        fars.append(fa)
        frrs.append(fr)
    diffs = np.abs(np.array(fars) - np.array(frrs))
    idx = diffs.argmin()
    return (fars[idx] + frrs[idx]) / 2, thresholds[idx]

################################################################################
# 4. Visualization
################################################################################

def visualize_pipeline(original: np.ndarray, **steps: Dict[str, np.ndarray]):
    titles = ["Original"] + list(steps.keys())
    images = [original] + list(steps.values())
    n = len(images)
    cols = 3
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(12, 4 * rows))
    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(rows, cols, i)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

################################################################################
# 5. Example test harness
################################################################################

def _process_image(img_path: str):
    img = load_image(img_path)
    norm = normalize(img)
    mask = segment(norm)
    ori = _gradient_orientation(norm)
    gabor = gabor_enhance(norm, ori)
    smqt_img = smqt(gabor)
    o_map, f_map, e_map = stft_features(smqt_img)
    bin_img, thin = binarize_thin(smqt_img)
    mins = extract_minutiae(thin * mask)
    cyls = create_cylinders(mins, o_map, (o_map, f_map, e_map))
    return {
        "norm": norm, "mask": mask, "gabor": gabor,
        "smqt": smqt_img, "thin": thin, "minutiae": mins,
        "cyl": cyls, "orient": o_map
    }


def evaluate_dataset(dir_path: str):
    imgs = sorted(glob(os.path.join(dir_path, '*.tif')) + glob(os.path.join(dir_path, '*.bmp')))
    templates = [ _process_image(p) for p in imgs ]
    genuine, impostor = [], []
    n = len(imgs)
    for i in range(n):
        for j in range(i + 1, n):
            score = match(templates[i]['cyl'], templates[j]['cyl'])
            if os.path.basename(imgs[i])[:3] == os.path.basename(imgs[j])[:3]:
                genuine.append(score)
            else:
                impostor.append(score)
    eer, thr = calculate_eer(genuine, impostor)
    print(f"EER={eer:.4f} @ threshold {thr:.3f}")

if __name__ == "__main__":
    tmp1 = _process_image(R"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002\FVC2002\db1_a\1_1.tif")
    tmp2 = _process_image(R"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002\FVC2002\db1_a\1_2.tif")
    score = match(tmp1['cyl'], tmp2['cyl'])
    print(f"Similarity: {score:.4f}")
    visualize_pipeline(load_image(tmp1), norm=tmp1['norm'], mask=tmp1['mask'],
                            gabor=tmp1['gabor'], smqt=tmp1['smqt'], thin=tmp1['thin'])
    # import argparse
    # ap = argparse.ArgumentParser(description="MTCC Fingerprint Matcher")
    # ap.add_argument('--img1', help='path to first image')
    # ap.add_argument('--img2', help='path to second image')
    # ap.add_argument('--dir', help='directory with dataset for EER test')
    # ap.add_argument('--viz', action='store_true')
    # args = ap.parse_args()

    # if args.dir:
    #     evaluate_dataset(args.dir)
    # elif args.img1 and args.img2:
    #     tmp1 = _process_image(args.img1)
    #     tmp2 = _process_image(args.img2)
    #     score = match(tmp1['cyl'], tmp2['cyl'])
    #     print(f"Similarity: {score:.4f}")
    #     if args.viz:
    #         visualize_pipeline(load_image(args.img1), norm=tmp1['norm'], mask=tmp1['mask'],
    #                            gabor=tmp1['gabor'], smqt=tmp1['smqt'], thin=tmp1['thin'])
    # else:
    #     ap.print_help()
