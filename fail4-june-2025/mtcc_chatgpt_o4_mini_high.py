import cv2
import numpy as np
from skimage.filters import gabor_kernel
from skimage.morphology import skeletonize
from scipy.signal import stft
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def load_image(path: str) -> np.ndarray:
    """Load fingerprint image in grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def normalize_image(img: np.ndarray, target_mean: float = 0, target_std: float = 1) -> np.ndarray:
    """Normalize image to given mean and std."""
    img_f = img.astype(np.float32)
    m, s = img_f.mean(), img_f.std()
    if s < 1e-5:
        return np.full_like(img_f, target_mean)
    norm = (img_f - m) / s
    return norm * target_std + target_mean


def segment_image(img: np.ndarray, block_size: int = 16, var_thresh: float = 0.01) -> np.ndarray:
    """Block-wise variance segmentation."""
    h, w = img.shape
    mask = np.zeros((h, w), bool)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            blk = img[y:y+block_size, x:x+block_size]
            if blk.std() > var_thresh:
                mask[y:y+block_size, x:x+block_size] = True
    return mask


def enhance_gabor(img: np.ndarray, orientations: np.ndarray, frequencies: np.ndarray,
                 kernel_size: int = 31) -> np.ndarray:
    """Gabor enhancement per-pixel using orientation and frequency maps."""
    h, w = img.shape
    enhanced = np.zeros_like(img, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            theta = orientations[y, x]
            freq = frequencies[y, x]
            if freq <= 0:
                continue
            kernel = np.real(gabor_kernel(freq, theta=theta, bandwidth=1, n_stds=3))
            k = kernel_size // 2
            if y-k<0 or y+k>=h or x-k<0 or x+k>=w:
                continue
            patch = img[y-k:y+k+1, x-k:x+k+1]
            enhanced[y, x] = np.sum(patch * kernel)
    return enhanced


def apply_smqt(img: np.ndarray) -> np.ndarray:
    """Successive Mean Quantization Transform normalization."""
    img = img.astype(np.float32)
    out = np.copy(img)
    for _ in range(3):
        m = out.mean()
        out = np.where(out < m, out + (m - out) * 0.5, out - (out - m) * 0.5)
    return out


def stft_analysis(img: np.ndarray, win_size: int = 32, overlap: float = 0.5):
    """Compute orientation and frequency maps via STFT."""
    h, w = img.shape
    step = int(win_size * (1 - overlap))
    ori = np.zeros((h, w), float)
    freq = np.zeros((h, w), float)
    for y in range(0, h-win_size+1, step):
        for x in range(0, w-win_size+1, step):
            win = img[y:y+win_size, x:x+win_size]
            f, t, Z = stft(win, nperseg=win_size, noverlap=int(win_size*0.9))
            mag = np.abs(Z)
            idx,_,_ = np.unravel_index(np.argmax(mag), mag.shape)
            fy, fx = idx
            ori_val = np.arctan2(fy - win_size/2, fx - win_size/2)
            freq_val = f[fy] if fy < len(f) else 0
            ori[y:y+win_size, x:x+win_size] = ori_val
            freq[y:y+win_size, x:x+win_size] = freq_val
    return ori, freq


def binarize_and_thin(img: np.ndarray, block_size: int = 16, thresh: float = 0.5):
    """Adaptive binarization plus thinning."""
    h, w = img.shape
    bin_img = np.zeros((h, w), bool)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            blk = img[y:y+block_size, x:x+block_size]
            t = blk.mean() * thresh
            bin_img[y:y+block_size, x:x+block_size] = blk > t
    thin = skeletonize(bin_img)
    return bin_img, thin


def extract_minutiae(thin: np.ndarray):
    """Simple end/bifurcation minutiae extraction."""
    minutiae = []
    h, w = thin.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if not thin[y, x]:
                continue
            n = np.sum(thin[y-1:y+2, x-1:x+2]) - 1
            if n == 1 or n == 3:
                minutiae.append((x, y, n))
    return minutiae


def create_cylinders(minutiae, ori, enhanced, R=65, Ns=18, Nd=5):
    """Generate MTCC descriptor cylinders for each minutia."""
    # Placeholder: detailed implementation per MTCC paper
    cylinders = []
    for m in minutiae:
        cyl = np.zeros((Ns, Ns, Nd), float)
        cylinders.append(cyl)
    return cylinders


def match_cylinders(cyls1, cyls2):
    """Compute similarity matrix and match score."""
    n1, n2 = len(cyls1), len(cyls2)
    sim = np.zeros((n1, n2), float)
    for i, c1 in enumerate(cyls1):
        for j, c2 in enumerate(cyls2):
            sim[i, j] = 1 - np.linalg.norm(c1 - c2) / (np.linalg.norm(c1) + np.linalg.norm(c2) + 1e-6)
    return sim.max()


def calculate_eer(scores, labels):
    """Calculate Equal Error Rate."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer


def match_two_fingers(path1: str, path2: str) -> float:
    """Full MTCC pipeline: returns similarity score."""
    img1 = load_image(path1)
    img2 = load_image(path2)
    n1 = normalize_image(img1)
    n2 = normalize_image(img2)
    ori1, freq1 = stft_analysis(n1)
    ori2, freq2 = stft_analysis(n2)
    g1 = enhance_gabor(n1, ori1, freq1)
    g2 = enhance_gabor(n2, ori2, freq2)
    _, t1 = binarize_and_thin(g1)
    _, t2 = binarize_and_thin(g2)
    m1 = extract_minutiae(t1)
    m2 = extract_minutiae(t2)
    c1 = create_cylinders(m1, ori1, g1)
    c2 = create_cylinders(m2, ori2, g2)
    return match_cylinders(c1, c2)


def visualize_steps(imgs: list, titles: list):
    """Show images side by side."""
    n = len(imgs)
    plt.figure(figsize=(4*n, 4))
    for i, (im, t) in enumerate(zip(imgs, titles), 1):
        plt.subplot(1, n, i)
        plt.imshow(im, cmap='gray')
        plt.title(t)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    img1 = R'C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2000\FVC2000\Db1_a\1_1.tif'
    img2 = R'C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2000\FVC2000\Db1_a\1_2.tif'
    score = match_two_fingers(img1, img2)
    