import numpy as np
import cv2
import os
import glob
from typing import Tuple, List
from skimage.morphology import skeletonize
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# --- Normalization ---
def normalize_image(img: np.ndarray, target_mean=0, target_std=1) -> np.ndarray:
    img = img.astype(np.float32)
    mean, std = img.mean(), img.std()
    if std < 1e-5:
        return np.zeros_like(img)
    normalized = (img - mean) / std
    return normalized * target_std + target_mean

# --- Segment Image ---
def segment_image(img: np.ndarray,
                  block_size: int = 16,
                  var_thresh: float = 0.01
                 ) -> np.ndarray:
    h, w = img.shape
    mask = np.zeros_like(img, dtype=np.uint8)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            blk = img[y:y+block_size, x:x+block_size]
            if blk.std() > var_thresh:
                mask[y:y+block_size, x:x+block_size] = 1
    # fill small holes
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

from scipy.ndimage import gaussian_filter

def smooth_orientation_field(orientation_blocks: np.ndarray, sigma: float = 2) -> np.ndarray:
    sin2 = np.sin(2 * orientation_blocks)
    cos2 = np.cos(2 * orientation_blocks)
    sin2_smooth = gaussian_filter(sin2, sigma)
    cos2_smooth = gaussian_filter(cos2, sigma)
    return 0.5 * np.arctan2(sin2_smooth, cos2_smooth)

# --- Orientation ---
def compute_orientation_field(img: np.ndarray, block_size: int = 16) -> np.ndarray:
    h, w = img.shape
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    orientation = np.zeros((h // block_size, w // block_size), dtype=np.float32)
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            gx = grad_x[i:i+block_size, j:j+block_size].flatten()
            gy = grad_y[i:i+block_size, j:j+block_size].flatten()
            v_x = np.sum(2 * gx * gy)
            v_y = np.sum(gx**2 - gy**2)
            orientation[i // block_size, j // block_size] = 0.5 * np.arctan2(v_x, v_y)
    return orientation

def interpolate_orientation_field(orientation_blocks: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(orientation_blocks, image_shape[::-1], interpolation=cv2.INTER_LINEAR)

# --- Gabor ---
def build_gabor_kernel(
    theta: float,
    frequency: float = 0.1,
    sigma_x: float = 4.0,
    sigma_y: float = 4.0,
    kernel_size: int = 21
) -> np.ndarray:
    half = kernel_size // 2
    y, x = np.meshgrid(np.arange(-half, half+1),
                       np.arange(-half, half+1))
    x_theta =  x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-0.5*(x_theta**2/sigma_x**2 + y_theta**2/sigma_y**2)) \
         * np.cos(2*np.pi*frequency*x_theta)
    return gb

def enhance_fingerprint_gabor(img: np.ndarray,
                              orientation_field: np.ndarray,
                              block_size: int = 16,
                              overlap: int = 8,
                              frequency: float = 0.1) -> np.ndarray:
    img = normalize_image(img)
    h, w = img.shape
    step = block_size - overlap
    output = np.zeros_like(img)
    weight_map = np.zeros_like(img)
    orientation_full = interpolate_orientation_field(orientation_field, img.shape)
    for i in range(0, h - block_size + 1, step):
        for j in range(0, w - block_size + 1, step):
            block = img[i:i+block_size, j:j+block_size]
            theta = np.median(orientation_full[i:i+block_size, j:j+block_size])
            kernel = build_gabor_kernel(
                    theta,
                    frequency=0.07,
                    sigma_x=8.0,
                    sigma_y=8.0,
                    kernel_size=31
                )
            filtered = cv2.filter2D(block, -1, kernel)
            output[i:i+block_size, j:j+block_size] += filtered
            weight_map[i:i+block_size, j:j+block_size] += 1
    weight_map[weight_map == 0] = 1
    return output / weight_map

# --- Binarization + Thinning ---
def binarize_and_thin(enhanced_img, mask=None):
    # scale to [0..255]
    mn, mx = enhanced_img.min(), enhanced_img.max()
    img8 = ((enhanced_img - mn)/(mx-mn)*255).astype(np.uint8)

    # Otsu + invert
    _, binary = cv2.threshold(img8, 0, 255,
                              cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary = 255 - binary

    # mask away background
    if mask is not None:
        binary[mask==0] = 0

    # small closing to bridge tiny gaps
    kernel = np.ones((3,3),np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # one-pixel skeleton
    skeleton = skeletonize(binary//255).astype(np.uint8)
    return binary, skeleton



# --- Minutiae ---
def extract_minutiae(thinned: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = thinned.shape
    coords, types = [], []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if thinned[y, x] != 1: continue
            window = thinned[y-1:y+2, x-1:x+2]
            cn = np.sum(np.abs(np.diff(window.flatten()[[0,1,2,5,8,7,6,3,0]]))) // 2
            if cn == 1:
                coords.append((x, y)); types.append('ending')
            elif cn == 3:
                coords.append((x, y)); types.append('bifurcation')
    return np.array(coords), np.array(types)

# --- Visualization ---
def visualize_pipeline(img: np.ndarray, norm: np.ndarray, enhanced: np.ndarray, binary: np.ndarray,
                       skeleton: np.ndarray, coords: np.ndarray, types: np.ndarray):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0, 0].imshow(img, cmap='gray'); axs[0, 0].set_title("Original")
    axs[0, 1].imshow(norm, cmap='gray'); axs[0, 1].set_title("Normalized")
    axs[0, 2].imshow(enhanced, cmap='gray'); axs[0, 2].set_title("Gabor Enhanced")
    axs[1, 0].imshow(binary, cmap='gray'); axs[1, 0].set_title("Binarized")
    axs[1, 1].imshow(skeleton, cmap='gray'); axs[1, 1].set_title("Thinned")
    m_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for (x, y), t in zip(coords, types):
        color = (0, 255, 0) if t == 'ending' else (0, 0, 255)
        cv2.circle(m_img, (x, y), 2, color, 1)
    axs[1, 2].imshow(m_img[..., ::-1]); axs[1, 2].set_title("Minutiae")
    for ax in axs.ravel(): ax.axis('off')
    plt.tight_layout(); plt.show()


# --- MTCC Descriptor ---
def generate_mtcc_descriptor(minutiae: np.ndarray,
                             orientation_img: np.ndarray,
                             texture_img: np.ndarray,
                             radius: int = 100,
                             sectors_per_ring: int = 16,
                             rings: int = 5,
                             orientations: int = 8) -> List[np.ndarray]:
    descriptors = []
    sector_angle = 2 * np.pi / sectors_per_ring
    band_width = radius / rings
    h, w = texture_img.shape
    for (x0, y0) in minutiae:
        theta0 = orientation_img[int(y0), int(x0)]
        vec = []
        for r in range(rings):
            r0, r1 = r * band_width, (r + 1) * band_width
            for s in range(sectors_per_ring):
                theta_s = s * sector_angle
                theta_e = (s + 1) * sector_angle
                pts = []
                for dr in np.linspace(r0, r1, 3):
                    for dt in np.linspace(theta_s, theta_e, 3):
                        a = theta0 + dt
                        x = int(x0 + dr * np.cos(a))
                        y = int(y0 + dr * np.sin(a))
                        if 0 <= x < w and 0 <= y < h:
                            pts.append((y, x))
                if not pts:
                    vec.extend([0.0] * orientations); continue
                values = np.array([texture_img[y, x] for y, x in pts])
                mean_val = np.mean(values)
                for _ in range(orientations):
                    vec.append(np.mean(np.abs(values - mean_val)))
        descriptors.append(np.array(vec, dtype=np.float32))
    return descriptors

def rotate_mtcc_descriptor(desc: np.ndarray, shift: int, orientations: int = 8) -> np.ndarray:
    sectors = len(desc) // orientations
    desc_reshaped = desc.reshape((sectors, orientations))
    return np.roll(desc_reshaped, shift=shift, axis=0).flatten()

def match_with_rotation_invariance(vec1: np.ndarray, vec2: np.ndarray, max_shifts: int = 16) -> float:
    return max(1 - np.linalg.norm(vec1 - rotate_mtcc_descriptor(vec2, s)) /
               (np.linalg.norm(vec1) + np.linalg.norm(vec2) + 1e-6) for s in range(max_shifts))

def relax_scores(sim: np.ndarray, top_n: int = 10) -> float:
    pairs = np.dstack(np.unravel_index(np.argsort(-sim.ravel()), sim.shape))[0][:top_n]
    score, used_a, used_b = 0, set(), set()
    for i, j in pairs:
        if i in used_a or j in used_b: continue
        score += sim[i, j]; used_a.add(i); used_b.add(j)
    return score / max(1, len(used_a))

# --- Matching Two Fingerprints ---
def load_fvc_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0

# --- Matching Two Fingerprints ---
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def smooth_orientation_field(orientation_blocks: np.ndarray, sigma: float = 2) -> np.ndarray:
    sin2 = np.sin(2 * orientation_blocks)
    cos2 = np.cos(2 * orientation_blocks)
    sin2_smooth = gaussian_filter(sin2, sigma)
    cos2_smooth = gaussian_filter(cos2, sigma)
    return 0.5 * np.arctan2(sin2_smooth, cos2_smooth)

def match_two_fingerprints(path1, path2, visualize=False):
    img1 = load_fvc_image(path1)
    img2 = load_fvc_image(path2)

    # Segment foreground
    m1 = segment_image(img1)
    m2 = segment_image(img2)

    # Normalize inside mask
    n1 = normalize_image(img1 * m1)
    n2 = normalize_image(img2 * m2)

    # --- Blockwise orientation (then smooth) ---
    o1 = compute_orientation_field(n1)
    o2 = compute_orientation_field(n2)
    o1 = smooth_orientation_field(o1)
    o2 = smooth_orientation_field(o2)

    # --- Interpolate full orientation field ---
    of1 = interpolate_orientation_field(o1, img1.shape)
    of2 = interpolate_orientation_field(o2, img2.shape)

    # --- Gabor enhancement ---
    g1 = enhance_fingerprint_gabor(n1, o1)
    g2 = enhance_fingerprint_gabor(n2, o2)

    # --- Segment (optionally re-run) ---
    m1 = segment_image(g1)
    m2 = segment_image(g2)

    # --- Binarize and thin ---
    b1, t1 = binarize_and_thin(g1, mask=m1)
    b2, t2 = binarize_and_thin(g2, mask=m2)

    # --- Minutiae extraction ---
    c1, ty1 = extract_minutiae(t1)
    c2, ty2 = extract_minutiae(t2)

    if visualize:
        visualize_pipeline(img1, n1, g1, b1, t1, c1, ty1)
        visualize_pipeline(img2, n2, g2, b2, t2, c2, ty2)

    # --- MTCC descriptors ---
    mt1 = generate_mtcc_descriptor(c1, of1, g1)
    mt2 = generate_mtcc_descriptor(c2, of2, g2)

    # --- Matching (rotation invariant, relaxed) ---
    sim = np.zeros((len(mt1), len(mt2)), np.float32)
    for i, d1 in enumerate(mt1):
        for j, d2 in enumerate(mt2):
            sim[i, j] = match_with_rotation_invariance(d1, d2)
    score = relax_scores(sim, top_n=10)
    print(f"Matching score: {score:.4f}")
    return score



def match_two_fingerprints(path1, path2, visualize=False) -> float:
    img1 = load_fvc_image(path1)
    img2 = load_fvc_image(path2)

    # ← new: get a mask for each
    m1 = segment_image(img1)
    m2 = segment_image(img2)

    # normalize & orient
    n1 = normalize_image(img1 * m1)
    n2 = normalize_image(img2 * m2)
    o1 = compute_orientation_field(n1)
    o2 = compute_orientation_field(n2)

    # gabor‐enhance
    g1 = enhance_fingerprint_gabor(n1, o1)
    g2 = enhance_fingerprint_gabor(n2, o2)

    # ← updated binarization/thinning
    b1, t1 = binarize_and_thin(g1, mask=m1)
    b2, t2 = binarize_and_thin(g2, mask=m2)

    # minutiae
    c1, ty1 = extract_minutiae(t1)
    c2, ty2 = extract_minutiae(t2)

    if visualize:
        visualize_pipeline(img1, n1, g1, b1, t1, c1, ty1)
        visualize_pipeline(img2, n2, g2, b2, t2, c2, ty2)

    # MTCC
    mt1 = generate_mtcc_descriptor(c1,
               interpolate_orientation_field(o1, img1.shape), g1)
    mt2 = generate_mtcc_descriptor(c2,
               interpolate_orientation_field(o2, img2.shape), g2)

    sim = np.zeros((len(mt1), len(mt2)), np.float32)
    for i, d1 in enumerate(mt1):
        for j, d2 in enumerate(mt2):
            sim[i,j] = match_with_rotation_invariance(d1, d2)

    score = relax_scores(sim, top_n=10)

    print(f"Matching score: {score:.4f}")   # ← or return and print at call‐site
    return score


# --- Batch Matching ---
def batch_evaluate_fvc_folder(folder: str, ext: str = '*.tif') -> Tuple[List[float], List[float]]:
    image_paths = sorted(glob.glob(os.path.join(folder, ext)))
    from collections import defaultdict
    subj_map = defaultdict(list)
    for path in image_paths:
        sid = os.path.basename(path).split('_')[0]
        subj_map[sid].append(path)
    genuine, impostor = [], []
    subs = list(subj_map.keys())
    for paths in subj_map.values():
        for i in range(len(paths)):
            for j in range(i+1, len(paths)):
                genuine.append(match_two_fingerprints(paths[i], paths[j]))
    for i in range(len(subs)):
        for j in range(i+1, len(subs)):
            impostor.append(match_two_fingerprints(subj_map[subs[i]][0], subj_map[subs[j]][0]))
    return genuine, impostor

# --- EER ---
def compute_eer(genuine_scores: List[float], impostor_scores: List[float]) -> float:
    y_true = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
    scores = np.array(genuine_scores + impostor_scores)
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100

match_two_fingerprints(R'C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2000\FVC2000\Db1_a\1_1.tif', R'C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2000\FVC2000\Db1_a\1_2.tif',visualize=True)
