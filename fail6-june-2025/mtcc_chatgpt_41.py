import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, binary_fill_holes
from scipy.signal import stft
from scipy.spatial.distance import cdist
from scipy.stats import norm
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32)

def normalize(img, mean=0.0, std=1.0):
    img = img.astype(np.float32)
    m, s = img.mean(), img.std()
    return ((img - m) / (s + 1e-8)) * std + mean

def segment(img, block_size=16, var_thresh=0.01):
    h, w = img.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    h_p, w_p = img_padded.shape

    mask = np.zeros((h_p, w_p), dtype=np.uint8)
    for y in range(0, h_p, block_size):
        for x in range(0, w_p, block_size):
            block = img_padded[y:y+block_size, x:x+block_size]
            block_var = np.var(block) / (255.0**2)
            if block_var > var_thresh:
                mask[y:y+block_size, x:x+block_size] = 1

    # Remove padding
    mask = mask[:h, :w]
    # Morphological postprocessing
    mask = binary_closing(binary_opening(mask, np.ones((5,5))), np.ones((5,5)))
    mask = binary_fill_holes(mask).astype(np.uint8)
    return mask


    # Now fill the mask blockwise
    mask = np.zeros((h_p, w_p), dtype=np.uint8)
    idx = 0
    for y in range(0, h_p, block_size):
        for x in range(0, w_p, block_size):
            mask[y:y+block_size, x:x+block_size] = mask_blocks[idx]
            idx += 1
    mask = mask[:img.shape[0], :img.shape[1]]
    mask = binary_closing(binary_opening(mask, np.ones((5,5))), np.ones((5,5)))
    mask = binary_fill_holes(mask).astype(np.uint8)
    return mask


def smqt(img, levels=8):
    def _smqt(level_img, level):
        if level == 0 or level_img.size == 0 or np.all(level_img == level_img[0]):
            return np.zeros_like(level_img)
        median = np.median(level_img)
        mask = level_img > median
        left = _smqt(level_img[~mask], level-1)
        right = _smqt(level_img[mask], level-1)
        out = np.zeros_like(level_img, dtype=np.uint8)
        out[~mask] = left
        out[mask] = right + (1 << (level-1))
        return out
    flat = img.flatten()
    smqt_flat = _smqt(flat, levels)
    return smqt_flat.reshape(img.shape)


def stft_features(img, window=16, overlap=8):
    h, w = img.shape
    orientation_map = np.zeros_like(img, dtype=np.float32)
    frequency_map = np.zeros_like(img, dtype=np.float32)
    energy_map = np.zeros_like(img, dtype=np.float32)
    for y in range(0, h - window + 1, overlap):
        for x in range(0, w - window + 1, overlap):
            patch = img[y:y+window, x:x+window]
            f = np.fft.fftshift(np.fft.fft2(patch * np.hanning(window)[:,None] * np.hanning(window)))
            mag = np.abs(f)
            cy, cx = np.unravel_index(np.argmax(mag), mag.shape)
            fy, fx = cy - window//2, cx - window//2
            freq = np.sqrt(fx**2 + fy**2) / window
            ori = np.arctan2(fy, fx)
            en = np.log1p(mag).sum()
            orientation_map[y:y+window, x:x+window] = ori
            frequency_map[y:y+window, x:x+window] = freq
            energy_map[y:y+window, x:x+window] = en
    # Smoothing
    orientation_map = gaussian_filter(orientation_map, 3)
    frequency_map = gaussian_filter(frequency_map, 3)
    energy_map = gaussian_filter(energy_map, 3)
    return orientation_map, frequency_map, energy_map

def gabor_enhance(img, orientation_map, freq_map, ksize=21):
    # Contextual Gabor filter ([8], [9], [24])
    enhanced = np.zeros_like(img, dtype=np.float32)
    half = ksize // 2
    for y in range(half, img.shape[0]-half):
        for x in range(half, img.shape[1]-half):
            theta = orientation_map[y,x]
            freq = freq_map[y,x]
            if freq < 0.05 or freq > 0.25:
                continue
            kernel = cv2.getGaborKernel((ksize, ksize), sigma=4.0, theta=theta, lambd=1.0/freq, gamma=0.5, psi=0)
            patch = img[y-half:y+half+1, x-half:x+half+1]
            enhanced[y,x] = np.sum(patch * kernel)
    # Normalize to 0-255
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced.astype(np.uint8)

def binarize_thin(img):
    # Adaptive threshold + Zhang-Suen thinning
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 7)
    thin = skeletonize(th > 0).astype(np.uint8) * 255
    return thin

def extract_minutiae(skeleton):
    # Crossing Number (CN) algorithm
    rows, cols = skeleton.shape
    minutiae = []
    offsets = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    for y in range(1, rows-1):
        for x in range(1, cols-1):
            if skeleton[y, x]:
                nb = [skeleton[y+dy, x+dx]>0 for dx,dy in offsets]
                cn = sum((nb[i] != nb[(i+1)%8]) for i in range(8)) // 2
                if cn == 1:
                    minutiae.append((x, y, 'ending'))
                elif cn == 3:
                    minutiae.append((x, y, 'bifurcation'))
    return np.array(minutiae)

def create_cylinders(minutiae, texture_maps, radius=70, ns=18, nd=5):
    # MTCC: replace angular with STFT features ([11][22])
    orientation_map, freq_map, energy_map = texture_maps
    cylinders = []
    for m in minutiae:
        x, y = int(m[0]), int(m[1])
        cylinder = np.zeros((ns, ns, nd), dtype=np.float32)
        for i in range(ns):
            for j in range(ns):
                for k in range(nd):
                    dx = (i - ns//2) * radius / ns
                    dy = (j - ns//2) * radius / ns
                    phi = 2 * np.pi * k / nd
                    cx, cy = int(x + dx*np.cos(phi) - dy*np.sin(phi)), int(y + dx*np.sin(phi) + dy*np.cos(phi))
                    if 0 <= cx < orientation_map.shape[1] and 0 <= cy < orientation_map.shape[0]:
                        f = freq_map[cy, cx]
                        o = orientation_map[cy, cx]
                        e = energy_map[cy, cx]
                        # Use orientation, frequency, or energy as per variant (here, orientation)
                        cylinder[i, j, k] = o
        cylinders.append(cylinder)
    return cylinders

def match(cylinders1, cylinders2):
    # Local Similarity Sort (LSS) ([11])
    sims = []
    for c1 in cylinders1:
        for c2 in cylinders2:
            # cell-wise cosine similarity (or sine or euclidean as per [22])
            valid = np.isfinite(c1) & np.isfinite(c2)
            if np.any(valid):
                sim = np.sum(np.cos(c1[valid] - c2[valid])) / np.sum(valid)
                sims.append(sim)
    if len(sims) == 0:
        return 0.0
    return np.max(sims)

def calculate_eer(genuine_scores, impostor_scores):
    from sklearn.metrics import roc_curve
    y_true = np.array([1]*len(genuine_scores) + [0]*len(impostor_scores))
    y_score = np.array(genuine_scores + impostor_scores)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    # EER is where FPR ~= FNR
    idx = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2

def visualize_pipeline(original, normalized, segmented, gabor, smqt_img,
                      stft_ori, stft_freq, stft_en, bin_img, thin_img, minutiae=None):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    imgs = [original, normalized, segmented*255, gabor, smqt_img,
            stft_ori, stft_freq, stft_en, bin_img]
    titles = ['Original', 'Normalized', 'Segmented Mask', 'Gabor Enhanced', 'SMQT',
              'STFT Orientation', 'STFT Frequency', 'STFT Energy', 'Binarized']
    for i, (im, t) in enumerate(zip(imgs, titles)):
        ax = axs[i//3, i%3]
        ax.imshow(im, cmap='gray')
        ax.set_title(t)
        ax.axis('off')
    axs[2,2].imshow(thin_img, cmap='gray')
    if minutiae is not None:
        yx = np.array([[m[1], m[0]] for m in minutiae])
        axs[2,2].scatter(yx[:,1], yx[:,0], c='r', s=10)
    axs[2,2].set_title('Thinned + Minutiae')
    axs[2,2].axis('off')
    plt.tight_layout()
    plt.show()

# === Scriptable FVC test harness ===

def process_fingerprint(img):
    norm = normalize(img)
    seg = segment(norm)
    smqt_img = smqt((norm*seg).astype(np.float32), levels=8)
    stft_ori, stft_freq, stft_en = stft_features(smqt_img)
    gabor = gabor_enhance(smqt_img, stft_ori, stft_freq)
    bin_img = cv2.adaptiveThreshold(gabor, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 7)
    thin_img = skeletonize(bin_img > 0).astype(np.uint8) * 255
    minutiae = extract_minutiae(thin_img)
    cyl = create_cylinders(minutiae, (stft_ori, stft_freq, stft_en))
    return {'original': img, 'normalized': norm, 'segmented': seg, 'gabor': gabor, 'smqt': smqt_img,
            'stft_ori': stft_ori, 'stft_freq': stft_freq, 'stft_en': stft_en, 'bin': bin_img, 'thin': thin_img,
            'minutiae': minutiae, 'cyl': cyl}

def run_fvc_protocol(image_paths, show_visuals=False):
    # Process all images
    features = [process_fingerprint(load_image(p)) for p in image_paths]
    # Compute scores for all genuine and impostor pairs
    genuine_scores, impostor_scores = [], []
    n = len(features)
    for i in range(n):
        for j in range(i+1, n):
            score = match(features[i]['cyl'], features[j]['cyl'])
            # Simple protocol: same prefix -> genuine, else impostor
            if image_paths[i].split('_')[0] == image_paths[j].split('_')[0]:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)
#    eer = calculate_eer(genuine_scores, impostor_scores)
#    print(f"EER: {eer:.4f}")
    if show_visuals:
        # Visualize one sample pipeline
        f = features[0]
        visualize_pipeline(f['original'], f['normalized'], f['segmented'], f['gabor'],
                           f['smqt'], f['stft_ori'], f['stft_freq'], f['stft_en'],
                           f['bin'], f['thin'], f['minutiae'])
#    return eer

import os
def get_file_paths_os(directory_path):
    """
    Returns a list of full file paths in the specified directory.
    Does not include files in subdirectories.
    """
    file_paths = []
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            file_paths.append(full_path)
    return file_paths

folder_path = R'C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002\Db1_a'
image_paths = get_file_paths_os(folder_path)
run_fvc_protocol(image_paths, show_visuals=True)

