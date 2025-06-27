import cv2
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt


def load_image(path):
    """Load a grayscale fingerprint image from disk."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def normalize(img):
    """Zero-mean, unit-variance normalization."""
    img = img.astype(np.float32)
    mean, std = img.mean(), img.std()
    if std < 1e-5:
        return np.zeros_like(img)
    return (img - mean) / std


def segment(img, block_size=16, var_thresh=0.01):
    """Block-wise variance segmentation: high-variance blocks are foreground."""
    h, w = img.shape
    mask = np.zeros_like(img, dtype=np.uint8)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            blk = img[y:y+block_size, x:x+block_size]
            if np.var(blk) > var_thresh:
                mask[y:y+block_size, x:x+block_size] = 255
    return mask


def compute_orientation_map(img, block_size=16):
    """Estimate local ridge orientation via Sobel gradients and smoothing."""
    img = img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    orientation = np.arctan2(gy, gx)
    kernel = np.ones((block_size, block_size), dtype=np.float32) / (block_size**2)
    orientation = cv2.filter2D(orientation, -1, kernel)
    return orientation


def gabor_enhance(img, orientation_map, freq=0.1, ksize=31):
    """Enhance ridges using oriented Gabor filters tuned by local orientation and frequency."""
    img_f = img.astype(np.float32)
    enhanced = np.zeros_like(img_f)
    half = ksize // 2
    for y in range(half, img.shape[0] - half):
        for x in range(half, img.shape[1] - half):
            theta = orientation_map[y, x]
            lam = 1.0 / freq
            kernel = cv2.getGaborKernel((ksize, ksize), sigma=4.0, theta=theta, lambd=lam, gamma=0.5, psi=0)
            patch = img_f[y-half:y+half+1, x-half:x+half+1]
            enhanced[y, x] = np.sum(patch * kernel)
    return normalize(enhanced)


def smqt(img, levels=8):
    """Successive Mean Quantization Transform."""
    img_u8 = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    output = np.zeros_like(img_u8, dtype=np.uint8)

    def recurse(sub, mask, level, lo, hi):
        if level >= levels or lo >= hi:
            return
        mid = (lo + hi) // 2
        m_low = mask & (sub <= mid)
        m_high = mask & (sub > mid)
        output[m_high] |= (1 << (levels - level - 1))
        recurse(sub, m_low, level+1, lo, mid)
        recurse(sub, m_high, level+1, mid+1, hi)

    mask_all = np.ones_like(img_u8, dtype=bool)
    recurse(img_u8, mask_all, 0, img_u8.min(), img_u8.max())
    return output


def stft_features(img, window=16, overlap=8):
    """Compute STFT-based texture maps: orientation, frequency, energy."""
    f, t, Z = stft(img.astype(np.float32), nperseg=window, noverlap=overlap, axis=-1)
    energy = np.abs(Z)
    orient = np.angle(Z)
    energy_avg = np.mean(energy, axis=-1)
    orient_avg = np.mean(orient, axis=-1)
    freq_avg = np.mean(f)
    energy_img = cv2.resize(energy_avg, (img.shape[1], img.shape[0]))
    orient_img = cv2.resize(orient_avg, (img.shape[1], img.shape[0]))
    freq_img = np.full_like(orient_img, freq_avg)
    return orient_img, freq_img, energy_img


def zhang_suen_thinning(bin_img):
    """Zhang-Suen skeletonization."""
    img = bin_img.copy() // 255
    changing = True
    while changing:
        changing = False
        to_remove = []
        rows, cols = img.shape
        # Sub-iteration 1
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                P = img[i-1:i+2, j-1:j+2].flatten()
                if img[i, j] == 1:
                    neighbors = P[[1,2,5,8,7,6,3,0]]
                    C = ((neighbors == 0) & (np.roll(neighbors, -1) == 1)).sum()
                    N = neighbors.sum()
                    if 2 <= N <= 6 and C == 1 and P[1]*P[5]*P[7] == 0 and P[5]*P[7]*P[3] == 0:
                        to_remove.append((i, j))
        if to_remove:
            for i,j in to_remove:
                img[i,j] = 0
            changing = True
        to_remove = []
        # Sub-iteration 2
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                P = img[i-1:i+2, j-1:j+2].flatten()
                if img[i, j] == 1:
                    neighbors = P[[1,2,5,8,7,6,3,0]]
                    C = ((neighbors == 0) & (np.roll(neighbors, -1) == 1)).sum()
                    N = neighbors.sum()
                    if 2 <= N <= 6 and C == 1 and P[1]*P[5]*P[3] == 0 and P[1]*P[7]*P[3] == 0:
                        to_remove.append((i, j))
        if to_remove:
            for i,j in to_remove:
                img[i,j] = 0
            changing = True
    return (img * 255).astype(np.uint8)


def binarize_thin(img):
    """Adaptive thresholding and thinning."""
    u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bin_img = cv2.adaptiveThreshold(u8, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, blockSize=15, C=-2)
    thin = zhang_suen_thinning(bin_img)
    return bin_img, thin


def extract_minutiae(skel):
    """Crossing Number (CN) to detect ridge endings and bifurcations."""
    minutiae = []
    rows, cols = skel.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if skel[i, j] == 255:
                P = skel[i-1:i+2, j-1:j+2] // 255
                nb = [P[1,0], P[0,0], P[0,1], P[0,2], P[1,2], P[2,2], P[2,1], P[2,0]]
                cn = sum((nb[k] == 0 and nb[(k+1)%8] == 1) for k in range(8))
                if cn == 1:
                    minutiae.append(('ending', (j, i)))
                elif cn == 3:
                    minutiae.append(('bifurcation', (j, i)))
    return minutiae


def create_cylinders(minutiae, texture_maps, radius=70, bins=8):
    """Generate MTCC descriptors by histogramming STFT textures around minutiae."""
    orient_map, freq_map, energy_map = texture_maps
    cylinders = []
    for mtype, (x, y) in minutiae:
        x, y = int(x), int(y)
        y0, y1 = max(0, y-radius), min(orient_map.shape[0], y+radius)
        x0, x1 = max(0, x-radius), min(orient_map.shape[1], x+radius)
        po = orient_map[y0:y1, x0:x1]
        pf = freq_map[y0:y1, x0:x1]
        pe = energy_map[y0:y1, x0:x1]
        ho, _ = np.histogram(po, bins=bins, range=(-np.pi, np.pi))
        hf, _ = np.histogram(pf, bins=bins)
        he, _ = np.histogram(pe, bins=bins)
        desc = np.hstack([ho, hf, he]).astype(np.float32)
        norm = np.linalg.norm(desc)
        if norm > 0:
            desc /= norm
        cylinders.append(desc)
    return cylinders


def match(cylinders1, cylinders2):
    """Similarity scoring by matching cylinder descriptors."""
    if not cylinders1 or not cylinders2:
        return 0.0
    sims = [max(np.dot(d1, d2) for d2 in cylinders2) for d1 in cylinders1]
    return float(np.mean(sims))


def calculate_eer(genuine, impostor):
    """Compute Equal Error Rate (EER) from score distributions."""
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    thr_vals = np.sort(np.unique(scores))
    fnrs, fprs = [], []
    for thr in thr_vals:
        fnr = np.sum(np.array(genuine) < thr) / len(genuine)
        fpr = np.sum(np.array(impostor) >= thr) / len(impostor)
        fnrs.append(fnr); fprs.append(fpr)
    fnrs = np.array(fnrs); fprs = np.array(fprs)
    idx = np.argmin(np.abs(fnrs - fprs))
    return float((fnrs[idx] + fprs[idx]) / 2)


def visualize_pipeline(original, normalized, segmented, gabor, smqt_im,
                       stft_energy, bin_img, thin, minutiae):
    """Show 3x3 grid of processing stages."""
    steps = [original, normalized, segmented, gabor, smqt_im,
             stft_energy, bin_img, thin]
    titles = ['Original','Normalized','Segmented','Gabor','SMQT',
              'STFT Energy','Binarized','Thinned','Minutiae']
    plt.figure(figsize=(10,10))
    for i, img in enumerate(steps, 1):
        plt.subplot(3,3,i)
        plt.imshow(img, cmap='gray'); plt.title(titles[i-1]); plt.axis('off')
    plt.subplot(3,3,9)
    plt.imshow(thin, cmap='gray'); plt.title('Minutiae'); plt.axis('off')
    for mtype, (x, y) in minutiae:
        c = 'r' if mtype=='ending' else 'b'
        plt.plot(x, y, 'o', color=c, markersize=3)
    plt.tight_layout()
    plt.show()


def match_and_visualize(path1, path2):
    """Load two FVC images, process, visualize each pipeline, and print match score."""
    img1 = load_image(path1)
    img2 = load_image(path2)
    # Process image 1
    norm1 = normalize(img1)
    seg1 = segment(norm1)
    ori1 = compute_orientation_map(norm1)
    gab1 = gabor_enhance(norm1, ori1)
    sm1 = smqt(gab1)
    so1, sf1, se1 = stft_features(sm1)
    bin1, thin1 = binarize_thin(se1)
    minutiae1 = extract_minutiae(thin1)
    cyl1 = create_cylinders(minutiae1, (so1, sf1, se1))
    # Process image 2
    norm2 = normalize(img2)
    seg2 = segment(norm2)
    ori2 = compute_orientation_map(norm2)
    gab2 = gabor_enhance(norm2, ori2)
    sm2 = smqt(gab2)
    so2, sf2, se2 = stft_features(sm2)
    bin2, thin2 = binarize_thin(se2)
    minutiae2 = extract_minutiae(thin2)
    cyl2 = create_cylinders(minutiae2, (so2, sf2, se2))
    # Visualize both
    print(f"Matching '{path1}' vs '{path2}'")
    print(f"Score: {match(cyl1, cyl2):.4f}")
    print("--- Pipeline for Image 1 ---")
    visualize_pipeline(img1, norm1, seg1, gab1, sm1, se1, bin1, thin1, minutiae1)
    print("--- Pipeline for Image 2 ---")
    visualize_pipeline(img2, norm2, seg2, gab2, sm2, se2, bin2, thin2, minutiae2)


if __name__ == '__main__':
    img1_path = R"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002\FVC2002\DB1_A\1_1.tif"
    img2_path = R"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002\FVC2002\DB1_A\1_2.tif"
    match_and_visualize(img1_path, img2_path)
