import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk, binary_erosion, binary_dilation, skeletonize
from scipy.interpolate import RectBivariateSpline

def normalize_image(img, M0=0.5, V0=0.01):
    """Proper Hong et al. (1998) mean-variance normalization, output as float [0,1]."""
    img = img.astype(np.float32) / 255.0  # Start with [0,1] float
    M = np.mean(img)
    V = np.var(img)
    if V == 0:
        return img  # Avoid division by zero
    diff_sq = (img - M)**2
    sqrt_term = np.sqrt(V0 * diff_sq / V)
    norm_img = np.where(img > M, M0 + sqrt_term, M0 - sqrt_term)
    return np.clip(norm_img, 0.0, 1.0)  # Float [0,1]

def segment_image(img, block_size=16, threshold=0.1):
    """Blockwise variance segmentation (as in paper)."""
    h, w = img.shape
    mask = np.zeros_like(img, dtype=bool)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            if block.std() > threshold * img.std():  # Variance threshold
                mask[i:i+block_size, j:j+block_size] = True
    # Smooth mask with morphology
    mask = binary_dilation(binary_erosion(mask, disk(3)), disk(5))
    return mask

def stft_enhancement(img, window_size=16, overlap=8, ridge_freq=1/5.0):
    """STFT enhancement (Chikkerur et al., 2005) with MTCC feature generation, output float [0,1]."""
    h, w = img.shape
    enhanced = np.zeros_like(img, dtype=np.float32)
    weights = np.zeros_like(img, dtype=np.float32)
    hann = np.hanning(window_size)[:, None] * np.hanning(window_size)
    # Initialize MTCC feature arrays based on block grid
    orientation = np.zeros((h // overlap + 1, w // overlap + 1))
    frequency = np.zeros_like(orientation)
    energy = np.zeros_like(orientation)
    
    # For debug: store sample block data
    sample_block = None
    sample_f = None
    sample_f_enh = None
    
    for i in range(0, h - window_size + 1, overlap):
        for j in range(0, w - window_size + 1, overlap):
            block = img[i:i+window_size, j:j+window_size] * hann
            if block.shape != (window_size, window_size):
                continue
            f = fftshift(fft2(block))
            # Wider bandpass filter to retain ridge energy
            rows, cols = np.ogrid[-window_size//2:window_size//2, -window_size//2:window_size//2]
            dist = np.sqrt(rows**2 + cols**2)
            bandpass = np.exp(-((dist - ridge_freq*window_size)**2) / (2 * (2.0*ridge_freq*window_size)**2))  # Wider bandwidth
            f_enh = f * bandpass
            enh_block = np.real(ifft2(fftshift(f_enh)))
            # Normalize enh_block to [0,1] per block
            enh_block = (enh_block - np.min(enh_block)) / (np.max(enh_block) - np.min(enh_block) + 1e-10)
            enhanced[i:i+window_size, j:j+window_size] += enh_block * hann
            weights[i:i+window_size, j:j+window_size] += hann
            
            # MTCC feature extraction (aligned with block indices)
            i_block = i // overlap
            j_block = j // overlap
            p_r_theta = np.abs(f_enh)**2 / np.sum(np.abs(f_enh)**2 + 1e-10)
            sin2theta = np.sum(p_r_theta * np.sin(2 * np.arctan2(cols, rows)))
            cos2theta = np.sum(p_r_theta * np.cos(2 * np.arctan2(cols, rows)))
            orientation[i_block, j_block] = 0.5 * np.arctan2(sin2theta, cos2theta)
            p_r = np.sum(p_r_theta, axis=0)
            frequency[i_block, j_block] = np.sum(p_r * (dist / window_size))
            energy[i_block, j_block] = np.log(np.sum(np.abs(f_enh)**2) + 1e-10)
            
            # Debug sample
            if sample_block is None:
                sample_block = block
                sample_f = np.log1p(np.abs(f))
                sample_f_enh = np.log1p(np.abs(f_enh))
    
    # Final normalization
    enhanced /= np.where(weights > 0, weights, 1)
    enhanced = np.clip(enhanced, 0, 1)
    
    # Interpolate MTCC features to full size
    spl_o = RectBivariateSpline(np.arange(0, h, overlap), np.arange(0, w, overlap), orientation)
    spl_f = RectBivariateSpline(np.arange(0, h, overlap), np.arange(0, w, overlap), frequency)
    spl_e = RectBivariateSpline(np.arange(0, h, overlap), np.arange(0, w, overlap), energy)
    orientation_full = spl_o(np.arange(h), np.arange(w))
    frequency_full = spl_f(np.arange(h), np.arange(w))
    energy_full = spl_e(np.arange(h), np.arange(w))
    
    # Debug visualization
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_block, cmap='gray', vmin=0, vmax=1)
    plt.title('Sample Block')
    plt.subplot(1, 3, 2)
    plt.imshow(sample_f, cmap='jet')
    plt.title('FFT Magnitude (Log)')
    plt.subplot(1, 3, 3)
    plt.imshow(sample_f_enh, cmap='jet')
    plt.title('Filtered FFT Magnitude (Log)')
    plt.show()
    
    print(f"Sample enh_block max: {np.max(enh_block):.4e}, min: {np.min(enh_block):.4e}")
    return enhanced, orientation_full, frequency_full, energy_full

def gabor_filter(img, freq=1/5.0, sigma_x=4, sigma_y=4, theta=np.pi/4):
    """Gabor filtering for smoothing (adapted from GitHub)."""
    kernel_size = 11
    gabor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma_x, theta, 1/freq, sigma_y/sigma_x, 0, ktype=cv2.CV_32F)
    gabor_img = cv2.filter2D(img.astype(np.float32), cv2.CV_8UC3, gabor_kernel)
    return np.clip(gabor_img, 0, 255).astype(np.float32) / 255.0  # Output [0,1]

def smqt(img, levels=8):
    """SMQT implementation (from StackOverflow, adapted)."""
    img = img.astype(np.float32)
    for _ in range(levels):
        mean = np.mean(img)
        img = (img > mean).astype(np.float32) * (2**levels - 1) / 2 + img / 2
    return np.clip(img, 0, 255).astype(np.float32) / 255.0  # Output [0,1]

def extract_minutiae(img):
    """Basic minutiae extractor: Binarize, thin, CN (MCC-style)."""
    _, bin_img = cv2.threshold((img * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thinned = skeletonize(bin_img // 255) * 255
    minutiae = []
    h, w = thinned.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if thinned[i, j] == 255:
                cn = 0
                for di, dj in [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]:
                    cn += 1 if thinned[i+di, j+dj] == 255 else 0
                if cn == 1:  # Ending
                    minutiae.append((j, i, 0))  # (x, y, angle=0 placeholder)
                elif cn == 3:  # Bifurcation
                    minutiae.append((j, i, 0))
    return minutiae

def build_mcc(minutiae, img_shape, R=63, Ns=18, Nd=5):
    """MCC cylinders (Section V)."""
    cylinders = []
    for m in minutiae:
        x_m, y_m, theta_m = m
        cylinder = np.zeros((Ns, Ns, Nd))
        for i in range(Ns):
            for j in range(Ns):
                for k in range(Nd):
                    dx = (i - (Ns+1)/2) * (2*R/Ns)
                    dy = (j - (Ns+1)/2) * (2*R/Ns)
                    p_x = x_m + dx * np.cos(theta_m) - dy * np.sin(theta_m)
                    p_y = y_m + dx * np.sin(theta_m) + dy * np.cos(theta_m)
                    if 0 <= p_x < img_shape[1] and 0 <= p_y < img_shape[0]:
                        contrib = 1.0  # Simplified
                        cylinder[i, j, k] = contrib
        cylinders.append(cylinder)
    return cylinders

def build_mtcc(minutiae, orientation_img, frequency_img, energy_img, R=63, Ns=18, Nd=5):
    """MTCC features (Sets 1 & 2). Returns dict of variants."""
    mtcc = {'MCC_o': [], 'MCC_f': [], 'MCC_e': [], 'MCC_co': [], 'MCC_cf': [], 'MCC_ce': []}
    for m in minutiae:
        x_m, y_m, theta_m = m
        for variant in mtcc.keys():
            cylinder = np.zeros((Ns, Ns, Nd))
            for i in range(Ns):
                for j in range(Ns):
                    for k in range(Nd):
                        dx = (i - (Ns+1)/2) * (2*R/Ns)
                        dy = (j - (Ns+1)/2) * (2*R/Ns)
                        p_x = int(x_m + dx * np.cos(theta_m) - dy * np.sin(theta_m))
                        p_y = int(y_m + dx * np.sin(theta_m) + dy * np.cos(theta_m))
                        if 0 <= p_x < orientation_img.shape[1] and 0 <= p_y < orientation_img.shape[0]:
                            if 'o' in variant: feat = orientation_img[p_y, p_x]
                            elif 'f' in variant: feat = frequency_img[p_y, p_x]
                            elif 'e' in variant: feat = energy_img[p_y, p_x]
                            contrib = np.exp(-((feat - theta_m)**2) / (2 * (np.pi/3)**2))
                            cylinder[i, j, k] = contrib
            mtcc[variant].append(cylinder)
    return mtcc

def lssr_matcher(cyl_a, cyl_b, delta_theta=np.pi/6, min_cells=50):
    """LSS with Relaxation (simplest from MCC)."""
    sim_matrix = np.zeros((len(cyl_a), len(cyl_b)))
    for i, ca in enumerate(cyl_a):
        for j, cb in enumerate(cyl_b):
            if np.abs(np.mean(ca) - np.mean(cb)) < delta_theta:
                dist = np.linalg.norm(ca - cb) / (np.linalg.norm(ca) + np.linalg.norm(cb) + 1e-10)
                sim_matrix[i, j] = 1 - dist if np.sum(ca > 0) > min_cells else 0
    top_pairs = np.argsort(sim_matrix.flatten())[::-1][:10]
    final_score = np.mean(sim_matrix.flatten()[top_pairs]) * (1 - 0.1 * len(top_pairs))
    return final_score

# Pipeline with visualizations
img = cv2.imread('C:/Users/Precision/Onus/Data/FVC-DataSets/DataSets/FVC2002/Db1_a/1_1.tif', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.axis('off')
plt.show()

norm_img = normalize_image(img)
plt.figure(figsize=(5, 5))
plt.imshow(norm_img, cmap='gray', vmin=0, vmax=1)
plt.title('Normalized Image')
plt.axis('off')
plt.show()

seg_mask = segment_image(norm_img)
plt.figure(figsize=(5, 5))
plt.imshow(seg_mask, cmap='gray')
plt.title('Segmentation Mask')
plt.axis('off')
plt.show()

segmented_img = cv2.bitwise_and(norm_img, norm_img, mask=seg_mask.astype(np.uint8))
plt.figure(figsize=(5, 5))
plt.imshow(segmented_img, cmap='gray', vmin=0, vmax=1)
plt.title('Segmented Image')
plt.axis('off')
plt.show()

# STFT Enhancement and MTCC Feature Generation
stft_img, orientation_img, frequency_img, energy_img = stft_enhancement(segmented_img)
plt.figure(figsize=(5, 5))
plt.imshow(stft_img, cmap='gray', vmin=0, vmax=1)
plt.title('STFT Enhanced')
plt.axis('off')
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(orientation_img, cmap='hsv')
plt.title('Orientation')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(frequency_img, cmap='jet')
plt.title('Frequency')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(energy_img, cmap='gray')
plt.title('Energy')
plt.axis('off')
plt.show()

# Gabor Filtering
gabor_img = gabor_filter((stft_img * 255).astype(np.uint8)) / 255.0
plt.figure(figsize=(5, 5))
plt.imshow(gabor_img, cmap='gray', vmin=0, vmax=1)
plt.title('Gabor Smoothed')
plt.axis('off')
plt.show()

# SMQT
smqt_img = smqt((gabor_img * 255).astype(np.uint8)) / 255.0
plt.figure(figsize=(5, 5))
plt.imshow(smqt_img, cmap='gray', vmin=0, vmax=1)
plt.title('SMQT Enhanced (Final Enhancement)')
plt.axis('off')
plt.show()

# Minutiae Extraction
minutiae_list = extract_minutiae((smqt_img * 255).astype(np.uint8))
plt.figure(figsize=(5, 5))
plt.imshow(smqt_img, cmap='gray', vmin=0, vmax=1)
for x, y, _ in minutiae_list:
    plt.plot(x, y, 'ro', markersize=5)
plt.title('Extracted Minutiae')
plt.axis('off')
plt.show()

# MCC Cylinders
mcc_cylinders = build_mcc(minutiae_list, smqt_img.shape)
print(f"MCC Cylinders built: {len(mcc_cylinders)}")
if mcc_cylinders:
    plt.figure(figsize=(5, 5))
    plt.imshow(mcc_cylinders[0][:,:,0], cmap='gray')
    plt.title('Sample MCC Cylinder Slice')
    plt.axis('off')
    plt.show()

# MTCC Dict
mtcc_dict = build_mtcc(minutiae_list, orientation_img, frequency_img, energy_img)
print("MTCC Variants built.")
if mtcc_dict['MCC_co']:
    plt.figure(figsize=(5, 5))
    plt.imshow(mtcc_dict['MCC_co'][0][:,:,0], cmap='gray')
    plt.title('Sample MTCC (co) Cylinder Slice')
    plt.axis('off')
    plt.show()

# Matching
score = lssr_matcher(mtcc_dict['MCC_co'], mtcc_dict['MCC_co'])
print("Pipeline complete. Self-match score:", score)