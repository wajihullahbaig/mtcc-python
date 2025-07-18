import numpy as np
from scipy import ndimage
from scipy.signal import stft
from scipy.ndimage import gaussian_filter
import cv2  # For image loading and basic processing
from skimage.filters import gabor_kernel  # For Gabor filtering
from skimage import morphology  # For morphological operations
import matplotlib.pyplot as plt  # For visualization
import os  # For handling file paths
from scipy.interpolate import griddata  # For interpolation if needed

# Constants from the MTCC paper and documents
R = 63  # Cylinder radius
NS = 18  # Number of sections
ND = 5   # Number of directions
SIGMA_S = 6  # Spatial Gaussian sigma
SIGMA_D = np.pi / 6  # Directional sigma (approx)
MIN_NEIGHBORS = 3  # Min neighbors for valid cylinder
MIN_VALID_CELLS = 0.5  # Percentage of valid cells
DELTA_THETA = np.pi / 6  # Global rotation angle for matching
MIN_MC = 0.3  # Min matchable cells

# Helper functions

def load_image(image_path):
    """Load grayscale image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    return img.astype(np.float32) / 255.0  # Normalize to [0,1]

def segmentation(img, block_size=16, threshold=0.05):
    """Blockwise variance-based segmentation."""
    h, w = img.shape
    mask = np.zeros_like(img, dtype=bool)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            if block.var() > threshold:
                mask[i:i+block_size, j:j+block_size] = True
    mask = morphology.remove_small_holes(mask, area_threshold=block_size**2)
    mask = morphology.remove_small_objects(mask, min_size=block_size**2)
    return mask

def polar_fft(fft_mag, max_r, num_theta=180):
    """Convert FFT magnitude to polar coordinates."""
    h, w = fft_mag.shape
    center = (h//2, w//2)
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    theta = np.arctan2(y - center[0], x - center[1])
    
    p_r_theta = np.zeros((max_r, num_theta))
    counts = np.zeros((max_r, num_theta))
    
    theta_idx = ((theta + np.pi) / (2 * np.pi) * num_theta).astype(int) % num_theta
    
    valid = r < max_r
    np.add.at(p_r_theta, (r[valid], theta_idx[valid]), fft_mag[valid]**2)
    np.add.at(counts, (r[valid], theta_idx[valid]), 1)
    
    p_r_theta /= np.maximum(counts, 1)
    return p_r_theta

def stft_enhancement(img, mask, block_size=16, overlap=8, sigma=1.0):
    """STFT enhancement with integrated feature computation."""
    h, w = img.shape
    enhanced = np.zeros_like(img)
    orientation = np.zeros_like(img)
    frequency = np.zeros_like(img)
    energy = np.zeros_like(img)
    count = np.zeros_like(img)
    
    window = np.outer(np.hanning(block_size), np.hanning(block_size))
    max_r = block_size // 2
    num_theta = 180
    
    for i in range(0, h - block_size + 1, block_size - overlap):
        for j in range(0, w - block_size + 1, block_size - overlap):
            block = img[i:i+block_size, j:j+block_size] * window
            if np.sum(mask[i:i+block_size, j:j+block_size]) < (block_size**2 / 2):
                continue
            
            fft_block = np.fft.fftshift(np.fft.fft2(block))
            abs_fft = np.abs(fft_block)
            
            if np.all(abs_fft == 0):
                continue
            
            # Energy
            en = np.log(np.sum(abs_fft**2) + 1e-6)
            
            # Polar transform
            p_r_theta = polar_fft(abs_fft, max_r, num_theta)
            
            # Orientation from p_theta
            p_theta = np.sum(p_r_theta, axis=0)
            theta_vals = np.linspace(0, np.pi, num_theta, endpoint=False)  # 0 to pi for orientation
            sin_2theta = np.sum(p_theta * np.sin(2 * theta_vals))
            cos_2theta = np.sum(p_theta * np.cos(2 * theta_vals))
            ori = 0.5 * np.arctan2(sin_2theta, cos_2theta)
            if ori < 0:
                ori += np.pi
            
            # Frequency from p_r
            p_r = np.sum(p_r_theta, axis=1)
            r_vals = np.arange(max_r)
            avg_r = np.sum(p_r * r_vals) / np.sum(p_r) if np.sum(p_r) > 0 else 0
            freq = avg_r / block_size  # Normalized frequency
            
            # Assign to maps
            slc = slice(i, i+block_size), slice(j, j+block_size)
            orientation[slc] += ori
            frequency[slc] += freq
            energy[slc] += en
            count[slc] += 1
            
            # Enhance block
            enhanced_block = np.real(np.fft.ifft2(np.fft.ifftshift(fft_block * gaussian_filter(abs_fft, sigma))))
            enhanced[slc] += enhanced_block * window  # Weight by window for proper overlap
    
    # Average maps where overlapped
    valid = count > 0
    orientation[valid] /= count[valid]
    frequency[valid] /= count[valid]
    energy[valid] /= count[valid]
    
    # Smooth maps
    orientation = gaussian_filter(orientation, sigma=1)
    frequency = gaussian_filter(frequency, sigma=1)
    energy = gaussian_filter(energy, sigma=1)
    
    enhanced /= np.maximum(enhanced.max(), 1e-6)
    return np.clip(enhanced, 0, 1), orientation, frequency, energy

def gabor_enhancement(img, orientation_img, frequency_img, kernel_size=11):
    """Gabor filtering for smoothing."""
    enhanced = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            theta = orientation_img[i, j]
            freq = frequency_img[i, j]
            if freq == 0:
                continue
            kernel = np.real(gabor_kernel(freq, theta, sigma_x=4, sigma_y=4))
            kh, kw = kernel.shape
            patch = img[max(0, i-kh//2):min(img.shape[0], i+kh//2+1),
                        max(0, j-kw//2):min(img.shape[1], j+kw//2+1)]
            if patch.shape != kernel.shape:
                continue
            enhanced[i, j] = np.sum(patch * kernel)
    return enhanced

def smqt(img, levels=8):
    """Successive Mean Quantization Transform."""
    img = img.copy()
    for _ in range(levels):
        mean = np.mean(img)
        img = np.where(img > mean, 1, 0) + img / 2
    return img

def extract_minutiae(img):
    """Placeholder for minutiae extraction. Returns list of (x, y, theta)."""
    binary = cv2.adaptiveThreshold((img * 255).astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thinned = morphology.thin(binary // 255)
    
    minutiae = []
    for i in range(1, thinned.shape[0]-1):
        for j in range(1, thinned.shape[1]-1):
            if thinned[i, j]:
                neighbors = np.sum(thinned[i-1:i+2, j-1:j+2]) - 1
                if neighbors == 1 or neighbors == 3:  # Ending or bifurcation
                    theta = 0.0  # Theta placeholder
                    minutiae.append((j, i, theta))  # (x, y, theta)
    return minutiae[:50]  # Limit for demo

def build_cylinder(minutia, all_minutiae, texture_imgs=None, feature_type='MCC_o'):
    """Build MCC or MTCC cylinder for a minutia."""
    x_m, y_m, theta_m = minutia
    cylinder = np.zeros((NS, NS, ND))
    valid_mask = np.ones((NS, NS, ND), dtype=bool)
    
    for i in range(NS):
        for j in range(NS):
            for k in range(ND):
                dx = (i - (NS + 1)/2) * (2 * R / NS)
                dy = (j - (NS + 1)/2) * (2 * R / NS)
                p_ij = np.array([x_m + dx * np.cos(theta_m) - dy * np.sin(theta_m),
                                 y_m + dx * np.sin(theta_m) + dy * np.cos(theta_m)])
                
                neighbors = [m for m in all_minutiae if m != minutia and np.linalg.norm([m[0]-p_ij[0], m[1]-p_ij[1]]) <= 3*SIGMA_S]
                
                if not neighbors:
                    valid_mask[i,j,k] = False
                    continue
                
                contrib = 0
                for neigh in neighbors:
                    spatial_contrib = np.exp(-np.linalg.norm([neigh[0]-p_ij[0], neigh[1]-p_ij[1]])**2 / (2*SIGMA_S**2))
                    
                    if feature_type == 'MCC_o':
                        dir_diff = np.abs(neigh[2] - theta_m)
                        dir_contrib = np.exp(-dir_diff**2 / (2 * SIGMA_D**2))
                    elif feature_type == 'MCC_f':
                        f_m = texture_imgs[1][int(y_m), int(x_m)] if texture_imgs else 0
                        f_n = texture_imgs[1][int(neigh[1]), int(neigh[0])] if texture_imgs else 0
                        dir_contrib = np.exp(-(f_m - f_n)**2 / (2 * SIGMA_D**2))
                    elif feature_type == 'MCC_e':
                        e_m = texture_imgs[2][int(y_m), int(x_m)] if texture_imgs else 0
                        e_n = texture_imgs[2][int(neigh[1]), int(neigh[0])] if texture_imgs else 0
                        dir_contrib = np.exp(-(e_m - e_n)**2 / (2 * SIGMA_D**2))
                    elif feature_type == 'MCC_co':
                        o_p = texture_imgs[0][int(p_ij[1]), int(p_ij[0])] if texture_imgs else 0
                        dir_contrib = np.exp(-(theta_m - o_p)**2 / (2 * SIGMA_D**2))
                    elif feature_type == 'MCC_cf':
                        f_p = texture_imgs[1][int(p_ij[1]), int(p_ij[0])] if texture_imgs else 0
                        f_m = texture_imgs[1][int(y_m), int(x_m)] if texture_imgs else 0
                        dir_contrib = np.exp(-(f_m - f_p)**2 / (2 * SIGMA_D**2))
                    elif feature_type == 'MCC_ce':
                        e_p = texture_imgs[2][int(p_ij[1]), int(p_ij[0])] if texture_imgs else 0
                        e_m = texture_imgs[2][int(y_m), int(x_m)] if texture_imgs else 0
                        dir_contrib = np.exp(-(e_m - e_p)**2 / (2 * SIGMA_D**2))
                    else:
                        raise ValueError(f"Unknown feature_type: {feature_type}")
                    
                    contrib += spatial_contrib * dir_contrib
                
                cylinder[i,j,k] = 1 / (1 + np.exp(-contrib + 0.5))  # Sigmoid approx Psi
    
    return cylinder, valid_mask

def build_descriptors(minutiae, texture_imgs, feature_type='MCC_o'):
    """Build list of cylinders for all minutiae."""
    descriptors = []
    for m in minutiae:
        cyl, mask = build_cylinder(m, minutiae, texture_imgs, feature_type)
        if np.sum(mask) / (NS*NS*ND) >= MIN_VALID_CELLS and len([n for n in minutiae if n != m and np.linalg.norm([n[0]-m[0], n[1]-m[1]]) <= R]) >= MIN_NEIGHBORS:
            descriptors.append((cyl, mask, m))
    return descriptors

def match_descriptors(desc_a, desc_b, distance_metric='euclidean'):
    """LSSR matching (simplified)."""
    scores = []
    for cyl_a, mask_a, m_a in desc_a:
        for cyl_b, mask_b, m_b in desc_b:
            valid_cells = np.logical_and(mask_a, mask_b)
            if np.sum(valid_cells) / (NS*NS*ND) < MIN_MC:
                continue
            diff = cyl_a - cyl_b
            if distance_metric == 'euclidean':
                norm_a = np.linalg.norm(cyl_a[valid_cells])
                norm_b = np.linalg.norm(cyl_b[valid_cells])
                if norm_a + norm_b == 0:
                    score = 0
                else:
                    score = 1 - np.linalg.norm(diff[valid_cells]) / (norm_a + norm_b)
            elif distance_metric == 'cosine':
                flat_a = cyl_a[valid_cells].flatten()
                flat_b = cyl_b[valid_cells].flatten()
                norm_a = np.linalg.norm(flat_a)
                norm_b = np.linalg.norm(flat_b)
                if norm_a * norm_b == 0:
                    score = 0
                else:
                    score = np.dot(flat_a, flat_b) / (norm_a * norm_b)
            elif distance_metric == 'sine':
                # Double angle sine distance (placeholder)
                score = np.mean(np.sin(2 * (cyl_a[valid_cells] - cyl_b[valid_cells])))
            scores.append((score, m_a, m_b))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    top_scores = scores[:min(10, len(scores))]  # Simplified nr=10
    final_score = np.mean([s[0] for s in top_scores[:5]]) if top_scores else 0  # np=5
    return final_score

def visualize_step(img, title, vis_flag):
    """Visualize image if flag is True."""
    if vis_flag:
        plt.figure(figsize=(5,5))
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

def process_fingerprint(image_path, feature_type='MCC_o', vis_flag=False):
    """End-to-end processing for one image with visualization flag."""
    img = load_image(image_path)
    visualize_step(img, 'Original Image', vis_flag)
    
    mask = segmentation(img)
    visualize_step(mask.astype(float), 'Segmentation Mask', vis_flag)
    
    enhanced_stft, o_img, f_img, e_img = stft_enhancement(img, mask)
    visualize_step(enhanced_stft, 'STFT Enhanced', vis_flag)
    visualize_step(o_img, 'Orientation Image', vis_flag)
    visualize_step(f_img, 'Frequency Image', vis_flag)
    visualize_step(e_img, 'Energy Image', vis_flag)
    
    enhanced_gabor = gabor_enhancement(enhanced_stft, o_img, f_img)
    visualize_step(enhanced_gabor, 'Gabor Enhanced', vis_flag)
    
    enhanced_smqt = smqt(enhanced_gabor)
    visualize_step(enhanced_smqt, 'SMQT Enhanced', vis_flag)
    
    texture_imgs = (o_img, f_img, e_img)
    minutiae = extract_minutiae(enhanced_smqt)
    descriptors = build_descriptors(minutiae, texture_imgs, feature_type)
    return descriptors

def compute_eer(scores_genuine, scores_impostor):
    """Simple EER calculation."""
    thresholds = np.linspace(0, 1, 100)
    frr = [np.sum(scores_genuine < t) / len(scores_genuine) for t in thresholds]
    far = [np.sum(scores_impostor > t) / len(scores_impostor) for t in thresholds]
    eer_idx = np.argmin(np.abs(np.array(frr) - np.array(far)))
    return (frr[eer_idx] + far[eer_idx]) / 2

def test_on_fvc_dataset(dataset_dir, feature_type='MCC_o', vis_flag=False):
    """Test on FVC-like dataset."""
    images = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.tif') or f.endswith('.png')]
    descriptors = {img: process_fingerprint(img, feature_type, vis_flag) for img in images}
    
    genuine_scores = []
    impostor_scores = []
    
    subjects = {}
    for img in images:
        base = os.path.basename(img)
        subject = base.split('_')[0]
        if subject not in subjects:
            subjects[subject] = []
        subjects[subject].append(img)
    
    for subj, imgs in subjects.items():
        for i in range(len(imgs)):
            for j in range(i+1, len(imgs)):
                score = match_descriptors(descriptors[imgs[i]], descriptors[imgs[j]])
                genuine_scores.append(score)
    
    subject_keys = list(subjects.keys())
    for idx1 in range(len(subject_keys)-1):
        s1 = subject_keys[idx1]
        for idx2 in range(idx1+1, len(subject_keys)):
            s2 = subject_keys[idx2]
            score = match_descriptors(descriptors[subjects[s1][0]], descriptors[subjects[s2][0]])
            impostor_scores.append(score)
    
    eer = compute_eer(np.array(genuine_scores), np.array(impostor_scores))
    print(f"EER for {feature_type}: {eer * 100:.2f}%")
    return eer

# Example usage:
test_on_fvc_dataset("C:/Users/Precision/Onus/Data/FVC-DataSets/DataSets/FVC2002/Db1_a/", 'MCC_o',True)
