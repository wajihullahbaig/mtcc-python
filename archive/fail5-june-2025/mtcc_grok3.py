import numpy as np
import cv2
from scipy import signal
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import uuid

def load_image(path: str) -> np.ndarray:
    """Load and preprocess fingerprint image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {path}")
    return img.astype(np.float32)

def normalize(img: np.ndarray) -> np.ndarray:
    """Normalize image to zero mean and unit variance."""
    mean = np.mean(img)
    var = np.var(img)
    if var == 0:
        return img
    return (img - mean) / np.sqrt(var) * 10 + 128  # Scale to reasonable range

def segment(img: np.ndarray, block_size: int = 16) -> np.ndarray:
    """Segment fingerprint using block-wise variance."""
    h, w = img.shape
    mask = np.zeros_like(img, dtype=np.uint8)
    
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = img[i:i+block_size, j:j+block_size]
            var = np.var(block)
            if var > 100:  # Threshold for foreground
                mask[i:i+block_size, j:j+block_size] = 255
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def gabor_enhance(img: np.ndarray, orientation_map: np.ndarray, freq: float = 0.1) -> np.ndarray:
    """Enhance image using Gabor filters tuned to orientation and frequency."""
    h, w = img.shape
    enhanced = np.zeros_like(img)
    
    for i in range(0, h, 16):
        for j in range(0, w, 16):
            theta = orientation_map[i, j]
            gabor_kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=4.0,
                theta=theta,
                lambd=1.0/freq,
                gamma=0.5
            )
            block = img[max(0, i-10):i+11, max(0, j-10):j+11]
            if block.shape == (21, 21):
                filtered = cv2.filter2D(block, -1, gabor_kernel)
                enhanced[max(0, i-10):i+11, max(0, j-10):j+11] = filtered
    
    return enhanced

def smqt(img: np.ndarray, levels: int = 8) -> np.ndarray:
    """Successive Mean Quantization Transform."""
    img = img.copy()
    h, w = img.shape
    output = np.zeros_like(img)
    
    for _ in range(levels):
        mean = np.mean(img)
        output = output * 2 + (img > mean).astype(np.uint8)
        img = np.where(img > mean, img - mean, img)
    
    return output * (255.0 / (2**levels - 1))

def stft_features(img: np.ndarray, window: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract orientation, frequency, and energy maps using STFT."""
    h, w = img.shape
    overlap = window // 2
    orientation = np.zeros_like(img)
    frequency = np.zeros_like(img)
    energy = np.zeros_like(img)
    
    window_func = signal.windows.hann(window)
    window_2d = np.outer(window_func, window_func)
    
    for i in range(0, h - window + 1, window - overlap):
        for j in range(0, w - window + 1, window - overlap):
            block = img[i:i+window, j:j+window]
            if block.shape != (window, window):
                continue
                
            block = block * window_2d
            f, _, Zxx = signal.stft(block.flatten(), nperseg=window)
            
            # Energy
            energy_block = np.log(np.sum(np.abs(Zxx)**2) + 1e-10)
            energy[i:i+window, j:j+window] = energy_block
            
            # Frequency (peak frequency)
            freq_idx = np.argmax(np.abs(Zxx), axis=0)
            freq_val = f[freq_idx].mean()
            frequency[i:i+window, j:j+window] = freq_val
            
            # Orientation (gradient-based)
            sobelx = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)
            theta = np.arctan2(sobely, sobelx) / 2
            orientation[i:i+window, j:j+window] = theta.mean()
    
    # Smooth orientation
    orientation = gaussian_filter(orientation, sigma=1)
    return orientation, frequency, energy

def binarize_thin(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Binarize and thin the image using adaptive thresholding and Zhang-Suen."""
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        img.astype(np.uint8), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    def zhang_suen(image):
        skeleton = image.copy()
        h, w = skeleton.shape
        changing = True
        
        while changing:
            changing = False
            to_white = []
            to_black = []
            
            # Step 1
            for i in range(1, h-1):
                for j in range(1, w-1):
                    if skeleton[i, j] == 255:
                        p = skeleton[i-1:i+2, j-1:j+2].flatten()[1:]  # Get 8 neighbors
                        if len(p) != 8:  # Ensure correct number of neighbors
                            continue
                        p2, p3, p4, p5, p6, p7, p8, p9 = p
                        m1 = (p2 * p3 * p4)
                        m2 = (p4 * p5 * p6)
                        a = sum(p) / 255
                        b = sum(p[k] * p[k+1] for k in range(7)) / 255 + p[7] * p[0] / 255
                        
                        if 2 <= a <= 6 and b == 1 and m1 == 0 and m2 == 0:
                            to_white.append((i, j))
            
            for i, j in to_white:
                skeleton[i, j] = 0
                changing = True
                
            to_white = []
            # Step 2
            for i in range(1, h-1):
                for j in range(1, w-1):
                    if skeleton[i, j] == 255:
                        p = skeleton[i-1:i+2, j-1:j+2].flatten()[1:]  # Get 8 neighbors
                        if len(p) != 8:
                            continue
                        p2, p3, p4, p5, p6, p7, p8, p9 = p
                        m1 = (p2 * p3 * p6)
                        m2 = (p6 * p7 * p8)
                        a = sum(p) / 255
                        b = sum(p[k] * p[k+1] for k in range(7)) / 255 + p[7] * p[0] / 255
                        
                        if 2 <= a <= 6 and b == 1 and m1 == 0 and m2 == 0:
                            to_white.append((i, j))
            
            for i, j in to_white:
                skeleton[i, j] = 0
                changing = True
                    
        return skeleton
    
    skeleton = zhang_suen(binary)
    return binary, skeleton

def extract_minutiae(skeleton: np.ndarray) -> List[Dict]:
    """Extract minutiae using Crossing Number algorithm."""
    h, w = skeleton.shape
    minutiae = []
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            if skeleton[i, j] == 255:
                neighbors = skeleton[i-1:i+2, j-1:j+2].flatten()[1:]
                cn = sum(abs(neighbors[k] - neighbors[k+1]) for k in range(7)) // 2 + abs(neighbors[7] - neighbors[0]) // 2
                
                if cn == 1 or cn == 3:  # Termination or bifurcation
                    orientation = np.arctan2(
                        skeleton[i-1:i+2, j-1:j+2][:, 1].sum(),
                        skeleton[i-1:i+2, j-1:j+2][1, :].sum()
                    ) / 2
                    minutiae.append({
                        'x': j,
                        'y': i,
                        'theta': orientation,
                        'type': 'termination' if cn == 1 else 'bifurcation'
                    })
    
    return minutiae

def create_cylinders(minutiae: List[Dict], texture_maps: Tuple[np.ndarray, np.ndarray, np.ndarray], radius: int = 70) -> List[np.ndarray]:
    """Create MTCC descriptors using texture features."""
    orientation_map, frequency_map, energy_map = texture_maps
    NS, ND = 18, 5
    delta_s = 2 * radius / NS
    delta_d = 2 * np.pi / ND
    sigma_s = delta_s * 0.5
    cylinders = []
    
    for m in minutiae:
        cylinder = np.zeros((NS, NS, ND))
        x_m, y_m, theta_m = m['x'], m['y'], m['theta']
        
        for i in range(NS):
            for j in range(NS):
                # Cell center
                cell_x = x_m + delta_s * (i - (NS + 1) / 2) * np.cos(theta_m) + delta_s * (j - (NS + 1) / 2) * np.sin(theta_m)
                cell_y = y_m - delta_s * (i - (NS + 1) / 2) * np.sin(theta_m) + delta_s * (j - (NS + 1) / 2) * np.cos(theta_m)
                
                if not (0 <= cell_x < orientation_map.shape[1] and 0 <= cell_y < orientation_map.shape[0]):
                    continue
                
                for k in range(ND):
                    d_k = -np.pi + (k - 0.5) * delta_d
                    
                    # Spatial contribution
                    spatial = np.exp(-((i - NS/2)**2 + (j - NS/2)**2) / (2 * sigma_s**2))
                    
                    # Texture contributions
                    cell_o = orientation_map[int(cell_y), int(cell_x)]
                    cell_f = frequency_map[int(cell_y), int(cell_x)]
                    cell_e = energy_map[int(cell_y), int(cell_x)]
                    
                    o_diff = np.abs((cell_o - d_k + np.pi) % (2 * np.pi) - np.pi)
                    f_diff = np.abs((cell_f - d_k + np.pi) % (2 * np.pi) - np.pi)
                    e_diff = np.abs((cell_e - d_k + np.pi) % (2 * np.pi) - np.pi)
                    
                    contrib_o = np.exp(-o_diff**2 / (2 * 0.5**2))
                    contrib_f = np.exp(-f_diff**2 / (2 * 0.5**2))
                    contrib_e = np.exp(-e_diff**2 / (2 * 0.5**2))
                    
                    cylinder[i, j, k] = spatial * (contrib_o + contrib_f + contrib_e) / 3
        
        cylinders.append(cylinder)
    
    return cylinders

def match(cylinders1: List[np.ndarray], cylinders2: List[np.ndarray]) -> float:
    """Match two sets of cylinders using LSS-R."""
    LSM = np.zeros((len(cylinders1), len(cylinders2)))
    
    for i, c1 in enumerate(cylinders1):
        for j, c2 in enumerate(cylinders2):
            valid_cells = (c1 > 0) & (c2 > 0)
            if np.sum(valid_cells) > 10:  # Minimum valid cells
                diff = c1[valid_cells] - c2[valid_cells]
                LSM[i, j] = 1 - np.linalg.norm(diff) / (np.linalg.norm(c1[valid_cells]) + np.linalg.norm(c2[valid_cells]))
    
    # Local Similarity Sort with Relaxation
    scores = np.sort(LSM.flatten())[::-1][:min(10, LSM.size)]
    return np.mean(scores) if scores.size > 0 else 0.0

def calculate_eer(genuine_scores: List[float], impostor_scores: List[float]) -> float:
    """Calculate Equal Error Rate."""
    thresholds = np.linspace(0, 1, 100)
    frr = []
    far = []
    
    for t in thresholds:
        frr.append(np.mean(np.array(genuine_scores) < t))
        far.append(np.mean(np.array(impostor_scores) > t))
    
    frr = np.array(frr)
    far = np.array(far)
    idx = np.argmin(np.abs(frr - far))
    return (frr[idx] + far[idx]) / 2

def visualize_pipeline(original: np.ndarray, *steps: np.ndarray) -> None:
    """Visualize processing pipeline in 3x3 grid."""
    titles = ['Original', 'Normalized', 'Segmented', 'Gabor Enhanced', 'SMQT', 'STFT Energy', 'Binarized', 'Thinned', 'Minutiae']
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i, (img, title) in enumerate(zip([original] + list(steps), titles)):
        ax = axes[i // 3, i % 3]
        if i == 8:  # Minutiae visualization
            ax.imshow(original, cmap='gray')
            minutiae = extract_minutiae(steps[-1])
            for m in minutiae:
                ax.plot(m['x'], m['y'], 'ro', markersize=5)
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example test harness
def test_pipeline(image_path: str):
    img = load_image(image_path)
    norm = normalize(img)
    mask = segment(norm)
    ori, freq, energy = stft_features(norm)
    gabor = gabor_enhance(norm, ori, freq.mean())
    smqt_img = smqt(gabor)
    ori, freq, energy = stft_features(smqt_img)
    binary, thin = binarize_thin(smqt_img)
    minutiae = extract_minutiae(thin)
    cylinders = create_cylinders(minutiae, (ori, freq, energy))
    
    visualize_pipeline(img, norm, mask, gabor, smqt_img, energy, binary, thin, thin)
    return cylinders

if __name__ == "__main__":
    img1_path = R'C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2000\FVC2000\Db1_a\1_1.tif'
    cylinders = test_pipeline(img1_path)