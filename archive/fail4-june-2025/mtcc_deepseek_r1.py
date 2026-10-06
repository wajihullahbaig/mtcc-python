import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize
import cv2
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter

### 1. Fingerprint Image Loading ###
def load_image(path):
    """Load grayscale fingerprint image"""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

### 2. Image Normalization ###
def normalize_image(img, M0=100, V0=100):
    """Normalize image to target mean (M0) and variance (V0)"""
    mean, var = np.mean(img), np.var(img)
    std = max(np.sqrt(var), 1)  # Avoid division by zero
    normalized = np.zeros_like(img, dtype=np.float32)
    mask = img > mean
    normalized[mask] = M0 + np.sqrt((V0 * (img[mask] - mean)**2) / var)
    normalized[~mask] = M0 - np.sqrt((V0 * (img[~mask] - mean)**2) / var)
    return np.clip(normalized, 0, 255).astype(np.uint8)

### 3. Segmentation (Variance-based) ###
def segment_image(img, block_size=16, threshold=0.1):
    """Segment fingerprint using block-wise variance"""
    h, w = img.shape
    mask = np.zeros((h, w))
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            if block.size > 0:
                var = np.var(block)
                mask[i:i+block_size, j:j+block_size] = var > threshold * np.max(img)
    return mask.astype(np.uint8)

### 4. Gabor Filter Enhancement ###
def gabor_filter(img, orientation, frequency):
    """Apply Gabor filter using orientation/frequency maps"""
    enhanced = np.zeros_like(img, dtype=np.float32)
    h, w = img.shape
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            theta = orientation[i, j]
            freq_val = frequency[i, j]
            
            # Avoid division by zero for frequency
            lambd = 1.0 / max(freq_val, 0.01)  # Wavelength (1/frequency)
            
            # Create Gabor kernel with correct parameters
            kernel = cv2.getGaborKernel(
                ksize=(8, 8), 
                sigma=5, 
                theta=theta, 
                lambd=lambd,
                gamma=0.5,  # Aspect ratio
                psi=0,      # Phase offset
                ktype=cv2.CV_32F
            )
            
            # Normalize kernel to prevent scaling
            kernel /= np.sum(np.abs(kernel))
            
            # Apply to current block
            block = img[i:i+8, j:j+8].astype(np.float32)
            filtered_block = cv2.filter2D(block, -1, kernel)
            enhanced[i:i+8, j:j+8] = filtered_block
    
    # Normalize final enhanced image
    return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

### 5. SMQT Enhancement ###
def smqt(img, levels=8):
    """Successive Mean Quantization Transform"""
    output = np.zeros_like(img)
    for level in range(levels):
        mean = np.mean(img)
        output[img >= mean] += 2**(levels - level - 1)
        img = img - mean
    return output.astype(np.uint8)

### 6. STFT Analysis for Texture Features ###
def stft_analysis(img, window_size=14, overlap=6):
    """Compute orientation, frequency, and energy maps via STFT"""
    h, w = img.shape
    orient_map = np.zeros((h, w))
    freq_map = np.zeros((h, w))
    energy_map = np.zeros((h, w))
    
    # Window function (raised cosine)
    window = np.outer(
        np.hanning(window_size),
        np.hanning(window_size)
    )
    
    for i in range(0, h - window_size, overlap):
        for j in range(0, w - window_size, overlap):
            block = img[i:i+window_size, j:j+window_size] * window
            fft = np.fft.fft2(block)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Find dominant frequency/orientation
            max_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
            max_val = magnitude[max_idx]
            
            # Calculate orientation and frequency
            dy = max_idx[0] - window_size//2
            dx = max_idx[1] - window_size//2
            angle = np.arctan2(dy, dx) if dx != 0 else np.pi/2
            freq = np.sqrt(dx**2 + dy**2) / window_size
            
            # Update maps
            orient_map[i:i+overlap, j:j+overlap] = angle
            freq_map[i:i+overlap, j:j+overlap] = freq
            energy_map[i:i+overlap, j:j+overlap] = np.log(np.sum(magnitude**2) + 1e-5)
    
    return orient_map, freq_map, energy_map

### 7. Minutiae Extraction (Simplified) ###
def extract_minutiae(img, mask):
    """Extract minutiae using crossing number method (simplified)"""
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    skeleton = skeletonize(binary).astype(np.uint8) 
    minutiae = []
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    for i in range(1, skeleton.shape[0]-1):
        for j in range(1, skeleton.shape[1]-1):
            if skeleton[i, j] == 0 or mask[i, j] == 0: 
                continue
            neighbors = skeleton[i-1:i+2, j-1:j+2]
            conv = signal.convolve2d(neighbors, kernel, boundary='symm', mode='valid')
            cn = (conv[0,0] - 10) // 10  # Crossing number
            if cn in [1, 3]:  # Termination or bifurcation
                minutiae.append((j, i, np.random.uniform(0, 2*np.pi)))  # x,y,angle
    return minutiae


def calculate_eer(far, frr):
    """Calculate Equal Error Rate"""
    return np.abs(far - frr).min() / 2  # Simplified

### Visualization ###
def visualize_pipeline(img, normalized, mask, enhanced, orient, freq, energy):
    """Plot all intermediate steps in a single figure"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,10))
    titles = ['Original', 'Normalized', 'Mask', 'Enhanced', 'Orientation', 'Frequency', 'Energy']
    images = [img, normalized, mask, enhanced, orient, freq, energy]
    
    for i in range(7):
        plt.subplot(2, 4, i+1)
        plt.imshow(images[i], cmap='gray' if i < 4 else 'viridis')
        plt.title(titles[i])
    plt.tight_layout()
    plt.show()

### Main Pipeline ###
def process_image(path):
    # 1. Load image
    img = load_image(path)
    
    # 2. Normalization
    normalized = normalize_image(img)
    
    # 3. Segmentation
    mask = segment_image(normalized)
    
    # 4. Enhancement
    orient, freq, energy = stft_analysis(normalized)
    gabor_enhanced = gabor_filter(normalized, orient, freq)
    smqt_enhanced = smqt(gabor_enhanced)
    
    # 5. Minutiae extraction
    minutiae = extract_minutiae(smqt_enhanced, mask)
    
    # 6. Cylinder creation (for first minutia)
    if minutiae:
        cylinder = create_cylinder(minutiae[0], minutiae, orient, freq)
    
    # Visualization
    visualize_pipeline(img, normalized, mask, smqt_enhanced, orient, freq, energy)
    
    return minutiae, cylinder



def create_cylinder(minutia, minutiae_list, orient_map, freq_map, energy_map, 
                   R=65, Ns=18, Nd=5, sigma_s=6, sigma_d=np.pi*5/36,
                   mu_psi=0.005, tau_psi=400, feature_type='MCC_co'):
    """
    Create Minutia Texture Cylinder Code (MTCC) for a central minutia
    
    Args:
        minutia: (x, y, theta) of central minutia
        minutiae_list: List of all minutiae in the fingerprint
        orient_map: Orientation field map (2D array)
        freq_map: Frequency field map (2D array)
        energy_map: Energy field map (2D array)
        R: Cylinder radius (default=65)
        Ns: Spatial divisions (default=18)
        Nd: Angular divisions (default=5)
        sigma_s: Spatial Gaussian standard deviation (default=6)
        sigma_d: Angular Gaussian standard deviation (default=5π/36)
        mu_psi: Sigmoid mean parameter (default=0.005)
        tau_psi: Sigmoid scale parameter (default=400)
        feature_type: Type of texture feature to use ('MCC_f', 'MCC_e', 'MCC_co', etc.)
    
    Returns:
        3D cylinder array (Ns x Ns x Nd)
    """
    def d_theta(angle1, angle2):
        """Normalized angular difference (equation 8)"""
        diff = angle1 - angle2
        if diff < -np.pi:
            return diff + 2*np.pi
        elif diff >= np.pi:
            return diff - 2*np.pi
        return diff

    # Unpack central minutia
    x0, y0, theta0 = minutia
    h, w = orient_map.shape
    
    # Pre-calculate neighbor positions for efficiency
    neighbor_pos = np.array([(m[0], m[1]) for m in minutiae_list])
    
    # Initialize cylinder
    cylinder = np.zeros((Ns, Ns, Nd))
    Delta_s = 2 * R / Ns
    center_idx = (Ns - 1) / 2.0
    
    # Rotation matrix for minutia orientation
    rot_mat = np.array([
        [np.cos(theta0), np.sin(theta0)],
        [-np.sin(theta0), np.cos(theta0)]
    ])
    
    # Build cylinder
    for i in range(Ns):
        for j in range(Ns):
            # Calculate cell center position (equation 2)
            offset = np.array([i - center_idx, j - center_idx]) * Delta_s
            rotated_offset = rot_mat @ offset
            px, py = x0 + rotated_offset[0], y0 + rotated_offset[1]
            
            # Skip cells outside image boundaries
            if not (0 <= px < w and 0 <= py < h):
                continue
                
            # Get texture features at cell center
            orient_val = orient_map[int(py), int(px)]
            freq_val = freq_map[int(py), int(px)]
            energy_val = energy_map[int(py), int(px)]
            
            # Find neighbors within 3*sigma_s radius
            distances = cdist([(px, py)], neighbor_pos)[0]
            neighbors = [m for idx, m in enumerate(minutiae_list) 
                         if distances[idx] < 3*sigma_s and m != minutia]
            
            total_contrib = 0
            for neighbor in neighbors:
                nx, ny, ntheta = neighbor
                
                # Spatial contribution (equation 6)
                dist = np.sqrt((nx - px)**2 + (ny - py)**2)
                C_s = np.exp(-dist**2 / (2 * sigma_s**2))
                
                # Directional contribution based on feature type
                if feature_type == 'MCC_f':  # Local frequency
                    diff = d_theta(theta0, freq_val)
                    C_d = np.exp(-diff**2 / (2 * sigma_d**2))
                elif feature_type == 'MCC_e':  # Local energy
                    diff = d_theta(theta0, energy_val)
                    C_d = np.exp(-diff**2 / (2 * sigma_d**2))
                elif feature_type == 'MCC_co':  # Cell-centered orientation
                    diff = d_theta(ntheta, orient_val)
                    C_d = np.exp(-diff**2 / (2 * sigma_d**2))
                elif feature_type == 'MCC_cf':  # Cell-centered frequency
                    diff = d_theta(ntheta, freq_val)
                    C_d = np.exp(-diff**2 / (2 * sigma_d**2))
                elif feature_type == 'MCC_ce':  # Cell-centered energy
                    diff = d_theta(ntheta, energy_val)
                    C_d = np.exp(-diff**2 / (2 * sigma_d**2))
                else:  # Default: original MCC angular feature
                    diff = d_theta(ntheta, theta0)
                    C_d = np.exp(-diff**2 / (2 * sigma_d**2))
                
                # Accumulate contributions (equation 3)
                total_contrib += C_s * C_d
            
            # Apply sigmoid function (equation 4)
            psi = 1 / (1 + np.exp(-tau_psi * (total_contrib - mu_psi)))
            
            # Angular component (equation 1)
            for k in range(Nd):
                d_phi_k = -np.pi + (k + 0.5) * (2*np.pi/Nd)
                # For simplicity, we'll store the same value for all angular bins
                cylinder[i, j, k] = psi
                
    return cylinder

def match_templates(template1, template2, min_valid_cells=0.2, delta_theta=np.pi*2/3):
    """
    Match two fingerprint templates using their MTCC cylinders
    
    Args:
        template1: List of cylinders for first fingerprint
        template2: List of cylinders for second fingerprint
        min_valid_cells: Minimum fraction of valid cells for matching
        delta_theta: Maximum rotation difference to consider
    
    Returns:
        Matching score between 0 (no match) and 1 (perfect match)
    """
    if not template1 or not template2:
        return 0.0
        
    best_score = 0
    # For simplicity, compare first cylinder from each template
    cyl1 = template1[0]
    cyl2 = template2[0]
    
    # Flatten cylinders for comparison
    vec1 = cyl1.flatten()
    vec2 = cyl2.flatten()
    
    # Count valid cells (non-zero)
    valid_mask = (vec1 != 0) & (vec2 != 0)
    valid_count = np.sum(valid_mask)
    
    if valid_count / len(vec1) < min_valid_cells:
        return 0.0
        
    # Extract valid components
    valid_vec1 = vec1[valid_mask]
    valid_vec2 = vec2[valid_mask]
    
    # Cosine similarity
    dot_product = np.dot(valid_vec1, valid_vec2)
    norm1 = np.linalg.norm(valid_vec1)
    norm2 = np.linalg.norm(valid_vec2)
    
    if norm1 > 0 and norm2 > 0:
        cosine_sim = dot_product / (norm1 * norm2)
        return max(0, cosine_sim)
    
    return 0.0



import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def main(image_path1, image_path2, feature_type='MCC_co', visualize=True):
    """
    Process two fingerprint images and compute their matching score
    
    Args:
        image_path1: Path to first fingerprint image
        image_path2: Path to second fingerprint image
        feature_type: MTCC feature type ('MCC_f', 'MCC_e', 'MCC_co', etc.)
        visualize: Show intermediate processing steps
    
    Returns:
        Matching score between 0-1 and processing visualizations
    """
    # Process first image
    img1 = load_image(image_path1)
    normalized1 = normalize_image(img1)
    mask1 = segment_image(normalized1)
    orient1, freq1, energy1 = stft_analysis(normalized1)
    gabor_enhanced1 = gabor_filter(normalized1, orient1, freq1)
    smqt_enhanced1 = smqt(gabor_enhanced1)
    minutiae1 = extract_minutiae(smqt_enhanced1, mask1)
    
    template1 = []
    for minutia in minutiae1[:5]:  # Use first 5 minutiae
        cylinder = create_cylinder(
            minutia, minutiae1, orient1, freq1, energy1, feature_type=feature_type
        )
        template1.append(cylinder)
    
    # Process second image
    img2 = load_image(image_path2)
    normalized2 = normalize_image(img2)
    mask2 = segment_image(normalized2)
    orient2, freq2, energy2 = stft_analysis(normalized2)
    gabor_enhanced2 = gabor_filter(normalized2, orient2, freq2)
    smqt_enhanced2 = smqt(gabor_enhanced2)
    minutiae2 = extract_minutiae(smqt_enhanced2, mask2)
    
    template2 = []
    for minutia in minutiae2[:5]:  # Use first 5 minutiae
        cylinder = create_cylinder(
            minutia, minutiae2, orient2, freq2, energy2, feature_type=feature_type
        )
        template2.append(cylinder)
    
    # Match templates
    score = match_templates(template1, template2)
    
    if visualize:
        # Visualize both processing pipelines
        fig, axes = plt.subplots(2, 7, figsize=(20, 8))
        titles = ['Original', 'Normalized', 'Mask', 'Enhanced', 'Orientation', 'Frequency', 'Energy']
        
        for i, img in enumerate([img1, normalized1, mask1, smqt_enhanced1, orient1, freq1, energy1]):
            axes[0, i].imshow(img, cmap='gray' if i < 4 else 'viridis')
            axes[0, i].set_title(titles[i])
            axes[0, i].axis('off')
        
        for i, img in enumerate([img2, normalized2, mask2, smqt_enhanced2, orient2, freq2, energy2]):
            axes[1, i].imshow(img, cmap='gray' if i < 4 else 'viridis')
            axes[1, i].set_title(titles[i])
            axes[1, i].axis('off')
        
        plt.suptitle(f"Fingerprint Matching Score: {score:.4f} ({feature_type})", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    return score

if __name__ == "__main__":
    # Example usage with FVC2002 images
    fvc_path = R'C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2000\FVC2000\Db1_a'
    image1 = os.path.join(fvc_path, "1_1.tif")
    image2 = os.path.join(fvc_path, "1_2.tif")  # Same finger
    image3 = os.path.join(fvc_path, "2_1.tif")   # Different finger
    
    # Match same finger
    same_score = main(image1, image2, feature_type='MCC_co')
    print(f"Same finger match score: {same_score:.4f}")
    
    # Match different fingers
    diff_score = main(image1, image3, feature_type='MCC_co', visualize=False)
    print(f"Different fingers match score: {diff_score:.4f}")