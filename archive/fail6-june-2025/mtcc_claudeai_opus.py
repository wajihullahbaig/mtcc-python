"""
MTCC (Minutiae Texture Cylinder Codes) Fingerprint Recognition System
Based on research papers incorporating STFT analysis and texture features
"""

import numpy as np
import cv2
from scipy import ndimage, signal
from scipy.fft import fft2, ifft2, fftshift
from skimage import morphology, filters
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 1: CORE PREPROCESSING PIPELINE
# ============================================================================

def load_and_preprocess(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and apply initial preprocessing including variance-based segmentation
    
    Returns:
        image: Preprocessed image
        mask: Binary mask of valid fingerprint region
    """
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = image_path
    
    if image is None:
        raise ValueError(f"Cannot load image from {image_path}")
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Variance-based segmentation
    mask = segment_fingerprint(image)
    
    # Morphological smoothing of mask
    mask = morphology.binary_closing(mask, morphology.disk(5))
    mask = morphology.binary_opening(mask, morphology.disk(3))
    
    # Remove small components
    mask = morphology.remove_small_objects(mask, min_size=1000)
    
    # Keep only largest component
    labels = morphology.label(mask)
    if labels.max() > 0:
        largest = np.argmax(np.bincount(labels.flat)[1:]) + 1
        mask = (labels == largest)
    
    return image, mask


def segment_fingerprint(image: np.ndarray, block_size: int = 16) -> np.ndarray:
    """
    Variance-based fingerprint segmentation
    """
    h, w = image.shape
    mask = np.zeros((h, w), dtype=bool)
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = image[i:i+block_size, j:j+block_size]
            variance = np.var(block)
            if variance > 0.01:  # Threshold for valid fingerprint region
                mask[i:i+block_size, j:j+block_size] = True
    
    return mask


def stft_enhance_and_analyze(img, mask, win=14, overlap=6):
    h, w = img.shape
    step = win - overlap
    orientation_map = np.zeros((h, w))
    frequency_map = np.zeros((h, w))
    energy_map = np.zeros((h, w))
    enhanced_img = np.zeros((h, w))
    weight_map = np.zeros((h, w))

    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / (win - 1))
    win2d = np.outer(window, window)

    for y in range(0, h - win + 1, step):
        for x in range(0, w - win + 1, step):
            if np.mean(mask[y:y+win, x:x+win]) < 0.5:
                continue

            block = img[y:y+win, x:x+win] * win2d
            f = np.fft.fft2(block)
            fshift = np.fft.fftshift(f)
            mag = np.abs(fshift)

            # Polar coordinates
            cy, cx = win // 2, win // 2
            yv, xv = np.mgrid[0:win, 0:win]
            fx = xv - cx
            fy = yv - cy
            r = np.sqrt(fx**2 + fy**2)
            theta = np.arctan2(fy, fx)

            # Only analyze frequencies in valid range
            rnorm = r / (win / 2)
            mask_freq = (rnorm > 0.08) & (rnorm < 0.4)

            # Dominant freq/orientation by max energy in masked region
            idx = np.argmax(mag[mask_freq])
            rr = r[mask_freq].flat[idx]
            tt = theta[mask_freq].flat[idx]
            freq = rr / win      # Normalized frequency
            orient = (tt + np.pi/2) % np.pi # ridges are perpendicular to freq

            # Contextual enhancement: Keep only dominant freq/orient
            enhance_block = np.zeros_like(fshift)
            peak_mask = (np.abs(r - rr) < 2) & (np.abs((theta-tt+np.pi)%(2*np.pi)-np.pi) < np.deg2rad(10))
            enhance_block[peak_mask] = fshift[peak_mask]
            rec_block = np.fft.ifft2(np.fft.ifftshift(enhance_block)).real

            # Block energy (log)
            energy = np.log(np.sum(mag**2) + 1e-8)

            # Place back
            enhanced_img[y:y+win, x:x+win] += rec_block * win2d
            orientation_map[y:y+win, x:x+win] += orient * win2d
            frequency_map[y:y+win, x:x+win] += freq * win2d
            energy_map[y:y+win, x:x+win] += energy * win2d
            weight_map[y:y+win, x:x+win] += win2d

    valid = weight_map > 0
    enhanced_img[valid] /= weight_map[valid]
    orientation_map[valid] /= weight_map[valid]
    frequency_map[valid] /= weight_map[valid]
    energy_map[valid] /= weight_map[valid]

    return {
        'enhanced_image': enhanced_img,
        'orientation_map': orientation_map,
        'frequency_map': frequency_map,
        'energy_map': energy_map,
    }


def smooth_orientation_map(orientation_map: np.ndarray, mask: np.ndarray, 
                          sigma: float = 3.0) -> np.ndarray:
    """Smooth orientation map using vector averaging"""
    # Convert to vector representation
    cos_2theta = np.cos(2 * orientation_map)
    sin_2theta = np.sin(2 * orientation_map)
    
    # Apply Gaussian smoothing
    cos_2theta_smooth = ndimage.gaussian_filter(cos_2theta * mask, sigma)
    sin_2theta_smooth = ndimage.gaussian_filter(sin_2theta * mask, sigma)
    weight_smooth = ndimage.gaussian_filter(mask.astype(float), sigma)
    
    # Normalize
    valid = weight_smooth > 0
    cos_2theta_smooth[valid] /= weight_smooth[valid]
    sin_2theta_smooth[valid] /= weight_smooth[valid]
    
    # Convert back to angle
    orientation_smooth = 0.5 * np.arctan2(sin_2theta_smooth, cos_2theta_smooth)
    
    return orientation_smooth


def compute_coherence(orientation_map: np.ndarray, mask: np.ndarray, 
                     window_size: int = 5) -> np.ndarray:
    """Compute angular coherence for adaptive filtering"""
    h, w = orientation_map.shape
    coherence = np.zeros((h, w))
    half_win = window_size // 2
    
    for i in range(half_win, h - half_win):
        for j in range(half_win, w - half_win):
            if not mask[i, j]:
                continue
                
            # Get local window
            window = orientation_map[i-half_win:i+half_win+1, j-half_win:j+half_win+1]
            window_mask = mask[i-half_win:i+half_win+1, j-half_win:j+half_win+1]
            
            if np.sum(window_mask) == 0:
                continue
            
            # Compute coherence
            center_angle = orientation_map[i, j]
            angle_diff = np.abs(window - center_angle)
            angle_diff = np.minimum(angle_diff, np.pi - angle_diff)
            
            coherence[i, j] = 1 - np.mean(angle_diff[window_mask]) / (np.pi / 2)
    
    return coherence


def contextual_filtering(image: np.ndarray, orientation_map: np.ndarray,
                        frequency_map: np.ndarray, coherence_map: np.ndarray,
                        mask: np.ndarray) -> np.ndarray:
    """Apply contextual filtering in frequency domain"""
    h, w = image.shape
    enhanced = np.zeros_like(image)
    
    # Process in overlapping blocks
    block_size = 32
    overlap = 16
    step = block_size - overlap
    
    for i in range(0, h - block_size + 1, step):
        for j in range(0, w - block_size + 1, step):
            if np.mean(mask[i:i+block_size, j:j+block_size]) < 0.5:
                continue
            
            # Get block properties
            block = image[i:i+block_size, j:j+block_size]
            orientation = np.mean(orientation_map[i:i+block_size, j:j+block_size])
            frequency = np.mean(frequency_map[i:i+block_size, j:j+block_size])
            coherence = np.mean(coherence_map[i:i+block_size, j:j+block_size])
            
            # Apply directional filter
            filtered_block = directional_filter(block, orientation, frequency, coherence)
            
            # Blend with raised cosine window
            window = create_raised_cosine_window(block_size)
            enhanced[i:i+block_size, j:j+block_size] += filtered_block * window
    
    return enhanced


def directional_filter(block: np.ndarray, orientation: float, frequency: float, 
                      coherence: float) -> np.ndarray:
    """Apply directional bandpass filter"""
    h, w = block.shape
    
    # FFT
    fft_block = fft2(block)
    fft_block = fftshift(fft_block)
    
    # Create frequency grid
    freq_x = np.fft.fftfreq(w, d=1)
    freq_y = np.fft.fftfreq(h, d=1)
    fx, fy = np.meshgrid(fftshift(freq_x), fftshift(freq_y))
    
    # Convert to polar
    r = np.sqrt(fx**2 + fy**2)
    theta = np.arctan2(fy, fx)
    
    # Radial bandpass filter (Butterworth)
    n = 4  # Filter order
    radial_filter = 1 / (1 + ((r - frequency) / (0.1 * frequency))**(2*n))
    
    # Angular filter (raised cosine)
    angle_diff = np.abs(theta - orientation)
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
    angular_bandwidth = np.pi / 6 * (2 - coherence)  # Adaptive bandwidth
    angular_filter = 0.5 * (1 + np.cos(np.pi * angle_diff / angular_bandwidth))
    angular_filter[angle_diff > angular_bandwidth] = 0
    
    # Combined filter
    combined_filter = radial_filter * angular_filter
    
    # Apply filter
    filtered_fft = fft_block * combined_filter
    
    # Inverse FFT
    filtered_block = np.real(ifft2(fftshift(filtered_fft)))
    
    return filtered_block


def create_raised_cosine_window(size: int) -> np.ndarray:
    """Create 2D raised cosine window"""
    window_1d = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    return window_1d[:, np.newaxis] * window_1d[np.newaxis, :]


def gabor_enhancement(image: np.ndarray, orientation_map: np.ndarray,
                     frequency_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply Gabor filtering for ridge enhancement"""
    enhanced = np.zeros_like(image)
    
    # Create Gabor filter bank
    num_orientations = 16
    orientations = np.linspace(0, np.pi, num_orientations, endpoint=False)
    
    # Process each pixel
    block_size = 16
    for i in range(0, image.shape[0] - block_size + 1, block_size):
        for j in range(0, image.shape[1] - block_size + 1, block_size):
            if not mask[i + block_size//2, j + block_size//2]:
                continue
            
            # Get local parameters
            local_orientation = orientation_map[i + block_size//2, j + block_size//2]
            local_frequency = frequency_map[i + block_size//2, j + block_size//2]
            
            # Find closest orientation in bank
            idx = np.argmin(np.abs(orientations - local_orientation))
            
            # Create Gabor kernel
            kernel = cv2.getGaborKernel(
                (block_size, block_size),
                sigma=4.0,
                theta=orientations[idx],
                lambd=1/local_frequency if local_frequency > 0 else 10,
                gamma=0.5,
                psi=0
            )
            
            # Apply filter
            filtered = cv2.filter2D(image[i:i+block_size, j:j+block_size], -1, kernel)
            enhanced[i:i+block_size, j:j+block_size] = filtered
    
    return enhanced * mask


def smqt_enhancement(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Successive Mean Quantization Transform for ridge enhancement"""
    # Number of levels
    levels = 4
    
    # Initialize
    enhanced = image.copy()
    
    for level in range(levels):
        # Compute mean
        mean_val = np.mean(enhanced[mask])
        
        # Quantize
        above_mean = enhanced >= mean_val
        below_mean = ~above_mean
        
        # Update values
        if np.any(above_mean & mask):
            enhanced[above_mean & mask] = np.mean(enhanced[above_mean & mask])
        if np.any(below_mean & mask):
            enhanced[below_mean & mask] = np.mean(enhanced[below_mean & mask])
    
    # Normalize
    enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)
    
    return enhanced * mask


# ============================================================================
# SECTION 2: ADVANCED TEXTURE FEATURE EXTRACTION
# ============================================================================

def extract_texture_features(stft_results: Dict) -> Dict:
    """
    Extract STFT-based texture features for MTCC descriptors
    
    Returns:
        Dictionary containing:
        - Io: Orientation image from STFT analysis
        - If: Frequency image from STFT analysis
        - Ie: Energy image (logarithmic) from STFT analysis
    """
    return {
        'Io': stft_results['orientation_map'],
        'If': stft_results['frequency_map'],
        'Ie': stft_results['energy_map']
    }


# ============================================================================
# SECTION 3: MINUTIAE EXTRACTION
# ============================================================================

def enhanced_minutiae_extraction(enhanced_image: np.ndarray, mask: np.ndarray) -> List[Dict]:
    """
    Extract minutiae using Crossing Number algorithm with quality assessment
    """
    # Binarize image
    binary = binarize_image(enhanced_image, mask)
    
    # Thin the image
    skeleton = skeletonize(binary)
    
    # Extract minutiae
    minutiae = []
    
    # Crossing number computation
    cn_filter = np.array([[1, 1, 1],
                         [1, 10, 1],
                         [1, 1, 1]])
    
    # Compute crossing numbers
    filtered = ndimage.convolve(skeleton.astype(int), cn_filter, mode='constant')
    
    # Find minutiae points
    # Ridge endings (CN = 1)
    ridge_endings = (filtered == 11) & skeleton
    
    # Bifurcations (CN = 3)
    bifurcations = (filtered == 13) & skeleton
    
    # Extract coordinates and properties
    for y, x in zip(*np.where(ridge_endings)):
        if mask[y, x]:
            minutiae.append({
                'x': x,
                'y': y,
                'type': 'ending',
                'angle': compute_minutia_angle(skeleton, x, y),
                'quality': compute_minutia_quality(enhanced_image, x, y, mask)
            })
    
    for y, x in zip(*np.where(bifurcations)):
        if mask[y, x]:
            minutiae.append({
                'x': x,
                'y': y,
                'type': 'bifurcation',
                'angle': compute_minutia_angle(skeleton, x, y),
                'quality': compute_minutia_quality(enhanced_image, x, y, mask)
            })
    
    # Sort by quality
    minutiae.sort(key=lambda m: m['quality'], reverse=True)
    
    return minutiae


def binarize_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Binarize fingerprint image"""
    # Local adaptive thresholding
    binary = image > filters.threshold_local(image, block_size=25, method='gaussian')
    return binary & mask


def compute_minutia_angle(skeleton: np.ndarray, x: int, y: int, 
                         search_radius: int = 5) -> float:
    """Compute minutia angle from skeleton"""
    angles = []
    
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            if dx == 0 and dy == 0:
                continue
            
            ny, nx = y + dy, x + dx
            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                if skeleton[ny, nx]:
                    angle = np.arctan2(dy, dx)
                    angles.append(angle)
    
    if angles:
        # Average angle
        return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    else:
        return 0


def compute_minutia_quality(image: np.ndarray, x: int, y: int, 
                           mask: np.ndarray, window_size: int = 15) -> float:
    """Compute minutia quality based on local clarity"""
    half_win = window_size // 2
    
    # Extract local window
    y_start = max(0, y - half_win)
    y_end = min(image.shape[0], y + half_win + 1)
    x_start = max(0, x - half_win)
    x_end = min(image.shape[1], x + half_win + 1)
    
    window = image[y_start:y_end, x_start:x_end]
    window_mask = mask[y_start:y_end, x_start:x_end]
    
    if np.sum(window_mask) == 0:
        return 0
    
    # Compute local variance as quality measure
    quality = np.var(window[window_mask])
    
    return quality


# ============================================================================
# SECTION 4: MTCC DESCRIPTOR GENERATION
# ============================================================================

def create_mtcc_cylinders(minutiae_list: List[Dict], texture_maps: Dict,
                         radius: int = 65, NS: int = 18, ND: int = 5) -> List[Dict]:
    """
    Generate MTCC (Minutiae Texture Cylinder Codes) descriptors
    """
    cylinders = []
    
    for minutia in minutiae_list:
        cylinder = create_single_mtcc_cylinder(minutia, minutiae_list, texture_maps, 
                                             radius, NS, ND)
        if cylinder is not None:
            cylinders.append({
                'minutia': minutia,
                'descriptor': cylinder
            })
    
    return cylinders


def create_single_mtcc_cylinder(central_minutia: Dict, all_minutiae: List[Dict],
                               texture_maps: Dict, radius: int, NS: int, ND: int) -> Optional[np.ndarray]:
    """Create MTCC cylinder for a single minutia"""
    # Initialize cylinder
    cylinder = np.zeros((NS, NS, ND))
    
    # Cell dimensions
    delta_S = 2 * radius / NS
    delta_D = 2 * np.pi / ND
    
    # Gaussian parameters
    sigma_S = 6
    sigma_D = 5 * np.pi / 36
    
    # Get neighboring minutiae
    neighbors = []
    for m in all_minutiae:
        if m == central_minutia:
            continue
        
        dx = m['x'] - central_minutia['x']
        dy = m['y'] - central_minutia['y']
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist <= 3 * sigma_S:
            neighbors.append(m)
    
    # Check validity
    if len(neighbors) < 4:  # Minimum neighbors
        return None
    
    # Fill cylinder cells
    valid_cells = 0
    
    for i in range(NS):
        for j in range(NS):
            # Cell center in minutia reference frame
            cell_x = (i - NS/2 + 0.5) * delta_S
            cell_y = (j - NS/2 + 0.5) * delta_S
            
            # Rotate to image coordinates
            theta = central_minutia['angle']
            rot_x = cell_x * np.cos(theta) - cell_y * np.sin(theta)
            rot_y = cell_x * np.sin(theta) + cell_y * np.cos(theta)
            
            # Cell center in image coordinates
            px = central_minutia['x'] + rot_x
            py = central_minutia['y'] + rot_y
            
            # Check if cell is valid (within image and mask)
            if not is_valid_cell(px, py, texture_maps['Io'].shape):
                continue
            
            valid_cells += 1
            
            # Get texture values at cell center
            cell_orientation = interpolate_value(texture_maps['Io'], px, py)
            cell_frequency = interpolate_value(texture_maps['If'], px, py)
            cell_energy = interpolate_value(texture_maps['Ie'], px, py)
            
            # Compute contributions for each angular sector
            for k in range(ND):
                angle_k = -np.pi + (k + 0.5) * delta_D
                
                # Spatial contribution (same as MCC)
                spatial_contrib = 0
                
                for neighbor in neighbors:
                    dx = neighbor['x'] - px
                    dy = neighbor['y'] - py
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    spatial = gaussian(dist, 0, sigma_S)
                    
                    # Different contributions based on feature type
                    # For MCCo (original): use minutia angles
                    # For MCCf: use frequency
                    # For MCCe: use energy
                    # For MCCco: use cell-centered orientation
                    
                    # Here we implement MCCco as example
                    angle_diff = angle_difference(angle_k, cell_orientation)
                    directional = gaussian(angle_diff, 0, sigma_D)
                    
                    spatial_contrib += spatial * directional
                
                # Store in cylinder
                cylinder[i, j, k] = sigmoid(spatial_contrib)
    
    # Check cylinder validity
    if valid_cells < 0.20 * NS * NS:  # Less than 20% valid cells
        return None
    
    return cylinder


def is_valid_cell(x: float, y: float, shape: Tuple[int, int]) -> bool:
    """Check if cell center is within valid image region"""
    h, w = shape
    return 0 <= x < w and 0 <= y < h


def interpolate_value(image: np.ndarray, x: float, y: float) -> float:
    """Bilinear interpolation"""
    h, w = image.shape
    
    # Boundary check
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        return 0
    
    # Integer and fractional parts
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    
    # Bilinear interpolation
    val = (1 - fx) * (1 - fy) * image[y0, x0] + \
          fx * (1 - fy) * image[y0, x0 + 1] + \
          (1 - fx) * fy * image[y0 + 1, x0] + \
          fx * fy * image[y0 + 1, x0 + 1]
    
    return val


def gaussian(x: float, mu: float, sigma: float) -> float:
    """Gaussian function"""
    return np.exp(-0.5 * ((x - mu) / sigma)**2)


def angle_difference(angle1: float, angle2: float) -> float:
    """Compute minimum angle difference"""
    diff = angle1 - angle2
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return abs(diff)


def sigmoid(x: float, mu: float = 0.005, tau: float = 400) -> float:
    """Sigmoid function for contribution normalization"""
    return 1 / (1 + np.exp(-tau * (x - mu)))


# ============================================================================
# SECTION 5: MATCHING WITH MULTIPLE DISTANCE METRICS
# ============================================================================

def mtcc_matching(cylinders1: List[Dict], cylinders2: List[Dict],
                 feature_type: str = 'MCCco') -> float:
    """
    MTCC matching using Local Similarity Sort with Relaxation (LSSR)
    """
    if not cylinders1 or not cylinders2:
        return 0
    
    # Compute local similarity matrix
    n1, n2 = len(cylinders1), len(cylinders2)
    lsm = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            # Compute similarity between cylinders
            sim = compute_cylinder_similarity(
                cylinders1[i]['descriptor'],
                cylinders2[j]['descriptor'],
                feature_type
            )
            lsm[i, j] = sim
    
    # Local Similarity Sort
    top_pairs = local_similarity_sort(lsm, n_pairs=min(20, n1*n2//4))
    
    # Relaxation
    relaxed_scores = apply_relaxation(top_pairs, cylinders1, cylinders2, lsm)
    
    # Global score
    if relaxed_scores:
        return np.mean(sorted(relaxed_scores, reverse=True)[:12])
    else:
        return 0


def compute_cylinder_similarity(cyl1: np.ndarray, cyl2: np.ndarray,
                               feature_type: str) -> float:
    """Compute similarity between two cylinders"""
    if cyl1 is None or cyl2 is None:
        return 0
    
    # Flatten cylinders
    c1_flat = cyl1.flatten()
    c2_flat = cyl2.flatten()
    
    # Different distance metrics based on feature type
    if feature_type == 'MCCo':
        # Euclidean distance
        dist = np.linalg.norm(c1_flat - c2_flat)
        similarity = 1 / (1 + dist)
    else:
        # Double angle distance for texture features
        # Convert to double angles
        c1_double = 2 * c1_flat
        c2_double = 2 * c2_flat
        
        # Compute cosine and sine distances
        cos_dist = np.linalg.norm(np.cos(c1_double) - np.cos(c2_double))
        sin_dist = np.linalg.norm(np.sin(c1_double) - np.sin(c2_double))
        
        # Combined distance
        dist = np.sqrt(cos_dist**2 + sin_dist**2) / 2
        similarity = 1 / (1 + dist)
    
    return similarity


def local_similarity_sort(lsm: np.ndarray, n_pairs: int) -> List[Tuple[int, int, float]]:
    """Select top matching pairs from similarity matrix"""
    # Get all pairs with scores
    pairs = []
    for i in range(lsm.shape[0]):
        for j in range(lsm.shape[1]):
            pairs.append((i, j, lsm[i, j]))
    
    # Sort by similarity
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Return top pairs
    return pairs[:n_pairs]


def apply_relaxation(top_pairs: List[Tuple[int, int, float]], 
                    cylinders1: List[Dict], cylinders2: List[Dict],
                    lsm: np.ndarray) -> List[float]:
    """Apply relaxation for structural compatibility"""
    relaxed_scores = []
    
    for idx, (i, j, score) in enumerate(top_pairs):
        # Get minutiae
        m1 = cylinders1[i]['minutia']
        m2 = cylinders2[j]['minutia']
        
        # Compute compatibility with other pairs
        compatibility = 0
        count = 0
        
        for idx2, (i2, j2, score2) in enumerate(top_pairs):
            if idx == idx2:
                continue
            
            m1_2 = cylinders1[i2]['minutia']
            m2_2 = cylinders2[j2]['minutia']
            
            # Check structural compatibility
            dist1 = np.sqrt((m1['x'] - m1_2['x'])**2 + (m1['y'] - m1_2['y'])**2)
            dist2 = np.sqrt((m2['x'] - m2_2['x'])**2 + (m2['y'] - m2_2['y'])**2)
            
            # Distance compatibility
            dist_compat = 1 - abs(dist1 - dist2) / (dist1 + dist2 + 1e-6)
            
            # Angle compatibility
            angle1 = np.arctan2(m1_2['y'] - m1['y'], m1_2['x'] - m1['x'])
            angle2 = np.arctan2(m2_2['y'] - m2['y'], m2_2['x'] - m2['x'])
            angle_compat = 1 - angle_difference(angle1, angle2) / np.pi
            
            compatibility += dist_compat * angle_compat * score2
            count += 1
        
        if count > 0:
            compatibility /= count
        
        # Relaxed score
        relaxed_score = score * (1 + 0.5 * compatibility)
        relaxed_scores.append(relaxed_score)
    
    return relaxed_scores


# ============================================================================
# SECTION 6: COMPREHENSIVE EVALUATION FRAMEWORK
# ============================================================================

def evaluate_fvc_dataset(dataset_path: str, method: str = 'MTCC') -> Dict:
    """
    FVC2002/2004 evaluation protocol
    
    - Genuine tests: [(8x7)/2] x 100 = 2800 per database
    - Impostor tests: [(100x99)/2] = 4950 per database
    """
    import os
    import glob
    
    genuine_scores = []
    impostor_scores = []
    templates = {}
    
    # Process all images
    image_files = sorted(glob.glob(os.path.join(dataset_path, "*.tif")))
    
    print(f"Processing {len(image_files)} images...")
    
    for idx, image_path in enumerate(image_files):
        print(f"Processing image {idx+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Extract template
        template = extract_mtcc_template(image_path, method)
        templates[image_path] = template
    
    print("Computing genuine scores...")
    # Genuine matching (same finger, different impressions)
    for i in range(0, len(image_files), 8):  # Each finger has 8 impressions
        for j in range(8):
            for k in range(j+1, 8):
                if i+j < len(image_files) and i+k < len(image_files):
                    score = mtcc_matching(
                        templates[image_files[i+j]],
                        templates[image_files[i+k]],
                        method
                    )
                    genuine_scores.append(score)
    
    print("Computing impostor scores...")
    # Impostor matching (different fingers, first impressions)
    for i in range(0, len(image_files), 8):
        for j in range(i+8, len(image_files), 8):
            if i < len(image_files) and j < len(image_files):
                score = mtcc_matching(
                    templates[image_files[i]],
                    templates[image_files[j]],
                    method
                )
                impostor_scores.append(score)
    
    # Calculate metrics
    eer, threshold = calculate_eer(genuine_scores, impostor_scores)
    
    return {
        'EER': eer,
        'threshold': threshold,
        'genuine_scores': genuine_scores,
        'impostor_scores': impostor_scores
    }


def extract_mtcc_template(image_path: str, method: str) -> List[Dict]:
    """Extract MTCC template from fingerprint image"""
    # Load and preprocess
    image, mask = load_and_preprocess(image_path)
    
    # STFT enhancement and analysis
    stft_results = stft_enhancement_analysis(image, mask)
    
    # Extract texture features
    texture_maps = extract_texture_features(stft_results)
    
    # Extract minutiae
    minutiae = enhanced_minutiae_extraction(stft_results['enhanced_image'], mask)
    
    # Generate MTCC descriptors
    cylinders = create_mtcc_cylinders(minutiae, texture_maps)
    
    return cylinders


def calculate_eer(genuine_scores: List[float], impostor_scores: List[float]) -> Tuple[float, float]:
    """Calculate Equal Error Rate"""
    if not genuine_scores or not impostor_scores:
        return 100.0, 0.0
    
    # Find threshold range
    all_scores = genuine_scores + impostor_scores
    min_score = min(all_scores)
    max_score = max(all_scores)
    
    # Search for EER
    best_eer = 100.0
    best_threshold = 0.0
    
    for threshold in np.linspace(min_score, max_score, 1000):
        # False Rejection Rate
        frr = sum(1 for s in genuine_scores if s < threshold) / len(genuine_scores)
        
        # False Acceptance Rate  
        far = sum(1 for s in impostor_scores if s >= threshold) / len(impostor_scores)
        
        # EER is where FRR = FAR
        if abs(frr - far) < abs(best_eer - far):
            best_eer = (frr + far) / 2
            best_threshold = threshold
    
    return best_eer * 100, best_threshold  # Return as percentage


# ============================================================================
# SECTION 7: VISUALIZATION AND DEBUGGING
# ============================================================================

def visualize_mtcc_pipeline(image_path: str, save_path: Optional[str] = None):
    """
    Comprehensive visualization of MTCC pipeline
    
    Grid layout (3x4):
    Row 1: Original → Segmented → STFT Enhanced → Curved Gabor Enhanced
    Row 2: Orientation Map → Frequency Map → Energy Map → Coherence Map
    Row 3: Binarized → Thinned → Minutiae → MTCC Cylinders Visualization
    """
    # Load and preprocess
    image, mask = load_and_preprocess(image_path)
    
    # STFT analysis
    stft_results = stft_enhance_and_analyze(image, mask)
    
    # Extract features
    texture_maps = extract_texture_features(stft_results)
    
    # Binarize and thin
    binary = binarize_image(stft_results['enhanced_image'], mask)
    skeleton = skeletonize(binary)
    
    # Extract minutiae
    minutiae = enhanced_minutiae_extraction(stft_results['enhanced_image'], mask)
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Enhancement pipeline
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Segmented Mask')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(stft_results['enhanced_image'], cmap='gray')
    axes[0, 2].set_title('STFT Enhanced')
    axes[0, 2].axis('off')
    
    # Gabor enhanced (for visualization)
    gabor_enhanced = gabor_enhancement(
        stft_results['enhanced_image'],
        stft_results['orientation_map'],
        stft_results['frequency_map'],
        mask
    )
    axes[0, 3].imshow(gabor_enhanced, cmap='gray')
    axes[0, 3].set_title('Gabor Enhanced')
    axes[0, 3].axis('off')
    
    # Row 2: Feature maps
    im1 = axes[1, 0].imshow(stft_results['orientation_map'], cmap='hsv')
    axes[1, 0].set_title('Orientation Map')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    im2 = axes[1, 1].imshow(stft_results['frequency_map'], cmap='jet')
    axes[1, 1].set_title('Frequency Map')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    im3 = axes[1, 2].imshow(stft_results['energy_map'], cmap='hot')
    axes[1, 2].set_title('Energy Map')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
    
    im4 = axes[1, 3].imshow(stft_results['coherence_map'], cmap='viridis')
    axes[1, 3].set_title('Coherence Map')
    axes[1, 3].axis('off')
    plt.colorbar(im4, ax=axes[1, 3], fraction=0.046)
    
    # Row 3: Minutiae extraction
    axes[2, 0].imshow(binary, cmap='gray')
    axes[2, 0].set_title('Binarized')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(skeleton, cmap='gray')
    axes[2, 1].set_title('Thinned')
    axes[2, 1].axis('off')
    
    # Minutiae visualization
    axes[2, 2].imshow(stft_results['enhanced_image'], cmap='gray')
    for m in minutiae[:50]:  # Show top 50 minutiae
        color = 'r' if m['type'] == 'ending' else 'b'
        axes[2, 2].plot(m['x'], m['y'], color+('o' if m['type'] == 'ending' else '^'), 
                       markersize=6)
        # Show direction
        dx = 10 * np.cos(m['angle'])
        dy = 10 * np.sin(m['angle'])
        axes[2, 2].arrow(m['x'], m['y'], dx, dy, head_width=3, head_length=2, 
                        fc=color, ec=color)
    axes[2, 2].set_title(f'Minutiae ({len(minutiae)} detected)')
    axes[2, 2].axis('off')
    
    # MTCC visualization (show cylinder structure)
    cylinders = create_mtcc_cylinders(minutiae[:5], texture_maps)  # First 5 minutiae
    if cylinders:
        # Visualize first cylinder
        cylinder = cylinders[0]['descriptor']
        if cylinder is not None:
            # Show middle slice
            slice_idx = cylinder.shape[2] // 2
            im5 = axes[2, 3].imshow(cylinder[:, :, slice_idx], cmap='hot')
            axes[2, 3].set_title('MTCC Cylinder (slice)')
            axes[2, 3].axis('off')
            plt.colorbar(im5, ax=axes[2, 3], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def visualize_mtcc_cylinders(minutiae: List[Dict], cylinders: List[Dict], 
                           texture_maps: Dict, num_examples: int = 3):
    """Visualize MTCC cylinder contents showing texture feature distributions"""
    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4*num_examples))
    
    for i in range(min(num_examples, len(cylinders))):
        cylinder = cylinders[i]['descriptor']
        minutia = cylinders[i]['minutia']
        
        if cylinder is None:
            continue
        
        # Show different slices
        for j in range(4):
            slice_idx = j * cylinder.shape[2] // 4
            im = axes[i, j].imshow(cylinder[:, :, slice_idx], cmap='hot')
            axes[i, j].set_title(f'Minutia {i+1}, Angular slice {j+1}')
            axes[i, j].axis('off')
            plt.colorbar(im, ax=axes[i, j], fraction=0.046)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 8: PARAMETER OPTIMIZATION
# ============================================================================

MTCC_PARAMETERS = {
    'cylinder_radius': 65,
    'spatial_sectors': 18,  # NS
    'angular_sectors': 5,   # ND
    'gaussian_spatial': 6,  # σS
    'gaussian_directional': 5*np.pi/36,  # σD
    'stft_window': 14,
    'stft_overlap': 6,
    'curved_region_lines': 33,
    'curved_region_points': 65,
    'gabor_sigma_x': 8.0,
    'gabor_sigma_y': 8.0,
}


# ============================================================================
# TESTING PROTOCOL
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("MTCC Fingerprint Recognition System")
    print("=" * 50)
    
    # Test with a single image
    test_image_path = R"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002\Db1_a/1_1.tif"
    
    # Visualize pipeline
    visualize_mtcc_pipeline(test_image_path, "mtcc_pipeline.png")
    
    # Test matching
    # Load FVC dataset
    dataset_path = "path/to/fvc2004/DB1_A"
    
    # Test multiple MTCC variants
    methods = ['MCCo', 'MCCf', 'MCCe', 'MCCco', 'MCCcf', 'MCCce']
    results = {}
    
    for method in methods:
        print(f"\nTesting {method}...")
        # eer_result = evaluate_fvc_dataset(dataset_path, method)
        # results[method] = eer_result
        # print(f"{method} EER: {eer_result['EER']:.2f}%")
    
    # Generate comparison plots
    # plot_det_curves(results)
    # plot_roc_curves(results)
    # generate_performance_table(results)