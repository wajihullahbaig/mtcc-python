import numpy as np
import cv2 # For efficient image loading, basic ops, adaptive thresholding
from scipy import ndimage, signal
from skimage.morphology import skeletonize, binary_opening, binary_closing, disk
from skimage.filters import gabor_kernel, gaussian
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.restoration import denoise_nl_means, estimate_sigma
from matplotlib import pyplot as plt
import os
import math
from collections import namedtuple

# --- Global Parameters and Constants ---
MTCC_PARAMETERS = {
    'cylinder_radius': 65, # R
    'spatial_sectors': 18, # NS (Number of spatial sectors)
    'angular_sectors': 5,  # ND (Number of directional sectors)
    'gaussian_spatial_sigma': 6.0, # σS for spatial contribution
    'gaussian_directional_sigma': 5 * np.pi / 36, # σD for directional contribution (approx 25 degrees)
    'stft_window_size': 14, # px, for square windows (W)
    'stft_overlap': 6,      # px, overlap between windows (O)
    'curved_region_lines': 33, # For curved ridge frequency estimation
    'curved_region_points': 65, # For curved ridge frequency estimation
    'gabor_sigma_x': 8.0, # Sigma along ridge flow for curved Gabor concept
    'gabor_sigma_y': 8.0, # Sigma across ridge flow for curved Gabor concept
    'gabor_lambda_min': 3.0, # Min ridge wavelength for Gabor
    'gabor_lambda_max': 20.0, # Max ridge wavelength for Gabor
    'minutiae_border_dist': 15, # Min distance from mask border for valid minutiae
    'minutiae_min_dist': 10, # Min distance between two minutiae
    'minutiae_spur_length': 10, # Max length of spurious branch to remove
    'segmentation_block_size': 16, # px
    'segmentation_variance_threshold': 100, # empirically tuned
    'morph_kernel_size': 5, # px for morphological operations
    'smqt_alpha': 2, # SMQT parameter
    'smqt_beta': 10, # SMQT parameter
    'relaxation_iterations': 5, # For LSSR
    'relaxation_decay_factor': 0.8, # For LSSR
    'minutiae_quality_thresh': 0.3 # Minimum quality for a minutia to be considered
}

# Minutia structure (x, y, angle_radians, type (1=end, 2=bif), quality)
Minutia = namedtuple('Minutia', ['x', 'y', 'angle', 'type', 'quality'])

# --- Utility Functions ---

def normalize_image_global(img):
    """Normalize image to 0-255 range."""
    img = np.array(img, dtype=np.float32)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val == 0:
        return np.zeros_like(img)
    return 255.0 * (img - min_val) / (max_val - min_val)

def normalize_image_local(img):
    """Normalize image locally using mean and variance."""
    mean = np.mean(img)
    std = np.std(img)
    normalized_img = (img - mean) / (std + 1e-8) # Add epsilon to prevent division by zero
    # Scale to a standard range, e.g., 0-255 after scaling to [-1, 1] for processing
    normalized_img = normalized_img * 127.5 + 127.5
    normalized_img[normalized_img < 0] = 0
    normalized_img[normalized_img > 255] = 255
    return normalized_img.astype(np.uint8)

def angle_diff(angle1, angle2):
    """Calculate the smallest angular difference between two angles in radians (0 to pi)."""
    diff = abs(angle1 - angle2)
    return min(diff, np.pi - diff) # For orientation, difference is 0 to pi

def orientation_diff_2pi(angle1, angle2):
    """Calculate the smallest angular difference between two angles in radians (0 to 2pi)."""
    diff = abs(angle1 - angle2)
    return min(diff, 2 * np.pi - diff)

def apply_mask(image, mask):
    """Apply mask to an image, setting non-mask regions to black."""
    if image.dtype != mask.dtype:
        mask = mask.astype(image.dtype)
    return image * mask

def double_angle_distance(angle1, angle2):
    """Compute the double angle distance between two orientation angles."""
    # Convert angles to double angle representation (0 to 2pi)
    da1 = 2 * angle1
    da2 = 2 * angle2
    # Compute Euclidean distance on unit circle (cos, sin)
    return np.sqrt((np.cos(da1) - np.cos(da2))**2 + (np.sin(da1) - np.sin(da2))**2)

# --- 1. CORE PREPROCESSING PIPELINE ---

def load_and_preprocess(image_path):
    """
    Load and apply initial preprocessing:
    - Variance-based segmentation with morphological smoothing
    - Create fingerprint mask for valid region constraint
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Global normalization for initial processing
    img = normalize_image_global(img) # This returns float32

    # Variance-based segmentation
    block_size = MTCC_PARAMETERS['segmentation_block_size']
    h, w = img.shape
    variance_map = np.zeros_like(img, dtype=np.float32)

    for r in range(0, h, block_size):
        for c in range(0, w, block_size):
            block = img[r:min(r+block_size, h), c:min(c+block_size, w)]
            if block.size > 0:
                variance_map[r:min(r+block_size, h), c:min(c+block_size, w)] = np.var(block)

    # Thresholding to create initial mask
    mask = (variance_map > MTCC_PARAMETERS['segmentation_variance_threshold']).astype(np.uint8) * 255

    # Morphological smoothing for the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MTCC_PARAMETERS['morph_kernel_size'], MTCC_PARAMETERS['morph_kernel_size']))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Fill small holes in the mask
    mask_labels = label(mask == 0)
    # Get background label, assuming the largest connected component of 0s is background
    if mask_labels.max() > 0:
        background_label = np.argmax(np.bincount(mask_labels.flat)[1:]) + 1
        mask[mask_labels != background_label] = 255 # Fill small holes (set to foreground)

    # Ensure mask is binary (0 or 255)
    mask[mask > 0] = 255
    mask = mask.astype(np.uint8)

    # Apply mask to image for a "clean" fingerprint area
    masked_img = apply_mask(img, mask)

    # Ensure the returned image is uint8 for consistent pipeline processing
    return masked_img.astype(np.uint8), mask.astype(bool) # Return boolean mask for easier use

def stft_enhancement_analysis(image, mask, window_size=14, overlap=6):
    """
    STFT-based enhancement and feature extraction (Papers 3-4)

    Returns:
        enhanced_image: STFT + Gabor + SMQT enhanced fingerprint
        orientation_map: Probabilistic orientation using vector averaging
        frequency_map: Ridge frequency from spectral analysis
        energy_map: Logarithmic energy content per block
        coherence_map: Angular coherence for adaptive filtering
    """
    h, w = image.shape
    step = window_size - overlap

    # Initialize maps
    orientation_map = np.zeros((h // step + 1, w // step + 1), dtype=np.float32)
    frequency_map = np.zeros_like(orientation_map, dtype=np.float32)
    energy_map = np.zeros_like(orientation_map, dtype=np.float32)
    coherence_map = np.zeros_like(orientation_map, dtype=np.float32)
    
    # Ensure image is float32 for FFT operations
    image_float = image.astype(np.float32)

    # Create a 2D raised cosine window
    win_1d = signal.windows.hann(window_size)
    win_2d = np.outer(win_1d, win_1d)

    # Iterate over overlapping windows
    for r_idx, r in enumerate(range(0, h - window_size + 1, step)):
        for c_idx, c in enumerate(range(0, w - window_size + 1, step)):
            block = image_float[r : r + window_size, c : c + window_size]
            block_mask = mask[r : r + window_size, c : c + window_size]

            # Only process blocks mostly within the fingerprint mask
            if np.sum(block_mask) < (window_size * window_size * 0.5): # At least 50% mask coverage
                continue

            # Apply window
            windowed_block = block * win_2d

            # 2D FFT
            f_transform = np.fft.fft2(windowed_block)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)

            # Analyze spectrum for dominant frequency, orientation, energy
            # We look for the peak in the positive frequency quadrant (after shifting origin)
            center_x, center_y = window_size // 2, window_size // 2
            # Set center (DC component) to zero to avoid it being dominant peak
            magnitude_spectrum[center_y, center_x] = 0

            # Find peak (excluding DC component)
            max_val_idx = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
            peak_y, peak_x = max_val_idx

            # Dominant Frequency: Distance from center
            # Frequencies are normalized by window_size
            freq_x = (peak_x - center_x) / window_size
            freq_y = (peak_y - center_y) / window_size
            dominant_freq = np.sqrt(freq_x**2 + freq_y**2)

            # Dominant Orientation: Angle of the peak from center, perpendicular to ridges
            dominant_orientation = np.arctan2(peak_y - center_y, peak_x - center_x)

            # Normalize orientation to 0-pi for ridge orientation
            # The angle of the spectral peak is perpendicular to the ridge flow
            dominant_orientation = (dominant_orientation + np.pi / 2) % np.pi # + 90 deg and normalize to 0-pi

            # Energy: Log sum of magnitudes (excluding DC)
            total_energy = np.sum(magnitude_spectrum**2)
            energy_map[r_idx, c_idx] = np.log(total_energy + 1e-8) # Add epsilon to avoid log(0)

            # Coherence: Measure of directional strength.
            coherence = (magnitude_spectrum[peak_y, peak_x] - magnitude_spectrum.mean()) / (magnitude_spectrum.std() + 1e-8)
            coherence_map[r_idx, c_idx] = np.clip(coherence, 0, 1) # Clip to 0-1 range

            orientation_map[r_idx, c_idx] = dominant_orientation
            frequency_map[r_idx, c_idx] = dominant_freq
            
    # Resize maps to original image size for pixel-wise access
    # Use linear interpolation for smoothness
    orientation_map = cv2.resize(orientation_map, (w, h), interpolation=cv2.INTER_LINEAR)
    frequency_map = cv2.resize(frequency_map, (w, h), interpolation=cv2.INTER_LINEAR)
    energy_map = cv2.resize(energy_map, (w, h), interpolation=cv2.INTER_LINEAR)
    coherence_map = cv2.resize(coherence_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize frequency map to a reasonable range (e.g., 0-1)
    frequency_map = np.clip(frequency_map, 0.05, 0.25) # Clip to typical fingerprint ridge frequencies
    frequency_map = normalize_image_global(frequency_map) / 255.0 # Normalize to 0-1 for convenience

    # Return the original image (or a basic normalized version) for curved Gabor processing.
    # The actual "enhanced_image" in this step is not yet the final Gabor-enhanced output.
    # We pass the original `image` (which is uint8) for consistency, as `curved_gabor_enhancement` will process it.
    return image, orientation_map, frequency_map, energy_map, coherence_map

def curved_gabor_enhancement(image, orientation_map, frequency_map, coherence_map):
    """
    Curved Gabor filter enhancement (Paper 5)
    """
    h, w = image.shape
    
    block_size = MTCC_PARAMETERS['stft_window_size'] # Re-use STFT block size
    step = MTCC_PARAMETERS['stft_window_size'] - MTCC_PARAMETERS['stft_overlap']

    # Convert image to float32 for processing
    image_float = image.astype(np.float32)

    # Convert normalized frequency_map (0-1) back to wavelength range (lambda)
    min_lambda = MTCC_PARAMETERS['gabor_lambda_min']
    max_lambda = MTCC_PARAMETERS['gabor_lambda_max']
    lambda_map = (1 - frequency_map) * (max_lambda - min_lambda) + min_lambda
    lambda_map[lambda_map < 1e-5] = 1e-5

    sigma_x = MTCC_PARAMETERS['gabor_sigma_x'] # Along ridge flow
    sigma_y = MTCC_PARAMETERS['gabor_sigma_y'] # Across ridge flow

    filtered_patches = np.zeros_like(image_float, dtype=np.float32)
    weights = np.zeros_like(image_float, dtype=np.float32)
    
    pad_size = int(max(sigma_x, sigma_y) * 3)
    padded_image = cv2.copyMakeBorder(image_float, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)
    padded_orientation = cv2.copyMakeBorder(orientation_map, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)
    padded_lambda = cv2.copyMakeBorder(lambda_map, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)
    padded_coherence = cv2.copyMakeBorder(coherence_map, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)

    for r in range(0, h, step):
        for c in range(0, w, step):
            center_r = r + pad_size
            center_c = c + pad_size

            local_orientation = padded_orientation[center_r, center_c]
            local_lambda = padded_lambda[center_r, center_c]
            local_coherence = padded_coherence[center_r, center_c]

            adaptive_sigma_y = sigma_y / (local_coherence + 0.1)
            adaptive_sigma_y = min(adaptive_sigma_y, sigma_y * 2)

            gabor_theta = local_orientation + np.pi / 2
            
            g_kernel_real = gabor_kernel(frequency=1.0/local_lambda, theta=gabor_theta,
                                         sigma_x=sigma_x, sigma_y=adaptive_sigma_y)
            g_kernel_real = np.real(g_kernel_real)
            
            img_patch = padded_image[center_r - block_size//2 : center_r + block_size//2 +1,
                                     center_c - block_size//2 : center_c + block_size//2 +1]
            
            # Ensure kernel and patch are compatible for convolution.
            # Using scipy.ndimage.convolve for simplicity and flexibility with kernel size.
            # Pad kernel if its size is very different from standard conv filter (e.g. 3x3)
            # Ensure img_patch is not empty
            if img_patch.shape[0] == 0 or img_patch.shape[1] == 0:
                continue

            filtered_block = ndimage.convolve(img_patch, g_kernel_real, mode='nearest')
            
            start_r_dest = max(0, r)
            end_r_dest = min(h, r + block_size)
            start_c_dest = max(0, c)
            end_c_dest = min(w, c + block_size)

            start_r_src = start_r_dest - r
            end_r_src = end_r_dest - r
            start_c_src = start_c_dest - c
            end_c_src = end_c_dest - c

            filtered_patches[start_r_dest:end_r_dest, start_c_dest:end_c_dest] += \
                filtered_block[start_r_src:end_r_src, start_c_src:end_c_src]

            weights[start_r_dest:end_r_dest, start_c_dest:end_c_dest] += 1 # Count overlaps

    enhanced_image = np.divide(filtered_patches, weights, out=np.zeros_like(filtered_patches), where=weights!=0)
    
    # Final normalization and SMQT to ensure 0-255 uint8 image
    enhanced_image = normalize_image_global(enhanced_image).astype(np.uint8)

    def smqt_simple(img, alpha=MTCC_PARAMETERS['smqt_alpha'], beta=MTCC_PARAMETERS['smqt_beta']):
        img = img.astype(np.float32)
        min_val, max_val = np.min(img), np.max(img)
        range_val = max_val - min_val
        if range_val == 0: return np.zeros_like(img, dtype=np.uint8)
        
        normalized = (img - min_val) / range_val
        enhanced = 255 * (normalized ** (1.0/alpha))
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    enhanced_image = smqt_simple(enhanced_image)

    return enhanced_image

# --- 2. ADVANCED TEXTURE FEATURE EXTRACTION ---

def extract_texture_features(enhanced_image, orientation_map, frequency_map, energy_map):
    """
    Extract STFT-based texture features for MTCC descriptors
    Features:
        - Io: Orientation image from STFT analysis (normalized 0-pi)
        - If: Frequency image from STFT analysis (normalized 0-1)
        - Ie: Energy image (logarithmic) from STFT analysis (normalized 0-1)

    These replace traditional minutiae angular information in cylinder codes.
    """
    energy_map_norm = normalize_image_global(energy_map) / 255.0

    return {
        'orientation': orientation_map, # radians (0 to pi)
        'frequency': frequency_map,     # normalized (0 to 1)
        'energy': energy_map_norm       # normalized (0 to 1)
    }

def estimate_ridge_frequency_curved_regions(image, orientation_map):
    """
    Ridge frequency estimation using curved regions (Paper 5)
    """
    h, w = image.shape
    frequency_map_curved = np.zeros_like(image, dtype=np.float32)
    block_size = MTCC_PARAMETERS['stft_window_size']
    step = MTCC_PARAMETERS['stft_window_size'] - MTCC_PARAMETERS['stft_overlap']

    # Ensure image is float32 for processing
    image_float = image.astype(np.float32)

    for r in range(0, h, step):
        for c in range(0, w, step):
            cx, cy = c + block_size // 2, r + block_size // 2

            if not (0 <= cy < h and 0 <= cx < w):
                continue

            local_orientation = orientation_map[cy, cx]

            num_points = MTCC_PARAMETERS['curved_region_points']
            profile = np.zeros(num_points)

            sample_angle = local_orientation + np.pi / 2

            for i in range(num_points):
                offset = (i - num_points // 2)
                px = int(cx + offset * np.cos(sample_angle))
                py = int(cy + offset * np.sin(sample_angle))

                if 0 <= py < h and 0 <= px < w:
                    profile[i] = image_float[py, px]
                else:
                    profile[i] = 0

            extrema_indices = []
            for i in range(1, num_points - 1):
                if profile[i] > profile[i-1] and profile[i] > profile[i+1]:
                    extrema_indices.append(i)
                elif profile[i] < profile[i-1] and profile[i] < profile[i+1]:
                    extrema_indices.append(i)

            if len(extrema_indices) < 2:
                frequency_map_curved[r : r + block_size, c : c + block_size] = 0
                continue

            inter_extrema_distances = []
            for i in range(len(extrema_indices) - 1):
                inter_extrema_distances.append(extrema_indices[i+1] - extrema_indices[i])

            if len(inter_extrema_distances) > 0:
                median_wavelength = np.median(inter_extrema_distances)

                if median_wavelength > 0:
                    local_frequency = 1.0 / median_wavelength
                    frequency_map_curved[r : r + block_size, c : c + block_size] = local_frequency
                else:
                    frequency_map_curved[r : r + block_size, c : c + block_size] = 0
            else:
                frequency_map_curved[r : r + block_size, c : c + block_size] = 0

    frequency_map_curved = cv2.resize(frequency_map_curved, (w, h), interpolation=cv2.INTER_LINEAR)
    frequency_map_curved = gaussian(frequency_map_curved, sigma=2)
    
    frequency_map_curved = np.clip(frequency_map_curved, 0.05, 0.25)
    frequency_map_curved = normalize_image_global(frequency_map_curved) / 255.0

    return frequency_map_curved

# --- 3. MINUTIAE EXTRACTION WITH QUALITY ASSESSMENT ---

def binarize_and_thin(image, mask):
    """
    Binarize and thin the enhanced fingerprint image.
    Uses adaptive thresholding and skeletonization.
    """
    # Apply mask before binarization to avoid processing background noise
    masked_img = image.copy()
    
    # Ensure masked_img is 8-bit unsigned integer before applying threshold
    if masked_img.dtype != np.uint8:
        masked_img = masked_img.astype(np.uint8)

    masked_img[~mask] = 255 # Set background to white for binarization

    # Adaptive thresholding (local binarization)
    binary_img = cv2.adaptiveThreshold(masked_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply mask again in case adaptive thresholding picked up something outside
    binary_img[~mask] = 0 # Ensure background is truly black

    # Convert to boolean for skimage.skeletonize
    binary_skimage = binary_img == 255

    # Skeletonization (thinning)
    skeleton = skeletonize(binary_skimage)

    return skeleton.astype(np.uint8) * 255 # Convert back to 0/255

def enhanced_minutiae_extraction(binary_skeleton, mask, orientation_map):
    """
    Extract minutiae using Crossing Number algorithm with quality assessment.
    """
    minutiae_list = []
    h, w = binary_skeleton.shape

    # Crossing Number (CN) map
    cn_map = np.zeros_like(binary_skeleton, dtype=np.uint8)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if binary_skeleton[y, x] == 255: # Only consider ridge pixels
                block = binary_skeleton[y-1:y+2, x-1:x+2] // 255 # Convert to 0/1
                
                cn = 0
                neighbors = [block[0,1], block[0,2], block[1,2], block[2,2],
                             block[2,1], block[2,0], block[1,0], block[0,0], block[0,1]] # P1 to P8, then P1 again
                for i in range(8):
                    cn += abs(neighbors[i] - neighbors[i+1])
                cn *= 0.5
                
                cn_map[y, x] = cn

                if cn == 1: # Ridge ending
                    minutiae_list.append(Minutia(x=x, y=y, angle=0, type=1, quality=0)) # Angle & quality placeholder
                elif cn == 3: # Bifurcation
                    minutiae_list.append(Minutia(x=x, y=y, angle=0, type=2, quality=0)) # Angle & quality placeholder

    # 1. Remove spurious minutiae (short branches, bridges)
    filtered_minutiae = []
    min_dist_sq = MTCC_PARAMETERS['minutiae_min_dist']**2
    for m in minutiae_list:
        if not mask[m.y, m.x] or \
           m.y < MTCC_PARAMETERS['minutiae_border_dist'] or \
           m.y > h - MTCC_PARAMETERS['minutiae_border_dist'] or \
           m.x < MTCC_PARAMETERS['minutiae_border_dist'] or \
           m.x > w - MTCC_PARAMETERS['minutiae_border_dist']:
            continue
        
        is_isolated = True
        for fm in filtered_minutiae:
            dist_sq = (m.x - fm.x)**2 + (m.y - fm.y)**2
            if dist_sq < min_dist_sq:
                is_isolated = False
                break
        if is_isolated:
            filtered_minutiae.append(m)

    # 2. Assign orientation and quality
    final_minutiae = []
    for m in filtered_minutiae:
        # Placeholder for final quality value (e.g., from 0.0 to 1.0)
        # Quality based on mask distance and local orientation map
        quality = 1.0
        if not mask[m.y, m.x]: quality = 0 # Must be in mask
        
        # Check coherence / clarity in a small window around the minutia
        # A simple check: if the standard deviation of orientation in a small neighborhood is high, quality is low.
        patch_size = 5 # small window around minutia
        y_start, y_end = max(0, m.y - patch_size // 2), min(h, m.y + patch_size // 2 + 1)
        x_start, x_end = max(0, m.x - patch_size // 2), min(w, m.x + patch_size // 2 + 1)
        
        local_orientation_patch = orientation_map[y_start:y_end, x_start:x_end]
        if local_orientation_patch.size > 0 and np.std(local_orientation_patch) > 0.5: # high std means low coherence
            quality *= 0.5

        final_quality = quality * (1 - (np.linalg.norm([m.x-w/2, m.y-h/2]) / np.linalg.norm([w/2, h/2]))) # Closer to center is better
        final_quality = np.clip(final_quality, 0.0, 1.0)
        
        if final_quality >= MTCC_PARAMETERS['minutiae_quality_thresh']:
            final_minutiae.append(Minutia(x=m.x, y=m.y, angle=orientation_map[m.y, m.x], type=m.type, quality=final_quality))

    return final_minutiae

# --- 4. MTCC DESCRIPTOR GENERATION (Core Innovation) ---

class MTCCCylinder:
    """Represents an MTCC descriptor for a single minutia."""
    def __init__(self, minutia, radius, ns, nd):
        self.minutia = minutia
        self.radius = radius
        self.ns = ns # Number of spatial sectors
        self.nd = nd # Number of directional sectors

        # Cylinder structure: (spatial_sector, directional_sector)
        # Each cell stores a vector of features: [spatial_contrib, freq_contrib, energy_contrib, orient_contrib]
        self.cells = np.zeros((ns, nd, 4), dtype=np.float32) # last dim: S, F, E, O (for MCCco)

    def get_cell_coords(self, spatial_idx, directional_idx):
        """Calculates the approximate center coordinates of a cell in image space."""
        r_inner = self.radius * spatial_idx / self.ns
        r_outer = self.radius * (spatial_idx + 1) / self.ns
        r_avg = (r_inner + r_outer) / 2.0

        angle_sector_width = 2 * np.pi / self.nd
        angle_avg = (directional_idx + 0.5) * angle_sector_width

        # Global angle relative to image x-axis
        # Minutia.angle is 0-pi. We add relative angle (0-2pi) to get a global 0-2pi angle.
        # This implicitly assumes a reference direction for the minutia.
        global_angle = self.minutia.angle + angle_avg # The minutia angle is the reference for the cylinder.

        cell_x = int(self.minutia.x + r_avg * np.cos(global_angle))
        cell_y = int(self.minutia.y + r_avg * np.sin(global_angle))
        return cell_x, cell_y, r_avg, global_angle

    def set_cell_value(self, s_idx, d_idx, spatial_contrib, freq_contrib, energy_contrib, orientation_contrib):
        """Sets the values for a specific cell."""
        self.cells[s_idx, d_idx] = [spatial_contrib, freq_contrib, energy_contrib, orientation_contrib]

def create_mtcc_cylinders(minutiae_list, texture_maps, radius=65):
    """
    Generate MTCC (Minutiae Texture Cylinder Codes) descriptors.
    """
    cylinders = []
    ns = MTCC_PARAMETERS['spatial_sectors']
    nd = MTCC_PARAMETERS['angular_sectors']
    sigma_s = MTCC_PARAMETERS['gaussian_spatial_sigma']

    img_h, img_w = texture_maps['orientation'].shape

    for minutia in minutiae_list:
        cylinder = MTCCCylinder(minutia, radius, ns, nd)

        for s_idx in range(ns):
            for d_idx in range(nd):
                cell_x, cell_y, r_avg, cell_global_angle = cylinder.get_cell_coords(s_idx, d_idx)

                if not (0 <= cell_y < img_h and 0 <= cell_x < img_w):
                    cylinder.set_cell_value(s_idx, d_idx, 0, 0, 0, 0)
                    continue

                spatial_contrib = np.exp(-(r_avg**2) / (2 * sigma_s**2))

                cell_freq = texture_maps['frequency'][cell_y, cell_x]
                cell_energy = texture_maps['energy'][cell_y, cell_x]
                cell_orient = texture_maps['orientation'][cell_y, cell_x] # 0 to pi

                freq_contrib = cell_freq
                energy_contrib = cell_energy
                orientation_contrib = cell_orient

                cylinder.set_cell_value(s_idx, d_idx, spatial_contrib, freq_contrib, energy_contrib, orientation_contrib)

        cylinders.append(cylinder)
    return cylinders

# --- 5. MATCHING WITH MULTIPLE DISTANCE METRICS ---

def compute_cylinder_distance(cylinder1, cylinder2, feature_type='MCCco'):
    """
    Computes distance between two MTCC cylinders based on specified feature type.
    """
    if cylinder1.ns != cylinder2.ns or cylinder1.nd != cylinder2.nd:
        return float('inf')

    total_distance_sq = 0
    valid_cells = 0

    for s_idx in range(cylinder1.ns):
        for d_idx in range(cylinder1.nd):
            contrib1 = cylinder1.cells[s_idx, d_idx]
            contrib2 = cylinder2.cells[s_idx, d_idx]

            if np.all(contrib1 == 0) or np.all(contrib2 == 0):
                continue

            valid_cells += 1

            if feature_type == 'MCCf' or feature_type == 'MCCcf': # Frequency-based
                distance = abs(contrib1[1] - contrib2[1])
            elif feature_type == 'MCCe' or feature_type == 'MCCce': # Energy-based
                distance = abs(contrib1[2] - contrib2[2])
            elif feature_type == 'MCCco': # Cell-centered orientation
                distance = double_angle_distance(contrib1[3], contrib2[3])
            else:
                distance = np.linalg.norm(contrib1 - contrib2)

            total_distance_sq += distance**2

    if valid_cells == 0:
        return float('inf')

    return np.sqrt(total_distance_sq / valid_cells)

def mtcc_matching(cylinders1, cylinders2, feature_type='MCCco'):
    """
    MTCC matching using Local Similarity Sort with Relaxation (LSSR)
    """
    if not cylinders1 or not cylinders2:
        return 0.0

    num_m1 = len(cylinders1)
    num_m2 = len(cylinders2)
    similarity_matrix = np.full((num_m1, num_m2), float('inf'))

    for i in range(num_m1):
        for j in range(num_m2):
            similarity_matrix[i, j] = compute_cylinder_distance(cylinders1[i], cylinders2[j], feature_type)

    candidate_pairs = []
    for i in range(num_m1):
        for j in range(num_m2):
            if similarity_matrix[i, j] < float('inf'):
                candidate_pairs.append((i, j, similarity_matrix[i, j]))

    candidate_pairs.sort(key=lambda x: x[2])

    N_INITIAL_PAIRS = min(50, len(candidate_pairs))
    top_pairs = candidate_pairs[:N_INITIAL_PAIRS]
    
    max_dist = np.max(similarity_matrix[similarity_matrix != float('inf')]) if np.any(similarity_matrix != float('inf')) else 1.0
    if max_dist == 0: max_dist = 1.0

    match_scores = {}
    for m1_idx, m2_idx, dist in top_pairs:
        match_scores[(m1_idx, m2_idx)] = 1.0 - (dist / max_dist)

    def get_k_nearest_neighbors(minutia_idx, cylinders, k=5):
        current_minutia = cylinders[minutia_idx].minutia
        distances = []
        for idx, cyl in enumerate(cylinders):
            if idx == minutia_idx: continue
            dist = np.sqrt((current_minutia.x - cyl.minutia.x)**2 + (current_minutia.y - cyl.minutia.y)**2)
            distances.append((dist, idx))
        distances.sort(key=lambda x: x[0])
        return [idx for d, idx in distances[:k]]

    current_match_scores = match_scores.copy()
    relaxation_iterations = MTCC_PARAMETERS['relaxation_iterations']
    decay_factor = MTCC_PARAMETERS['relaxation_decay_factor']

    for iter_count in range(relaxation_iterations):
        next_match_scores = current_match_scores.copy()
        
        for (m1_i, m2_j), score_ij in current_match_scores.items():
            if score_ij == 0: continue

            neighbors1 = get_k_nearest_neighbors(m1_i, cylinders1, k=5)
            neighbors2 = get_k_nearest_neighbors(m2_j, cylinders2, k=5)

            support = 0
            for m1_k_idx in neighbors1:
                for m2_l_idx in neighbors2:
                    if (m1_k_idx, m2_l_idx) in current_match_scores:
                        score_kl = current_match_scores[(m1_k_idx, m2_l_idx)]
                        
                        dist_ik = np.sqrt((cylinders1[m1_i].minutia.x - cylinders1[m1_k_idx].minutia.x)**2 +
                                          (cylinders1[m1_i].minutia.y - cylinders1[m1_k_idx].minutia.y)**2)
                        dist_jl = np.sqrt((cylinders2[m2_j].minutia.x - cylinders2[m2_l_idx].minutia.x)**2 +
                                          (cylinders2[m2_j].minutia.y - cylinders2[m2_l_idx].minutia.y)**2)
                        
                        angle_ik = np.arctan2(cylinders1[m1_k_idx].minutia.y - cylinders1[m1_i].minutia.y,
                                              cylinders1[m1_k_idx].minutia.x - cylinders1[m1_i].minutia.x)
                        angle_jl = np.arctan2(cylinders2[m2_l_idx].minutia.y - cylinders2[m2_j].minutia.y,
                                              cylinders2[m2_l_idx].minutia.x - cylinders2[m2_j].minutia.x)
                        
                        angle_ik = (angle_ik + 2 * np.pi) % (2 * np.pi)
                        angle_jl = (angle_jl + 2 * np.pi) % (2 * np.pi)

                        dist_compatibility = np.exp(-abs(dist_ik - dist_jl) / 10.0)
                        angle_compatibility = np.exp(-orientation_diff_2pi(angle_ik, angle_jl) / (np.pi/8))
                        
                        compatibility_factor = dist_compatibility * angle_compatibility
                        
                        support += score_kl * compatibility_factor

            next_match_scores[(m1_i, m2_j)] = score_ij * (1 + decay_factor * support)
            
        current_match_scores = next_match_scores
        
        max_score = max(current_match_scores.values()) if current_match_scores else 1.0
        if max_score > 0:
            for key in current_match_scores:
                current_match_scores[key] /= max_score
        
    FINAL_SCORE_THRESHOLD = 0.5

    final_score = 0
    for (m1_idx, m2_idx), score in current_match_scores.items():
        if score > FINAL_SCORE_THRESHOLD:
            final_score += score
            
    max_possible_final_score = min(num_m1, num_m2)
    if max_possible_final_score > 0:
        final_score /= max_possible_final_score

    return final_score

# --- 6. COMPREHENSIVE EVALUATION FRAMEWORK ---

def read_fvc_dataset_paths(base_fvc_path, database_name='DB1_A', num_subjects=100, num_impressions=8, image_extension='.tif'):
    """
    Reads image paths from a standard FVC dataset structure.

    Args:
        base_fvc_path (str): The root directory where FVC databases are located (e.g., "FVC2002").
                             If your structure is like `my_data/101_1.tif`, then `base_fvc_path`
                             would be `my_data` and `database_name` would be an empty string or None.
        database_name (str): The specific database subfolder to read (e.g., "DB1_A").
                             Set to '' or None if images are directly in `base_fvc_path`.
        num_subjects (int): Number of subjects in the database (e.g., 100 for FVC2002).
        num_impressions (int): Number of impressions per subject (e.g., 8).
        image_extension (str): The file extension of the images (e.g., ".tif", ".png", ".bmp").

    Returns:
        list: A list of full paths to the image files.
    """
    image_files = []
    
    if database_name: # If a specific database subfolder is given
        db_path = os.path.join(base_fvc_path, database_name)
    else: # If images are directly in the base_fvc_path
        db_path = base_fvc_path

    if not os.path.isdir(db_path):
        print(f"Error: Dataset path not found: {db_path}. Please check the path and database_name.")
        return []

    # Attempt to read filenames based on FVC naming convention
    # e.g., 101_1.tif, 101_2.tif, ..., 101_8.tif
    print(f"Attempting to read images from: {db_path}")
    for s_idx in range(1, num_subjects + 1):
        for i_idx in range(1, num_impressions + 1):
            image_filename = f"{s_idx:03d}_{i_idx}{image_extension}"
            image_path = os.path.join(db_path, image_filename)
            
            if os.path.exists(image_path):
                image_files.append(image_path)
            # else:
            #     print(f"Debug: Image not found: {image_path}") # Uncomment for debugging missing files
    
    if not image_files:
        print(f"No images found matching FVC naming convention ({num_subjects} subjects, {num_impressions} impressions, {image_extension}).")
        print("Falling back to reading all images in the directory.")
        # Fallback: Read all image files in the directory if FVC convention not found
        for root, _, files in os.walk(db_path):
            for file in files:
                if file.lower().endswith(image_extension):
                    image_files.append(os.path.join(root, file))
        if not image_files:
            print(f"Still no {image_extension} images found in {db_path}. Ensure the path and extension are correct.")

    return sorted(image_files) # Sort to ensure consistent processing order

def evaluate_fvc_dataset(image_files, method='MCCco'):
    """
    FVC2002/2004 evaluation protocol.
    
    Args:
        image_files (list): A list of full paths to the fingerprint images.
        method (str): The MTCC variant to use for matching (e.g., 'MCCco').

    Returns:
        tuple: (eer, genuine_scores, impostor_scores)
    """
    print(f"\n--- Starting MTCC Matching Evaluation for method: {method} ---")
    
    templates = {}
    
    print(f"Processing {len(image_files)} images...")
    for idx, image_path in enumerate(image_files):
        try:
            img, mask = load_and_preprocess(image_path)
            enhanced, orient_map, freq_map_stft, energy_map, coherence = stft_enhancement_analysis(img, mask) 
            enhanced_final = curved_gabor_enhancement(enhanced, orient_map, freq_map_stft, coherence)
            
            freq_map_curved = estimate_ridge_frequency_curved_regions(enhanced_final, orient_map)
            
            texture_maps = extract_texture_features(enhanced_final, orient_map, freq_map_curved, energy_map)
            skeleton = binarize_and_thin(enhanced_final, mask)
            minutiae = enhanced_minutiae_extraction(skeleton, mask, orient_map)
            
            cylinders = create_mtcc_cylinders(minutiae, texture_maps)
            
            templates[image_path] = cylinders
            print(f"  Processed [{idx+1}/{len(image_files)}] {os.path.basename(image_path)}. Minutiae: {len(minutiae)}, Cylinders: {len(cylinders)}")
        except Exception as e:
            print(f"  Error processing {os.path.basename(image_path)}: {e}")
            templates[image_path] = []
    
    genuine_scores = []
    impostor_scores = []

    image_ids = list(templates.keys())
    
    for i in range(len(image_ids)):
        path1 = image_ids[i]
        
        # Parse subject ID from filename (e.g., "101_1.tif" -> "101")
        try:
            filename1 = os.path.basename(path1)
            # Assuming file names like `XXX_Y.ext` or `XXX_YY.ext` or `XXX.ext` if no impressions
            # More robust parsing for FVC-like naming
            subj1_part = filename1.split('_')[0]
            if not subj1_part.isdigit(): # If not purely numeric (e.g., "finger_01_1.png")
                subj1_part = filename1.split('_')[-2] # Try second to last part
            subj1 = int(subj1_part)
        except (ValueError, IndexError):
            print(f"Warning: Could not robustly parse subject ID from {filename1}. Using string comparison for subject ID. This may affect genuine/impostor classification.")
            subj1 = filename1.split(os.sep)[-2] if os.sep in filename1 else "unknown_subj_1" # Fallback to folder name or generic
            
        for j in range(i + 1, len(image_ids)):
            path2 = image_ids[j]
            
            try:
                filename2 = os.path.basename(path2)
                subj2_part = filename2.split('_')[0]
                if not subj2_part.isdigit():
                    subj2_part = filename2.split('_')[-2]
                subj2 = int(subj2_part)
            except (ValueError, IndexError):
                subj2 = filename2.split(os.sep)[-2] if os.sep in filename2 else "unknown_subj_2"
                
            if not templates[path1] or not templates[path2]:
                continue

            score = mtcc_matching(templates[path1], templates[path2], method)
            
            if subj1 == subj2:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    print(f"  Total Genuine Scores: {len(genuine_scores)}")
    print(f"  Total Impostor Scores: {len(impostor_scores)}")

    eer = calculate_eer(genuine_scores, impostor_scores)
    print(f"--- {method} EER: {eer:.2f}% ---")
    return eer, genuine_scores, impostor_scores

def calculate_eer(genuine_scores, impostor_scores):
    """Calculate Equal Error Rate (EER)."""
    if not genuine_scores or not impostor_scores:
        return float('nan')

    thresholds = np.linspace(0, 1, 1000)
    frr = []
    far = []

    for t in thresholds:
        frr.append(np.sum(np.array(genuine_scores) < t) / len(genuine_scores))
        far.append(np.sum(np.array(impostor_scores) >= t) / len(impostor_scores))

    min_diff_idx = np.argmin(np.abs(np.array(frr) - np.array(far)))
    eer_value = (frr[min_diff_idx] + far[min_diff_idx]) / 2 * 100

    return eer_value

def plot_det_curves(results):
    """Plot Detection Error Tradeoff (DET) curves."""
    plt.figure(figsize=(10, 8))
    for method, res in results.items():
        genuine_scores = res['genuine']
        impostor_scores = res['impostor']
        
        if not genuine_scores or not impostor_scores:
            continue

        thresholds = np.linspace(0, 1, 1000)
        frr = []
        far = []
        for t in thresholds:
            frr.append(np.sum(np.array(genuine_scores) < t) / len(genuine_scores))
            far.append(np.sum(np.array(impostor_scores) >= t) / len(impostor_scores))

        plt.plot(far, frr, label=f'{method} (EER: {res["EER"]:.2f}%)')

    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.title('DET Curves')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               ['0.1%', '1%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    plt.yticks([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               ['0.1%', '1%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    plt.legend()
    plt.tight_layout()
    plt.savefig('det_curves.png')
    plt.show()

def plot_roc_curves(results):
    """Plot Receiver Operating Characteristic (ROC) curves."""
    plt.figure(figsize=(10, 8))
    for method, res in results.items():
        genuine_scores = res['genuine']
        impostor_scores = res['impostor']
        
        if not genuine_scores or not impostor_scores:
            continue

        thresholds = np.linspace(0, 1, 1000)
        tpr = []
        fpr = []

        for t in thresholds:
            tpr.append(np.sum(np.array(genuine_scores) >= t) / len(genuine_scores))
            fpr.append(np.sum(np.array(impostor_scores) >= t) / len(impostor_scores))

        plt.plot(fpr, tpr, label=f'{method} (EER: {res["EER"]:.2f}%)')

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves')
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.legend()
    plt.tight_layout()
    plt.savefig('roc_curves.png')
    plt.show()

def generate_performance_table(results):
    """Generate a table of EER results."""
    print("\n--- Performance Summary ---")
    print("{:<10} {:<10}".format("Method", "EER (%)"))
    print("-" * 20)
    for method, res in results.items():
        print("{:<10} {:<10.2f}".format(method, res['EER']))
    print("-" * 20)


# --- 7. VISUALIZATION AND DEBUGGING ---

def visualize_mtcc_pipeline(image_path, save_debug=True):
    """
    Comprehensive visualization of MTCC pipeline.
    """
    print(f"\n--- Visualizing Pipeline for {image_path} ---")
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Pipeline steps
    img, mask = load_and_preprocess(image_path)
    enhanced_stft, orient_map, freq_map_stft, energy_map, coherence = stft_enhancement_analysis(img, mask) 
    enhanced_curved_gabor = curved_gabor_enhancement(enhanced_stft, orient_map, freq_map_stft, coherence)
    freq_map_curved = estimate_ridge_frequency_curved_regions(enhanced_curved_gabor, orient_map)
    texture_maps = extract_texture_features(enhanced_curved_gabor, orient_map, freq_map_curved, energy_map)

    skeleton = binarize_and_thin(enhanced_curved_gabor, mask)
    minutiae = enhanced_minutiae_extraction(skeleton, mask, orient_map)
    cylinders = create_mtcc_cylinders(minutiae, texture_maps)

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'MTCC Pipeline Visualization for {os.path.basename(image_path)}', fontsize=16)

    # Row 1
    axes[0, 0].imshow(original_img, cmap='gray'); axes[0, 0].set_title('1. Original'); axes[0, 0].axis('off')
    axes[0, 1].imshow(img, cmap='gray'); axes[0, 1].set_title('2. Segmented'); axes[0, 1].axis('off')
    axes[0, 2].imshow(enhanced_stft, cmap='gray'); axes[0, 2].set_title('3. STFT Enhanced (Input)'); axes[0, 2].axis('off')
    axes[0, 3].imshow(enhanced_curved_gabor, cmap='gray'); axes[0, 3].set_title('4. Curved Gabor Enhanced'); axes[0, 3].axis('off')

    # Row 2 (Maps)
    axes[1, 0].imshow(orient_map, cmap='hsv', vmin=0, vmax=np.pi); axes[1, 0].set_title('5. Orientation Map (rad)'); axes[1, 0].axis('off')
    axes[1, 1].imshow(freq_map_curved, cmap='viridis'); axes[1, 1].set_title('6. Frequency Map (curved)'); axes[1, 1].axis('off')
    axes[1, 2].imshow(energy_map, cmap='hot'); axes[1, 2].set_title('7. Energy Map'); axes[1, 2].axis('off')
    axes[1, 3].imshow(coherence, cmap='gray'); axes[1, 3].set_title('8. Coherence Map'); axes[1, 3].axis('off')

    # Row 3 (Minutiae and Descriptors)
    axes[2, 0].imshow(skeleton, cmap='gray'); axes[2, 0].set_title('9. Binarized/Thinned'); axes[2, 0].axis('off')
    
    minutiae_img = skeleton.copy()
    minutiae_img = cv2.cvtColor(minutiae_img, cv2.COLOR_GRAY2BGR)
    for m in minutiae:
        color = (0, 0, 255) if m.type == 1 else (255, 0, 0)
        cv2.circle(minutiae_img, (m.x, m.y), 3, color, -1)
        length = 10
        end_x = int(m.x + length * np.cos(m.angle))
        end_y = int(m.y + length * np.sin(m.angle))
        cv2.line(minutiae_img, (m.x, m.y), (end_x, end_y), color, 1)
    axes[2, 1].imshow(minutiae_img); axes[2, 1].set_title(f'10. Minutiae ({len(minutiae)})'); axes[2, 1].axis('off')

    visualize_mtcc_cylinders(axes[2, 2], minutiae, cylinders, texture_maps, image_path)
    axes[2, 2].set_title('11. MTCC Cylinders (Conceptual)'); axes[2, 2].axis('off')

    axes[2, 3].text(0.5, 0.5, '12. Reserved / Additional Info', ha='center', va='center', fontsize=12, color='gray')
    axes[2, 3].axis('off')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_debug:
        plt.savefig(f'pipeline_visualization_{os.path.basename(image_path)}.png')
    plt.show()

def visualize_mtcc_cylinders(ax, minutiae, cylinders, texture_maps, image_path):
    """
    Visualize MTCC cylinder contents showing texture feature distributions.
    """
    if not minutiae or not cylinders:
        ax.text(0.5, 0.5, 'No minutiae/cylinders to visualize.', ha='center', va='center', fontsize=10, color='red')
        return

    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        ax.text(0.5, 0.5, 'Background image not loaded.', ha='center', va='center', fontsize=10, color='red')
        original_img = np.zeros((200,200))
    
    vis_img = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8)

    minutia_to_vis = minutiae[0]
    cylinder_to_vis = None
    for cyl in cylinders:
        if cyl.minutia == minutia_to_vis:
            cylinder_to_vis = cyl
            break
    
    if cylinder_to_vis is None:
        ax.text(0.5, 0.5, 'Cylinder for chosen minutia not found.', ha='center', va='center', fontsize=10, color='red')
        return

    ns = cylinder_to_vis.ns
    nd = cylinder_to_vis.nd
    radius = cylinder_to_vis.radius
    
    cv2.circle(vis_img, (minutia_to_vis.x, minutia_to_vis.y), 5, (0, 255, 255), -1)

    cmap = plt.cm.hsv

    for s_idx in range(ns):
        r_inner = radius * s_idx / ns
        r_outer = radius * (s_idx + 1) / ns
        
        for d_idx in range(nd):
            angle_sector_width = 2 * np.pi / nd
            angle_start = (d_idx * angle_sector_width) + (minutia_to_vis.angle)
            angle_end = ((d_idx + 1) * angle_sector_width) + (minutia_to_vis.angle)

            orientation_value = cylinder_to_vis.cells[s_idx, d_idx, 3]
            
            color_norm = orientation_value / np.pi
            rgb_color = cmap(color_norm)[:3]
            rgb_color_bgr = (int(rgb_color[2]*255), int(rgb_color[1]*255), int(rgb_color[0]*255))

            center = (minutia_to_vis.x, minutia_to_vis.y)
            
            num_segments = 20
            points = [center]
            for i in range(num_segments + 1):
                seg_angle = angle_start + (i / num_segments) * (angle_end - angle_start)
                px = int(center[0] + r_outer * np.cos(seg_angle))
                py = int(center[1] + r_outer * np.sin(seg_angle))
                points.append((px, py))
            points = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(vis_img, [points], rgb_color_bgr)

            cv2.ellipse(vis_img, center, (int(r_outer), int(r_outer)), 0, np.degrees(angle_start), np.degrees(angle_end), (200,200,200), 1)
            cv2.ellipse(vis_img, center, (int(r_inner),int(r_inner)), 0, np.degrees(angle_start), np.degrees(angle_end), (200,200,200), 1)

    ax.imshow(vis_img)
    ax.set_title(f'MTCC for Minutia at ({minutia_to_vis.x},{minutia_to_vis.y})')


# --- New Function to run full evaluation ---
def run_mtcc_evaluation(dataset_folder_path, database_name='DB1_A', num_subjects=100, num_impressions=8, image_extension='.tif', run_visualization=True):
    """
    Runs the complete MTCC fingerprint recognition system evaluation for a given dataset.

    Args:
        dataset_folder_path (str): The root path to your FVC-like dataset.
                                   e.g., if your images are in `FVC2002/DB1_A/`, then pass `FVC2002`.
                                   If your images are directly in `my_dataset/`, then pass `my_dataset`
                                   and set `database_name=''`.
        database_name (str): The subfolder name for the database (e.g., 'DB1_A').
                             Set to '' or None if images are directly in `dataset_folder_path`.
        num_subjects (int): Expected number of subjects (fingerprints) in the dataset.
        num_impressions (int): Expected number of impressions per subject.
        image_extension (str): The file extension of the fingerprint images (e.g., '.tif', '.png').
        run_visualization (bool): Whether to run the visualization pipeline for a sample image.
    """
    print(f"Starting MTCC Evaluation for dataset: {dataset_folder_path}/{database_name}")

    # Step 1: Read all image paths from the dataset
    all_image_paths = read_fvc_dataset_paths(
        base_fvc_path=dataset_folder_path,
        database_name=database_name,
        num_subjects=num_subjects,
        num_impressions=num_impressions,
        image_extension=image_extension
    )

    if not all_image_paths:
        print("No image paths found. Please check your dataset path and parameters.")
        return

    # Step 2: Run visualization for a sample image (optional)
    if run_visualization and all_image_paths:
        sample_image_path = all_image_paths[0] # Pick the first image for visualization
        visualize_mtcc_pipeline(sample_image_path, save_debug=True)

    # Step 3: Run the full evaluation for specified MTCC variants
    methods_to_evaluate = ['MCCf', 'MCCe', 'MCCco']
    results = {}
    
    for method in methods_to_evaluate:
        print(f"\nEvaluating with {method} method...")
        eer, genuine, impostor = evaluate_fvc_dataset(all_image_paths, method)
        results[method] = {'EER': eer, 'genuine': genuine, 'impostor': impostor}
        print(f"Final {method} EER: {eer:.2f}%")
    
    # Step 4: Generate performance plots and table
    if any(res['genuine'] and res['impostor'] for res in results.values()):
        plot_det_curves(results)
        plot_roc_curves(results)
        generate_performance_table(results)
    else:
        print("\nNot enough genuine/impostor scores to generate plots/table.")
        print("This could be due to a small dataset or errors during processing.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Create Mock FVC Dataset for Demonstration ---
    fvc_mock_base_path = R"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002"
    mock_db_name = "DB1_A"
    mock_num_subjects = 3
    mock_num_impressions = 2
    mock_image_ext = '.tif'

    mock_db_full_path = os.path.join(fvc_mock_base_path, mock_db_name)
    if not os.path.exists(mock_db_full_path):
        os.makedirs(mock_db_full_path)
        print(f"Created mock FVC directory: {mock_db_full_path}")
        for s_idx in range(1, mock_num_subjects + 1):
            for i_idx in range(1, mock_num_impressions + 1):
                dummy_img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
                img_filename = f"{s_idx:03d}_{i_idx}{mock_image_ext}"
                cv2.imwrite(os.path.join(mock_db_full_path, img_filename), dummy_img)
        print(f"Created {mock_num_subjects * mock_num_impressions} dummy images in {mock_db_full_path}.")
    else:
        print(f"Mock FVC directory already exists: {mock_db_full_path}")
    
    # --- How to use the new `run_mtcc_evaluation` function ---
    # To run the full evaluation, simply call this function with your dataset path.
    # IMPORTANT: Adjust the parameters (database_name, num_subjects, num_impressions, image_extension)
    #            to match your *actual* FVC dataset's structure and size.

    # Example for running with the mock data:
    run_mtcc_evaluation(
        dataset_folder_path=fvc_mock_base_path,
        database_name=mock_db_name,
        num_subjects=mock_num_subjects,
        num_impressions=mock_num_impressions,
        image_extension=mock_image_ext,
        run_visualization=True
    )

    # Example if your images are directly in `my_dataset_folder` (no subfolder like DB1_A):
    # create a dummy flat dataset for this example
    # flat_dataset_path = "Flat_Mock_Data"
    # if not os.path.exists(flat_dataset_path):
    #     os.makedirs(flat_dataset_path)
    #     print(f"Created mock flat directory: {flat_dataset_path}")
    #     for s_idx in range(1, 4): # 3 subjects
    #         for i_idx in range(1, 3): # 2 impressions
    #             dummy_img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    #             # Naming: subjID_impID.ext
    #             cv2.imwrite(os.path.join(flat_dataset_path, f"{s_idx:03d}_{i_idx}.png"), dummy_img)
    # run_mtcc_evaluation(
    #     dataset_folder_path=flat_dataset_path,
    #     database_name='', # Important: Set to empty string if no subfolder
    #     num_subjects=3,   # Adjust as per your flat dataset
    #     num_impressions=2,# Adjust as per your flat dataset
    #     image_extension='.png',
    #     run_visualization=False
    # )