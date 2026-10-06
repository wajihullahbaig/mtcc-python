import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from skimage.morphology import skeletonize, thin # skeletonize is generally better than thin

# --- Configuration Parameters (Based on papers and common practices) ---
# General image processing
BLOCK_SIZE = 16 # For segmentation, general block processing
GAUSSIAN_KERNEL_SIZE = 5 # For smoothing filters
SMOOTH_SIGMA = 1 # For general Gaussian smoothing of features

# Gabor Filter Parameters (Mudegaonkar & Adgaonkar / common Gabor usage)
NUM_GABOR_ORIENTATIONS = 8 # 0, 22.5, ..., 157.5 degrees
GABOR_FREQUENCIES = [0.1] # A typical ridge frequency (1/wavelength). If multiple, average results.
GABOR_KERNEL_SIZE = 31 # Must be odd for cv2.getGaborKernel

# STFT Analysis Parameters (Shimna & Neethu / Baig et al.)
STFT_WINDOW_SIZE = 32 # Square window size for STFT blocks
STFT_OVERLAP = STFT_WINDOW_SIZE // 2 # 50% overlap typically

# MTCC Parameters (Baig et al., Table IV and Section V)
R_CYLINDER = 65 # Radius of the cylinder around the central minutia
NS = 18 # Number of spatial sectors along each axis (for Ns x Ns grid)
# ND = 5 (Number of directional sectors) is for original MCC angular bins.
# For MTCC, feature type replaces this dimension.
DELTA_S = 2 * R_CYLINDER / NS # Spatial cell size (distance unit)

# Gaussian parameters for spatial/directional contributions (Baig et al., Table IV)
SIGMA_S = 6 # sigma_R for Gs
SIGMA_D = np.deg2rad(36) # sigma_phi for GD (converted to radians)

# --- Helper Functions ---

def load_image(image_path, gray=True):
    """Loads an image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return img

def normalize_image(img, mean=128, variance=128): # Typical range for 8-bit images
    """Normalizes image to a desired mean and variance."""
    # From Mudegaonkar & Adgaonkar, Eq 11, applied *pixel-wise* in *each sector*.
    # For initial global normalization, we use a simpler form, then sector-wise can be applied later.
    norm_img = img.copy().astype(np.float32)
    current_mean = np.mean(norm_img)
    current_std = np.std(norm_img)

    if current_std == 0: # Avoid division by zero for flat images
        norm_img = np.full_like(norm_img, mean)
    else:
        norm_img = (norm_img - current_mean) * np.sqrt(variance) / current_std + mean
    
    return np.clip(norm_img, 0, 255).astype(np.uint8)

def segment_image(img, block_size=BLOCK_SIZE, variance_threshold_ratio=0.1, morphological_ops=True):
    """
    Segments the fingerprint foreground from background based on local variance.
    As per Bazen & Gerez.
    """
    height, width = img.shape
    segmentation_mask = np.zeros_like(img, dtype=np.uint8)

    # Compute block-wise variance
    for r in range(0, height, block_size):
        for c in range(0, width, block_size):
            block = img[r:min(r + block_size, height), c:min(c + block_size, width)]
            
            # Calculate variance for the block. A threshold (e.g., 100 for 0-255 scale) can determine foreground.
            # Convert img to float for variance calculation to prevent overflow with large pixel values
            block_var = np.var(block.astype(np.float32))
            
            # A good threshold depends on image scale. For 0-255, variance can be up to ~128^2.
            # Using a ratio of max possible variance for 8-bit image for adaptability.
            max_possible_variance = (255 / 2)**2 # If pixel values are half 0, half 255
            
            if block.size > 0 and block_var > variance_threshold_ratio * max_possible_variance:
                segmentation_mask[r:min(r + block_size, height), c:min(c + block_size, width)] = 255

    if morphological_ops:
        # Apply morphological operations for smoothing (opening and closing)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # Slightly larger kernel for better smoothing
        segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return segmentation_mask

def create_gabor_filter_bank(num_orientations=NUM_GABOR_ORIENTATIONS, frequencies=GABOR_FREQUENCIES, ksize=GABOR_KERNEL_SIZE):
    """
    Creates a bank of Gabor filters.
    Orientations are evenly spaced from 0 to 180 degrees (as used in fingerprint analysis).
    """
    gabor_bank = []
    orientations_rad = np.arange(0, np.pi, np.pi / num_orientations) # 0 to pi radians
    
    # Sigma is the standard deviation of the Gaussian envelope.
    # Common heuristic: sigma ~ 0.56 * lambda (wavelength) or ksize/6.
    # Let's use a fixed sigma relative to kernel size or try to relate to freq.
    sigma = ksize / 6.0 # Standard deviation for the Gaussian envelope
    gamma = 0.5 # Spatial aspect ratio (sigma_y / sigma_x). Typical for elongated ridges.

    for freq in frequencies:
        lmbda = 1.0 / freq # Wavelength
        for theta in orientations_rad:
            # ksize, sigma, theta, lambda, gamma, psi (phase offset, 0 for even symmetric)
            gabor_filter = cv2.getGaborKernel((ksize, ksize), sigma, theta, lmbda, gamma, 0, ktype=cv2.CV_32F)
            gabor_bank.append(gabor_filter)
    return gabor_bank, [np.rad2deg(o) for o in orientations_rad]

def apply_gabor_filter_bank(img, gabor_bank):
    """Applies Gabor filter bank to an image and returns an average enhanced image."""
    filtered_images = []
    for gabor_filter in gabor_bank:
        # Apply filter and sum up results for multiple orientations/frequencies
        # Using cv2.CV_32F for output to avoid clipping before averaging
        filtered_img = cv2.filter2D(img.astype(np.float32), cv2.CV_32F, gabor_filter)
        filtered_images.append(filtered_img)
    
    # Average the filtered images for combined enhancement
    if filtered_images:
        enhanced_img = np.mean(np.array(filtered_images), axis=0)
        enhanced_img = normalize_image(enhanced_img) # Re-normalize to 0-255 range
    else:
        enhanced_img = img.copy() # Return original if no filters applied
    return enhanced_img

def smqt_enhancement(img):
    """
    Successive Mean Quantization Transform (SMQT) proxy.
    The true SMQT is complex. Adaptive Histogram Equalization (CLAHE)
    is a good practical substitute for enhancing local contrast.
    """
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(img)
    return enhanced_img

def compute_stft_features(img, mask, window_size=STFT_WINDOW_SIZE, overlap=STFT_OVERLAP):
    """
    Computes Orientation (Io), Frequency (If), and Energy (Ie) images using STFT block-wise.
    As described in Shimna & Neethu and Baig et al.
    """
    height, width = img.shape
    Io = np.zeros(img.shape, dtype=np.float32) # Orientation (radians, [0, pi))
    If = np.zeros(img.shape, dtype=np.float32) # Frequency magnitude
    Ie = np.zeros(img.shape, dtype=np.float32) # Energy

    step = window_size - overlap

    # Iterate over image in overlapping blocks
    for r in range(0, height - window_size + 1, step):
        for c in range(0, width - window_size + 1, step):
            block = img[r:r + window_size, c:c + window_size].astype(np.float32)
            block_mask = mask[r:r + window_size, c:c + window_size]

            # Skip if block is mostly background (e.g., less than 50% foreground pixels)
            if np.sum(block_mask) < (block_mask.size * 0.5):
                continue

            # Apply window function (e.g., Hann window) and mean subtraction
            window = np.outer(cv2.createHanningWindow((window_size, 1), cv2.CV_32F),
                              cv2.createHanningWindow((window_size, 1), cv2.CV_32F).T)
            block_windowed = (block - np.mean(block)) * window

            # Compute FFT
            fourier_transform = np.fft.fft2(block_windowed)
            fourier_transform_shifted = np.fft.fftshift(fourier_transform)
            magnitude_spectrum = np.abs(fourier_transform_shifted)
            power_spectrum = magnitude_spectrum**2

            # Compute Energy (Ie): sum of power spectrum (log-scaled for dynamic range)
            energy = np.log(np.sum(power_spectrum) + 1e-10) # Add epsilon to avoid log(0)
            Ie[r:r + window_size, c:c + window_size] = energy

            # Compute Orientation (Io) and Frequency (If) from dominant peak
            # Exclude DC component (center) for peak finding
            peak_mask = np.ones_like(power_spectrum, dtype=bool)
            center_x, center_y = window_size // 2, window_size // 2
            # Exclude a small area around DC component
            peak_mask[center_y-2:center_y+3, center_x-2:center_x+3] = False 
            
            # Find coordinates of the maximum power (dominant frequency)
            max_idx = np.unravel_index(np.argmax(power_spectrum * peak_mask), power_spectrum.shape)
            
            # Calculate frequency components relative to center
            fx = max_idx[1] - center_x
            fy = max_idx[0] - center_y

            # Orientation: angle in radians [0, pi) (fingerprint ridges have 180-degree symmetry)
            orientation_rad = 0.5 * np.arctan2(fy, fx) # Gives angle in [-pi/2, pi/2] from xy gradient
            if orientation_rad < 0:
                orientation_rad += np.pi # Normalize to [0, pi)

            # Frequency: magnitude of the frequency vector (normalized by window size)
            frequency_mag = np.sqrt(fx**2 + fy**2) / window_size # Cycles per pixel
            
            # Assign computed values to the entire block region
            Io[r:r + window_size, c:c + window_size] = orientation_rad
            If[r:r + window_size, c:c + window_size] = frequency_mag

    # Smoothening of orientation, frequency, and energy fields using Gaussian filter
    Io = filters.gaussian_filter(Io, sigma=SMOOTH_SIGMA)
    If = filters.gaussian_filter(If, sigma=SMOOTH_SIGMA)
    Ie = filters.gaussian_filter(Ie, sigma=SMOOTH_SIGMA)

    return Io, If, Ie

def binarize_image(img_8u, block_size=11, c=2):
    """
    Binarizes the image using adaptive thresholding (Gaussian method).
    Input image must be 8-bit unsigned integer.
    """
    # Ensure input is 8-bit for adaptiveThreshold
    if img_8u.dtype != np.uint8:
        img_8u = cv2.normalize(img_8u, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Adaptive thresholding often performs better for varying illumination
    binary_img = cv2.adaptiveThreshold(img_8u, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, block_size, c)
    return binary_img

def thin_image(binary_img):
    """
    Thins the binary image to single-pixel width ridges using skeletonization.
    Input must be a binary image (0 or 255).
    """
    # skimage.skeletonize expects a boolean image (True for foreground).
    thinned_img = skeletonize(binary_img / 255).astype(np.uint8) * 255
    return thinned_img

def extract_minutiae(thinned_img, mask, orientation_map):
    """
    Extracts minutiae (ridge endings and bifurcations) using the Crossing Number (CN) method.
    Assigns orientation from the provided orientation_map.
    """
    minutiae = [] # List of {'x': x, 'y': y, 'type': 'E'/'B', 'orientation': angle_rad, 'quality': 1.0}
    height, width = thinned_img.shape

    # Pad image to simplify neighborhood access and avoid border issues
    padded_img = cv2.copyMakeBorder(thinned_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    for y in range(1, height - 1): # Iterate through actual image coordinates
        for x in range(1, width - 1):
            if padded_img[y, x] == 255: # If it's a ridge pixel (foreground)
                # 3x3 neighborhood pixels, clockwise starting from top-left
                p = [padded_img[y-1, x-1], padded_img[y-1, x], padded_img[y-1, x+1],
                     padded_img[y, x+1], padded_img[y+1, x+1], padded_img[y+1, x],
                     padded_img[y+1, x-1], padded_img[y, x-1], padded_img[y-1, x-1]] # Closing the loop

                # Convert to 0/1 for CN calculation
                p_binary = [1 if val == 255 else 0 for val in p]

                cn = 0
                for i in range(8):
                    cn += abs(p_binary[i] - p_binary[i+1])
                cn /= 2 # Crossing Number formula

                # Only consider minutiae within the foreground mask
                if mask[y-1, x-1] == 255: # Adjust for padding
                    minutia_obj = {'x': x-1, 'y': y-1, 'orientation': 0.0, 'quality': 1.0}
                    if cn == 1: # Ridge ending
                        minutia_obj['type'] = 'E'
                    elif cn == 3: # Bifurcation
                        minutia_obj['type'] = 'B'
                    else:
                        continue # Not a minutia (e.g., normal ridge pixel, bridge, etc.)
                    
                    # Assign orientation from STFT map at minutia location
                    # Ensure coordinates are valid for orientation_map
                    if 0 <= minutia_obj['y'] < orientation_map.shape[0] and 0 <= minutia_obj['x'] < orientation_map.shape[1]:
                        minutia_obj['orientation'] = orientation_map[minutia_obj['y'], minutia_obj['x']]
                    
                    minutiae.append(minutia_obj)
    return minutiae

# --- MTCC Specific Functions ---

def euclidean_distance(p1, p2):
    """Euclidean distance between two 2D points (x, y)."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def gaussian_contribution(d, sigma):
    """General Gaussian function for contributions (spatial or directional)."""
    # Represents Gs (spatial) or GD (directional) from Baig et al.
    return np.exp(-(d**2) / (2 * sigma**2))

def angular_difference_normalized(theta1, theta2):
    """
    Calculates normalized angular difference do(theta1, theta2) as per Baig et al. Eq 8.
    Ensures difference is in [-pi, pi).
    """
    diff = theta1 - theta2
    # Normalize to [-pi, pi)
    while diff >= np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff

def create_mtcc_cylinder(center_minutia, all_minutiae, stft_features):
    """
    Creates the six types of MTCC cylinder descriptors for a given central minutia.
    stft_features is a tuple (Io, If_scaled, Ie_scaled) where If/Ie are scaled to [-pi, pi].
    """
    Io, If_scaled, Ie_scaled = stft_features
    xm, ym, theta_m = center_minutia['x'], center_minutia['y'], center_minutia['orientation']
    
    # Initialize cylinder arrays (Ns x Ns grid) for each MTCC type
    cylinder_mcco = np.zeros((NS, NS), dtype=np.float32)
    cylinder_mccf = np.zeros((NS, NS), dtype=np.float32)
    cylinder_mcce = np.zeros((NS, NS), dtype=np.float32)
    cylinder_mcc_co = np.zeros((NS, NS), dtype=np.float32)
    cylinder_mcc_cf = np.zeros((NS, NS), dtype=np.float32)
    cylinder_mcc_ce = np.zeros((NS, NS), dtype=np.float32)

    # Filter neighboring minutiae within the cylinder radius R_CYLINDER
    neighboring_minutiae = [
        mt for mt in all_minutiae 
        if mt != center_minutia and euclidean_distance((xm, ym), (mt['x'], mt['y'])) <= R_CYLINDER
    ]
    
    # Pre-calculate rotation matrix for the central minutia's orientation
    cos_theta_m = np.cos(theta_m)
    sin_theta_m = np.sin(theta_m)

    for i in range(NS):
        for j in range(NS):
            # Calculate cell center p_ij (spatial coordinates relative to (xm,ym)) as per Baig et al. Eq 2
            # (i - (NS - 1) / 2) & (j - (NS - 1) / 2) centers the grid around 0
            x_prime_local = DELTA_S * (i - (NS - 1) / 2) 
            y_prime_local = DELTA_S * (j - (NS - 1) / 2)
            
            # Rotate and translate to absolute image coordinates
            p_ij_x = xm + (x_prime_local * cos_theta_m - y_prime_local * sin_theta_m)
            p_ij_y = ym + (x_prime_local * sin_theta_m + y_prime_local * cos_theta_m)
            
            # Convert to integer coordinates for image lookup
            p_ij_x_int, p_ij_y_int = int(round(p_ij_x)), int(round(p_ij_y))
            
            # Check if cell center is within image bounds
            if not (0 <= p_ij_y_int < Io.shape[0] and 0 <= p_ij_x_int < Io.shape[1]):
                continue # Skip cells outside image boundaries

            # --- Calculate contributions for each MTCC type ---

            # For MCCo, MCCf, MCCe: Sum contributions from neighboring minutiae
            for mt in neighboring_minutiae:
                # Spatial distance d_s between cell center p_ij and neighbor minutia mt
                d_s = euclidean_distance((p_ij_x, p_ij_y), (mt['x'], mt['y']))
                gs_contrib = gaussian_contribution(d_s, SIGMA_S)

                # Ensure neighbor minutia coords are valid for STFT feature lookup
                mt_x_int, mt_y_int = int(mt['x']), int(mt['y'])
                if not (0 <= mt_y_int < Io.shape[0] and 0 <= mt_x_int < Io.shape[1]):
                    continue # Skip if neighbor's features cannot be sampled
                
                # MCCo: Original Minutiae Angle contribution
                # Angular difference between neighbor minutia's orientation and central minutia's orientation
                d_theta_mcco = angular_difference_normalized(mt['orientation'], theta_m)
                gd_mcco = gaussian_contribution(d_theta_mcco, SIGMA_D)
                cylinder_mcco[i, j] += gs_contrib * gd_mcco

                # MCCf: Frequency map contribution (features treated as angles)
                freq_m = If_scaled[int(ym), int(xm)] # Central minutia's scaled frequency
                freq_mt = If_scaled[mt_y_int, mt_x_int] # Neighbor minutia's scaled frequency
                d_theta_mccf = angular_difference_normalized(freq_m, freq_mt)
                gd_mccf = gaussian_contribution(d_theta_mccf, SIGMA_D)
                cylinder_mccf[i, j] += gs_contrib * gd_mccf

                # MCCe: Energy map contribution (features treated as angles)
                energy_m = Ie_scaled[int(ym), int(xm)] # Central minutia's scaled energy
                energy_mt = Ie_scaled[mt_y_int, mt_x_int] # Neighbor minutia's scaled energy
                d_theta_mcce = angular_difference_normalized(energy_m, energy_mt)
                gd_mcce = gaussian_contribution(d_theta_mcce, SIGMA_D)
                cylinder_mcce[i, j] += gs_contrib * gd_mcce
            
            # Cell-centered MTCC types (MCC_co, MCC_cf, MCC_ce)
            # Contributions are based directly on the STFT feature value at the cell's center p_ij
            # No sum over neighbors, as the value is "picked up" from the map at p_ij.
            # This is interpretation based on Baig et al. Section VI.B.
            
            cylinder_mcc_co[i, j] = Io[p_ij_y_int, p_ij_x_int]
            cylinder_mcc_cf[i, j] = If_scaled[p_ij_y_int, p_ij_x_int]
            cylinder_mcc_ce[i, j] = Ie_scaled[p_ij_y_int, p_ij_x_int]

    # Normalize each cylinder to create a fixed-length feature vector
    # L2 normalization is common for feature vectors
    mcco_code = cylinder_mcco.flatten()
    mccf_code = cylinder_mccf.flatten()
    mcce_code = cylinder_mcce.flatten()
    mcc_co_code = cylinder_mcc_co.flatten()
    mcc_cf_code = cylinder_mcc_cf.flatten()
    mcc_ce_code = cylinder_mcc_ce.flatten()

    # Apply L2 normalization to each feature vector
    mcco_code = mcco_code / (np.linalg.norm(mcco_code) + 1e-10)
    mccf_code = mccf_code / (np.linalg.norm(mccf_code) + 1e-10)
    mcce_code = mcce_code / (np.linalg.norm(mcce_code) + 1e-10)
    mcc_co_code = mcc_co_code / (np.linalg.norm(mcc_co_code) + 1e-10)
    mcc_cf_code = mcc_cf_code / (np.linalg.norm(mcc_cf_code) + 1e-10)
    mcc_ce_code = mcc_ce_code / (np.linalg.norm(mcc_ce_code) + 1e-10)

    return {
        'mcco': mcco_code,
        'mccf': mccf_code,
        'mcce': mcce_code,
        'mcc_co': mcc_co_code,
        'mcc_cf': mcc_cf_code,
        'mcc_ce': mcc_ce_code,
    }

def compare_mtcc_descriptors(desc1, desc2):
    """
    Compares two MTCC descriptors (dictionaries of feature vectors).
    Returns Euclidean distance for each feature type. Lower distance means higher similarity.
    """
    scores = {}
    for key in desc1:
        dist = np.linalg.norm(desc1[key] - desc2[key])
        scores[key] = dist
    return scores

# --- Matching and Evaluation (Conceptual / Placeholder) ---

def two_finger_matching(img1_path, img2_path):
    """
    High-level function to perform matching between two fingerprint images.
    Returns average similarity scores for each MTCC type.
    This is a simplified conceptual implementation of LSSR (Local Similarity Sort with Relaxation).
    """
    print(f"Processing '{img1_path}' for MTCC feature generation...")
    results1, _ = process_fingerprint_pipeline(img1_path)
    minutiae1 = results1['minutiae']
    stft_features1_scaled = (results1['io'], results1['if_scaled'], results1['ie_scaled'])

    print(f"Processing '{img2_path}' for MTCC feature generation...")
    results2, _ = process_fingerprint_pipeline(img2_path)
    minutiae2 = results2['minutiae']
    stft_features2_scaled = (results2['io'], results2['if_scaled'], results2['ie_scaled'])

    # Generate MTCC descriptors for all valid minutiae in both images
    mtcc_descriptors1 = []
    for m in minutiae1:
        # Ensure minutia orientation is not None (means it was within valid STFT map)
        if m['orientation'] is not None: 
            mtcc_descriptors1.append(create_mtcc_cylinder(m, minutiae1, stft_features1_scaled))

    mtcc_descriptors2 = []
    for m in minutiae2:
        if m['orientation'] is not None:
            mtcc_descriptors2.append(create_mtcc_cylinder(m, minutiae2, stft_features2_scaled))

    if not mtcc_descriptors1 or not mtcc_descriptors2:
        print("Not enough minutiae with valid descriptors for matching. Skipping matching.")
        return {key: float('inf') for key in ['mcco', 'mccf', 'mcce', 'mcc_co', 'mcc_cf', 'mcc_ce']}

    # Simplistic matching: For each descriptor in image 1, find the minimum Euclidean distance
    # to any descriptor in image 2. Then average these minimum distances.
    # This is a basic proxy for "global" similarity.
    avg_match_scores = {key: [] for key in mtcc_descriptors1[0]}

    for desc1 in mtcc_descriptors1:
        min_dists_for_desc1_per_type = {key: float('inf') for key in desc1}
        for desc2 in mtcc_descriptors2:
            current_dists = compare_mtcc_descriptors(desc1, desc2)
            for key in current_dists:
                min_dists_for_desc1_per_type[key] = min(min_dists_for_desc1_per_type[key], current_dists[key])
        
        # Add the best match distance for this desc1 to the list for averaging
        for key in min_dists_for_desc1_per_type:
            if min_dists_for_desc1_per_type[key] != float('inf'): # Ensure a match was found
                avg_match_scores[key].append(min_dists_for_desc1_per_type[key])
    
    final_scores = {key: np.mean(scores_list) if scores_list else float('inf') for key, scores_list in avg_match_scores.items()}
    return final_scores

def calculate_eer(genuine_scores, impostor_scores):
    """
    Conceptual placeholder for EER (Equal Error Rate) calculation.
    Requires lists of similarity scores from genuine and impostor matches.
    """
    print("\n--- EER Calculation (Conceptual Placeholder) ---")
    print("To calculate EER meaningfully, a large database of genuine and impostor scores is required.")
    print("This function demonstrates where such an evaluation step would fit in the pipeline.")
    print("In a real scenario, you would:")
    print("1. Perform all-to-all comparisons within and between different fingerprints in a database.")
    print("2. Collect 'genuine scores' (matches of same finger) and 'impostor scores' (matches of different fingers).")
    print("3. Iterate through possible thresholds to compute False Acceptance Rate (FAR) and False Rejection Rate (FRR).")
    print("4. EER is the point where FAR approximately equals FRR.")
    print("Accuracy values (e.g., 98.22% mentioned in one paper) are derived from such comprehensive evaluations.")
    return None

# --- Main Pipeline and Visualization ---

def process_fingerprint_pipeline(image_path):
    """
    Executes the full fingerprint processing pipeline for a single image and collects data for visualization.
    """
    print(f"Starting pipeline for: {image_path}")
    original_img = load_image(image_path)
    if original_img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    vis_data = {'original': original_img}
    pipeline_results = {}

    # 1. Image Loading (already done)
    # The original image is the starting point.

    # 2. Image Normalization
    normalized_img = normalize_image(original_img)
    vis_data['normalized'] = normalized_img
    pipeline_results['normalized_img'] = normalized_img

    # 3. Segmentation
    segmentation_mask = segment_image(normalized_img)
    vis_data['segmentation_mask'] = segmentation_mask
    pipeline_results['segmentation_mask'] = segmentation_mask

    # 4. Gabor Filter Enhancement
    gabor_bank, _ = create_gabor_filter_bank()
    # Apply Gabor to the normalized image, masked by the segmentation mask
    # Convert mask to float for multiplication
    masked_normalized_img = normalized_img.astype(np.float32) * (segmentation_mask / 255.0)
    gabor_enhanced_img = apply_gabor_filter_bank(masked_normalized_img, gabor_bank)
    vis_data['gabor_enhanced'] = gabor_enhanced_img
    pipeline_results['gabor_enhanced_img'] = gabor_enhanced_img

    # 5. SMQT (Successive Mean Quantization Transform) - proxy with CLAHE
    # Applied to the Gabor-enhanced image for final visual quality.
    smqt_enhanced_img = smqt_enhancement(gabor_enhanced_img)
    vis_data['smqt_enhanced'] = smqt_enhanced_img
    pipeline_results['smqt_enhanced_img'] = smqt_enhanced_img

    # 6. STFT Analysis for creating features for MTCC
    # Applied to the SMQT-enhanced image, masked by segmentation.
    masked_smqt_enhanced_img = smqt_enhanced_img.astype(np.float32) * (segmentation_mask / 255.0)
    io_img, if_img, ie_img = compute_stft_features(masked_smqt_enhanced_img, segmentation_mask)
    vis_data['io'] = io_img # Raw STFT orientation
    vis_data['if'] = if_img # Raw STFT frequency magnitude
    vis_data['ie'] = ie_img # Raw STFT energy

    pipeline_results['io'] = io_img
    pipeline_results['if'] = if_img
    pipeline_results['ie'] = ie_img

    # Scale If and Ie to [-pi, pi] as implied by Baig et al. for angular_difference_normalized
    # Min/max for these can vary significantly, so ensure non-zero range for scaling
    if_min, if_max = np.min(if_img), np.max(if_img)
    ie_min, ie_max = np.min(ie_img), np.max(ie_img)

    if_range = if_max - if_min
    ie_range = ie_max - ie_min

    # Scaling to [-pi, pi]
    if if_range > 1e-10:
        if_scaled = (if_img - if_min) / if_range * (2 * np.pi) - np.pi
    else: # Handle flat image if_img
        if_scaled = np.zeros_like(if_img)

    if ie_range > 1e-10:
        ie_scaled = (ie_img - ie_min) / ie_range * (2 * np.pi) - np.pi
    else: # Handle flat image ie_img
        ie_scaled = np.zeros_like(ie_img)
        
    pipeline_results['if_scaled'] = if_scaled
    pipeline_results['ie_scaled'] = ie_scaled

    # 7. Image binarization and thinning (for minutiae extraction)
    # Binarize the SMQT enhanced image, masked.
    binarized_img = binarize_image(smqt_enhanced_img)
    # Apply mask after binarization for clean edges
    binarized_img = binarized_img * (segmentation_mask // 255)
    vis_data['binarized'] = binarized_img
    pipeline_results['binarized_img'] = binarized_img
    
    thinned_img = thin_image(binarized_img)
    vis_data['thinned'] = thinned_img
    pipeline_results['thinned_img'] = thinned_img

    # 8. Minutiae extraction
    # Pass the raw orientation map (Io) for minutiae orientation assignment
    minutiae = extract_minutiae(thinned_img, segmentation_mask, io_img)
    vis_data['minutiae'] = minutiae # Minutiae data for visualization overlay
    pipeline_results['minutiae'] = minutiae

    # 9. Cylinder creation (MTCC features per minutia)
    # This generates the MTCC descriptors for each extracted minutia.
    # The actual MTCC descriptors are the numerical output for matching.
    all_mtcc_descriptors = []
    for m in minutiae:
        if m['orientation'] is not None:
            # Pass the STFT features (Io, If_scaled, Ie_scaled)
            descriptor = create_mtcc_cylinder(m, minutiae, (io_img, if_scaled, ie_scaled))
            all_mtcc_descriptors.append(descriptor)
    pipeline_results['all_mtcc_descriptors'] = all_mtcc_descriptors

    return pipeline_results, vis_data

def visualize_pipeline(vis_data):
    """
    Visualizes key steps of the fingerprint processing pipeline in a single figure.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    titles = [
        '1. Original Image',
        '2. Normalized Image',
        '3. Segmentation Mask',
        '4. Gabor Enhanced',
        '5. SMQT Enhanced',
        '6. STFT Orientation (Io)',
        '6. STFT Frequency (If)',
        '7. Binarized Image',
        '8. Thinned Image w/ Minutiae'
    ]

    images_to_show = [
        vis_data['original'],
        vis_data['normalized'],
        vis_data['segmentation_mask'],
        vis_data['gabor_enhanced'],
        vis_data['smqt_enhanced'],
        vis_data['io'], # Plotting raw Io
        vis_data['if'], # Plotting raw If
        vis_data['binarized'],
        vis_data['thinned']
    ]
    
    # Custom colormaps for orientation and frequency
    for i, ax in enumerate(axes):
        cmap = 'gray'
        if titles[i] == '6. STFT Orientation (Io)':
            cmap = 'hsv' # Hue for angle
        elif titles[i] == '6. STFT Frequency (If)':
            cmap = 'viridis' # Color gradient for magnitude

        im = ax.imshow(images_to_show[i], cmap=cmap)
        ax.set_title(titles[i])
        ax.axis('off')

        # Add colorbar for STFT features
        if titles[i] in ['6. STFT Orientation (Io)', '6. STFT Frequency (If)']:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Overlay minutiae on the thinned image
        if i == 8 and 'minutiae' in vis_data:
            minutiae = vis_data['minutiae']
            for m in minutiae:
                # Plot minutia point
                color = 'red' if m['type'] == 'E' else 'blue' # Red for ending, Blue for bifurcation
                marker = 'o' if m['type'] == 'E' else 'x'
                ax.plot(m['x'], m['y'], marker=marker, color=color, markersize=6, mew=1.5, linestyle='None')
                
                # Draw orientation line
                if m['orientation'] is not None:
                    length = 15 # Length of the orientation line
                    # Convert orientation (radians in [0, pi)) to a direction for plotting
                    # Fingerprint orientation often means direction of ridge flow.
                    # Angle is perpendicular to ridge direction. So 90 deg rotation.
                    # Or, more simply, the orientation map already implies the ridge direction.
                    # Just use orientation directly.
                    
                    # For ridges, the orientation is typically 0 to pi.
                    # Line extends in both directions for 180-degree ambiguity.
                    # For visualization, draw a line segment centered at minutia in direction 'orientation'
                    dx = length * np.cos(m['orientation'])
                    dy = length * np.sin(m['orientation'])
                    
                    # Draw a line segment (not an arrow) showing 180-degree orientation
                    ax.plot([m['x'] - dx/2, m['x'] + dx/2], [m['y'] - dy/2, m['y'] + dy/2],
                            color='green', linewidth=1.5)

    plt.tight_layout()
    plt.show()


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Sample Image Setup ---
    # For robust testing, provide paths to actual fingerprint images (e.g., from FVC2002/2004).
    # If not found, synthetic images will be generated for demonstration.
    
    img_path1 = R'C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2000\FVC2000\Db1_a\1_1.tif'
    img_path2 = R'C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2000\FVC2000\Db1_a\1_2.tif'
    

    # Generate simple synthetic images if files don't exist
    try:
        test_img1 = load_image(img_path1)
        test_img2 = load_image(img_path2)
    except FileNotFoundError:
        print(f"Sample images '{img_path1}' or '{img_path2}' not found. Creating synthetic ones for demonstration.")
        img_h, img_w = 200, 200

        # Create synthetic image 1 (simple parallel lines with some curvature)
        synth_img1 = np.zeros((img_h, img_w), dtype=np.uint8)
        for i in range(10, img_h - 10, 10):
            cv2.line(synth_img1, (0, i), (img_w - 1, i + int(30 * np.sin(i / 25.0))), 255, 1)
        synth_img1 = cv2.GaussianBlur(synth_img1, (3,3), 0)
        cv2.imwrite(img_path1, synth_img1)
        test_img1 = synth_img1
        
        # Create synthetic image 2 (similar but slightly different pattern for matching)
        synth_img2 = np.zeros((img_h, img_w), dtype=np.uint8)
        for i in range(15, img_h - 15, 10):
            cv2.line(synth_img2, (0, i), (img_w - 1, i + int(25 * np.cos(i / 20.0))), 255, 1)
        synth_img2 = cv2.GaussianBlur(synth_img2, (3,3), 0)
        cv2.imwrite(img_path2, synth_img2)
        test_img2 = synth_img2

    try:
        # --- Run the full pipeline for one image and visualize ---
        print("\n--- Running pipeline for Image 1 ---")
        pipeline_results1, vis_data1 = process_fingerprint_pipeline(img_path1)
        visualize_pipeline(vis_data1)
        
        # --- Demonstrate two-finger matching ---
        print("\n--- Demonstrating Two-Finger Matching (Conceptual) ---")
        match_scores = two_finger_matching(img_path1, img_path2)
        print("\nMatching Scores (Average Euclidean Distance for each MTCC type):")
        for k, v in match_scores.items():
            print(f"  {k}: {v:.4f}")
        print("\nNote: Lower scores indicate higher similarity.")

        # --- EER calculation placeholder ---
        calculate_eer(None, None)

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        print("Please ensure image files exist or are correctly generated.")