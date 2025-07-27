import numpy as np
import cv2
from scipy.ndimage import binary_erosion, binary_dilation, label, find_objects
from scipy.signal import windows
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import os 
from skimage.morphology import skeletonize 

# --- 1. CORE PREPROCESSING PIPELINE ---

def load_and_preprocess(image_path):
    """
    Load fingerprint image and apply initial preprocessing.
    - Variance-based segmentation with morphological smoothing.
    - Create fingerprint mask for valid region constraint.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Normalize intensity to 0-255 range if not already
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- Segmentation (Conceptual: Variance-based with morphological smoothing) ---
    # Paper uses "blockwise variance based image segmentation method, [23]".
    # Morphological smoothing is used for the mask.

    block_size = 16 
    
    # Calculate local standard deviation
    img_float = img.astype(np.float32)
    mean_sq = cv2.boxFilter(img_float * img_float, -1, (block_size, block_size))
    mean_val = cv2.boxFilter(img_float, -1, (block_size, block_size))
    local_std = np.sqrt(np.maximum(0, mean_sq - mean_val**2))

    # Threshold local standard deviation to get a foreground mask
    # Threshold chosen empirically; should be tuned for datasets
    mask = (local_std > np.mean(local_std) * 0.5).astype(np.uint8) * 255

    # Morphological smoothing of the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Apply mask to image (set background to black)
    preprocessed_img = cv2.bitwise_and(img, img, mask=mask)

    return preprocessed_img, mask

def stft_feature_extraction(image, window_size=None, overlap=None):
    """
    STFT-based feature extraction (Papers 3-4).
    This function primarily extracts feature maps (orientation, frequency, energy, coherence)
    and provides a raw STFT-reconstructed image. Gabor and AHE are applied separately.
    
    Returns:
        stft_reconstructed_image: Image reconstructed from STFT, before Gabor/AHE.
        orientation_map: Probabilistic orientation using vector averaging
        frequency_map: Ridge frequency from spectral analysis
        energy_map: Logarithmic energy content per block
        coherence_map: Angular coherence for adaptive filtering
    """
    if window_size is None: window_size = MTCC_PARAMETERS['stft_window']
    if overlap is None: overlap = MTCC_PARAMETERS['stft_overlap']

    h, w = image.shape
    step_size = window_size - overlap
    
    # Initialize feature maps
    orientation_map = np.zeros_like(image, dtype=np.float32)
    frequency_map = np.zeros_like(image, dtype=np.float32)
    energy_map = np.zeros_like(image, dtype=np.float32)
    coherence_map = np.zeros_like(image, dtype=np.float32) # Placeholder
    
    stft_reconstruction_accumulator = np.zeros_like(image, dtype=np.float32)
    reconstruction_weights = np.zeros_like(image, dtype=np.float32)
    
    # Raised cosine spectral window [3, Eq 5]
    win_1d = windows.cosine(window_size)
    spectral_window = np.outer(win_1d, win_1d) # 2D separable window

    if step_size <= 0:
        step_size = 1 # Fallback to 1 pixel step for robustness

    for y in range(0, h - window_size + 1, step_size):
        for x in range(0, w - window_size + 1, step_size):
            block = image[y:y+window_size, x:x+window_size].astype(np.float32)

            # Remove DC content [4, Algo Step 1a]
            block -= np.mean(block)

            # Multiply by spectral window [4, Algo Step 1b]
            windowed_block = block * spectral_window

            # Obtain FFT of the block [4, Algo Step 1c]
            F = fftshift(fft2(windowed_block))
            magnitude_spectrum = np.abs(F)
            
            # Energy: Sum of magnitude squared (logarithmic) [3, Eq 11]
            block_energy = np.log(np.sum(magnitude_spectrum**2) + 1e-10) # Add epsilon to avoid log(0)
            
            # Find dominant frequency and orientation (simplified for concept)
            if np.max(magnitude_spectrum) > 0:
                dom_freq_idx = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
                center_y, center_x = window_size // 2, window_size // 2
                
                k_y = dom_freq_idx[0] - center_y
                k_x = dom_freq_idx[1] - center_x
                
                # Orientation is perpendicular to frequency vector
                orientation = np.arctan2(k_y, k_x) / 2.0 + np.pi/2 # Double angle representation
                frequency = np.sqrt(k_x**2 + k_y**2) / window_size # Normalized frequency

                # Store features for this block. Overlapping regions will be averaged later.
                orientation_map[y:y+window_size, x:x+window_size] = orientation
                frequency_map[y:y+window_size, x:x+window_size] = frequency
                energy_map[y:y+window_size, x:x+window_size] = block_energy

            # Reconstruct block from FFT (before Gabor/AHE)
            block_recon = np.real(ifft2(ifftshift(F)))
            stft_reconstruction_accumulator[y:y+window_size, x:x+window_size] += block_recon
            reconstruction_weights[y:y+window_size, x:x+window_size] += 1 # For averaging overlaps

    # Normalize accumulated blocks to get the initial STFT reconstructed image
    # Handle division by zero for regions with no contributions
    reconstruction_weights[reconstruction_weights == 0] = 1 # Avoid division by zero
    stft_reconstructed_image = stft_reconstruction_accumulator / reconstruction_weights
    stft_reconstructed_image = cv2.normalize(stft_reconstructed_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Smooth feature maps by Gaussian filter for conceptual purpose
    orientation_map = cv2.GaussianBlur(orientation_map, (5,5), 0)
    frequency_map = cv2.GaussianBlur(frequency_map, (5,5), 0)
    energy_map = cv2.GaussianBlur(energy_map, (5,5), 0)
    
    # Coherence map (placeholder, actual coherence calculation is more complex)
    coherence_map = np.ones_like(image, dtype=np.float32) * 0.8 

    return stft_reconstructed_image, orientation_map, frequency_map, energy_map, coherence_map

def create_gabor_filter(orientation, frequency, sigma_x, sigma_y, kernel_size):
    """Creates a Gabor filter kernel based on local orientation and frequency."""
    theta = orientation
    # Wavelength is reciprocal of frequency. Handle zero frequency.
    lamda = 1.0 / (frequency + 1e-6) if frequency > 0 else 10.0 
    lamda = np.clip(lamda, 5, 20) # Ensure wavelength is within reasonable bounds for visual effect
    
    if kernel_size % 2 == 0:
        kernel_size += 1 # Ensure kernel size is odd

    # Gamma (aspect ratio) for OpenCV's Gabor kernel is sigma_y / sigma_x
    gamma = sigma_y / sigma_x 
    
    # Phase offset (psi) is typically 0 for enhancing ridges/valleys, as per paper's cosine term
    psi = 0 

    gabor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma_x, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
    return gabor_kernel

def apply_adaptive_gabor_filter(image, orientation_map, frequency_map, sigma_x=None, sigma_y=None, kernel_size=None):
    """
    Applies adaptive Gabor filtering based on local orientation and frequency maps.
    This uses an overlap-add method with cosine windowing to reduce blocking artifacts.
    This method corresponds to the "filtering with Gabor functions" described in the paper. [3]
    """
    if sigma_x is None: sigma_x = MTCC_PARAMETERS['gabor_sigma_x']
    if sigma_y is None: sigma_y = MTCC_PARAMETERS['gabor_sigma_y']
    if kernel_size is None: kernel_size = MTCC_PARAMETERS['gabor_kernel_size']

    h, w = image.shape
    block_size = 16 # Common block size for local feature averaging

    gabor_accumulator = np.zeros_like(image, dtype=np.float32)
    gabor_weights = np.zeros_like(image, dtype=np.float32)

    half_kernel = kernel_size // 2

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            current_block_y_start = y
            current_block_y_end = min(y + block_size, h)
            current_block_x_start = x
            current_block_x_end = min(x + block_size, w)

            # Get average orientation and frequency for the current block's region
            avg_orientation = np.mean(orientation_map[current_block_y_start:current_block_y_end, current_block_x_start:current_block_x_end])
            avg_frequency = np.mean(frequency_map[current_block_y_start:current_block_y_end, current_block_x_start:current_block_x_end])

            # Create Gabor kernel for this block's local features
            gabor_kernel = create_gabor_filter(avg_orientation, avg_frequency, sigma_x, sigma_y, kernel_size)
            
            # Define a larger region around the current block for filtering
            filter_region_y_start = max(0, current_block_y_start - half_kernel)
            filter_region_y_end = min(h, current_block_y_end + half_kernel)
            filter_region_x_start = max(0, current_block_x_start - half_kernel)
            filter_region_x_end = min(w, current_block_x_end + half_kernel)

            region_to_filter = image[filter_region_y_start:filter_region_y_end, 
                                     filter_region_x_start:filter_region_x_end].astype(np.float32)

            if region_to_filter.size == 0: 
                continue

            filtered_extended_region = cv2.filter2D(region_to_filter, cv2.CV_32F, gabor_kernel, borderType=cv2.BORDER_REPLICATE)
            
            # Extract the portion of the filtered_extended_region that corresponds to the
            # current block's original extent in the image.
            slice_y_start_in_filtered = current_block_y_start - filter_region_y_start
            slice_y_end_in_filtered = slice_y_start_in_filtered + (current_block_y_end - current_block_y_start)
            slice_x_start_in_filtered = current_block_x_start - filter_region_x_start
            slice_x_end_in_filtered = slice_x_start_in_filtered + (current_block_x_end - current_block_x_start)

            filtered_block_content = filtered_extended_region[slice_y_start_in_filtered:slice_y_end_in_filtered, 
                                                              slice_x_start_in_filtered:slice_x_end_in_filtered]
            
            # --- Apply an output window for smooth blending of overlapping filtered blocks ---
            current_block_height = current_block_y_end - current_block_y_start
            current_block_width = current_block_x_end - current_block_x_start
            
            if current_block_height == 0 or current_block_width == 0:
                continue

            output_win_y = windows.cosine(current_block_height)
            output_win_x = windows.cosine(current_block_width)
            output_window = np.outer(output_win_y, output_win_x)

            # Accumulate the windowed block and the window itself for proper normalization
            gabor_accumulator[current_block_y_start:current_block_y_end, current_block_x_start:current_block_x_end] += filtered_block_content * output_window
            gabor_weights[current_block_y_start:current_block_y_end, current_block_x_start:current_block_x_end] += output_window

    gabor_weights[gabor_weights == 0] = 1 
    enhanced_img = gabor_accumulator / gabor_weights
    enhanced_img = cv2.normalize(enhanced_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return enhanced_img

def apply_ahe_normalization(image):
    """
    Applies Adaptive Histogram Equalization (AHE) for contrast enhancement,
    as specified in the paper's normalization step. [3]
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

# --- 2. ADVANCED TEXTURE FEATURE EXTRACTION ---

def extract_texture_features(enhanced_image, orientation_map, frequency_map, energy_map):
    """
    Extract STFT-based texture features for MTCC descriptors.
    
    Features:
        - Io: Orientation image from STFT analysis
        - If: Frequency image from STFT analysis  
        - Ie: Energy image (logarithmic) from STFT analysis
    
    These replace traditional minutiae angular information in cylinder codes.
    """
    texture_maps = {
        'orientation': orientation_map, # Io
        'frequency': frequency_map,     # If
        'energy': energy_map            # Ie
    }
    return texture_maps

# --- 3. MINUTIAE EXTRACTION WITH QUALITY ASSESSMENT ---

def binarize_and_thin(image):
    """
    Binarize the image and apply thinning to get a skeleton.
    Uses skimage.morphology.skeletonize.
    """
    # Otsu's thresholding for binarization
    _, binary_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert for thinning (ridges are foreground, usually white on black, skeletonize expects True for foreground)
    # Convert to boolean for skimage.skeletonize
    # Assuming the binarization makes ridges black (0) and background white (255) if the input was inverted by Gabor.
    # If ridges are white (255) and background black (0) after enhancement, use (binary_img == 255).
    # Standard Gabor output usually has ridges as high intensity (white). Otsu would threshold them to white.
    binary_for_thinning = (binary_img == 255) 

    # Perform thinning
    thinned_image_bool = skeletonize(binary_for_thinning)
    
    # Convert back to uint8 (255 for ridge, 0 for background)
    thinned_image = thinned_image_bool.astype(np.uint8) * 255
    
    return thinned_image

def enhanced_minutiae_extraction(thinned_skeleton, mask):
    """
    Extract minutiae using Crossing Number algorithm with quality assessment.
    - Apply FingerJetFXOSE-style quality sorting (conceptual).
    - Filter minutiae based on local ridge quality.
    - Ensure minutiae are within valid fingerprint mask.
    """
    minutiae_list = [] # List of (x, y, orientation, type, quality)
    
    rows, cols = thinned_skeleton.shape
    skeleton_binary = thinned_skeleton // 255 # Convert to 0/1

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if skeleton_binary[r, c] == 1: # Only process ridge pixels (which are now 1-pixel wide)
                # 3x3 neighborhood (P1 to P8, P0 is center)
                # P2 P3 P4
                # P1 P0 P5
                # P8 P7 P6
                
                p = [skeleton_binary[r-1, c-1], skeleton_binary[r-1, c], skeleton_binary[r-1, c+1],
                     skeleton_binary[r, c+1], skeleton_binary[r+1, c+1], skeleton_binary[r+1, c],
                     skeleton_binary[r+1, c-1], skeleton_binary[r, c-1]]
                
                # Crossing Number calculation
                cn = 0.5 * np.sum(np.abs(np.array(p) - np.roll(np.array(p), -1)))
                
                minutia_type = None
                if cn == 1:
                    minutia_type = "ending"
                elif cn == 3:
                    minutia_type = "bifurcation"
                
                if minutia_type:
                    if mask[r, c] == 255: # Check if within valid fingerprint mask
                        # Placeholder for orientation and quality.
                        # In a real system, minutiae orientation would be derived from the local
                        # ridge orientation map (e.g., orientation_map[r,c]).
                        # Quality would be derived from local image quality metrics.
                        minutiae_list.append({'x': c, 'y': r, 'type': minutia_type, 'orientation': 0.0, 'quality': 1.0})

    minutiae_list.sort(key=lambda m: m['quality'], reverse=True)

    filtered_minutiae = []
    min_dist_sq = 10**2 # Minimum squared distance for minutiae separation (e.g., 10 pixels)
    for m1 in minutiae_list:
        is_spurious = False
        for m2 in filtered_minutiae:
            dist_sq = (m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2
            if dist_sq < min_dist_sq:
                is_spurious = True
                break
        if not is_spurious:
            filtered_minutiae.append(m1)

    return filtered_minutiae

# --- 4. MTCC DESCRIPTOR GENERATION (Core Innovation) ---

def create_3d_cylinder(minutia, radius, NS, ND):
    """
    Helper to conceptually represent a 3D cylinder structure.
    It defines the cell grid based on spatial and angular sectors.
    """
    cylinder = {
        'center_minutia': minutia,
        'radius': radius,
        'spatial_sectors': NS,
        'angular_sectors': ND,
        'cells': [] 
    }
    
    delta_s = (2 * radius) / NS 
    delta_d = (2 * np.pi) / ND 

    for s_idx in range(NS):
        for d_idx in range(ND):
            rel_x = -radius + s_idx * delta_s + delta_s / 2
            rel_y = -radius + d_idx * delta_s + delta_s / 2 
            
            # Rotate relative coordinates based on minutia orientation (conceptual)
            cos_theta = np.cos(minutia['orientation'])
            sin_theta = np.sin(minutia['orientation'])
            
            cell_center_x = minutia['x'] + rel_x * cos_theta - rel_y * sin_theta
            cell_center_y = minutia['y'] + rel_x * sin_theta + rel_y * cos_theta
            
            cylinder['cells'].append({
                's_idx': s_idx,
                'd_idx': d_idx,
                'center_x': int(cell_center_x),
                'center_y': int(cell_center_y),
                'contributions': {} 
            })
    return cylinder

def calculate_spatial_contribution(minutia, neighbor_minutia, radius):
    """
    Calculate spatial contribution based on distance between minutia and neighbor.
    Based on Gs(ds(mt, p^m_i,j)) in Paper 2, Eq 6 (Gaussian function).
    """
    dist = np.sqrt((minutia['x'] - neighbor_minutia['x'])**2 + \
                   (minutia['y'] - neighbor_minutia['y'])**2)
    sigma_s = MTCC_PARAMETERS['gaussian_spatial'] 
    return np.exp(-0.5 * (dist / sigma_s)**2)

def calculate_texture_contribution(value1, value2, sigma_d):
    """
    Generic function to calculate contribution based on difference in texture values,
    using a Gaussian function.
    """
    diff = np.abs(value1 - value2)
    return np.exp(-0.5 * (diff / sigma_d)**2)


def create_mtcc_cylinders(minutiae_list, texture_maps, radius=None, NS=None, ND=None):
    """
    Generate MTCC (Minutiae Texture Cylinder Codes) descriptors.
    Based on Paper 2 methodology.
    """
    if radius is None: radius = MTCC_PARAMETERS['cylinder_radius']
    if NS is None: NS = MTCC_PARAMETERS['spatial_sectors']
    if ND is None: ND = MTCC_PARAMETERS['angular_sectors']

    mtcc_descriptors = []
    
    orientation_map = texture_maps['orientation']
    frequency_map = texture_maps['frequency']
    energy_map = texture_maps['energy']
    
    img_h, img_w = orientation_map.shape 

    for central_minutia in minutiae_list:
        cylinder = create_3d_cylinder(central_minutia, radius, NS, ND)
        
        neighbor_minutiae_in_range = []
        for other_minutia in minutiae_list:
            if other_minutia == central_minutia:
                continue
            dist_sq = (central_minutia['x'] - other_minutia['x'])**2 + \
                      (central_minutia['y'] - other_minutia['y'])**2
            if dist_sq <= radius**2:
                neighbor_minutiae_in_in_range.append(other_minutia)

        for cell in cylinder['cells']:
            cell_x, cell_y = cell['center_x'], cell['center_y']

            if not (0 <= cell_y < img_h and 0 <= cell_x < img_w):
                continue 

            sampled_orient_cell = orientation_map[cell_y, cell_x]
            sampled_freq_cell = frequency_map[cell_y, cell_x]
            sampled_energy_cell = energy_map[cell_y, cell_x]

            cell['contributions']['MCCo'] = 0.0 
            cell['contributions']['MCCf'] = 0.0 
            cell['contributions']['MCCe'] = 0.0 
            cell['contributions']['MCCco'] = 0.0 
            cell['contributions']['MCCcf'] = 0.0 
            cell['contributions']['MCCce'] = 0.0 

            for neighbor_minutia in neighbor_minutiae_in_range:
                spatial_contrib = calculate_spatial_contribution(central_minutia, neighbor_minutia, radius)

                n_x, n_y = neighbor_minutia['x'], neighbor_minutia['y']
                if not (0 <= n_y < img_h and 0 <= n_x < img_w): continue
                
                sampled_orient_neighbor = orientation_map[n_y, n_x]
                sampled_freq_neighbor = frequency_map[n_y, n_x]
                sampled_energy_neighbor = energy_map[n_y, n_x]

                sigma_d_orient = MTCC_PARAMETERS['gaussian_directional']
                sigma_d_freq = MTCC_PARAMETERS['gaussian_frequency']
                sigma_d_energy = MTCC_PARAMETERS['gaussian_energy']

                # Traditional MCCo (using minutia's intrinsic orientation, placeholder 0.0)
                diff_orient_minutia = np.abs(central_minutia['orientation'] - neighbor_minutia['orientation'])
                cell['contributions']['MCCo'] += spatial_contrib * calculate_texture_contribution(diff_orient_minutia, 0, sigma_d_orient)
                
                # MCCf: Based on difference in frequency maps
                freq_diff = np.abs(frequency_map[central_minutia['y'], central_minutia['x']] - sampled_freq_neighbor)
                cell['contributions']['MCCf'] += spatial_contrib * calculate_texture_contribution(freq_diff, 0, sigma_d_freq)

                # MCCe: Based on difference in energy maps
                energy_diff = np.abs(energy_map[central_minutia['y'], central_minutia['x']] - sampled_energy_neighbor)
                cell['contributions']['MCCe'] += spatial_contrib * calculate_texture_contribution(energy_diff, 0, sigma_d_energy)
                
                # MCCco (Cell-centered orientation contributions)
                orient_diff_cell_center = np.abs(central_minutia['orientation'] - sampled_orient_cell)
                cell['contributions']['MCCco'] += spatial_contrib * calculate_texture_contribution(orient_diff_cell_center, 0, sigma_d_orient)
                
                # MCCcf (Cell-centered frequency contributions)
                freq_diff_cell_center = np.abs(frequency_map[central_minutia['y'], central_minutia['x']] - sampled_freq_cell)
                cell['contributions']['MCCcf'] += spatial_contrib * calculate_texture_contribution(freq_diff_cell_center, 0, sigma_d_freq)

                # MCCce (Cell-centered energy contributions)
                energy_diff_cell_center = np.abs(energy_map[central_minutia['y'], central_minutia['x']] - sampled_energy_cell)
                cell['contributions']['MCCce'] += spatial_contrib * calculate_texture_contribution(energy_diff_cell_center, 0, sigma_d_energy)

        mtcc_descriptors.append(cylinder)
        
    return mtcc_descriptors

# --- 5. MATCHING WITH MULTIPLE DISTANCE METRICS ---

def compute_local_similarity_matrix(cylinders1, cylinders2, feature_type):
    """
    Computes a local similarity matrix between two sets of cylinders.
    """
    n1 = len(cylinders1)
    n2 = len(cylinders2)
    similarity_matrix = np.zeros((n1, n2))
    
    for i, cyl1 in enumerate(cylinders1):
        for j, cyl2 in enumerate(cylinders2):
            score = 0
            if cyl1 and cyl2 and cyl1.get('cells') and cyl2.get('cells'):
                for cell1 in cyl1['cells']:
                    for cell2 in cyl2['cells']:
                        if feature_type in cell1['contributions'] and feature_type in cell2['contributions']:
                            score += np.abs(cell1['contributions'][feature_type] - cell2['contributions'][feature_type])
                            
            similarity_matrix[i, j] = 1.0 / (score + 1e-6) 

    return similarity_matrix

def select_top_matching_pairs(similarity_matrix, top_n=20):
    """
    Selects top matching pairs from the similarity matrix (Local Similarity Sort).
    """
    flat_indices = np.argsort(similarity_matrix.flatten())[::-1] 
    
    top_pairs = []
    matched_cyl1_indices = set()
    matched_cyl2_indices = set()
    
    for idx in flat_indices:
        r, c = np.unravel_index(idx, similarity_matrix.shape)
        if r not in matched_cyl1_indices and c not in matched_cyl2_indices:
            top_pairs.append({'cyl1_idx': r, 'cyl2_idx': c, 'score': similarity_matrix[r, c]})
            matched_cyl1_indices.add(r)
            matched_cyl2_indices.add(c)
            if len(top_pairs) >= top_n:
                break
    return top_pairs

def apply_relaxation(top_pairs):
    """
    Apply relaxation-based compatibility scoring (conceptual placeholder).
    """
    relaxed_scores = [p['score'] for p in top_pairs]
    return relaxed_scores

def compute_global_score(relaxed_scores):
    """
    Compute the final global matching score.
    """
    if not relaxed_scores:
        return 0.0
    return np.mean(relaxed_scores) 

def mtcc_matching(cylinders1, cylinders2, feature_type='MCCco'):
    """
    MTCC matching using Local Similarity Sort with Relaxation (LSSR).
    """
    if not cylinders1 or not cylinders2 or any(c is None for c in cylinders1) or any(c is None for c in cylinders2):
        return 0.0 

    similarity_matrix = compute_local_similarity_matrix(cylinders1, cylinders2, feature_type)
    top_pairs = select_top_matching_pairs(similarity_matrix) 
    
    relaxed_scores = apply_relaxation(top_pairs)
    
    final_score = compute_global_score(relaxed_scores)
    return final_score

# --- 6. COMPREHENSIVE EVALUATION FRAMEWORK ---

def calculate_eer(genuine_scores, impostor_scores):
    """
    Calculate Equal Error Rate (EER).
    """
    if not genuine_scores and not impostor_scores:
        return 100.0 

    scores = np.concatenate((genuine_scores, impostor_scores))
    labels = np.concatenate((np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))))

    sorted_indices = np.argsort(scores)
    scores = scores[sorted_indices]
    labels = labels[sorted_indices]

    far = []
    frr = []
    thresholds = scores

    for T in thresholds:
        fa = np.sum((scores >= T) & (labels == 0)) 
        tr = np.sum((scores >= T) & (labels == 1)) 
        fr = np.sum((scores < T) & (labels == 1))  
        tn = np.sum((scores < T) & (labels == 0))  

        current_far = fa / (fa + tn + 1e-6) 
        current_frr = fr / (fr + tr + 1e-6) 
        
        far.append(current_far)
        frr.append(current_frr)

    far = np.array(far)
    frr = np.array(frr)

    diffs = np.abs(far - frr)
    if len(diffs) == 0:
        return 100.0 
    eer_idx = np.argmin(diffs)
    eer = (far[eer_idx] + frr[eer_idx]) / 2.0
    
    return eer * 100 

def evaluate_fvc_dataset(dataset_path, method='MCCco'):
    """
    FVC2002/2004 evaluation protocol.
    This function processes images from the dataset and performs matching.
    """
    print(f"Starting FVC dataset evaluation for {method}...")
    
    num_users = 100       
    images_per_user = 8   
    
    all_cylinders = [[] for _ in range(num_users)] 
    
    print("Generating MTCC descriptors for FVC2002...")
    for finger_idx in range(num_users):
        current_finger_id = finger_idx + 1
        finger_id_str = str(current_finger_id)

        for impression_idx in range(images_per_user):
            current_impression_id = impression_idx + 1
            impression_id_str = str(current_impression_id)

            img_filename = f"{finger_id_str}_{impression_id_str}.tif"
            img_path = os.path.join(dataset_path, img_filename)

            if not os.path.exists(img_path):
                print(f"Warning: Image not found at {img_path}. Skipping this impression.")
                all_cylinders[finger_idx].append(None) 
                continue

            try:
                img, mask = load_and_preprocess(img_path)
                stft_recon_img, orient_map, freq_map, energy_map, coherence = \
                    stft_feature_extraction(img) # Only feature extraction and raw reconstruction
                
                gabor_enhanced = apply_adaptive_gabor_filter(stft_recon_img, orient_map, freq_map) # Apply adaptive Gabor
                enhanced_final = apply_ahe_normalization(stft_recon_img) # Apply AHE (Adaptive Histogram Equalization)
                
                texture_maps = extract_texture_features(enhanced_final, orient_map, freq_map, energy_map)
                skeleton = binarize_and_thin(enhanced_final)
                minutiae = enhanced_minutiae_extraction(skeleton, mask)
                
                cylinders = create_mtcc_cylinders(minutiae, texture_maps)
                all_cylinders[finger_idx].append(cylinders)
            except FileNotFoundError as e:
                print(f"Error processing {img_path}: {e}")
                all_cylinders[finger_idx].append(None) 
            except Exception as e:
                print(f"An unexpected error occurred during processing {img_path}: {e}")
                all_cylinders[finger_idx].append(None) 
    print("MTCC descriptors generation complete.")

    genuine_scores = []
    impostor_scores = []

    print("Starting matching process...")
    # Simulate Genuine Matches (all impressions of a finger against each other)
    for u_idx in range(num_users):
        for i_idx in range(images_per_user):
            for j_idx in range(i_idx + 1, images_per_user):
                cyl1 = all_cylinders[u_idx][i_idx]
                cyl2 = all_cylinders[u_idx][j_idx]
                
                if cyl1 is not None and cyl2 is not None:
                    score = mtcc_matching(cyl1, cyl2, feature_type=method)
                    genuine_scores.append(score)
    
    # Simulate Impostor Matches (first impression of each finger against first impression of every other finger)
    for u1_idx in range(num_users):
        for u2_idx in range(u1_idx + 1, num_users):
            cyl1 = all_cylinders[u1_idx][0]
            cyl2 = all_cylinders[u2_idx][0]
            
            if cyl1 is not None and cyl2 is not None:
                score = mtcc_matching(cyl1, cyl2, feature_type=method)
                impostor_scores.append(score)
    print("Matching complete.")

    eer = calculate_eer(genuine_scores, impostor_scores)
    return eer, genuine_scores, impostor_scores

# --- 7. VISUALIZATION AND DEBUGGING ---
import matplotlib.pyplot as plt

def visualize_mtcc_pipeline(original_image_path, save_debug=True):
    """
    Comprehensive visualization of MTCC pipeline.
    """
    print(f"Visualizing MTCC pipeline for {original_image_path}...")
    try:
        original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            print(f"Error: Could not load image {original_image_path}. Please check the path and file.")
            print("Skipping visualization.")
            return
        
        # Step 1: Core Preprocessing Pipeline
        preprocessed_img, mask = load_and_preprocess(original_image_path)
        
        # Step 2: STFT-based Feature Extraction (no Gabor/AHE yet, just reconstruction and maps)
        stft_reconstructed_img, orientation_map, frequency_map, energy_map, coherence_map = \
            stft_feature_extraction(preprocessed_img) 

        # Step 3: Apply Adaptive Gabor Enhancement
        gabor_enhanced_img = apply_adaptive_gabor_filter(stft_reconstructed_img, orientation_map, frequency_map)

        # Step 4: Apply AHE Normalization
        final_enhanced_img = apply_ahe_normalization(stft_reconstructed_img)

        # Step 5: Texture Feature Extraction (maps are already generated)
        texture_features = extract_texture_features(final_enhanced_img, orientation_map, frequency_map, energy_map)

        # Step 6: Binarize and Thin (for minutiae extraction)
        binarized_img = binarize_and_thin(final_enhanced_img) 
        
        # Step 7: Minutiae Extraction
        minutiae_list = enhanced_minutiae_extraction(binarized_img, mask)

        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        fig.suptitle("MTCC Fingerprint Recognition Pipeline Visualization", fontsize=16)

        # Row 1: Processing Stages
        axes[0, 0].imshow(original_img, cmap='gray'); axes[0, 0].set_title("1. Original Image")
        axes[0, 1].imshow(preprocessed_img, cmap='gray'); axes[0, 1].set_title("2. Segmented")
        axes[0, 2].imshow(stft_reconstructed_img, cmap='gray'); axes[0, 2].set_title("3. STFT Reconstructed")
        axes[0, 3].imshow(gabor_enhanced_img, cmap='gray'); axes[0, 3].set_title("4. Adaptive Gabor Enhanced")

        # Row 2: Feature Maps
        axes[1, 0].imshow(orientation_map, cmap='hsv'); axes[1, 0].set_title("5. Orientation Map")
        axes[1, 1].imshow(frequency_map, cmap='viridis'); axes[1, 1].set_title("6. Frequency Map")
        axes[1, 2].imshow(energy_map, cmap='magma'); axes[1, 2].set_title("7. Energy Map")
        axes[1, 3].imshow(coherence_map, cmap='cividis'); axes[1, 3].set_title("8. Coherence Map")

        # Row 3: Minutiae and Cylinder Structure (conceptual)
        axes[2, 0].imshow(binarized_img, cmap='gray'); axes[2, 0].set_title("9. Binarized/Thinned")
        
        # Visualize minutiae on the thinned image
        minutiae_img = binarized_img.copy()
        if minutiae_img.ndim == 2:
            minutiae_img = cv2.cvtColor(minutiae_img, cv2.COLOR_GRAY2BGR)

        for m in minutiae_list:
            # Minutiae orientation (currently placeholder 0.0) would typically be shown as a line segment
            # For visualization, just show the point.
            color = (0, 0, 255) if m['type'] == 'ending' else (255, 0, 0) # Red for ending, Blue for bifurcation
            cv2.circle(minutiae_img, (m['x'], m['y']), 3, color, -1)
        axes[2, 1].imshow(cv2.cvtColor(minutiae_img, cv2.COLOR_BGR2RGB)); axes[2, 1].set_title(f"10. Minutiae ({len(minutiae_list)} found)")

        axes[2, 2].text(0.5, 0.5, "11. MTCC Cylinders\n(Conceptual)", horizontalalignment='center', verticalalignment='center', transform=axes[2, 2].transAxes, fontsize=12)
        axes[2, 2].axis('off')

        axes[2, 3].text(0.5, 0.5, "12. Texture-based Contributions\n(Conceptual Slices)", horizontalalignment='center', verticalalignment='center', transform=axes[2, 3].transAxes, fontsize=12)
        axes[2, 3].axis('off')

        for ax_row in axes:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_debug:
            plt.savefig("mtcc_pipeline_visualization.png")
        plt.show()

    except Exception as e:
        print(f"Error during visualization: {e}")

# --- 8. PARAMETER OPTIMIZATION ---
MTCC_PARAMETERS = {
    'cylinder_radius': 65,
    'spatial_sectors': 18,  # NS
    'angular_sectors': 5,   # ND  
    'gaussian_spatial': 6,  # σS for spatial contribution (e.g., in calculate_spatial_contribution)
    'gaussian_directional': np.pi/10, # σD for angular/texture difference contribution, scaled from 5*np.pi/36
    'gaussian_frequency': 2, # σF for frequency difference contribution (placeholder, not explicitly in Table IV)
    'gaussian_energy': 2,    # σE for energy difference contribution (placeholder, not explicitly in Table IV)
    'stft_window': 14,
    'stft_overlap': 6,
    'gabor_sigma_x': 8.0, 
    'gabor_sigma_y': 8.0, 
    'gabor_kernel_size': 31, 
    'curved_region_lines': 33, # Conceptual, not directly implemented in current Gabor
    'curved_region_points': 65, # Conceptual
}

# --- Performance Plotting Functions ---
def plot_det_curves(results):
    """Generates and plots DET curves for comparison."""
    plt.figure(figsize=(10, 8))
    plt.title("DET Curves for MTCC Variants (Computed Match Results)")
    plt.xlabel("False Acceptance Rate (FAR) (%)")
    plt.ylabel("False Rejection Rate (FRR) (%)")
    plt.grid(True)
    
    # These are conceptual DET curves. For true DET, you would need the full FAR/FRR arrays
    # returned from calculate_eer across all thresholds, not just the single EER point.
    for method, data in results.items():
        if data['genuine'] and data['impostor']:
            scores = np.concatenate((data['genuine'], data['impostor']))
            labels = np.concatenate((np.ones(len(data['genuine'])), np.zeros(len(data['impostor']))))
            
            # Sort scores and labels for calculating full FAR/FRR for plotting
            sorted_indices = np.argsort(scores)
            scores_sorted = scores[sorted_indices]
            labels_sorted = labels[sorted_indices]

            far_plot = []
            frr_plot = []
            for T in scores_sorted:
                fa = np.sum((scores_sorted >= T) & (labels_sorted == 0)) 
                tr = np.sum((scores_sorted >= T) & (labels_sorted == 1)) 
                fr = np.sum((scores_sorted < T) & (labels_sorted == 1))  
                tn = np.sum((scores_sorted < T) & (labels_sorted == 0))  

                current_far = fa / (fa + tn + 1e-6) 
                current_frr = fr / (fr + tr + 1e-6) 
                
                far_plot.append(current_far * 100) # Convert to percentage
                frr_plot.append(current_frr * 100) # Convert to percentage
            
            plt.plot(far_plot, frr_plot, label=f"{method} (EER: {data['EER']:.2f}%)")
            plt.scatter(data['EER'], data['EER'], marker='x', s=100, color='red') # Mark EER point
        else:
            print(f"Warning: Not enough scores to plot DET curve for {method}.")

    plt.legend()
    plt.xscale('log') 
    plt.xlim(0.1, 100)
    plt.ylim(0.1, 100)
    plt.show()

def plot_roc_curves(results):
    """Generates and plots ROC curves for comparison."""
    plt.figure(figsize=(10, 8))
    plt.title("ROC Curves for MTCC Variants (Computed Match Results)")
    plt.xlabel("False Acceptance Rate (FAR) (%)")
    plt.ylabel("Genuine Attempts Accepted (1-FRR) (%)")
    plt.grid(True)

    for method, data in results.items():
        if data['genuine'] and data['impostor']:
            scores = np.concatenate((data['genuine'], data['impostor']))
            labels = np.concatenate((np.ones(len(data['genuine'])), np.zeros(len(data['impostor']))))
            
            sorted_indices = np.argsort(scores)
            scores_sorted = scores[sorted_indices]
            labels_sorted = labels[sorted_indices]

            far_plot = []
            gar_plot = [] # Genuine Acceptance Rate = 1 - FRR
            for T in scores_sorted:
                fa = np.sum((scores_sorted >= T) & (labels_sorted == 0)) 
                tr = np.sum((scores_sorted >= T) & (labels_sorted == 1)) 
                fr = np.sum((scores_sorted < T) & (labels_sorted == 1))  
                tn = np.sum((scores_sorted < T) & (labels_sorted == 0))  

                current_far = fa / (fa + tn + 1e-6) 
                current_gar = tr / (fr + tr + 1e-6) # GAR = (Genuine Accepted) / Total Genuine
                
                far_plot.append(current_far * 100) # Convert to percentage
                gar_plot.append(current_gar * 100) # Convert to percentage
            
            plt.plot(far_plot, gar_plot, label=f"{method} (EER: {data['EER']:.2f}%)")
            plt.scatter(data['EER'], 100 - data['EER'], marker='x', s=100, color='red') # Mark EER point (GAR at EER)
        else:
            print(f"Warning: Not enough scores to plot ROC curve for {method}.")

    plt.legend()
    plt.xscale('log') 
    plt.xlim(0.1, 100)
    plt.ylim(0, 100)
    plt.show()

def generate_performance_table(results):
    """Prints a performance table."""
    print("\n--- Performance Benchmarks (Based on Actual Descriptor Matching) ---")
    print(f"{'Method':<10} | {'EER (%)':<10}")
    print("-" * 23)
    for method, data in results.items():
        print(f"{method:<10} | {data['EER']:<10.2f}")
    print("-" * 23)
    print("\nNote: The accuracy of the EER depends on the completeness and robustness of conceptual components (e.g., minutiae extraction, texture map accuracy, matching logic).")


# --- TESTING PROTOCOL ---
if __name__ == "__main__":
    # Example Usage:

    # 1. Visualization of the pipeline (requires an actual image file)
    # IMPORTANT: Update this path to a real fingerprint image from your FVC2002 dataset.
    sample_image_path = 'C:/Users/Precision/Onus/Data/FVC-DataSets/DataSets/FVC2002/Db1_a/1_1.tif' # <--- ADJUST THIS TO YOUR REAL IMAGE PATH!
    
    # Ensure the path exists for visualization
    if not os.path.exists(sample_image_path):
        print(f"Error: Visualization image not found at '{sample_image_path}'.")
        print("Please update 'sample_image_path' to a valid fingerprint image to run visualization.")
    else:
        visualize_mtcc_pipeline(sample_image_path, save_debug=True)

    # 2. FVC Dataset Evaluation
    # IMPORTANT: Update this path to your actual FVC2002 Db1_a directory.
    # Images are typically named 001_1.tif, 001_2.tif, ..., 100_8.tif
    dataset_path = 'C:/Users/Precision/Onus/Data/FVC-DataSets/DataSets/FVC2002/Db1_a' # <--- ADJUST THIS TO YOUR REAL DATASET PATH!
    
    methods_to_test = ['MCCo', 'MCCf', 'MCCe', 'MCCco', 'MCCcf', 'MCCce']
    simulated_results = {}
    
    print("\n--- Starting FVC Dataset Evaluation ---")
    # Check if the dataset path exists before proceeding with evaluation
    if not os.path.exists(dataset_path):
        print(f"Error: FVC dataset path not found at '{dataset_path}'.")
        print("Please update 'dataset_path' to a valid FVC2002 Db1_a directory to run the evaluation.")
    else:
        for method in methods_to_test:
            print(f"\nTesting {method}...")
            eer, genuine, impostor = evaluate_fvc_dataset(dataset_path, method)
            simulated_results[method] = {'EER': eer, 'genuine': genuine, 'impostor': impostor}
            print(f"Computed {method} EER: {eer:.2f}%")
        
        # Generate comparison plots and table for the computed results
        plot_det_curves(simulated_results)
        plot_roc_curves(simulated_results)
        generate_performance_table(simulated_results)