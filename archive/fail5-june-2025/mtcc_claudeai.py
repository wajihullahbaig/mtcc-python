import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
import warnings
warnings.filterwarnings('ignore')

def load_image(path):
    """Load and convert image to grayscale."""
    if isinstance(path, str):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = path  # Assume it's already an array
    
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    
    return img.astype(np.float64)

def normalize(img):
    """Apply zero-mean, unit variance normalization."""
    mean_val = np.mean(img)
    std_val = np.std(img)
    
    if std_val == 0:
        return np.zeros_like(img)
    
    normalized = (img - mean_val) / std_val
    return normalized

def segment(img, block_size=16):
    """Block-wise variance segmentation to create foreground mask."""
    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate variance for each block
    variances = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = img[i:i+block_size, j:j+block_size]
            var = np.var(block)
            variances.append(var)
    
    # Use Otsu's threshold on variances
    if len(variances) > 0:
        threshold = np.percentile(variances, 30)  # Lower threshold for fingerprint regions
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = img[i:i+block_size, j:j+block_size]
                if np.var(block) > threshold:
                    mask[i:i+block_size, j:j+block_size] = 255
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def calculate_orientation_map(img, block_size=16):
    """Calculate orientation map using gradient-based method."""
    # Sobel operators
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    h, w = img.shape
    orientation = np.zeros((h, w))
    
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            gx = grad_x[i:i+block_size, j:j+block_size]
            gy = grad_y[i:i+block_size, j:j+block_size]
            
            # Calculate structure tensor components
            gxx = np.sum(gx * gx)
            gyy = np.sum(gy * gy)
            gxy = np.sum(gx * gy)
            
            # Calculate orientation
            denom = gxx - gyy
            if abs(denom) < 1e-10:
                theta = np.pi / 4
            else:
                theta = 0.5 * np.arctan2(2 * gxy, denom)
            
            orientation[i:i+block_size, j:j+block_size] = theta
    
    return orientation

def gabor_enhance(img, orientation_map, freq=0.1):
    """Apply Gabor filtering with overlapping windows and smoothing."""
    h, w = img.shape
    enhanced = np.zeros_like(img)
    weight_map = np.zeros_like(img)
    
    # Gabor kernel parameters
    sigma_x, sigma_y = 3, 3
    kernel_size = 21
    block_size = 16
    overlap = block_size // 2  # 50% overlap
    
    # Create weight matrix for overlapping regions (raised cosine window)
    window_weight = np.zeros((block_size, block_size))
    center = block_size // 2
    for i in range(block_size):
        for j in range(block_size):
            dist_x = abs(i - center) / center
            dist_y = abs(j - center) / center
            # Raised cosine weighting
            weight_x = 0.5 * (1 + np.cos(np.pi * min(dist_x, 1)))
            weight_y = 0.5 * (1 + np.cos(np.pi * min(dist_y, 1)))
            window_weight[i, j] = weight_x * weight_y
    
    # Process with overlapping windows
    for i in range(0, h - block_size + 1, overlap):
        for j in range(0, w - block_size + 1, overlap):
            # Get local region
            end_i = min(i + block_size, h)
            end_j = min(j + block_size, w)
            actual_h = end_i - i
            actual_w = end_j - j
            
            if actual_h < block_size//2 or actual_w < block_size//2:
                continue
                
            local_img = img[i:end_i, j:end_j]
            
            # Get local orientation (average over the block)
            orient_block = orientation_map[i:end_i, j:end_j]
            local_orient = np.mean(orient_block)
            
            # Create adaptive Gabor kernel based on local statistics
            local_freq = freq
            local_variance = np.var(local_img)
            if local_variance > 0:
                # Adapt frequency based on local ridge spacing
                grad_x = cv2.Sobel(local_img, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(local_img, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                if np.mean(grad_mag) > 0:
                    local_freq = min(0.2, max(0.05, freq * (1 + 0.5 * np.mean(grad_mag) / 255)))
            
            # Create Gabor kernel
            y, x = np.mgrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
            
            # Rotate coordinates
            x_rot = x * np.cos(local_orient) + y * np.sin(local_orient)
            y_rot = -x * np.sin(local_orient) + y * np.cos(local_orient)
            
            # Gabor formula with adaptive parameters
            exp_term = np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))
            cos_term = np.cos(2 * np.pi * local_freq * x_rot)
            gabor_kernel = exp_term * cos_term
            
            # Apply Gabor filter
            filtered = cv2.filter2D(local_img, -1, gabor_kernel)
            
            # Apply window weighting for smooth blending
            current_weight = window_weight[:actual_h, :actual_w]
            enhanced[i:end_i, j:end_j] += filtered * current_weight
            weight_map[i:end_i, j:end_j] += current_weight
    
    # Normalize by weight map to handle overlapping regions
    weight_map[weight_map == 0] = 1  # Avoid division by zero
    enhanced = enhanced / weight_map
    
    # Post-processing smoothing to eliminate remaining artifacts
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
    
    return enhanced

def smqt(img, levels=8):
    """Successive Mean Quantization Transform for enhancement."""
    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10)
    img_scaled = (img_norm * (levels - 1)).astype(np.uint8)
    
    enhanced = np.zeros_like(img)
    
    for level in range(levels):
        mask = (img_scaled == level)
        if np.any(mask):
            mean_val = np.mean(img[mask])
            enhanced[mask] = mean_val
    
    # Apply local contrast enhancement
    kernel = np.ones((3, 3)) / 9
    local_mean = cv2.filter2D(enhanced, -1, kernel)
    enhanced = enhanced + 0.5 * (enhanced - local_mean)
    
    return enhanced

def stft_features(img, window=16):
    """Extract orientation, frequency, and energy maps using STFT analysis."""
    h, w = img.shape
    
    # Initialize feature maps
    orientation_map = np.zeros((h, w))
    frequency_map = np.zeros((h, w))
    energy_map = np.zeros((h, w))
    
    # Overlap parameters
    overlap = window // 2
    
    for i in range(0, h - window + 1, overlap):
        for j in range(0, w - window + 1, overlap):
            block = img[i:i+window, j:j+window]
            
            # Apply window function
            window_func = np.hanning(window)[:, None] * np.hanning(window)[None, :]
            windowed_block = block * window_func
            
            # Compute 2D FFT
            fft_block = np.fft.fft2(windowed_block)
            fft_shifted = np.fft.fftshift(fft_block)
            magnitude = np.abs(fft_shifted)
            
            # Convert to polar coordinates
            center = window // 2
            y_coords, x_coords = np.mgrid[:window, :window]
            y_coords = y_coords - center
            x_coords = x_coords - center
            
            # Calculate radius and angle
            radius = np.sqrt(x_coords**2 + y_coords**2)
            angles = np.arctan2(y_coords, x_coords)
            
            # Exclude DC component
            valid_mask = radius > 1
            
            if np.any(valid_mask):
                # Calculate dominant orientation
                weighted_angles = angles[valid_mask] * magnitude[valid_mask]
                orientation = np.arctan2(np.sum(np.sin(2 * weighted_angles)), 
                                       np.sum(np.cos(2 * weighted_angles))) / 2
                
                # Calculate dominant frequency
                weighted_radius = radius[valid_mask] * magnitude[valid_mask]
                frequency = np.sum(weighted_radius) / np.sum(magnitude[valid_mask])
                frequency = frequency / window  # Normalize
                
                # Calculate energy
                energy = np.log(np.sum(magnitude**2) + 1)
                
                # Fill the maps
                end_i = min(i + window, h)
                end_j = min(j + window, w)
                orientation_map[i:end_i, j:end_j] = orientation
                frequency_map[i:end_i, j:end_j] = frequency
                energy_map[i:end_i, j:end_j] = energy
    
    return orientation_map, frequency_map, energy_map

def binarize_thin(img):
    """Improved adaptive thresholding and thinning to reduce bloating."""
    # Normalize image to 0-255
    img_norm = ((img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10) * 255).astype(np.uint8)
    
    # Apply bilateral filter to reduce noise while preserving edges
    img_filtered = cv2.bilateralFilter(img_norm, 9, 75, 75)
    
    # Multi-scale adaptive thresholding for better ridge preservation
    # Method 1: Local adaptive threshold
    binary1 = cv2.adaptiveThreshold(img_filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 15, 8)
    
    # Method 2: Gaussian adaptive threshold  
    binary2 = cv2.adaptiveThreshold(img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 15, 8)
    
    # Combine both methods
    binary_combined = cv2.bitwise_and(binary1, binary2)
    
    # Invert so ridges are white
    binary_combined = 255 - binary_combined
    
    # Advanced morphological operations to reduce bloating
    # Use smaller, more conservative kernels
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Opening to remove small noise
    binary_clean = cv2.morphologyEx(binary_combined, cv2.MORPH_OPEN, kernel_small)
    
    # Very light closing to connect broken ridges without over-bloating
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel_small)
    
    # Distance transform based thinning preprocessing
    # This helps achieve better skeletonization results
    dist_transform = cv2.distanceTransform(binary_clean, cv2.DIST_L2, 5)
    
    # Create a refined binary image using distance transform
    # Keep only pixels that are on the ridge centerlines
    threshold_dist = 0.3 * dist_transform.max()
    binary_refined = (dist_transform > threshold_dist).astype(np.uint8) * 255
    
    # Alternative approach: Use medial axis transform for better ridge centerlines
    from scipy import ndimage
    
    # Apply median filter to smooth before skeletonization
    binary_median = ndimage.median_filter(binary_refined, size=2)
    
    # Skeletonization using scikit-image with better preprocessing
    skeleton_input = binary_median > 0
    
    # Apply morphological thinning in multiple passes for better results
    skeleton = skeletonize(skeleton_input, method='zhang').astype(np.uint8) * 255
    
    # Post-process skeleton to remove artifacts
    # Remove isolated pixels
    kernel_clean = np.ones((3, 3), np.uint8)
    skeleton_clean = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel_clean)
    
    # Remove short spurs (length < 5 pixels)
    skeleton_final = remove_short_branches(skeleton_clean, min_length=5)
    
    return binary_clean, skeleton_final

def remove_short_branches(skeleton, min_length=5):
    """Remove short branches from skeleton to clean up artifacts."""
    h, w = skeleton.shape
    cleaned = skeleton.copy()
    
    # Find all endpoints
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
                 (1, 1), (1, 0), (1, -1), (0, -1)]
    
    changed = True
    while changed:
        changed = False
        endpoints = []
        
        # Find current endpoints
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if cleaned[i, j] > 0:
                    neighbor_count = sum(1 for di, dj in neighbors 
                                       if cleaned[i + di, j + dj] > 0)
                    if neighbor_count == 1:  # Endpoint
                        endpoints.append((i, j))
        
        # Trace short branches from endpoints
        for start_i, start_j in endpoints:
            if cleaned[start_i, start_j] == 0:  # Already removed
                continue
                
            # Trace branch
            path = [(start_i, start_j)]
            current_i, current_j = start_i, start_j
            
            while len(path) < min_length:
                # Find next pixel in branch
                next_found = False
                for di, dj in neighbors:
                    ni, nj = current_i + di, current_j + dj
                    if (0 <= ni < h and 0 <= nj < w and 
                        cleaned[ni, nj] > 0 and (ni, nj) not in path):
                        
                        # Check if this is continuation of branch
                        neighbor_count = sum(1 for ddi, ddj in neighbors 
                                           if cleaned[ni + ddi, nj + ddj] > 0)
                        
                        if neighbor_count <= 2:  # Still on a branch
                            path.append((ni, nj))
                            current_i, current_j = ni, nj
                            next_found = True
                            break
                
                if not next_found:
                    break
            
            # Remove short branch
            if len(path) < min_length:
                for pi, pj in path:
                    cleaned[pi, pj] = 0
                changed = True
    
    return cleaned

def extract_minutiae(skeleton):
    """Extract minutiae using Crossing Number (CN) algorithm."""
    h, w = skeleton.shape
    minutiae = []
    
    # 8-connected neighborhood offsets
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
                 (1, 1), (1, 0), (1, -1), (0, -1)]
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if skeleton[i, j] > 0:  # Ridge pixel
                # Calculate crossing number
                cn = 0
                neighbor_values = []
                
                for di, dj in neighbors:
                    ni, nj = i + di, j + dj
                    val = 1 if skeleton[ni, nj] > 0 else 0
                    neighbor_values.append(val)
                
                # Calculate crossing number
                for k in range(8):
                    cn += abs(neighbor_values[k] - neighbor_values[(k + 1) % 8])
                cn = cn // 2
                
                # Classify minutiae
                if cn == 1:  # Ridge ending
                    # Calculate orientation
                    orientation = calculate_minutia_orientation(skeleton, i, j)
                    minutiae.append({'x': j, 'y': i, 'type': 'ending', 'orientation': orientation})
                elif cn == 3:  # Ridge bifurcation
                    orientation = calculate_minutia_orientation(skeleton, i, j)
                    minutiae.append({'x': j, 'y': i, 'type': 'bifurcation', 'orientation': orientation})
    
    return minutiae

def calculate_minutia_orientation(skeleton, y, x):
    """Calculate orientation of minutia point."""
    # Use a local window to estimate orientation
    window_size = 5
    h, w = skeleton.shape
    
    y_start = max(0, y - window_size)
    y_end = min(h, y + window_size + 1)
    x_start = max(0, x - window_size)
    x_end = min(w, x + window_size + 1)
    
    local_region = skeleton[y_start:y_end, x_start:x_end]
    
    # Find ridge pixels
    ridge_points = np.where(local_region > 0)
    
    if len(ridge_points[0]) > 2:
        # Fit line to ridge points
        y_coords = ridge_points[0] + y_start - y
        x_coords = ridge_points[1] + x_start - x
        
        # Calculate orientation using least squares
        if len(x_coords) > 1:
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            try:
                m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
                orientation = np.arctan(m)
            except:
                orientation = 0
        else:
            orientation = 0
    else:
        orientation = 0
    
    return orientation

def create_cylinders(minutiae, texture_maps, radius=70):
    """Create MTCC descriptors using texture features."""
    orientation_map, frequency_map, energy_map = texture_maps
    h, w = orientation_map.shape
    
    # Cylinder parameters
    ns = 16  # Number of sectors radially
    nd = 6   # Number of sectors angularly
    cylinders = []
    
    for minutia in minutiae:
        cx, cy = minutia['x'], minutia['y']
        
        # Skip minutiae too close to border
        if cx < radius or cy < radius or cx >= w - radius or cy >= h - radius:
            continue
        
        # Create cylinder descriptor
        cylinder = np.zeros((ns, nd, 3))  # 3 channels for orientation, frequency, energy
        
        for r_idx in range(ns):
            for theta_idx in range(nd):
                # Calculate actual position
                r = (r_idx + 1) * radius / ns
                theta = theta_idx * 2 * np.pi / nd
                
                # Convert to Cartesian coordinates relative to minutia
                x_offset = int(r * np.cos(theta))
                y_offset = int(r * np.sin(theta))
                
                sample_x = cx + x_offset
                sample_y = cy + y_offset
                
                # Check bounds
                if 0 <= sample_x < w and 0 <= sample_y < h:
                    # Sample texture features
                    cylinder[r_idx, theta_idx, 0] = orientation_map[sample_y, sample_x]
                    cylinder[r_idx, theta_idx, 1] = frequency_map[sample_y, sample_x]
                    cylinder[r_idx, theta_idx, 2] = energy_map[sample_y, sample_x]
        
        cylinders.append({
            'minutia': minutia,
            'descriptor': cylinder
        })
    
    return cylinders

def match(cylinders1, cylinders2):
    """Calculate similarity score between two sets of cylinders."""
    if len(cylinders1) == 0 or len(cylinders2) == 0:
        return 0.0
    
    # Create distance matrix between all cylinder pairs
    n1, n2 = len(cylinders1), len(cylinders2)
    distances = np.zeros((n1, n2))
    
    for i, cyl1 in enumerate(cylinders1):
        for j, cyl2 in enumerate(cylinders2):
            # Calculate Euclidean distance between descriptors
            desc1 = cyl1['descriptor'].flatten()
            desc2 = cyl2['descriptor'].flatten()
            
            # Normalize descriptors
            desc1 = desc1 / (np.linalg.norm(desc1) + 1e-10)
            desc2 = desc2 / (np.linalg.norm(desc2) + 1e-10)
            
            distance = np.linalg.norm(desc1 - desc2)
            distances[i, j] = distance
    
    # Find best matches using Hungarian algorithm approximation
    # For simplicity, use greedy matching
    total_distance = 0
    matched_pairs = 0
    used_j = set()
    
    for i in range(n1):
        best_j = -1
        best_dist = float('inf')
        
        for j in range(n2):
            if j not in used_j and distances[i, j] < best_dist:
                best_dist = distances[i, j]
                best_j = j
        
        if best_j != -1 and best_dist < 1.0:  # Threshold for valid match
            total_distance += best_dist
            matched_pairs += 1
            used_j.add(best_j)
    
    # Calculate similarity score
    if matched_pairs > 0:
        avg_distance = total_distance / matched_pairs
        similarity = 1.0 / (1.0 + avg_distance)
        # Weight by number of matched pairs
        similarity *= min(matched_pairs / min(len(cylinders1), len(cylinders2)), 1.0)
    else:
        similarity = 0.0
    
    return similarity

def calculate_eer(genuine_scores, impostor_scores):
    """Calculate Equal Error Rate."""
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return 1.0
    
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 1000)
    
    min_diff = float('inf')
    eer = 1.0
    
    for threshold in thresholds:
        far = np.sum(impostor_scores >= threshold) / len(impostor_scores)
        frr = np.sum(genuine_scores < threshold) / len(genuine_scores)
        
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2
    
    return eer

def visualize_pipeline(original, *steps):
    """Visualize all processing stages in a 3x3 grid."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # List of stage names
    stage_names = [
        'Original', 'Normalized', 'Segmented', 'Gabor Enhanced',
        'SMQT Enhanced', 'STFT Energy', 'Binarized', 'Skeleton', 'Minutiae'
    ]
    
    # Prepare all images
    all_images = [original] + list(steps)
    
    for i, (img, name) in enumerate(zip(all_images, stage_names)):
        if i < len(axes):
            if len(img.shape) == 3:  # Handle RGB images
                axes[i].imshow(img, cmap='viridis')
            else:
                axes[i].imshow(img, cmap='gray')
            axes[i].set_title(name)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(all_images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def process_fingerprint(img_path, visualize=False):
    """Complete fingerprint processing pipeline."""
    # Load and process image
    img = load_image(img_path)
    
    # Step 1: Normalize
    normalized = normalize(img)
    
    # Step 2: Segment
    mask = segment(normalized)
    
    # Step 3: Calculate orientation for Gabor
    orientation_map = calculate_orientation_map(normalized)
    
    # Step 4: Gabor enhancement
    gabor_enhanced = gabor_enhance(normalized, orientation_map)
    
    # Step 5: SMQT enhancement
    smqt_enhanced = smqt(gabor_enhanced)
    
    # Step 6: STFT features
    stft_orientation, stft_frequency, stft_energy = stft_features(smqt_enhanced)
    
    # Step 7: Binarization and thinning
    binary, skeleton = binarize_thin(smqt_enhanced)
    
    # Step 8: Extract minutiae
    minutiae = extract_minutiae(skeleton)
    
    # Step 9: Create minutiae image for visualization
    minutiae_img = skeleton.copy()
    for m in minutiae:
        cv2.circle(minutiae_img, (m['x'], m['y']), 3, 255, -1)
    
    # Step 10: Create MTCC descriptors
    texture_maps = (stft_orientation, stft_frequency, stft_energy)
    cylinders = create_cylinders(minutiae, texture_maps)
    
    if visualize:
        visualize_pipeline(
            img, normalized, mask, gabor_enhanced, 
            smqt_enhanced, stft_energy, binary, skeleton, minutiae_img
        )
    
    return {
        'minutiae': minutiae,
        'cylinders': cylinders,
        'texture_maps': texture_maps,
        'steps': {
            'original': img,
            'normalized': normalized,
            'mask': mask,
            'gabor': gabor_enhanced,
            'smqt': smqt_enhanced,
            'stft_energy': stft_energy,
            'binary': binary,
            'skeleton': skeleton,
            'minutiae_img': minutiae_img
        }
    }

import os
import glob
from collections import defaultdict
import time
import json

def test_fvc_database(fvc_path, db_name="DB1_A", max_images=None, save_results=True, visualize_samples=False):
    """
    Test MTCC system on FVC database with standard evaluation protocol.
    
    Args:
        fvc_path: Path to FVC database directory
        db_name: Database name (e.g., "DB1_A", "DB2_A", etc.)
        max_images: Maximum number of images to process (None for all)
        save_results: Save detailed results to JSON
        visualize_samples: Show processing pipeline for first few images
    
    Returns:
        dict: Complete evaluation results including EER, timings, etc.
    """
    print(f"Testing MTCC system on FVC {db_name}")
    print("=" * 50)
    
    # Find image files
    db_path = os.path.join(fvc_path, db_name)
    if not os.path.exists(db_path):
        print(f"Error: Database path {db_path} not found")
        return None
    
    # Standard FVC naming: XXX_Y.tif where XXX is finger ID, Y is impression
    image_files = glob.glob(os.path.join(db_path, "*.tif"))
    if not image_files:
        image_files = glob.glob(os.path.join(db_path, "*.bmp"))
    if not image_files:
        image_files = glob.glob(os.path.join(db_path, "*.png"))
    
    if not image_files:
        print(f"No image files found in {db_path}")
        return None
    
    print(f"Found {len(image_files)} images")
    
    if max_images:
        image_files = sorted(image_files)[:max_images]
        print(f"Processing first {len(image_files)} images")
    
    # Parse image metadata
    database = defaultdict(list)
    for img_path in image_files:
        filename = os.path.basename(img_path)
        # Extract finger ID and impression number
        try:
            # Standard format: 001_1.tif, 001_2.tif, etc.
            finger_id = filename.split('_')[0]
            impression = filename.split('_')[1].split('.')[0]
            database[finger_id].append({
                'path': img_path,
                'finger_id': finger_id,
                'impression': impression,
                'filename': filename
            })
        except:
            print(f"Warning: Could not parse filename {filename}")
    
    print(f"Found {len(database)} unique fingers")
    print(f"Impressions per finger: {[len(impressions) for impressions in database.values()]}")
    
    # Process all images
    print("\nProcessing images...")
    processed_data = {}
    processing_times = []
    failed_images = []
    
    total_images = sum(len(impressions) for impressions in database.values())
    current_image = 0
    
    for finger_id, impressions in database.items():
        for impression_data in impressions:
            current_image += 1
            img_path = impression_data['path']
            filename = impression_data['filename']
            
            print(f"Processing {filename} ({current_image}/{total_images})")
            
            try:
                start_time = time.time()
                
                # Show visualization for first few images
                show_viz = visualize_samples and current_image <= 3
                
                result = process_fingerprint(img_path, visualize=show_viz)
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Store results
                processed_data[filename] = {
                    'finger_id': finger_id,
                    'impression': impression_data['impression'],
                    'minutiae_count': len(result['minutiae']),
                    'cylinder_count': len(result['cylinders']),
                    'cylinders': result['cylinders'],
                    'processing_time': processing_time
                }
                
                print(f"  -> {len(result['minutiae'])} minutiae, {len(result['cylinders'])} cylinders, {processing_time:.2f}s")
                
            except Exception as e:
                print(f"  -> FAILED: {str(e)}")
                failed_images.append(filename)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(processed_data)}/{total_images}")
    print(f"Failed images: {len(failed_images)}")
    print(f"Average processing time: {np.mean(processing_times):.2f}s")
    
    # Matching phase
    print("\nPerforming matching...")
    genuine_scores = []
    impostor_scores = []
    matching_times = []
    
    # Generate genuine pairs (same finger, different impressions)
    genuine_pairs = []
    for finger_id, impressions in database.items():
        if len(impressions) >= 2:
            for i in range(len(impressions)):
                for j in range(i + 1, len(impressions)):
                    file1 = impressions[i]['filename']
                    file2 = impressions[j]['filename']
                    if file1 in processed_data and file2 in processed_data:
                        genuine_pairs.append((file1, file2))
    
    print(f"Genuine pairs: {len(genuine_pairs)}")
    
    # Generate impostor pairs (different fingers, first impression each)
    impostor_pairs = []
    finger_list = list(database.keys())
    for i in range(len(finger_list)):
        for j in range(i + 1, min(i + 50, len(finger_list))):  # Limit impostor pairs
            finger1, finger2 = finger_list[i], finger_list[j]
            if len(database[finger1]) > 0 and len(database[finger2]) > 0:
                file1 = database[finger1][0]['filename']  # First impression
                file2 = database[finger2][0]['filename']  # First impression
                if file1 in processed_data and file2 in processed_data:
                    impostor_pairs.append((file1, file2))
    
    print(f"Impostor pairs: {len(impostor_pairs)}")
    
    # Perform genuine matching
    print("Computing genuine scores...")
    for i, (file1, file2) in enumerate(genuine_pairs):
        if i % 50 == 0:
            print(f"  Genuine: {i}/{len(genuine_pairs)}")
        
        start_time = time.time()
        cylinders1 = processed_data[file1]['cylinders']
        cylinders2 = processed_data[file2]['cylinders']
        score = match(cylinders1, cylinders2)
        matching_time = time.time() - start_time
        
        genuine_scores.append(score)
        matching_times.append(matching_time)
    
    # Perform impostor matching
    print("Computing impostor scores...")
    for i, (file1, file2) in enumerate(impostor_pairs):
        if i % 50 == 0:
            print(f"  Impostor: {i}/{len(impostor_pairs)}")
        
        start_time = time.time()
        cylinders1 = processed_data[file1]['cylinders']
        cylinders2 = processed_data[file2]['cylinders']
        score = match(cylinders1, cylinders2)
        matching_time = time.time() - start_time
        
        impostor_scores.append(score)
        matching_times.append(matching_time)
    
    # Calculate metrics
    eer = calculate_eer(genuine_scores, impostor_scores)
    
    # Additional metrics
    genuine_mean = np.mean(genuine_scores) if genuine_scores else 0
    genuine_std = np.std(genuine_scores) if genuine_scores else 0
    impostor_mean = np.mean(impostor_scores) if impostor_scores else 0
    impostor_std = np.std(impostor_scores) if impostor_scores else 0
    
    avg_matching_time = np.mean(matching_times) if matching_times else 0
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    
    # Compile results
    results = {
        'database': db_name,
        'total_images': total_images,
        'processed_images': len(processed_data),
        'failed_images': len(failed_images),
        'unique_fingers': len(database),
        'genuine_pairs': len(genuine_pairs),
        'impostor_pairs': len(impostor_pairs),
        'eer': eer,
        'genuine_scores': {
            'mean': genuine_mean,
            'std': genuine_std,
            'min': min(genuine_scores) if genuine_scores else 0,
            'max': max(genuine_scores) if genuine_scores else 0,
            'count': len(genuine_scores)
        },
        'impostor_scores': {
            'mean': impostor_mean,
            'std': impostor_std,
            'min': min(impostor_scores) if impostor_scores else 0,
            'max': max(impostor_scores) if impostor_scores else 0,
            'count': len(impostor_scores)
        },
        'timing': {
            'avg_processing_time': avg_processing_time,
            'avg_matching_time': avg_matching_time,
            'total_processing_time': sum(processing_times),
            'total_matching_time': sum(matching_times)
        },
        'minutiae_stats': {
            'avg_minutiae': np.mean([data['minutiae_count'] for data in processed_data.values()]),
            'avg_cylinders': np.mean([data['cylinder_count'] for data in processed_data.values()])
        }
    }
    
    # Print results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Database: {db_name}")
    print(f"Images processed: {len(processed_data)}/{total_images}")
    print(f"Equal Error Rate (EER): {eer:.4f} ({eer*100:.2f}%)")
    print(f"Genuine scores: {genuine_mean:.4f} ± {genuine_std:.4f}")
    print(f"Impostor scores: {impostor_mean:.4f} ± {impostor_std:.4f}")
    print(f"Average minutiae per image: {results['minutiae_stats']['avg_minutiae']:.1f}")
    print(f"Average cylinders per image: {results['minutiae_stats']['avg_cylinders']:.1f}")
    print(f"Average processing time: {avg_processing_time:.2f}s")
    print(f"Average matching time: {avg_matching_time:.4f}s")
    
    # Save detailed results
    if save_results:
        output_file = f"mtcc_results_{db_name}_{len(processed_data)}images.json"
        
        # Prepare data for JSON (remove non-serializable objects)
        json_results = results.copy()
        json_results['genuine_scores_list'] = genuine_scores
        json_results['impostor_scores_list'] = impostor_scores
        json_results['failed_images'] = failed_images
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")
    
    return results

def test_multiple_fvc_databases(fvc_base_path, databases=None, max_images_per_db=None):
    """Test multiple FVC databases and compare results."""
    if databases is None:
        databases = ["DB1_A", "DB2_A", "DB3_A", "DB4_A"]
    
    all_results = {}
    
    for db_name in databases:
        print(f"\n{'='*60}")
        print(f"TESTING DATABASE: {db_name}")
        print(f"{'='*60}")
        
        db_path = os.path.join(fvc_base_path, db_name)
        if not os.path.exists(db_path):
            print(f"Skipping {db_name} - path not found: {db_path}")
            continue
        
        results = test_fvc_database(fvc_base_path, db_name, max_images_per_db)
        if results:
            all_results[db_name] = results
    
    # Summary comparison
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        print(f"{'Database':<10} {'EER':<8} {'Genuine':<10} {'Impostor':<10} {'Minutiae':<10}")
        print("-" * 60)
        
        for db_name, results in all_results.items():
            eer = results['eer']
            genuine_mean = results['genuine_scores']['mean']
            impostor_mean = results['impostor_scores']['mean']
            avg_minutiae = results['minutiae_stats']['avg_minutiae']
            
            print(f"{db_name:<10} {eer:.4f}   {genuine_mean:.4f}    {impostor_mean:.4f}    {avg_minutiae:.1f}")
    
    return all_results

def quick_fvc_test(fvc_path, max_images=20):
    """Quick test with limited images for development."""
    print("Quick FVC Test (limited images)")
    return test_fvc_database(fvc_path, "DB1_A", max_images=max_images, visualize_samples=True)

# Example usage and testing
def test_mtcc_system():
    """Test the MTCC system with synthetic data."""
    print("Testing MTCC Fingerprint Recognition System...")
    
    # Create synthetic fingerprint-like image
    img = np.zeros((200, 200))
    
    # Add some ridge-like patterns
    for i in range(0, 200, 8):
        if i % 16 == 0:
            img[i:i+4, :] = 128
    
    # Add some noise
    noise = np.random.normal(0, 10, img.shape)
    img = img + noise
    
    # Process the image
    result = process_fingerprint(img, visualize=True)
    
    print(f"Found {len(result['minutiae'])} minutiae")
    print(f"Created {len(result['cylinders'])} MTCC descriptors")
    
    # Test matching with itself (should give high score)
    if len(result['cylinders']) > 0:
        self_match_score = match(result['cylinders'], result['cylinders'])
        print(f"Self-matching score: {self_match_score:.4f}")
    
    return result

if __name__ == "__main__":
    # Example usage:
    # For quick testing:
    quick_fvc_test(R"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002\FVC2002", max_images=2)
    
    # For full database testing:
    # test_fvc_database("/path/to/FVC2004", "DB1_A")
    
    # For multiple databases:
    # test_multiple_fvc_databases("/path/to/FVC2004")
    
    test_mtcc_system()