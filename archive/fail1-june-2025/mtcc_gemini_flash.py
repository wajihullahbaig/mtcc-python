import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import uniform_filter # Can be used for smoothing, but Gabor is specific
import matplotlib.pyplot as plt
import os
from glob import glob
# Removed math.atan2 as np.arctan2 is used for array operations
from math import pi, sqrt, cos, sin, exp, log 

# --- Helper function for angle normalization (Eq 8, 9) ---
def get_directional_difference(angle1, angle2):
    """
    Calculates normalized angular difference in radians.
    Ensures the difference is in the range [-pi, pi).
    (Ref: Eq 8, 9)
    """
    diff = angle1 - angle2
    if diff < -pi:
        diff += 2 * pi
    elif diff >= pi:
        diff -= 2 * pi
    return diff

# --- Image Processing Helper Functions ---

def segment_image(img_float, config, viz):
    """
    Performs block-wise variance segmentation and creates a mask.
    (Ref: IV.A. "blockwise variance based image segmentation method")
    Input: img_float (float32, 0.0-1.0)
    Output: segmented_img (float32, 0.0-1.0), mask (uint8, 0 or 255)
    """
    rows, cols = img_float.shape
    # CHANGED: Use a more appropriate block size for segmentation variance
    # This prevents the variance image from being too flat, leading to a black mask.
    block_s_seg = config.block_size * 2 # Changed from * 4 to * 2 (e.g., 32x32)
    variance_img = np.zeros_like(img_float, dtype=np.float32)

    for r in range(0, rows, block_s_seg):
        for c in range(0, cols, block_s_seg):
            block = img_float[r:min(r + block_s_seg, rows), c:min(c + block_s_seg, cols)]
            if block.size > 0:
                var = np.var(block)
                variance_img[r:min(r + block_s_seg, rows), c:min(c + block_s_seg, cols)] = var

    # Normalize variance image to 0-255 for thresholding
    variance_img_uint8 = cv2.normalize(variance_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu's thresholding for mask
    _, mask = cv2.threshold(variance_img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Smooth mask using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Apply mask to float image. Ensure areas outside mask are truly zero.
    segmented_img = img_float * (mask / 255.0) 
    
    viz.show_image(segmented_img, f"3. Image Segmented: {img_float.shape}", "image_segmented")
    viz.show_image(mask, f"4. Image Mask: {img_float.shape}", "image_mask")
    return segmented_img, mask

def stft_feature_extraction(img_float, config, viz):
    """
    Performs Short-Time Fourier Transform (STFT) analysis to generate
    smooth orientation, frequency, and energy fields for the entire image.
    (Ref: IV.C. "STFT Analysis - orientation, frequency and energy image generation")
    Input: img_float (float32, 0.0-1.0)
    Output: orientation_field (float32, radians), frequency_field (float32, 0-1), energy_field (float32, 0-1)
    """
    rows, cols = img_float.shape
    block_size = config.block_size
    overlap_stride = config.overlap_stride

    # Determine dimensions for low-resolution maps (block-wise features)
    num_blocks_r = (rows - block_size) // overlap_stride + 1
    num_blocks_c = (cols - block_size) // overlap_stride + 1
    
    # Initialize low-resolution feature accumulators
    block_orient_cos_sum = np.zeros((num_blocks_r, num_blocks_c), dtype=np.float32)
    block_orient_sin_sum = np.zeros((num_blocks_r, num_blocks_c), dtype=np.float32)
    block_freq_sum = np.zeros((num_blocks_r, num_blocks_c), dtype=np.float32)
    block_energy_sum = np.zeros((num_blocks_r, num_blocks_c), dtype=np.float32)
    block_count = np.zeros((num_blocks_r, num_blocks_c), dtype=np.int32)

    window_1d = np.hanning(block_size)
    window_2d = np.outer(window_1d, window_1d)

    for r_idx in range(num_blocks_r):
        r_start = r_idx * overlap_stride
        for c_idx in range(num_blocks_c):
            c_start = c_idx * overlap_stride
            block = img_float[r_start : r_start + block_size, c_start : c_start + block_size]

            # REMOVED: If the block is mostly flat (e.g., background), skip to avoid noise
            # Removing this ensures feature fields are calculated for all blocks,
            # letting the segmentation mask handle invalid regions later.
            # if np.std(block) < 1e-3: 
            #     continue 

            block_fft = np.fft.fft2(block * window_2d)
            block_fft_shifted = np.fft.fftshift(block_fft)
            magnitude_spectrum = np.abs(block_fft_shifted)

            center_x, center_y = block_size // 2, block_size // 2
            # Mask out the center (DC component) to find dominant *ridge* frequency
            # Use a slightly larger mask for robustness
            magnitude_spectrum[center_x - 2 : center_x + 3, center_y - 2 : center_y + 3] = 0

            # Find peak in magnitude spectrum
            max_mag = np.max(magnitude_spectrum)
            if max_mag > 1e-6: # Check for non-trivial peak
                peak_r, peak_c = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
                freq_r_norm = (peak_r - center_x) / block_size
                freq_c_norm = (peak_c - center_y) / block_size

                # Ridge orientation is perpendicular to dominant frequency direction
                # FIXED: Use np.arctan2 for array-like inputs
                orientation_raw = np.arctan2(freq_c_norm, freq_r_norm) / 2
                
                frequency = sqrt(freq_r_norm**2 + freq_c_norm**2)
                energy = log(max_mag + 1e-10) # Using max_mag as energy, add epsilon for log(0)

                # Store orientation as complex numbers for proper averaging
                block_orient_cos_sum[r_idx, c_idx] += cos(2 * orientation_raw)
                block_orient_sin_sum[r_idx, c_idx] += sin(2 * orientation_raw)
                block_freq_sum[r_idx, c_idx] += frequency
                block_energy_sum[r_idx, c_idx] += energy
                block_count[r_idx, c_idx] += 1

    # Handle blocks with no contribution (avoid division by zero)
    # Set to 1 to prevent division by zero for untouched blocks; their sum remains 0.
    block_count[block_count == 0] = 1 

    # Average low-resolution maps
    avg_block_orient_cos = block_orient_cos_sum / block_count
    avg_block_orient_sin = block_orient_sin_sum / block_count
    avg_block_freq = block_freq_sum / block_count
    avg_block_energy = block_energy_sum / block_count

    # Convert averaged complex orientation back to angle
    # FIXED: Use np.arctan2 for array-like inputs
    low_res_orient = np.arctan2(avg_block_orient_sin, avg_block_orient_cos) / 2

    # Interpolate to full image size for smooth fields (using cubic interpolation)
    # Ensure interpolation respects the original image dimensions
    orientation_field = cv2.resize(low_res_orient, (cols, rows), interpolation=cv2.INTER_CUBIC)
    frequency_field = cv2.resize(avg_block_freq, (cols, rows), interpolation=cv2.INTER_CUBIC)
    energy_field = cv2.resize(avg_block_energy, (cols, rows), interpolation=cv2.INTER_CUBIC)

    # Normalize frequency and energy fields to 0-1 range
    # Clamp values for robust normalization if interpolation creates out-of-range values
    frequency_field = np.clip(frequency_field, 0, np.max(frequency_field))
    energy_field = np.clip(energy_field, 0, np.max(energy_field))

    # Perform normalization after clipping
    frequency_field = cv2.normalize(frequency_field, None, 0, 1, cv2.NORM_MINMAX)
    energy_field = cv2.normalize(energy_field, None, 0, 1, cv2.NORM_MINMAX)

    # --- Diagnostic prints ---
    print(f"Orientation field range (radians): [{np.min(orientation_field):.4f}, {np.max(orientation_field):.4f}]")
    print(f"Frequency field range (0-1 normalized): [{np.min(frequency_field):.4f}, {np.max(frequency_field):.4f}]")
    print(f"Energy field range (0-1 normalized): [{np.min(energy_field):.4f}, {np.max(energy_field):.4f}]")

    # Visualize results (Grayscale as requested).
    # Convert radian orientation to a displayable grayscale: map [-pi/2, pi/2] to [0, 1].
    # This range is then displayed by imshow, where 0 is black, 1 is white.
    orientation_display = (orientation_field + (pi/2)) / pi 
    viz.show_image(orientation_display, f"STFT Orientation Field: {img_float.shape}", "stft_orientation", cmap='gray')
    viz.show_image(frequency_field, f"STFT Frequency Field: {img_float.shape}", "stft_frequency", cmap='gray')
    viz.show_image(energy_field, f"STFT Energy Field: {img_float.shape}", "stft_energy", cmap='gray')
                   
    return orientation_field, frequency_field, energy_field

def gabor_enhance(img_float, orientation_field, frequency_field, config, viz):
    """
    Applies Gabor filtering for ridge enhancement and smoothing,
    using spatially varying orientation and frequency fields.
    This implementation creates a bank of kernels and applies the best match per pixel
    to minimize blockiness.
    (Ref: IV.B. "Gabor filtering")
    Input: img_float (float32, 0.0-1.0), orientation_field (float32, radians),
           frequency_field (float32, 0.0-1.0)
    Output: enhanced_img (float32, 0.0-1.0)
    """
    rows, cols = img_float.shape
    enhanced_img = np.zeros_like(img_float, dtype=np.float32)

    # Gabor parameters from config
    ksize = config.gabor_ksize
    sigma = config.gabor_sigma
    gamma = config.gabor_gamma
    
    # Create a bank of Gabor kernels for discrete orientations (0 to pi for ridge orientation)
    orientations_bank = np.linspace(0, pi, config.gabor_theta_steps, endpoint=False) 

    # Pre-compute Gabor filter responses for the entire image for each angle in the bank
    filtered_images_per_angle = []
    for theta_val in orientations_bank:
        lambda_val = config.gabor_lambd_base 
        
        kernel = cv2.getGaborKernel(ksize, sigma, theta_val, lambda_val, gamma, 0, ktype=cv2.CV_32F)
        # Normalize kernel to have zero mean (important for enhancement)
        kernel -= kernel.sum() / (ksize[0] * ksize[1])
        
        # Apply filter to the full image with border replication
        filtered_img_single_angle = cv2.filter2D(img_float, cv2.CV_32F, kernel, borderType=cv2.BORDER_REPLICATE)
        filtered_images_per_angle.append(filtered_img_single_angle)

    # For each pixel, select the Gabor response from the best-matching filter
    for r in range(rows):
        for c in range(cols):
            # Ensure pixel is within the mask area for meaningful orientation
            if (r >= config.img_mask.shape[0] or c >= config.img_mask.shape[1] or config.img_mask[r,c] == 0):
                enhanced_img[r, c] = 0.5 # Set to mid-gray for masked out areas
                continue

            # Local orientation from the STFT field, maps [-pi/2, pi/2]
            local_theta = orientation_field[r,c]
            # Adjust STFT's [-pi/2, pi/2] to [0, pi) for matching Gabor kernels' orientation
            # This handles the pi-periodicity of ridge orientation.
            local_theta_normalized_for_gabor = (local_theta + pi/2) % pi 
            
            # Find closest pre-defined orientation in the bank
            # We use the angular difference taking into account pi-periodicity
            angular_distances = [min(abs(local_theta_normalized_for_gabor - b_theta), 
                                     pi - abs(local_theta_normalized_for_gabor - b_theta)) 
                                 for b_theta in orientations_bank]
            closest_idx = np.argmin(angular_distances)
            
            enhanced_img[r, c] = filtered_images_per_angle[closest_idx][r, c]

    # Normalize enhanced_img to 0-1 range
    enhanced_img = cv2.normalize(enhanced_img, None, 0, 1, cv2.NORM_MINMAX)
    
    viz.show_image(enhanced_img, f"Gabor Enhanced: {img_float.shape}", "gabor_enhanced")
    return enhanced_img

def clahe_enhance(img_float, config, viz):
    """
    Approximation of Successive Mean Quantize Transform (SMQT) using CLAHE.
    (Ref: IV.B. "SMQT is applied to produce the final fingerprint image")
    Input: img_float (float32, 0.0-1.0)
    Output: enhanced_img (float32, 0.0-1.0)
    """
    # CLAHE operates on uint8. Convert float to uint8 for CLAHE.
    img_uint8 = (img_float * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=config.clahe_tile_grid_size)
    enhanced_img_uint8 = clahe.apply(img_uint8)
    enhanced_img_float = enhanced_img_uint8.astype(np.float32) / 255.0 # Convert back to float
    
    viz.show_image(enhanced_img_float, f"2. Image Enhanced (CLAHE): {img_float.shape}", "clahe_enhanced")
    return enhanced_img_float

def binarize_image(img_float, mask, config, viz):
    """
    Binarizes the enhanced image and applies the segmentation mask.
    Input: img_float (float32, 0.0-1.0), mask (uint8, 0 or 255)
    Output: binary_img (uint8, 0 or 255)
    """
    # Adaptive thresholding expects uint8.
    img_uint8 = (img_float * 255).astype(np.uint8)
    binary_img = cv2.adaptiveThreshold(img_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 2)
    # Apply mask
    binary_img = cv2.bitwise_and(binary_img, binary_img, mask=mask)
    
    viz.show_image(binary_img, f"5. Binary Image: {img_float.shape}", "binary_image")
    return binary_img

def thin_image(binary_img, config, viz):
    """
    Thins the binary image using skeletonization.
    Input: binary_img (uint8, 0 or 255)
    Output: thinned_img (uint8, 0 or 255)
    """
    # skeletonize expects boolean image (True for foreground)
    thinned_img = skeletonize(binary_img == 255)
    thinned_img = thinned_img.astype(np.uint8) * 255 # Convert back to uint8 (0 or 255)
    
    viz.show_image(thinned_img, f"Thinned Image: {binary_img.shape}", "thinned_image")
    return thinned_img

# --- Minutiae Extraction Helper Function ---

def extract_minutiae(thinned_img, mask, orientation_field, config, viz):
    """
    Extracts minutiae (ridge endings and bifurcations) from a thinned image.
    Minutia format: (x, y, angle, type, quality) where x=row, y=col.
    Angle is based on the local orientation field.
    """
    minutiae = []
    rows, cols = thinned_img.shape
    # Pad image to handle border pixels for 8-neighbor analysis
    padded_img = cv2.copyMakeBorder(thinned_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if padded_img[r, c] == 255: # If it's a ridge pixel
                # Get 8-neighbor sum (crossing number concept)
                # Neighbors in clockwise order from top-left:
                # N1 N2 N3
                # N8    N4
                # N7 N6 N5
                neighbors_vals = [
                    padded_img[r - 1, c - 1], padded_img[r - 1, c], padded_img[r - 1, c + 1],
                    padded_img[r, c + 1], padded_img[r + 1, c + 1], padded_img[r + 1, c],
                    padded_img[r + 1, c - 1], padded_img[r, c - 1]
                ]
                # Convert to 0/1 for counting
                neighbors_bin = [1 if n == 255 else 0 for n in neighbors_vals]
                
                cn = 0
                for i_nb in range(8):
                    cn += abs(neighbors_bin[i_nb] - neighbors_bin[(i_nb + 1) % 8])
                cn /= 2

                m_type = None
                if cn == 1: # Ridge ending
                    m_type = 'ending'
                elif cn == 3: # Bifurcation
                    m_type = 'bifurcation'
                else:
                    continue # Not a minutia point

                # Minutia coordinates (unpad them to original image coords)
                minutia_x, minutia_y = r - 1, c - 1

                # Ensure minutia is within the valid mask area
                if mask[minutia_x, minutia_y] == 0: # Check mask before using orientation field
                    continue

                # Minutiae angle: Use the orientation field at the minutia point.
                # orientation_field is in radians (e.g., [-pi/2, pi/2])
                m_angle = orientation_field[minutia_x, minutia_y]

                # Placeholder for quality (simplified)
                m_quality = config.minutiae_quality_threshold + 0.1 

                minutiae.append((minutia_x, minutia_y, m_angle, m_type, m_quality))
    
    return minutiae

# --- 0. Configuration Class (SOLID: SRP, OCP for future extension) ---
class Config:
    """Manages all system parameters and visualization flags."""

    def __init__(self):
        # Image Processing Parameters
        self.block_size = 16  # STFT block size
        self.overlap_stride = 8 # STFT overlap stride (half of block_size for good overlap)
        
        self.gabor_ksize = (31, 31) # Gabor kernel size (larger for smoothing)
        self.gabor_sigma = 4.0 # Gabor sigma (standard deviation of Gaussian envelope)
        self.gabor_theta_steps = 16 # Number of Gabor orientations (0 to pi, finer steps for bank)
        self.gabor_gamma = 0.5 # Gabor aspect ratio (width to height of Gaussian envelope)
        # CHANGED: Adjusted Gabor wavelength for better ridge enhancement
        self.gabor_lambd_base = 18.0 # Base wavelength for Gabor (pixels per ridge cycle)

        self.clahe_clip_limit = 2.0 # CLAHE clip limit (approximation for SMQT)
        self.clahe_tile_grid_size = (8, 8) # CLAHE tile grid size

        # Minutiae Extraction Parameters
        self.minutiae_quality_threshold = 0.5 # Simplified quality threshold

        # MCC/MTCC Parameters (based on Table IV)
        self.R = 65            # Radius of the cylinder
        self.Ns = 18           # Number of spatial divisions (Ns x Ns grid)
        self.ND = 5            # Number of directional divisions
        self.sigma_s = 6       # Sigma for spatial Gaussian
        # Sigma for directional Gaussian, paper uses 36 degrees (pi/5 rad)
        self.sigma_d = 36 * pi / 180 
        self.sigma_f = pi / 4  # Sigma for frequency Gaussian (custom, tuned for cyclic interpretation)
        self.sigma_e = pi / 4  # Sigma for energy Gaussian (custom, tuned for cyclic interpretation)
        self.delta_s = (2 * self.R) / self.Ns # Spatial cell size
        self.delta_d = (2 * pi) / self.ND # Directional cell size (2pi for full circle)

        self.min_neighbors = 4 # Min NP in table IV (min neighbors for valid cylinder)
        self.min_valid_cells = 0.20 # minME in table IV (min % of valid cells for cylinder)

        # Matching Parameters (Simplified LSSR)
        self.match_radius_threshold = 20 # Max distance for initial minutia pairing (unused in current LSS)
        self.top_nr_matches = 10 # Top nr matching minutiae for LSS initial candidates
        self.final_top_np_matches = 5 # Final np matches after relaxation (simplified selection)

        # Dataset Parameters
        self.dataset_root = r"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2000\FVC2000\\"
        self.dataset_name = "db1_a" # Default dataset to load

        # Placeholder for image mask, populated during processing
        self.img_mask = None 

        # Visualization Control (Dictionary for precise step control)
        self._show_step_viz = {
            "image_loaded": True,
            "image_segmented": True,
            "image_mask": True,
            "stft_orientation": True, # Grayscale
            "stft_frequency": True,   # Grayscale
            "stft_energy": True,      # Grayscale
            "gabor_enhanced": True,
            "clahe_enhanced": True, # Renamed from smqt_enhanced
            "binary_image": True,
            "thinned_image": True,
            "minutiae_plot": True,
            "cylinder_code_visualization": True,
            "matching_process": True,
            "final_match_score": True,
        }

    def should_show_viz(self, step_name):
        return self._show_step_viz.get(step_name, False)

# --- 1. Visualizer Class (SOLID: SRP) ---
class Visualizer:
    """Handles all visualization aspects."""

    def __init__(self, config: Config):
        self.config = config

    def show_image(self, img, title, step_name, cmap='gray'):
        """Displays an image if its corresponding visualization flag is enabled."""
        if self.config.should_show_viz(step_name):
            plt.figure(figsize=(6, 6))
            # Ensure img is suitable for imshow (0-1 float or 0-255 uint8)
            if img.dtype == np.float32:
                plt.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
            else: # Assume uint8
                plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    def plot_minutiae(self, img, minutiae, title, step_name):
        """Plots minutiae on an image."""
        if self.config.should_show_viz(step_name):
            plt.figure(figsize=(8, 8))
            # Ensure img is suitable for imshow
            if img.dtype == np.float32:
                plt.imshow(img, cmap='gray', vmin=0.0, vmax=1.0)
            else: # Assume uint8
                plt.imshow(img, cmap='gray', vmin=0, vmax=255)
            plt.title(title)
            for m in minutiae:
                x, y, angle, m_type, quality = m
                color = 'red' if m_type == 'ending' else 'blue'
                plt.plot(y, x, 'o', color=color, markersize=5)
                # Draw angle line from minutia point
                length = 20 # Length of the orientation line
                # angle is from horizontal (column) axis. x is row, y is column.
                # dx_plot = length * cos(angle) means change in plot's X (column)
                # dy_plot = length * sin(angle) means change in plot's Y (row)
                # Matplotlib's arrow expects (x_start, y_start, dx, dy)
                plt.arrow(y, x, length * cos(angle), length * sin(angle),
                          head_width=3, head_length=3, fc=color, ec=color)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    def plot_cylinder_slice(self, cylinder_code, title, step_name):
        """Visualizes a 2D slice of the cylinder code (e.g., sum over directional bins)."""
        if self.config.should_show_viz(step_name):
            # Sum contributions over directional bins (k-axis)
            spatial_slice = np.sum(cylinder_code, axis=2)
            plt.figure(figsize=(6, 6))
            plt.imshow(spatial_slice, cmap='viridis', origin='lower')
            plt.title(title)
            plt.colorbar(label='Contribution')
            plt.xlabel('Spatial X (j-bin)')
            plt.ylabel('Spatial Y (i-bin)')
            plt.tight_layout()
            plt.show()

# --- 2. Data Loader Class (SOLID: SRP) ---
class DataLoader:
    """Loads fingerprint images from a specified directory."""

    def __init__(self, config: Config):
        self.config = config

    def load_fingerprint_images(self):
        """
        Loads all fingerprint images from the configured dataset directory.
        Returns a dictionary: {image_name: image_data}.
        """
        dataset_path = os.path.join(self.config.dataset_root, self.config.dataset_name)
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path not found: {dataset_path}")
            return {}

        image_files = glob(os.path.join(dataset_path, "*.tif")) + \
                      glob(os.path.join(dataset_path, "*.bmp")) + \
                      glob(os.path.join(dataset_path, "*.png"))

        images = {}
        for filepath in sorted(image_files):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images[os.path.basename(filepath)] = img
            else:
                print(f"Warning: Could not load image {filepath}")
        return images

# --- 5. Base Cylinder Code Generator (SOLID: ISP, LSP) ---
class BaseCylinderCodeGenerator:
    """Abstract base class for cylinder code generation."""

    def __init__(self, config: Config, minutiae_list, img_shape, mask):
        self.config = config
        self.minutiae_list = minutiae_list
        self.img_shape = img_shape
        self.mask = mask # Segmentation mask

    def _get_cell_center(self, central_minutia_pos, i, j):
        """
        Calculates the spatial center (row, col) of a cell (i, j) relative to
        the central minutia. (Ref: Eq 2)
        Input central_minutia_pos: (row_m, col_m, theta_m) where theta_m is central minutia angle.
        Output: (cell_center_row, cell_center_col) in image coordinates.
        """
        row_m, col_m, theta_m = central_minutia_pos

        # Calculate delta_row, delta_col for the cell's center in a local coordinate system
        # (i, j) are bin indices for (row, col) spatial bins.
        # (Ns/2) shifts origin to center of cylinder's grid.
        delta_row_local = (i - (self.config.Ns / 2)) * self.config.delta_s
        delta_col_local = (j - (self.config.Ns / 2)) * self.config.delta_s

        # Rotate these deltas by the central minutia's angle (theta_m)
        # Assuming theta_m is angle from horizontal (column) axis.
        cos_theta = cos(theta_m)
        sin_theta = sin(theta_m)

        # Rotated delta_col, delta_row in global image coordinates
        # Rotation matrix applied to (delta_col_local, delta_row_local)
        delta_col_rotated = delta_col_local * cos_theta - delta_row_local * sin_theta
        delta_row_rotated = delta_col_local * sin_theta + delta_row_local * cos_theta

        cell_center_row = int(round(row_m + delta_row_rotated))
        cell_center_col = int(round(col_m + delta_col_rotated))

        return cell_center_row, cell_center_col

    def _get_spatial_contribution(self, dist):
        """
        Calculates spatial contribution using Gaussian function. (Ref: Eq 6)
        """
        return exp(-(dist**2) / (2 * self.config.sigma_s**2))

    def _get_directional_contribution(self, angle_diff, sigma_directional):
        """
        Calculates directional contribution using Gaussian function. (Ref: Eq 7)
        """
        return exp(-(angle_diff**2) / (2 * sigma_directional**2))

    def _get_cell_validity(self, cell_center_row, cell_center_col):
        """Checks if a cell center is within image boundaries and mask."""
        rows, cols = self.img_shape
        if not (0 <= cell_center_row < rows and 0 <= cell_center_col < cols):
            return False
        # Mask is uint8 (0 or 255)
        return self.mask[cell_center_row, cell_center_col] == 255

    def generate_code(self, central_minutia_idx):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError

    def _get_neighboring_minutiae(self, central_minutia_idx):
        """
        Identifies neighboring minutiae within the cylinder's radius 'R'. (Ref: Eq 5)
        Returns a list of dictionaries with minutia data and relative properties.
        """
        central_minutia = self.minutiae_list[central_minutia_idx]
        row_m, col_m, theta_m = central_minutia[0:3]

        neighbors = []
        for i, neighbor_minutia in enumerate(self.minutiae_list):
            if i == central_minutia_idx:
                continue

            row_n, col_n, theta_n = neighbor_minutia[0:3]
            dist = sqrt((row_n - row_m)**2 + (col_n - col_m)**2)

            if dist <= self.config.R:
                # Relative angle of neighbor's orientation relative to central minutia's orientation
                angle_diff_neighbor_orient_from_central_orient = get_directional_difference(theta_n, theta_m)

                neighbors.append({
                    'minutia': neighbor_minutia,
                    'dist_from_center': dist,
                    'angle_diff_neighbor_orient_from_central_orient': angle_diff_neighbor_orient_from_central_orient,
                })
        return neighbors

# --- 6. MCC Generator (Original Minutia Cylinder Codes) ---
class MCCGenerator(BaseCylinderCodeGenerator):
    """Generates original Minutia Cylinder Codes (MCCo)."""

    def generate_code(self, central_minutia_idx):
        """
        Generates the MCC descriptor for a given central minutia.
        The cylinder is (Ns x Ns x ND) in size.
        (Ref: Eq 3, 4, 5, 6, 7)
        """
        central_minutia = self.minutiae_list[central_minutia_idx]
        central_minutia_pos = central_minutia[0:3] # (row_m, col_m, theta_m)
        
        # Initialize cylinder code (Ns spatial bins x Ns spatial bins x ND directional bins)
        cylinder_code = np.zeros((self.config.Ns, self.config.Ns, self.config.ND), dtype=np.float32)

        # Get neighboring minutiae within radius R
        neighbors_in_radius = self._get_neighboring_minutiae(central_minutia_idx)

        valid_cell_count = 0
        total_cells = self.config.Ns * self.config.Ns * self.config.ND

        for i in range(self.config.Ns): # Spatial Y (rows)
            for j in range(self.config.Ns): # Spatial X (cols)
                cell_center_row, cell_center_col = self._get_cell_center(central_minutia_pos, i, j)

                # Check cell validity (within image bounds and mask)
                if not self._get_cell_validity(cell_center_row, cell_center_col):
                    continue

                cell_has_contribution = False
                
                for neighbor_info in neighbors_in_radius:
                    n_minutia = neighbor_info['minutia']
                    row_n, col_n, theta_n = n_minutia[0:3]

                    # Spatial distance from neighbor to cell center
                    dist_to_cell_center = sqrt((row_n - cell_center_row)**2 + (col_n - cell_center_col)**2)
                    spatial_contrib = self._get_spatial_contribution(dist_to_cell_center)

                    # Directional difference (theta_n - theta_m)
                    directional_diff_original = neighbor_info['angle_diff_neighbor_orient_from_central_orient']
                    
                    for k in range(self.config.ND):
                        # angle_k_center is the center angle of the k-th directional bin.
                        # Maps bin index k to angle in [-pi, pi) for 2*pi range
                        angle_k_center = k * self.config.delta_d - pi

                        # Calculate directional contribution based on how neighbor's relative angle
                        # aligns with the bin's central angle.
                        angular_diff_for_bin = get_directional_difference(angle_k_center, directional_diff_original)
                        directional_contrib = self._get_directional_contribution(angular_diff_for_bin, self.config.sigma_d)
                        
                        cylinder_code[i, j, k] += spatial_contrib * directional_contrib
                        cell_has_contribution = True

                if cell_has_contribution:
                    valid_cell_count += 1
        
        # Apply validity constraints for the whole cylinder
        if len(neighbors_in_radius) < self.config.min_neighbors or \
           (valid_cell_count / total_cells) < self.config.min_valid_cells:
            return None # Cylinder is not valid
        
        # Normalize the cylinder code to ensure values are between 0 and 1
        max_val = np.max(cylinder_code)
        if max_val > 0:
            cylinder_code /= max_val
            
        return cylinder_code

# --- 7. MTCC Generator (Minutia Texture Cylinder Codes) ---
class MTCCGenerator(BaseCylinderCodeGenerator):
    """Generates Minutia Texture Cylinder Codes (MTCC) variants."""

    def __init__(self, config: Config, minutiae_list, img_shape, mask,
                 orientation_field, frequency_field, energy_field):
        super().__init__(config, minutiae_list, img_shape, mask)
        self.orientation_field = orientation_field # Io (float, radians)
        self.frequency_field = frequency_field     # If (float, 0-1)
        self.energy_field = energy_field           # Ie (float, 0-1)

    def generate_code(self, central_minutia_idx, code_type='MCCo'):
        """
        Generates MTCC descriptor for a central minutia, supporting various types.
        code_type: 'MCCo' (original), 'MCCf' (frequency), 'MCCe' (energy),
                   'MCCco' (cell-centered orientation), 'MCCcf' (cell-centered frequency),
                   'MCCce' (cell-centered energy).
        """
        central_minutia = self.minutiae_list[central_minutia_idx]
        central_minutia_pos = central_minutia[0:3] # (row_m, col_m, theta_m)

        cylinder_code = np.zeros((self.config.Ns, self.config.Ns, self.config.ND), dtype=np.float32)
        neighbors_in_radius = self._get_neighboring_minutiae(central_minutia_idx)

        valid_cell_count = 0
        total_cells = self.config.Ns * self.config.Ns * self.config.ND

        for i in range(self.config.Ns):
            for j in range(self.config.Ns):
                cell_center_row, cell_center_col = self._get_cell_center(central_minutia_pos, i, j)

                if not self._get_cell_validity(cell_center_row, cell_center_col):
                    continue

                cell_has_contribution = False

                if code_type in ['MCCco', 'MCCcf', 'MCCce']:
                    # Cell-centered features: Feature value is picked directly from the field at cell center.
                    # Ensure coordinates are within image bounds before lookup.
                    if not (0 <= cell_center_row < self.img_shape[0] and 0 <= cell_center_col < self.img_shape[1]):
                        continue

                    feature_value_at_cell_center = 0.0
                    sigma_directional_for_type = 0.0

                    if code_type == 'MCCco':
                        feature_value_at_cell_center = self.orientation_field[cell_center_row, cell_center_col]
                        sigma_directional_for_type = self.config.sigma_d # Orientation uses sigma_d
                    elif code_type == 'MCCcf':
                        # Frequency is 0-1 normalized. Map to an angular range for get_directional_difference.
                        # Mapping 0-1 to -pi to pi for compatibility with angular difference function.
                        feature_value_at_cell_center = (self.frequency_field[cell_center_row, cell_center_col] * 2 * pi) - pi
                        sigma_directional_for_type = self.config.sigma_f
                    elif code_type == 'MCCce':
                        # Energy is 0-1 normalized. Map to an angular range.
                        feature_value_at_cell_center = (self.energy_field[cell_center_row, cell_center_col] * 2 * pi) - pi
                        sigma_directional_for_type = self.config.sigma_e
                    
                    spatial_contrib = 1.0 # For cell-centered, spatial contribution is often considered uniform or 1.0
                    
                    for k in range(self.config.ND):
                        angle_k_center = k * self.config.delta_d - pi
                        
                        angular_diff = get_directional_difference(angle_k_center, feature_value_at_cell_center)
                        directional_contrib = self._get_directional_contribution(angular_diff, sigma_directional_for_type)
                        
                        cylinder_code[i, j, k] += spatial_contrib * directional_contrib
                        cell_has_contribution = True

                else: # Minutia-centered features (MCCo, MCCf, MCCe - similar to original MCC but with texture values)
                    for neighbor_info in neighbors_in_radius:
                        n_minutia = neighbor_info['minutia']
                        row_n, col_n, theta_n = n_minutia[0:3]

                        dist_to_cell_center = sqrt((row_n - cell_center_row)**2 + (col_n - cell_center_col)**2)
                        spatial_contrib = self._get_spatial_contribution(dist_to_cell_center)

                        feature_value_from_neighbor_location = 0.0
                        sigma_directional_for_type = 0.0

                        # Ensure neighbor coordinates are within image bounds for field lookup
                        if not (0 <= row_n < self.img_shape[0] and 0 <= col_n < self.img_shape[1]):
                            continue

                        if code_type == 'MCCo':
                            feature_value_from_neighbor_location = neighbor_info['angle_diff_neighbor_orient_from_central_orient']
                            sigma_directional_for_type = self.config.sigma_d
                        elif code_type == 'MCCf':
                            feature_value_from_neighbor_location = (self.frequency_field[row_n, col_n] * 2 * pi) - pi
                            sigma_directional_for_type = self.config.sigma_f
                        elif code_type == 'MCCe':
                            feature_value_from_neighbor_location = (self.energy_field[row_n, col_n] * 2 * pi) - pi
                            sigma_directional_for_type = self.config.sigma_e
                        
                        for k in range(self.config.ND):
                            angle_k_center = k * self.config.delta_d - pi
                            angular_diff = get_directional_difference(angle_k_center, feature_value_from_neighbor_location)
                            directional_contrib = self._get_directional_contribution(angular_diff, sigma_directional_for_type)
                            cylinder_code[i, j, k] += spatial_contrib * directional_contrib
                            cell_has_contribution = True
                
                if cell_has_contribution:
                    valid_cell_count += 1
        
        # Apply validity constraints for the whole cylinder
        if len(neighbors_in_radius) < self.config.min_neighbors or \
           (valid_cell_count / total_cells) < self.config.min_valid_cells:
            return None
        
        # Normalize the cylinder code to ensure values are between 0 and 1
        max_val = np.max(cylinder_code)
        if max_val > 0:
            cylinder_code /= max_val
            
        return cylinder_code

# --- 8. Matcher Class (SOLID: SRP) ---
class Matcher:
    """Performs fingerprint matching using Minutia Cylinder Codes and LSSR."""

    def __init__(self, config: Config, visualizer: Visualizer):
        self.config = config
        self.viz = visualizer

    def _euclidean_distance(self, cyl1, cyl2):
        """Calculates Euclidean distance between two cylinder codes. (Ref: Eq 15)"""
        if cyl1 is None or cyl2 is None:
            return float('inf')
        if cyl1.shape != cyl2.shape:
            return float('inf')
        
        dist_sq = 0.0
        common_valid_cells = 0
        for i in range(self.config.Ns):
            for j in range(self.config.Ns):
                for k in range(self.config.ND):
                    # Consider contribution if EITHER cell has a non-zero value, as per MCC implementation
                    if cyl1[i, j, k] > 0 or cyl2[i, j, k] > 0: 
                        dist_sq += (cyl1[i, j, k] - cyl2[i, j, k])**2
                        common_valid_cells += 1
        
        if common_valid_cells == 0:
            return float('inf')

        return sqrt(dist_sq) / common_valid_cells


    def _double_angle_distance(self, cyl1, cyl2):
        """
        Calculates double angle distance between two cylinder codes. (Ref: Eq 18)
        This is a specific metric for angular features (like in MTCC).
        dy(Ca, Cb) = sqrt(Cosd(Ca, Cb)^2 + Sind(Ca, Cb)^2)
        """
        if cyl1 is None or cyl2 is None:
            return float('inf')
        if cyl1.shape != cyl2.shape:
            return float('inf')

        dist_sum_cos_sq = 0.0
        dist_sum_sin_sq = 0.0
        common_valid_cells = 0

        for i in range(self.config.Ns):
            for j in range(self.config.Ns):
                for k in range(self.config.ND):
                    val1 = cyl1[i, j, k]
                    val2 = cyl2[i, j, k]
                    
                    if val1 > 0 or val2 > 0: # If either cell has contribution
                        # Values in cylinder codes are normalized 0-1. Map them to an angular range.
                        # The paper implies these are used in trigonometric functions of double angle.
                        # Let's map normalized [0,1] values to [-pi, pi] to represent angles.
                        angle_val1 = (val1 * 2 * pi) - pi
                        angle_val2 = (val2 * 2 * pi) - pi

                        dist_sum_cos_sq += (cos(2 * angle_val1) - cos(2 * angle_val2))**2
                        dist_sum_sin_sq += (sin(2 * angle_val1) - sin(2 * angle_val2))**2
                        common_valid_cells += 1
        
        if common_valid_cells == 0:
            return float('inf')
        
        return sqrt(dist_sum_cos_sq + dist_sum_sin_sq) / common_valid_cells


    def match_templates(self, minutiae_templates_A, minutiae_templates_B, viz_name="Matching"):
        """
        Matches two sets of minutiae templates using Local Similarity Sort.
        (Relaxation simplified for non-bloated code).
        (Ref: VII. "Matching - Local Similarity Sort With Relaxation.")
        """
        if self.config.should_show_viz("matching_process"):
            print(f"\n--- {viz_name} ---")
            print(f"Template A has {len(minutiae_templates_A)} minutia cylinders.")
            print(f"Template B has {len(minutiae_templates_B)} minutia cylinders.")

        # Local Similarity Matrix (LSM)
        # Stores (distance, idx_A, idx_B, m_A_original, m_B_original)
        lsm = []
        for i, cyl_A_data in enumerate(minutiae_templates_A):
            cyl_A, m_A_original, type_A = cyl_A_data
            if cyl_A is None: continue

            for j, cyl_B_data in enumerate(minutiae_templates_B):
                cyl_B, m_B_original, type_B = cyl_B_data
                if cyl_B is None: continue
                
                # Match only if descriptor types are the same
                if type_A != type_B:
                    continue

                # Determine distance metric based on type of cylinder code
                if type_A == 'MCCo': # Original MCC
                    distance_func = self._euclidean_distance
                else: # All MTCC variants use double angle distance
                    distance_func = self._double_angle_distance

                dist = distance_func(cyl_A, cyl_B)
                if dist != float('inf'):
                    lsm.append((dist, i, j, m_A_original, m_B_original))
        
        # Sort LSM by distance (smaller distance means higher similarity)
        lsm.sort(key=lambda x: x[0])

        if self.config.should_show_viz("matching_process"):
            print(f"Generated {len(lsm)} potential minutia pairs.")

        # Simple relaxation: select unique best matches for each minutia from A to B.
        final_matches = []
        matched_minutiae_A_indices = set()
        matched_minutiae_B_indices = set()
        
        # Select top `top_nr_matches` from LSM, then refine for uniqueness
        candidate_matches = lsm[:min(self.config.top_nr_matches, len(lsm))]

        for dist, idx_A, idx_B, m_A_orig, m_B_orig in candidate_matches:
            if idx_A not in matched_minutiae_A_indices and idx_B not in matched_minutiae_B_indices:
                final_matches.append((dist, idx_A, idx_B, m_A_orig, m_B_orig))
                matched_minutiae_A_indices.add(idx_A)
                matched_minutiae_B_indices.add(idx_B)
                if len(final_matches) >= self.config.final_top_np_matches:
                    break

        # Compute final matching score (e.g., inverse of average distance of selected matches)
        if len(final_matches) == 0:
            final_score = 0.0
        else:
            avg_distance = sum(m[0] for m in final_matches) / len(final_matches)
            
            # Use max observed distance in filtered matches for normalization, or a default
            max_dist_in_matches = max(m[0] for m in final_matches) if final_matches else 1.0
            if max_dist_in_matches == 0: max_dist_in_matches = 1.0 # Fallback

            normalized_avg_distance = avg_distance / max_dist_in_matches
            final_score = max(0, 1.0 - normalized_avg_distance) # Similarity is 1 - normalized_distance

        if self.config.should_show_viz("final_match_score"):
            print(f"Final match score: {final_score:.4f}")
            print(f"Number of final matched pairs: {len(final_matches)}")

        return final_score, final_matches


# --- Main Execution Logic ---
if __name__ == "__main__":
    config = Config()
    viz = Visualizer(config)
    data_loader = DataLoader(config)
    # FIXED: Instantiate Matcher object before it's used
    matcher = Matcher(config, viz) 

    print(f"Loading images from: {os.path.join(config.dataset_root, config.dataset_name)}")
    images = data_loader.load_fingerprint_images()

    if not images:
        print("No images found. Please ensure your 'fingerprint_dataset' directory and subfolders are correctly set up with images.")
    else:
        # Get image names
        img_names = list(images.keys())
        
        if len(img_names) < 2:
            print("Need at least two images in the dataset for matching demonstration.")
            print("Processing and showing features for the first available image.")
            
            first_img_name = img_names[0]
            img_original = images[first_img_name]

            # 0-1 normalize the image first
            img_float = img_original.astype(np.float32) / 255.0
            viz.show_image(img_float, f"1. Image Loaded (Normalized): {first_img_name}", "image_loaded")

            # --- Image Processing Pipeline (using functions) ---
            segmented_img_float, mask = segment_image(img_float, config, viz)
            config.img_mask = mask # Store mask in config for Gabor
            orientation_field, frequency_field, energy_field = stft_feature_extraction(segmented_img_float, config, viz)
            gabor_enhanced_img_float = gabor_enhance(segmented_img_float, orientation_field, frequency_field, config, viz)
            clahe_enhanced_img_float = clahe_enhance(gabor_enhanced_img_float, config, viz)
            binary_img = binarize_image(clahe_enhanced_img_float, mask, config, viz)
            thinned_img = thin_image(binary_img, config, viz) # Corrected arguments
            
            # --- Minutiae Extraction ---
            minutiae = extract_minutiae(thinned_img, mask, orientation_field, config, viz)
            viz.plot_minutiae(img_float, minutiae, f"6. Minutiae Plot: {first_img_name}", "minutiae_plot")
            print(f"Extracted {len(minutiae)} minutiae for {first_img_name}.")
            
        else:
            # Example: Match first two images from the dataset
            img1_name, img2_name = img_names[0], img_names[1]
            img1_original, img2_original = images[img1_name], images[img2_name]

            # 0-1 normalize both images
            img1_float = img1_original.astype(np.float32) / 255.0
            img2_float = img2_original.astype(np.float32) / 255.0
            viz.show_image(img1_float, f"1. Image Loaded (Normalized): {img1_name}", "image_loaded")
            viz.show_image(img2_float, f"1. Image Loaded (Normalized): {img2_name}", "image_loaded")


            # --- Process Image 1 ---
            print(f"\n--- Processing Image 1: {img1_name} ---")
            segmented_img1, mask1 = segment_image(img1_float, config, viz)
            config.img_mask = mask1 # Set mask for Gabor use
            orientation_field1, frequency_field1, energy_field1 = stft_feature_extraction(segmented_img1, config, viz)
            gabor_enhanced_img1 = gabor_enhance(segmented_img1, orientation_field1, frequency_field1, config, viz)
            clahe_enhanced_img1 = clahe_enhance(gabor_enhanced_img1, config, viz)
            binary_img1 = binarize_image(clahe_enhanced_img1, mask1, config, viz)
            thinned_img1 = thin_image(binary_img1, config, viz) # FIXED: Removed extra args
            minutiae1 = extract_minutiae(thinned_img1, mask1, orientation_field1, config, viz)
            viz.plot_minutiae(img1_float, minutiae1, f"6. Minutiae Plot: {img1_name}", "minutiae_plot")
            print(f"Extracted {len(minutiae1)} minutiae for {img1_name}.")

            # --- Process Image 2 ---
            print(f"\n--- Processing Image 2: {img2_name} ---")
            segmented_img2, mask2 = segment_image(img2_float, config, viz)
            config.img_mask = mask2 # Set mask for Gabor use
            orientation_field2, frequency_field2, energy_field2 = stft_feature_extraction(segmented_img2, config, viz)
            gabor_enhanced_img2 = gabor_enhance(segmented_img2, orientation_field2, frequency_field2, config, viz)
            clahe_enhanced_img2 = clahe_enhance(gabor_enhanced_img2, config, viz)
            binary_img2 = binarize_image(clahe_enhanced_img2, mask2, config, viz)
            thinned_img2 = thin_image(binary_img2, config, viz) # FIXED: Removed extra args (again)
            minutiae2 = extract_minutiae(thinned_img2, mask2, orientation_field2, config, viz)
            viz.plot_minutiae(img2_float, minutiae2, f"6. Minutiae Plot: {img2_name}", "minutiae_plot")
            print(f"Extracted {len(minutiae2)} minutiae for {img2_name}.")

            # --- Feature Extraction (Cylinder Codes) ---
            print("\n--- Generating Cylinder Codes ---")
            mcc_generator = MCCGenerator(config, minutiae1, img1_float.shape, mask1)
            mtcc_generator = MTCCGenerator(config, minutiae1, img1_float.shape, mask1,
                                            orientation_field1, frequency_field1, energy_field1)

            # Generate cylinders for Image 1 (Query)
            cylinders_img1 = []
            for i, m in enumerate(minutiae1):
                # MCCo (Original Minutia Cylinder Codes)
                mcco_cyl = mcc_generator.generate_code(i)
                if mcco_cyl is not None:
                    cylinders_img1.append((mcco_cyl, m, "MCCo"))
                    if viz.config.should_show_viz("cylinder_code_visualization") and i == 0:
                        viz.plot_cylinder_slice(mcco_cyl, f"MCCo Cylinder Slice for {img1_name} (Minutia {i})", "cylinder_code_visualization")
                
                # MCCco (Cell-Centered Orientation MTCC)
                mcco_co_cyl = mtcc_generator.generate_code(i, code_type='MCCco')
                if mcco_co_cyl is not None:
                    cylinders_img1.append((mcco_co_cyl, m, "MCCco"))
                    if viz.config.should_show_viz("cylinder_code_visualization") and i == 0: # Only plot first minutia for brevity
                        viz.plot_cylinder_slice(mcco_co_cyl, f"MCCco Cylinder Slice for {img1_name} (Minutia {i})", "cylinder_code_visualization")

            # Generate cylinders for Image 2 (Template)
            mcc_generator_temp = MCCGenerator(config, minutiae2, img2_float.shape, mask2)
            mtcc_generator_temp = MTCCGenerator(config, minutiae2, img2_float.shape, mask2,
                                                orientation_field2, frequency_field2, energy_field2)
            cylinders_img2 = []
            for i, m in enumerate(minutiae2):
                mcco_cyl = mcc_generator_temp.generate_code(i)
                if mcco_cyl is not None:
                    cylinders_img2.append((mcco_cyl, m, "MCCo"))

                mcco_co_cyl = mtcc_generator_temp.generate_code(i, code_type='MCCco')
                if mcco_co_cyl is not None:
                    cylinders_img2.append((mcco_co_cyl, m, "MCCco"))

            print(f"Generated {len(cylinders_img1)} cylinder codes for {img1_name}.")
            print(f"Generated {len(cylinders_img2)} cylinder codes for {img2_name}.")

            # --- Matching ---
            print("\n--- Performing Matching ---")
            
            # Match original MCC (MCCo)
            mcco_cyls_1 = [c for c in cylinders_img1 if c[2] == "MCCo"]
            mcco_cyls_2 = [c for c in cylinders_img2 if c[2] == "MCCo"]
            score_mcco, _ = matcher.match_templates(mcco_cyls_1, mcco_cyls_2, f"Matching {img1_name} and {img2_name} (MCCo)")

            # Match cell-centered orientation MTCC (MCCco)
            mcco_co_cyls_1 = [c for c in cylinders_img1 if c[2] == "MCCco"]
            mcco_co_cyls_2 = [c for c in cylinders_img2 if c[2] == "MCCco"]
            score_mcco_co, _ = matcher.match_templates(mcco_co_cyls_1, mcco_co_cyls_2, f"Matching {img1_name} and {img2_name} (MCCco)")

            print(f"\nFinal Scores:")
            print(f"MCCo Matching Score: {score_mcco:.4f}")
            print(f"MCCco (MTCC) Matching Score: {score_mcco_co:.4f}")
            print("\nNote: A higher score indicates better similarity. Scores will vary based on images and simplified relaxation.")

    print("\nProcessing complete.")