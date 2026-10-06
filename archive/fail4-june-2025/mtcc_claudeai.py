import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import threshold_otsu
import math
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class MTCCFingerprintSystem:
    """
    Minutiae Texture Cylinder Codes (MTCC) Fingerprint Recognition System
    Based on research papers for fingerprint enhancement and matching using
    Gabor filters, STFT analysis, and texture features.
    """
    
    def __init__(self, block_size: int = 16, overlap: int = 8):
        """Initialize MTCC system with default parameters."""
        self.block_size = block_size
        self.overlap = overlap
        self.cylinder_radius = 65
        self.cylinder_sectors = 80
        self.num_directions = 8
        
        # Gabor filter parameters
        self.gabor_frequencies = [1/8, 1/12, 1/16]
        self.gabor_angles = np.linspace(0, np.pi, 8, endpoint=False)
        
    def load_fingerprint(self, image_path: str) -> np.ndarray:
        """Load and preprocess fingerprint image."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Normalize to 0-255 range
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image.astype(np.float64)
    
    def normalize_image(self, image: np.ndarray, 
                       target_mean: float = 100, 
                       target_var: float = 100) -> np.ndarray:
        """Normalize image to target mean and variance."""
        mean_val = np.mean(image)
        var_val = np.var(image)
        
        if var_val < 1e-6:  # Avoid division by zero
            return np.full_like(image, target_mean)
        
        normalized = target_mean + np.sqrt(target_var) * (image - mean_val) / np.sqrt(var_val)
        return np.clip(normalized, 0, 255)
    
    def segment_fingerprint(self, image: np.ndarray, 
                           block_size: int = 16,
                           threshold: float = 0.1) -> np.ndarray:
        """Segment fingerprint from background using variance-based method."""
        h, w = image.shape
        mask = np.zeros((h, w), dtype=bool)
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = image[i:i+block_size, j:j+block_size]
                if np.var(block) > threshold * np.var(image):
                    mask[i:i+block_size, j:j+block_size] = True
        
        # Morphological operations to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask.astype(bool)
    
    def create_gabor_kernel(self, sigma_x: float, sigma_y: float, 
                           theta: float, freq: float, 
                           size: int = 31) -> np.ndarray:
        """Create Gabor filter kernel."""
        x = np.linspace(-size//2, size//2, size)
        y = np.linspace(-size//2, size//2, size)
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        X_theta = X * np.cos(theta) + Y * np.sin(theta)
        Y_theta = -X * np.sin(theta) + Y * np.cos(theta)
        
        # Gabor formula
        gb = np.exp(-(X_theta**2/(2*sigma_x**2) + Y_theta**2/(2*sigma_y**2))) * \
             np.cos(2 * np.pi * freq * X_theta)
        
        return gb
    
    def gabor_filter_bank(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply bank of Gabor filters to image."""
        filtered_images = []
        
        for freq in self.gabor_frequencies:
            for angle in self.gabor_angles:
                kernel = self.create_gabor_kernel(4, 4, angle, freq)
                filtered = cv2.filter2D(image, -1, kernel)
                filtered_images.append(filtered)
        
        return filtered_images
    
    def gabor_enhancement(self, image: np.ndarray, 
                         orientation_map: np.ndarray,
                         frequency_map: np.ndarray) -> np.ndarray:
        """Enhanced Gabor filtering using orientation and frequency maps."""
        h, w = image.shape
        enhanced = np.zeros_like(image)
        
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                # Get block
                block = image[i:i+self.block_size, j:j+self.block_size]
                if block.size == 0:
                    continue
                
                # Get local orientation and frequency
                oi = min(i + self.block_size//2, h-1)
                oj = min(j + self.block_size//2, w-1)
                
                orientation = orientation_map[oi, oj]
                frequency = frequency_map[oi, oj]
                
                # Create adaptive Gabor filter
                if frequency > 0:
                    kernel = self.create_gabor_kernel(2, 2, orientation, frequency, 15)
                    filtered_block = cv2.filter2D(block, -1, kernel)
                    
                    # Place filtered block back
                    end_i = min(i + self.block_size, h)
                    end_j = min(j + self.block_size, w)
                    enhanced[i:end_i, j:end_j] = filtered_block[:end_i-i, :end_j-j]
        
        return enhanced
    
    def smqt_enhancement(self, image: np.ndarray, levels: int = 3) -> np.ndarray:
        """Successive Mean Quantization Transform for enhancement."""
        enhanced = image.copy().astype(np.float64)
        
        for level in range(levels):
            # Calculate mean of non-zero regions
            valid_pixels = enhanced > 0
            if np.sum(valid_pixels) == 0:
                break
                
            mean_val = np.mean(enhanced[valid_pixels])
            std_val = np.std(enhanced[valid_pixels])
            
            # More conservative enhancement
            alpha = 0.3  # Reduced from 0.1 for more subtle changes
            
            # Enhance contrast around mean
            enhanced = np.where(enhanced > mean_val,
                              enhanced + (255 - enhanced) * alpha * 0.5,
                              enhanced * (1 + alpha * 0.5))
            
            # Apply local contrast enhancement
            enhanced = enhanced + alpha * (enhanced - mean_val)
            
            # Normalize to valid range
            enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced
    
    def stft_analysis(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Short-Time Fourier Transform analysis for orientation, frequency, and energy."""
        h, w = image.shape
        orientation_map = np.zeros((h, w))
        frequency_map = np.zeros((h, w))
        energy_map = np.zeros((h, w))
        
        window = signal.windows.hann(self.block_size)
        window_2d = np.outer(window, window)
        
        for i in range(0, h - self.block_size + 1, self.overlap):
            for j in range(0, w - self.block_size + 1, self.overlap):
                # Extract block
                block = image[i:i+self.block_size, j:j+self.block_size]
                
                # Apply window
                windowed_block = block * window_2d
                
                # FFT
                fft_block = np.fft.fft2(windowed_block)
                fft_shifted = np.fft.fftshift(fft_block)
                magnitude = np.abs(fft_shifted)
                
                # Convert to polar coordinates
                center = self.block_size // 2
                y_coords, x_coords = np.ogrid[:self.block_size, :self.block_size]
                x_coords = x_coords - center
                y_coords = y_coords - center
                
                # Calculate orientation and frequency
                angles = np.arctan2(y_coords, x_coords)
                radii = np.sqrt(x_coords**2 + y_coords**2)
                
                # Weight by magnitude
                weighted_angles = angles * magnitude
                weighted_radii = radii * magnitude
                
                total_weight = np.sum(magnitude)
                
                if total_weight > 0:
                    # Dominant orientation (vector averaging)
                    sin_sum = np.sum(magnitude * np.sin(2 * angles))
                    cos_sum = np.sum(magnitude * np.cos(2 * angles))
                    dominant_orientation = 0.5 * np.arctan2(sin_sum, cos_sum)
                    
                    # Dominant frequency
                    dominant_frequency = np.sum(weighted_radii) / total_weight / self.block_size
                    
                    # Energy
                    energy = np.log(total_weight + 1)
                else:
                    dominant_orientation = 0
                    dominant_frequency = 0
                    energy = 0
                
                # Fill maps
                end_i = min(i + self.block_size, h)
                end_j = min(j + self.block_size, w)
                
                orientation_map[i:end_i, j:end_j] = dominant_orientation
                frequency_map[i:end_i, j:end_j] = dominant_frequency
                energy_map[i:end_i, j:end_j] = energy
        
        return orientation_map, frequency_map, energy_map
    
    def binarize_and_thin(self, image: np.ndarray) -> np.ndarray:
        """Binarize and thin the fingerprint image."""
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Apply Gaussian smoothing first
        smoothed = gaussian_filter(image, sigma=1.0)
        
        # Use adaptive thresholding instead of Otsu for better local contrast
        binary = cv2.adaptiveThreshold(
            smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert if necessary (ridges should be white)
        if np.mean(binary) > 127:
            binary = 255 - binary
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Convert to boolean for skeletonization
        binary_bool = (binary > 128).astype(bool)
        
        # Remove small objects
        binary_bool = remove_small_objects(binary_bool, min_size=20)
        
        # Skeletonization
        try:
            skeleton = skeletonize(binary_bool)
            skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        except:
            # Fallback: use erosion-based thinning
            skeleton_uint8 = self._simple_thinning(binary)
        
        return skeleton_uint8
    
    def _simple_thinning(self, binary_image: np.ndarray) -> np.ndarray:
        """Simple thinning algorithm as fallback."""
        # Convert to binary
        binary = (binary_image > 128).astype(np.uint8)
        
        # Apply erosion iteratively until convergence
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        prev_skeleton = binary.copy()
        
        for _ in range(10):  # Maximum iterations
            eroded = cv2.erode(prev_skeleton, kernel, iterations=1)
            dilated = cv2.dilate(eroded, kernel, iterations=1)
            skeleton = cv2.subtract(prev_skeleton, dilated)
            
            if np.array_equal(skeleton, prev_skeleton):
                break
            prev_skeleton = skeleton
        
        return skeleton
    
    def extract_minutiae(self, skeleton: np.ndarray) -> List[Dict]:
        """Extract minutiae from thinned fingerprint image."""
        minutiae = []
        h, w = skeleton.shape
        
        # Ensure skeleton is binary
        if skeleton.max() > 1:
            binary_skeleton = (skeleton > 128).astype(np.uint8)
        else:
            binary_skeleton = skeleton.astype(np.uint8)
        
        # Apply additional cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        binary_skeleton = cv2.morphologyEx(binary_skeleton, cv2.MORPH_OPEN, kernel)
        
        # Use crossing number method for minutiae detection
        for i in range(2, h-2):
            for j in range(2, w-2):
                if binary_skeleton[i, j] == 1:  # Ridge pixel
                    # Calculate crossing number
                    cn = self._crossing_number(binary_skeleton, i, j)
                    
                    if cn == 1:  # Ridge ending
                        angle = self._calculate_minutiae_angle(binary_skeleton, i, j, 'ending')
                        quality = self._calculate_quality(skeleton, i, j)
                        
                        # Check if this minutia is too close to existing ones
                        if self._is_valid_minutia(minutiae, j, i, min_distance=15):
                            minutiae.append({
                                'x': j, 'y': i, 'angle': angle, 'type': 'ending',
                                'quality': quality
                            })
                    
                    elif cn == 3:  # Ridge bifurcation
                        angle = self._calculate_minutiae_angle(binary_skeleton, i, j, 'bifurcation')
                        quality = self._calculate_quality(skeleton, i, j)
                        
                        # Check if this minutia is too close to existing ones
                        if self._is_valid_minutia(minutiae, j, i, min_distance=15):
                            minutiae.append({
                                'x': j, 'y': i, 'angle': angle, 'type': 'bifurcation',
                                'quality': quality
                            })
        
        # Sort by quality and keep best ones
        minutiae.sort(key=lambda x: x['quality'], reverse=True)
        return minutiae[:30]  # Keep top 30 minutiae
    
    def _crossing_number(self, binary_skeleton: np.ndarray, y: int, x: int) -> int:
        """Calculate crossing number for minutiae detection."""
        # 8-connected neighbors in circular order
        neighbors = [
            binary_skeleton[y-1, x-1], binary_skeleton[y-1, x], binary_skeleton[y-1, x+1],
            binary_skeleton[y, x+1], binary_skeleton[y+1, x+1], binary_skeleton[y+1, x],
            binary_skeleton[y+1, x-1], binary_skeleton[y, x-1]
        ]
        
        # Add first neighbor at the end for circular calculation
        neighbors.append(neighbors[0])
        
        # Calculate crossing number
        cn = 0
        for i in range(8):
            cn += abs(neighbors[i] - neighbors[i+1])
        
        return cn // 2
    
    def _is_valid_minutia(self, existing_minutiae: List[Dict], x: int, y: int, min_distance: int = 15) -> bool:
        """Check if a new minutia is valid (not too close to existing ones)."""
        for minutia in existing_minutiae:
            dist = np.sqrt((minutia['x'] - x)**2 + (minutia['y'] - y)**2)
            if dist < min_distance:
                return False
        return True
    
    def _calculate_minutiae_angle(self, binary_skeleton: np.ndarray, 
                                 y: int, x: int, minutiae_type: str) -> float:
        """Calculate minutiae angle using local ridge direction."""
        h, w = binary_skeleton.shape
        
        # Find direction to nearest ridge pixel
        for radius in range(3, 10):
            angles = []
            for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                ni = int(y + radius * np.sin(angle))
                nj = int(x + radius * np.cos(angle))
                
                if 0 <= ni < h and 0 <= nj < w and binary_skeleton[ni, nj] == 1:
                    angles.append(angle)
            
            if angles:
                return np.mean(angles)
        
        return 0.0
    
    def _calculate_quality(self, image: np.ndarray, y: int, x: int, 
                          window_size: int = 16) -> float:
        """Calculate minutiae quality based on local variance."""
        h, w = image.shape
        y1 = max(0, y - window_size//2)
        y2 = min(h, y + window_size//2)
        x1 = max(0, x - window_size//2)
        x2 = min(w, x + window_size//2)
        
        local_region = image[y1:y2, x1:x2]
        return np.var(local_region) if local_region.size > 0 else 0
    
    def create_mcc_cylinder(self, minutiae: List[Dict], 
                           orientation_map: np.ndarray,
                           frequency_map: np.ndarray,
                           energy_map: np.ndarray,
                           central_minutia: Dict) -> np.ndarray:
        """Create MTCC cylinder for a central minutia."""
        # Cylinder parameters
        radius = self.cylinder_radius
        ns = 16  # Spatial sectors
        nd = 5   # Angular sectors
        
        cylinder = np.zeros((ns, ns, nd))
        
        cx, cy = central_minutia['x'], central_minutia['y']
        central_angle = central_minutia['angle']
        
        # Create tessellation
        for i in range(ns):
            for j in range(ns):
                for k in range(nd):
                    # Calculate cell center
                    r = (i - ns//2 + 0.5) * (2 * radius / ns)
                    theta = (j - ns//2 + 0.5) * (2 * radius / ns)
                    
                    # Rotate according to central minutia
                    cell_x = cx + r * np.cos(central_angle) - theta * np.sin(central_angle)
                    cell_y = cy + r * np.sin(central_angle) + theta * np.cos(central_angle)
                    
                    # Cell angular sector
                    sector_angle = -np.pi + (k + 0.5) * (2 * np.pi / nd)
                    
                    # Calculate contributions from neighboring minutiae
                    contribution = 0
                    
                    for minutia in minutiae:
                        if minutia == central_minutia:
                            continue
                        
                        mx, my = minutia['x'], minutia['y']
                        dist = np.sqrt((mx - cell_x)**2 + (my - cell_y)**2)
                        
                        if dist <= radius and dist > 0:
                            # Spatial contribution (Gaussian)
                            spatial_contrib = np.exp(-(dist**2) / (2 * (radius/3)**2))
                            
                            # For MTCC, use texture features instead of minutiae angles
                            if 0 <= int(cell_y) < orientation_map.shape[0] and \
                               0 <= int(cell_x) < orientation_map.shape[1]:
                                
                                # Use orientation, frequency, or energy from STFT
                                texture_angle = orientation_map[int(cell_y), int(cell_x)]
                                
                                # Angular contribution
                                angle_diff = self._normalize_angle(sector_angle - texture_angle)
                                angular_contrib = np.exp(-(angle_diff**2) / (2 * (np.pi/4)**2))
                                
                                contribution += spatial_contrib * angular_contrib
                    
                    cylinder[i, j, k] = contribution
        
        return cylinder.flatten()
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def match_cylinders(self, cylinder1: np.ndarray, cylinder2: np.ndarray) -> float:
        """Match two MTCC cylinders using cosine similarity."""
        if len(cylinder1) == 0 or len(cylinder2) == 0:
            return 0.0
        
        # Normalize cylinders
        norm1 = np.linalg.norm(cylinder1)
        norm2 = np.linalg.norm(cylinder2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cylinder1_norm = cylinder1 / norm1
        cylinder2_norm = cylinder2 / norm2
        
        # Calculate cosine similarity
        similarity = np.dot(cylinder1_norm, cylinder2_norm)
        return max(0, similarity)  # Ensure non-negative
    
    def match_fingerprints(self, minutiae1: List[Dict], minutiae2: List[Dict],
                          features1: Tuple[np.ndarray, np.ndarray, np.ndarray],
                          features2: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
        """Match two fingerprints using MTCC features."""
        if len(minutiae1) == 0 or len(minutiae2) == 0:
            return 0.0
        
        orientation_map1, frequency_map1, energy_map1 = features1
        orientation_map2, frequency_map2, energy_map2 = features2
        
        total_score = 0
        valid_matches = 0
        
        # Match top minutiae
        max_minutiae = min(10, len(minutiae1), len(minutiae2))
        
        for i in range(max_minutiae):
            minutia1 = minutiae1[i]
            cylinder1 = self.create_mcc_cylinder(
                minutiae1, orientation_map1, frequency_map1, energy_map1, minutia1
            )
            
            best_match_score = 0
            
            for j in range(max_minutiae):
                minutia2 = minutiae2[j]
                cylinder2 = self.create_mcc_cylinder(
                    minutiae2, orientation_map2, frequency_map2, energy_map2, minutia2
                )
                
                score = self.match_cylinders(cylinder1, cylinder2)
                best_match_score = max(best_match_score, score)
            
            total_score += best_match_score
            valid_matches += 1
        
        return total_score / valid_matches if valid_matches > 0 else 0.0
    
    def calculate_eer(self, genuine_scores: List[float], 
                     impostor_scores: List[float]) -> Tuple[float, float]:
        """Calculate Equal Error Rate."""
        if not genuine_scores or not impostor_scores:
            return 0.0, 0.0
        
        # Create sorted score list
        all_scores = sorted(set(genuine_scores + impostor_scores))
        
        best_eer = 1.0
        best_threshold = 0.0
        
        for threshold in all_scores:
            # False Rejection Rate (genuine scores below threshold)
            frr = sum(1 for score in genuine_scores if score < threshold) / len(genuine_scores)
            
            # False Acceptance Rate (impostor scores above threshold)
            far = sum(1 for score in impostor_scores if score >= threshold) / len(impostor_scores)
            
            # EER is when FRR ≈ FAR
            if abs(frr - far) < abs(best_eer - 0.5):
                best_eer = (frr + far) / 2
                best_threshold = threshold
        
        return best_eer, best_threshold
    
    def process_fingerprint(self, image_path: str) -> Tuple[List[Dict], 
                                                          Tuple[np.ndarray, np.ndarray, np.ndarray],
                                                          Dict[str, np.ndarray]]:
        """Complete fingerprint processing pipeline."""
        # Step 1: Load image
        image = self.load_fingerprint(image_path)
        
        # Step 2: Normalize
        normalized = self.normalize_image(image)
        
        # Step 3: Segment
        mask = self.segment_fingerprint(normalized)
        
        # Step 4: STFT Analysis
        orientation_map, frequency_map, energy_map = self.stft_analysis(normalized)
        
        # Step 5: Gabor enhancement
        enhanced = self.gabor_enhancement(normalized, orientation_map, frequency_map)
        
        # Step 6: SMQT enhancement
        smqt_enhanced = self.smqt_enhancement(enhanced)
        
        # Step 7: Binarize and thin
        skeleton = self.binarize_and_thin(smqt_enhanced)
        
        # Step 8: Extract minutiae
        minutiae = self.extract_minutiae(skeleton)
        
        # Store intermediate results for visualization
        intermediate_results = {
            'original': image,
            'normalized': normalized,
            'mask': mask.astype(np.uint8) * 255,
            'orientation': orientation_map,
            'frequency': frequency_map,
            'energy': energy_map,
            'gabor_enhanced': enhanced,
            'smqt_enhanced': smqt_enhanced,
            'skeleton': skeleton
        }
        
        return minutiae, (orientation_map, frequency_map, energy_map), intermediate_results
    
    def visualize_processing_steps(self, intermediate_results: Dict[str, np.ndarray],
                                  minutiae: List[Dict], save_path: Optional[str] = None):
        """Visualize all processing steps in a single figure."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        # Define visualization steps
        steps = [
            ('original', 'Original Image', 'gray'),
            ('normalized', 'Normalized', 'gray'),
            ('mask', 'Segmentation Mask', 'gray'),
            ('orientation', 'Orientation Map', 'hsv'),
            ('frequency', 'Frequency Map', 'viridis'),
            ('energy', 'Energy Map', 'hot'),
            ('gabor_enhanced', 'Gabor Enhanced', 'gray'),
            ('smqt_enhanced', 'SMQT Enhanced', 'gray'),
            ('skeleton', 'Skeleton + Minutiae', 'gray')
        ]
        
        for i, (key, title, cmap) in enumerate(steps):
            if i < len(axes):
                im = axes[i].imshow(intermediate_results[key], cmap=cmap)
                axes[i].set_title(title, fontsize=12, fontweight='bold')
                axes[i].axis('off')
                
                # Add colorbar for feature maps
                if key in ['orientation', 'frequency', 'energy']:
                    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                
                # Add minutiae visualization on skeleton
                if key == 'skeleton' and minutiae:
                    endings = [m for m in minutiae if m['type'] == 'ending']
                    bifurcations = [m for m in minutiae if m['type'] == 'bifurcation']
                    
                    # Plot endings in red
                    if endings:
                        ending_x = [m['x'] for m in endings[:15]]
                        ending_y = [m['y'] for m in endings[:15]]
                        axes[i].scatter(ending_x, ending_y, c='red', s=30, marker='o', 
                                      alpha=0.8, label=f'Endings ({len(endings)})')
                    
                    # Plot bifurcations in blue
                    if bifurcations:
                        bifur_x = [m['x'] for m in bifurcations[:15]]
                        bifur_y = [m['y'] for m in bifurcations[:15]]
                        axes[i].scatter(bifur_x, bifur_y, c='blue', s=30, marker='s', 
                                      alpha=0.8, label=f'Bifurcations ({len(bifurcations)})')
                    
                    if minutiae:
                        axes[i].legend(loc='upper right', fontsize=8)
        
        plt.suptitle('MTCC Fingerprint Processing Pipeline', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print processing statistics
        print(f"\n=== Processing Statistics ===")
        print(f"Total minutiae extracted: {len(minutiae)}")
        if minutiae:
            endings = len([m for m in minutiae if m['type'] == 'ending'])
            bifurcations = len([m for m in minutiae if m['type'] == 'bifurcation'])
            print(f"  - Endings: {endings}")
            print(f"  - Bifurcations: {bifurcations}")
            avg_quality = np.mean([m['quality'] for m in minutiae])
            print(f"  - Average quality: {avg_quality:.2f}")
        
        for key in ['orientation', 'frequency', 'energy']:
            if key in intermediate_results:
                data = intermediate_results[key]
                print(f"{key.capitalize()} map - Range: [{data.min():.3f}, {data.max():.3f}]")


# Example usage and testing functions
def test_mtcc_system():
    """Test the MTCC system with sample data."""
    # Initialize system
    mtcc = MTCCFingerprintSystem()
    
    # Create a synthetic fingerprint for testing
    def create_synthetic_fingerprint(size=(256, 256)):
        """Create a more realistic synthetic fingerprint for testing."""
        h, w = size
        y, x = np.ogrid[:h, :w]
        
        # Create multiple sinusoidal patterns with different orientations
        fingerprint = np.zeros((h, w))
        
        # Add multiple ridge patterns
        for i in range(3):
            frequency = 0.08 + i * 0.02
            orientation = np.pi / 6 * (i + 1)
            
            pattern = 127 + 60 * np.sin(
                2 * np.pi * frequency * (
                    x * np.cos(orientation) + y * np.sin(orientation)
                )
            )
            
            # Add weight based on distance from center
            center_x, center_y = w // 2, h // 2
            weight = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (w/4)**2))
            
            fingerprint += pattern * weight / 3
        
        # Add some ridge breaks and noise for realism
        noise = np.random.normal(0, 15, size)
        fingerprint += noise
        
        # Add some ridge breaks
        for _ in range(20):
            break_x = np.random.randint(w//4, 3*w//4)
            break_y = np.random.randint(h//4, 3*h//4)
            cv2.circle(fingerprint, (break_x, break_y), 3, 50, -1)
        
        return np.clip(fingerprint, 0, 255).astype(np.uint8)
    
    # Create test images
    test_image = create_synthetic_fingerprint()
    
    # Save test image
    cv2.imwrite('test_fingerprint.png', test_image)
    
    print("Testing MTCC Fingerprint Recognition System...")
    
    try:
        # Process fingerprint
        minutiae, features, intermediate = mtcc.process_fingerprint('test_fingerprint.png')
        
        print(f"Extracted {len(minutiae)} minutiae")
        print(f"Orientation map shape: {features[0].shape}")
        print(f"Frequency map shape: {features[1].shape}")
        print(f"Energy map shape: {features[2].shape}")
        
        # Visualize results
        mtcc.visualize_processing_steps(intermediate, minutiae)
        
        # Test matching (self-matching should give high score)
        match_score = mtcc.match_fingerprints(minutiae, minutiae, features, features)
        print(f"Self-matching score: {match_score:.3f}")
        
        # Test EER calculation with dummy data
        genuine_scores = [0.8, 0.9, 0.7, 0.85, 0.92]
        impostor_scores = [0.2, 0.3, 0.15, 0.25, 0.35]
        eer, threshold = mtcc.calculate_eer(genuine_scores, impostor_scores)
        print(f"Test EER: {eer:.3f} at threshold: {threshold:.3f}")
        
        print("MTCC system test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mtcc_system()