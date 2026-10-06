import numpy as np
import cv2
from scipy import ndimage, signal
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import os
from dataclasses import dataclass

@dataclass
class Minutia:
    x: float
    y: float
    angle: float
    quality: float = 1.0

class ImageSegmentation:
    def __init__(self, block_size: int = 16, variance_threshold: float = 100):
        self.block_size = block_size
        self.variance_threshold = variance_threshold
    
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment fingerprint using block-wise variance"""
        h, w = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(0, h - self.block_size, self.block_size):
            for j in range(0, w - self.block_size, self.block_size):
                block = image[i:i+self.block_size, j:j+self.block_size]
                if np.var(block) > self.variance_threshold:
                    mask[i:i+self.block_size, j:j+self.block_size] = 255
        
        # Morphological operations to smooth mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask, image * (mask / 255.0)

class ImageNormalization:
    def __init__(self, target_mean: float = 100, target_variance: float = 100):
        self.target_mean = target_mean
        self.target_variance = target_variance
    
    def normalize(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Normalize image to target mean and variance"""
        if mask is not None:
            roi_pixels = image[mask > 0]
            if len(roi_pixels) == 0:
                return image
            mean_val = np.mean(roi_pixels)
            var_val = np.var(roi_pixels)
        else:
            mean_val = np.mean(image)
            var_val = np.var(image)
        
        if var_val < 1e-6:
            return image
        
        normalized = np.zeros_like(image, dtype=np.float32)
        normalized = self.target_mean + np.sqrt(self.target_variance) * (image - mean_val) / np.sqrt(var_val)
        
        return np.clip(normalized, 0, 255).astype(np.uint8)

class STFTAnalysis:
    def __init__(self, window_size: int = 32, overlap: int = 16):
        self.window_size = window_size
        self.overlap = overlap
        self.step = window_size - overlap
    
    def analyze(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """STFT analysis to extract orientation, frequency, and energy"""
        h, w = image.shape
        orientation = np.zeros((h, w))
        frequency = np.zeros((h, w))
        energy = np.zeros((h, w))
        
        # Ensure we cover the entire image
        for i in range(0, h, self.step):
            for j in range(0, w, self.step):
                # Handle edge cases - adjust window size if needed
                i_end = min(i + self.window_size, h)
                j_end = min(j + self.window_size, w)
                
                window = image[i:i_end, j:j_end]
                
                # Pad window if it's smaller than expected
                if window.shape[0] < self.window_size or window.shape[1] < self.window_size:
                    padded_window = np.zeros((self.window_size, self.window_size))
                    padded_window[:window.shape[0], :window.shape[1]] = window
                    window = padded_window
                
                # Calculate orientation using gradient method (more reliable than FFT peak)
                orient, freq, enrg = self._analyze_window(window)
                
                # Fill the regions - ensure we don't go out of bounds
                orientation[i:i_end, j:j_end] = orient
                frequency[i:i_end, j:j_end] = freq
                energy[i:i_end, j:j_end] = enrg
        
        # Fill any remaining areas with nearest neighbor interpolation
        orientation = self._fill_missing_areas(orientation)
        frequency = self._fill_missing_areas(frequency)
        energy = self._fill_missing_areas(energy)
        
        # Proper normalization
        # Orientation: keep in [-π, π] range (no clipping needed)
        frequency = np.clip(frequency, 0, 1)
        
        # Normalize energy to [0, 1] range
        if np.max(energy) > np.min(energy):
            energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
        else:
            energy = np.ones_like(energy) * 0.5
        
        return orientation, frequency, energy
    
    def _analyze_window(self, window: np.ndarray) -> Tuple[float, float, float]:
        """Analyze a single window to extract orientation, frequency, and energy"""
        # Calculate gradients using Sobel operators
        grad_x = cv2.Sobel(window.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(window.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate orientation using the standard fingerprint method
        # Sum of squared gradients weighted approach
        Gxx = np.sum(grad_x * grad_x)
        Gyy = np.sum(grad_y * grad_y) 
        Gxy = np.sum(grad_x * grad_y)
        
        # Ridge orientation calculation (standard method from fingerprint literature)
        # Orientation is perpendicular to gradient direction
        if Gxx != Gyy:
            orientation = 0.5 * np.arctan2(2 * Gxy, Gxx - Gyy)
        else:
            orientation = np.pi / 4 if Gxy > 0 else -np.pi / 4
        
        # Ensure orientation is in [-π, π] range
        while orientation > np.pi:
            orientation -= 2 * np.pi
        while orientation < -np.pi:
            orientation += 2 * np.pi
        
        # Calculate frequency using autocorrelation in ridge direction
        # Rotate window to align with ridge direction
        center_x, center_y = window.shape[1] // 2, window.shape[0] // 2
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), 
                                                  np.degrees(orientation), 1.0)
        rotated_window = cv2.warpAffine(window, rotation_matrix, window.shape[::-1])
        
        # Take a slice along the ridge direction and compute autocorrelation
        center_slice = rotated_window[center_y, :]
        
        # Simple frequency estimation using zero crossings
        mean_val = np.mean(center_slice)
        zero_crossings = np.where(np.diff(np.sign(center_slice - mean_val)))[0]
        
        if len(zero_crossings) > 1:
            # Estimate frequency from zero crossings
            avg_period = 2 * len(center_slice) / len(zero_crossings)
            frequency = 1.0 / avg_period if avg_period > 0 else 0.1
        else:
            frequency = 0.1  # Default frequency
        
        # Normalize frequency to [0, 1] range
        frequency = np.clip(frequency, 0, 1)
        
        # Calculate energy as variance of the window
        energy = np.var(window)
        
        return orientation, frequency, energy
    
    def _fill_missing_areas(self, array: np.ndarray) -> np.ndarray:
        """Fill any zero areas with nearest neighbor interpolation"""
        mask = array == 0
        if np.any(mask):
            # Simple nearest neighbor filling
            from scipy.ndimage import distance_transform_edt
            
            # Find nearest non-zero values
            indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
            array[mask] = array[tuple(indices[:, mask])]
        
        return array

class GaborFilter:
    def __init__(self, sigma_x: float = 2.0, sigma_y: float = 2.0):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
    
    def create_filter(self, size: int, frequency: float, orientation: float) -> np.ndarray:
        """Create Gabor filter kernel"""
        if size % 2 == 0:
            size += 1
        
        x, y = np.meshgrid(np.arange(-size//2, size//2+1), np.arange(-size//2, size//2+1))
        
        # Rotate coordinates
        x_rot = x * np.cos(orientation) + y * np.sin(orientation)
        y_rot = -x * np.sin(orientation) + y * np.cos(orientation)
        
        # Gaussian envelope
        gaussian = np.exp(-(x_rot**2/(2*self.sigma_x**2) + y_rot**2/(2*self.sigma_y**2)))
        
        # Sinusoidal component with proper frequency scaling
        sinusoid = np.cos(2 * np.pi * frequency * x_rot)
        
        gabor = gaussian * sinusoid
        # Zero mean
        gabor = gabor - np.mean(gabor)
        
        return gabor
    
    def enhance(self, image: np.ndarray, orientations: np.ndarray, 
                frequencies: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Enhance image using locally adaptive Gabor filters"""
        h, w = image.shape
        enhanced = np.zeros_like(image, dtype=np.float32)
        
        # Use smaller blocks for better local adaptation
        block_size = 16
        overlap = block_size // 2
        
        for i in range(0, h - block_size + 1, overlap):
            for j in range(0, w - block_size + 1, overlap):
                # Get local orientation and frequency at block center
                center_i = i + block_size // 2
                center_j = j + block_size // 2
                
                if center_i < h and center_j < w:
                    local_orient = orientations[center_i, center_j]
                    local_freq = frequencies[center_i, center_j]
                    
                    # Scale frequency appropriately (typical ridge frequency is 0.1-0.2 cycles/pixel)
                    scaled_freq = 0.1 + local_freq * 0.1
                    
                    # Create Gabor filter
                    gabor_kernel = self.create_filter(kernel_size, scaled_freq, local_orient)
                    
                    # Extract block
                    block = image[i:i+block_size, j:j+block_size].astype(np.float32)
                    
                    # Apply filter
                    if block.shape == (block_size, block_size):
                        filtered = cv2.filter2D(block, -1, gabor_kernel)
                        
                        # Add to enhanced image (weighted by center)
                        weight = 1.0  # Could use Gaussian weighting for smoother blending
                        enhanced[i:i+block_size, j:j+block_size] += filtered * weight
        
        # Normalize and convert back to uint8
        if np.max(enhanced) > np.min(enhanced):
            enhanced = (enhanced - np.min(enhanced)) / (np.max(enhanced) - np.min(enhanced)) * 255
        enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced.astype(np.uint8)

class MinutiaeExtractor:
    def __init__(self, quality_threshold: float = 0.5):
        self.quality_threshold = quality_threshold
    
    def extract(self, image: np.ndarray, mask: np.ndarray = None) -> List[Minutia]:
        """Extract minutiae using proper binarization and thinning"""
        # Binarize image (ridges should be white/255)
        binary = self._binarize(image)
        
        # Apply mask if provided
        if mask is not None:
            binary = binary & (mask > 0)
        
        # Thin to single pixel ridges
        thinned = self._thin_image(binary)
        
        # Find minutiae points
        minutiae = self._find_minutiae(thinned, image)
        
        return minutiae
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Proper binarization for fingerprint ridges"""
        # Use local adaptive thresholding
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 
            blockSize=15, C=5
        )
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def _thin_image(self, binary: np.ndarray) -> np.ndarray:
        """Zhang-Suen thinning algorithm for single pixel ridges"""
        # Convert to single channel if needed
        if len(binary.shape) == 3:
            binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
        
        # Ensure ridges are 1, background is 0
        binary = (binary > 127).astype(np.uint8)
        
        thinned = binary.copy()
        
        while True:
            # Sub-iteration 1
            marked = self._zhang_suen_iteration(thinned, 0)
            thinned[marked] = 0
            
            # Sub-iteration 2  
            marked = self._zhang_suen_iteration(thinned, 1)
            thinned[marked] = 0
            
            # Check if any pixels were removed
            if not np.any(marked):
                break
        
        return thinned * 255  # Convert back to 0-255 range
    
    def _zhang_suen_iteration(self, image: np.ndarray, iteration: int) -> np.ndarray:
        """One iteration of Zhang-Suen thinning"""
        h, w = image.shape
        marked = np.zeros((h, w), dtype=bool)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                if image[i, j] == 1:  # Only consider ridge pixels
                    # Get 8-neighborhood (clockwise from top)
                    p = [image[i-1, j], image[i-1, j+1], image[i, j+1], image[i+1, j+1],
                         image[i+1, j], image[i+1, j-1], image[i, j-1], image[i-1, j-1]]
                    
                    # Condition 1: 2 <= N(p1) <= 6
                    N_p1 = sum(p)
                    if not (2 <= N_p1 <= 6):
                        continue
                    
                    # Condition 2: S(p1) = 1 (exactly one 0-1 transition)
                    S_p1 = 0
                    for k in range(8):
                        if p[k] == 0 and p[(k+1) % 8] == 1:
                            S_p1 += 1
                    if S_p1 != 1:
                        continue
                    
                    # Condition 3 & 4 (different for each iteration)
                    if iteration == 0:
                        # p2 * p4 * p6 = 0 AND p4 * p6 * p8 = 0
                        if (p[0] * p[2] * p[4] == 0) and (p[2] * p[4] * p[6] == 0):
                            marked[i, j] = True
                    else:
                        # p2 * p4 * p8 = 0 AND p2 * p6 * p8 = 0  
                        if (p[0] * p[2] * p[6] == 0) and (p[0] * p[4] * p[6] == 0):
                            marked[i, j] = True
        
        return marked
    
    def _find_minutiae(self, thinned: np.ndarray, original: np.ndarray) -> List[Minutia]:
        """Find minutiae in thinned image"""
        minutiae = []
        h, w = thinned.shape
        
        # Convert to binary
        binary = (thinned > 127).astype(np.uint8)
        
        # Remove noise and artifacts with size filtering
        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Filter out small components (noise)
        min_component_size = 10
        filtered_binary = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_component_size:
                filtered_binary[labels == i] = 1
        
        binary = filtered_binary
        
        for i in range(2, h-2):
            for j in range(2, w-2):
                if binary[i, j] == 1:  # Ridge pixel
                    # Count 8-connected neighbors
                    neighbors = []
                    neighbor_coords = []
                    
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            if binary[i+di, j+dj] == 1:
                                neighbors.append((i+di, j+dj))
                                neighbor_coords.append((di, dj))
                    
                    # Minutiae: termination (1 neighbor) or bifurcation (3+ neighbors)
                    num_neighbors = len(neighbors)
                    if num_neighbors == 1 or num_neighbors >= 3:
                        # Additional validation - check if it's a real minutia
                        if self._validate_minutia(binary, i, j, num_neighbors):
                            # Calculate angle based on ridge direction
                            angle = self._calculate_ridge_angle(binary, i, j, neighbor_coords)
                            
                            # Calculate quality
                            quality = self._calculate_quality(original, i, j)
                            
                            if quality > self.quality_threshold:
                                minutiae.append(Minutia(j, i, angle, quality))
        
        return minutiae
    
    def _validate_minutia(self, binary: np.ndarray, x: int, y: int, num_neighbors: int) -> bool:
        """Additional validation to remove false minutiae"""
        # Check if minutia is near image boundary (often artifacts)
        h, w = binary.shape
        border_distance = 10
        if x < border_distance or x >= h - border_distance or y < border_distance or y >= w - border_distance:
            return False
        
        # For bifurcations, ensure they're not just noise
        if num_neighbors >= 3:
            # Check in a larger neighborhood to ensure it's a real bifurcation
            large_neighborhood = binary[x-2:x+3, y-2:y+3]
            if np.sum(large_neighborhood) < 4:  # Too sparse
                return False
        
        return True
    
    def _calculate_ridge_angle(self, binary: np.ndarray, x: int, y: int, neighbor_coords: List) -> float:
        """Calculate ridge direction at minutia point"""
        if not neighbor_coords:
            return 0.0
        
        # For termination: use direction to the single neighbor
        if len(neighbor_coords) == 1:
            dx, dy = neighbor_coords[0]
            return np.arctan2(dy, dx)
        
        # For bifurcation: use average direction
        angles = []
        for dx, dy in neighbor_coords:
            angles.append(np.arctan2(dy, dx))
        
        # Convert to complex numbers for circular averaging
        complex_angles = [np.exp(1j * angle) for angle in angles]
        avg_complex = np.mean(complex_angles)
        return np.angle(avg_complex)
    
    def _calculate_quality(self, image: np.ndarray, x: int, y: int, radius: int = 8) -> float:
        """Calculate minutia quality based on local coherence"""
        x1, y1 = max(0, x-radius), max(0, y-radius)
        x2, y2 = min(image.shape[0], x+radius), min(image.shape[1], y+radius)
        
        local_region = image[x1:x2, y1:y2]
        
        # Quality based on local variance and mean
        variance = np.var(local_region)
        mean_val = np.mean(local_region)
        
        # Normalize quality score
        quality = min(variance / 1000.0, 1.0) * min(mean_val / 255.0, 1.0)
        return quality

class MTCCDescriptor:
    def __init__(self, radius: int = 65, ns: int = 18, nd: int = 5):
        self.radius = radius
        self.ns = ns  # Number of spatial divisions
        self.nd = nd  # Number of angular divisions
        self.delta_s = 2 * radius / ns
        self.delta_d = 2 * np.pi / nd
    
    def extract_mcc_features(self, minutiae: List[Minutia], orientation_img: np.ndarray,
                           frequency_img: np.ndarray, energy_img: np.ndarray,
                           image_shape: Tuple[int, int], feature_type: str = 'orientation') -> np.ndarray:
        """Extract MTCC features for all minutiae"""
        features = []
        
        for minutia in minutiae:
            cylinder = self._create_cylinder(minutia, orientation_img, frequency_img, 
                                          energy_img, image_shape, feature_type)
            features.append(cylinder.flatten())
        
        return np.array(features)
    
    def _create_cylinder(self, minutia: Minutia, orientation_img: np.ndarray,
                        frequency_img: np.ndarray, energy_img: np.ndarray,
                        image_shape: Tuple[int, int], feature_type: str) -> np.ndarray:
        """Create 3D cylinder for a single minutia"""
        cylinder = np.zeros((self.ns, self.ns, self.nd))
        
        # Select feature image based on type
        if feature_type == 'orientation':
            feature_img = orientation_img
        elif feature_type == 'frequency':
            feature_img = frequency_img
        elif feature_type == 'energy':
            feature_img = energy_img
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Create rotation matrix
        cos_theta = np.cos(minutia.angle)
        sin_theta = np.sin(minutia.angle)
        
        for i in range(self.ns):
            for j in range(self.ns):
                # Cell center in local coordinates
                local_x = (i - self.ns//2) * self.delta_s
                local_y = (j - self.ns//2) * self.delta_s
                
                # Transform to global coordinates
                global_x = minutia.x + local_x * cos_theta - local_y * sin_theta
                global_y = minutia.y + local_x * sin_theta + local_y * cos_theta
                
                # Check bounds
                if (0 <= global_x < image_shape[1] and 0 <= global_y < image_shape[0]):
                    gx, gy = int(global_x), int(global_y)
                    
                    for k in range(self.nd):
                        # Angular bin
                        bin_angle = -np.pi + (k + 0.5) * self.delta_d
                        
                        # Get feature value at this location
                        feature_val = feature_img[gy, gx]
                        
                        # Calculate contribution based on feature type
                        if feature_type == 'orientation':
                            # Use angular difference for orientation
                            angle_diff = self._angle_difference(feature_val, bin_angle)
                            contribution = np.exp(-angle_diff**2 / (2 * (np.pi/8)**2))
                        else:
                            # Use Gaussian for frequency/energy
                            contribution = np.exp(-(feature_val - 0.5)**2 / (2 * 0.25**2))
                        
                        cylinder[i, j, k] = contribution
        
        return cylinder
    
    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """Calculate normalized angle difference"""
        diff = angle1 - angle2
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return abs(diff)

class MTCCMatcher:
    def __init__(self, distance_type: str = 'euclidean'):
        self.distance_type = distance_type
    
    def match(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Match two sets of MTCC features"""
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        
        # Local Similarity Sort (LSS)
        similarity_matrix = self._compute_similarity_matrix(features1, features2)
        
        # Select top matches
        max_matches = min(len(features1), len(features2), 10)
        top_scores = []
        
        for _ in range(max_matches):
            max_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
            if similarity_matrix[max_idx] > 0:
                top_scores.append(similarity_matrix[max_idx])
                # Zero out row and column to avoid re-matching
                similarity_matrix[max_idx[0], :] = 0
                similarity_matrix[:, max_idx[1]] = 0
            else:
                break
        
        return np.mean(top_scores) if top_scores else 0.0
    
    def _compute_similarity_matrix(self, features1: np.ndarray, features2: np.ndarray) -> np.ndarray:
        """Compute similarity matrix between feature sets"""
        similarity_matrix = np.zeros((len(features1), len(features2)))
        
        for i, feat1 in enumerate(features1):
            for j, feat2 in enumerate(features2):
                if self.distance_type == 'euclidean':
                    dist = euclidean(feat1, feat2)
                    similarity = 1.0 / (1.0 + dist)
                elif self.distance_type == 'cosine':
                    dot_product = np.dot(feat1, feat2)
                    norms = np.linalg.norm(feat1) * np.linalg.norm(feat2)
                    similarity = dot_product / (norms + 1e-8)
                else:
                    similarity = 0.0
                
                similarity_matrix[i, j] = similarity
        
        return similarity_matrix

class MTCCVisualizer:
    def __init__(self, show_plots: bool = True):
        self.show_plots = show_plots
    
    def visualize_segmentation(self, image: np.ndarray, mask: np.ndarray, segmented: np.ndarray):
        """Visualize segmentation results"""
        if not self.show_plots:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        axes[2].imshow(segmented, cmap='gray')
        axes[2].set_title('Segmented Image')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_enhancement(self, original: np.ndarray, enhanced: np.ndarray):
        """Visualize enhancement results"""
        if not self.show_plots:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(enhanced, cmap='gray')
        axes[1].set_title('Enhanced')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_stft_features(self, orientation: np.ndarray, frequency: np.ndarray, energy: np.ndarray):
        """Visualize STFT analysis results"""
        if not self.show_plots:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        im1 = axes[0].imshow(orientation, cmap='gray')
        axes[0].set_title('Orientation')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(frequency, cmap='gray')
        axes[1].set_title('Frequency')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(energy, cmap='gray')
        axes[2].set_title('Energy')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.show()
    
    def visualize_minutiae(self, image: np.ndarray, minutiae: List[Minutia]):
        """Visualize detected minutiae"""
        if not self.show_plots:
            return
            
        plt.figure(figsize=(10, 8))
        plt.imshow(image, cmap='gray')
        
        for minutia in minutiae:
            # Plot minutia location
            plt.plot(minutia.x, minutia.y, 'ro', markersize=5)
            
            # Plot minutia direction
            length = 15
            end_x = minutia.x + length * np.cos(minutia.angle)
            end_y = minutia.y + length * np.sin(minutia.angle)
            plt.arrow(minutia.x, minutia.y, end_x - minutia.x, end_y - minutia.y,
                     head_width=3, head_length=3, fc='red', ec='red')
        
        plt.title(f'Detected Minutiae ({len(minutiae)} points)')
        plt.axis('off')
        plt.show()
    
    def visualize_binarization(self, original: np.ndarray, binary: np.ndarray, thinned: np.ndarray):
        """Visualize binarization and thinning process"""
        if not self.show_plots:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(binary, cmap='gray')
        axes[1].set_title('Binarized')
        axes[1].axis('off')
        
        axes[2].imshow(thinned, cmap='gray')
        axes[2].set_title('Thinned')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

class FVCDataLoader:
    @staticmethod
    def load_image(filepath: str) -> np.ndarray:
        """Load fingerprint image"""
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {filepath}")
        return image
    
    @staticmethod
    def test_image_pair(image1_path: str, image2_path: str, visualize: bool = False) -> Dict:
        """Test matching between two fingerprint images"""
        # Load images
        img1 = FVCDataLoader.load_image(image1_path)
        img2 = FVCDataLoader.load_image(image2_path)
        
        # Initialize components
        segmenter = ImageSegmentation()
        normalizer = ImageNormalization()
        stft = STFTAnalysis()
        gabor = GaborFilter()
        extractor = MinutiaeExtractor()
        descriptor = MTCCDescriptor()
        matcher = MTCCMatcher()
        visualizer = MTCCVisualizer(show_plots=visualize)
        
        results = {}
        
        # Process both images
        for idx, img in enumerate([img1, img2], 1):
            # Segmentation
            mask, segmented = segmenter.segment(img)
            
            # Normalization
            normalized = normalizer.normalize(segmented, mask)
            
            # STFT Analysis on normalized image (BEFORE enhancement)
            orientation, frequency, energy = stft.analyze(normalized)
            
            # Gabor Enhancement using the orientation map
            enhanced = gabor.enhance(normalized, orientation, frequency)
            
            # Minutiae Extraction on enhanced image (binarize AFTER enhancement)
            minutiae = extractor.extract(enhanced, mask)
            
            # Show binarization process if visualizing
            if visualize:
                binary = extractor._binarize(enhanced)
                thinned = extractor._thin_image(binary)
                visualizer.visualize_binarization(enhanced, binary, thinned)
            
            # MTCC Feature Extraction (using orientation features)
            mtcc_features = descriptor.extract_mcc_features(
                minutiae, orientation, frequency, energy, img.shape, 'orientation'
            )
            
            results[f'image{idx}'] = {
                'original': img,
                'mask': mask,
                'segmented': segmented,
                'normalized': normalized,
                'orientation': orientation,
                'frequency': frequency,
                'energy': energy,
                'enhanced': enhanced,
                'minutiae': minutiae,
                'features': mtcc_features
            }
            
            # Visualizations
            if visualize:
                print(f"\n--- Image {idx} Processing ---")
                visualizer.visualize_segmentation(img, mask, segmented)
                visualizer.visualize_stft_features(orientation, frequency, energy)
                visualizer.visualize_enhancement(normalized, enhanced)
                visualizer.visualize_minutiae(enhanced, minutiae)
        
        # Matching
        match_score = matcher.match(results['image1']['features'], results['image2']['features'])
        results['match_score'] = match_score
        
        print(f"\nMatching Results:")
        print(f"Image 1 minutiae: {len(results['image1']['minutiae'])}")
        print(f"Image 2 minutiae: {len(results['image2']['minutiae'])}")
        print(f"Match Score: {match_score:.4f}")
        
        return results

# Example usage function
def example_usage():
    """Example of how to use the MTCC implementation"""
    
    # Test with two fingerprint images (provide your own paths)
    image1_path = r"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002\FVC2002\Db1_a\1_1.tif"
    image2_path = r"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002\FVC2002\Db1_a\1_2.tif"
    
    try:
        # Test image pair matching with visualization
        results = FVCDataLoader.test_image_pair(image1_path, image2_path, visualize=True)
        
        # Print summary
        print(f"\nProcessing Summary:")
        print(f"Match Score: {results['match_score']:.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please provide valid fingerprint image paths")
        
        # Create sample synthetic data for demonstration
        print("\nCreating synthetic demonstration...")
        img1 = np.random.randint(0, 255, (300, 300)).astype(np.uint8)
        img2 = img1 + np.random.randint(-20, 20, (300, 300)).astype(np.uint8)
        
        # Save temporary images
        cv2.imwrite('temp_img1.jpg', img1)
        cv2.imwrite('temp_img2.jpg', img2)
        
        # Test with synthetic images
        results = FVCDataLoader.test_image_pair('temp_img1.jpg', 'temp_img2.jpg', visualize=False)
        
        # Cleanup
        os.remove('temp_img1.jpg')
        os.remove('temp_img2.jpg')

if __name__ == "__main__":
    example_usage()