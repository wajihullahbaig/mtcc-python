import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import cv2
from scipy import ndimage
from scipy.fft import fft2, ifft2
from scipy.signal import windows
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap


@dataclass
class Minutia:
    """Represents a minutia point with coordinates and angle"""
    x: float
    y: float
    angle: float
    quality: float = 1.0


@dataclass
class MTCCDescriptor:
    """Container for MTCC feature descriptor"""
    minutia: Minutia
    cylinder: np.ndarray
    valid: bool = True


class VisualizationManager:
    """Manages visualization of different processing steps"""
    
    def __init__(self, show_config: Dict[str, bool] = None):
        self.default_config = {
            'image_loaded': True,
            'image_enhanced': True,
            'image_segmented': True,
            'image_masks': True,
            'binary_image': True,
            'minutiae_plots': True
        }
        
        self.show_config = show_config if show_config else self.default_config
        self.figure_counter = 0
    
    def show_image_loaded(self, image: np.ndarray, title: str = "Original Image"):
        """Visualize the loaded fingerprint image"""
        if not self.show_config.get('image_loaded', False):
            return
        
        plt.figure(figsize=(8, 6))
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    
    def show_image_enhanced(self, original: np.ndarray, enhanced: np.ndarray, 
                           title: str = "Image Enhancement"):
        """Visualize original vs enhanced image"""
        if not self.show_config.get('image_enhanced', False):
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(enhanced, cmap='gray')
        axes[1].set_title('Enhanced Image')
        axes[1].axis('off')
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def show_image_segmented(self, image: np.ndarray, segmented: np.ndarray,
                           title: str = "Image Segmentation"):
        """Visualize original vs segmented image"""
        if not self.show_config.get('image_segmented', False):
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(segmented, cmap='gray')
        axes[1].set_title('Segmented Image')
        axes[1].axis('off')
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def show_image_masks(self, image: np.ndarray, mask: np.ndarray, 
                    orientation: np.ndarray = None, frequency: np.ndarray = None,
                    energy: np.ndarray = None, title: str = "Segmentation Masks and Texture Features"):
        """Visualize segmentation mask and texture feature maps in grayscale"""
        if not self.show_config.get('image_masks', False):
            return
        
        has_texture = orientation is not None and frequency is not None and energy is not None
        cols = 5 if has_texture else 2
        
        fig, axes = plt.subplots(1, cols, figsize=(4*cols, 4))
        if cols == 2:
            axes = [axes[0], axes[1]]
        
        # Original with mask overlay
        axes[0].imshow(image, cmap='gray')
        mask_overlay = np.ma.masked_where(mask == 0, mask)
        axes[0].imshow(mask_overlay, cmap='Reds', alpha=0.3)
        axes[0].set_title('Image with Mask Overlay')
        axes[0].axis('off')
        
        # Mask only
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        if has_texture:
            # Orientation map in grayscale (normalize from [-π, π] to [0, 1])
            orientation_normalized = (orientation + np.pi) / (2 * np.pi)
            axes[2].imshow(orientation_normalized, cmap='gray')
            axes[2].set_title('Orientation Map')
            axes[2].axis('off')
            
            # Frequency map in grayscale
            axes[3].imshow(frequency, cmap='gray')
            axes[3].set_title('Frequency Map')
            axes[3].axis('off')
            
            # Energy map in grayscale
            axes[4].imshow(energy, cmap='gray')
            axes[4].set_title('Energy Map')
            axes[4].axis('off')
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def show_binary_image(self, original: np.ndarray, binary: np.ndarray, 
                         skeleton: np.ndarray = None, title: str = "Binary Processing"):
        """Visualize binary processing steps"""
        if not self.show_config.get('binary_image', False):
            return
        
        cols = 3 if skeleton is not None else 2
        fig, axes = plt.subplots(1, cols, figsize=(4*cols, 4))
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Enhanced Image')
        axes[0].axis('off')
        
        axes[1].imshow(binary, cmap='gray')
        axes[1].set_title('Binary Image')
        axes[1].axis('off')
        
        if skeleton is not None:
            axes[2].imshow(skeleton, cmap='gray')
            axes[2].set_title('Skeleton/Thinned')
            axes[2].axis('off')
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def show_minutiae_plots(self, image: np.ndarray, minutiae: List[Minutia],
                           binary: np.ndarray = None, title: str = "Minutiae Detection"):
        """Visualize detected minutiae on the fingerprint"""
        if not self.show_config.get('minutiae_plots', False):
            return
        
        cols = 2 if binary is not None else 1
        fig, axes = plt.subplots(1, cols, figsize=(6*cols, 6))
        
        if cols == 1:
            axes = [axes]
        
        # Minutiae on enhanced image
        axes[0].imshow(image, cmap='gray')
        self._plot_minutiae_on_axis(axes[0], minutiae)
        axes[0].set_title(f'Minutiae on Enhanced Image ({len(minutiae)} detected)')
        axes[0].axis('off')
        
        if binary is not None:
            # Minutiae on binary/skeleton
            axes[1].imshow(binary, cmap='gray')
            self._plot_minutiae_on_axis(axes[1], minutiae)
            axes[1].set_title(f'Minutiae on Binary Image ({len(minutiae)} detected)')
            axes[1].axis('off')
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def _plot_minutiae_on_axis(self, ax, minutiae: List[Minutia]):
        """Helper to plot minutiae points and directions on an axis"""
        for minutia in minutiae:
            # Plot minutia point
            circle = patches.Circle((minutia.x, minutia.y), radius=3, 
                                  color='red', fill=True, alpha=0.8)
            ax.add_patch(circle)
            
            # Plot direction arrow
            arrow_length = 15
            dx = arrow_length * np.cos(minutia.angle)
            dy = arrow_length * np.sin(minutia.angle)
            
            ax.arrow(minutia.x, minutia.y, dx, dy, 
                    head_width=2, head_length=3, fc='blue', ec='blue', alpha=0.8)
            
            # Add quality text
            ax.text(minutia.x + 8, minutia.y - 8, f'{minutia.quality:.2f}',
                   fontsize=8, color='green', weight='bold')
    
    def show_cylinder_visualization(self, minutia: Minutia, cylinder: np.ndarray,
                                  title: str = "Cylinder Visualization"):
        """Visualize a single minutia cylinder"""
        if not self.show_config.get('minutiae_plots', False):
            return
        
        fig = plt.figure(figsize=(12, 4))
        
        # Show different slices of the cylinder
        for i in range(min(3, cylinder.shape[2])):
            ax = fig.add_subplot(1, 3, i+1)
            im = ax.imshow(cylinder[:, :, i], cmap='viridis')
            ax.set_title(f'Cylinder Slice {i+1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        fig.suptitle(f'{title} - Minutia at ({minutia.x:.1f}, {minutia.y:.1f})')
        plt.tight_layout()
        plt.show()
    
    def show_matching_visualization(self, image1: np.ndarray, minutiae1: List[Minutia],
                                  image2: np.ndarray, minutiae2: List[Minutia],
                                  matches: List[Tuple[int, int, float]] = None,
                                  title: str = "Fingerprint Matching"):
        """Visualize fingerprint matching with correspondences"""
        if not self.show_config.get('minutiae_plots', False):
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # First fingerprint
        axes[0].imshow(image1, cmap='gray')
        self._plot_minutiae_on_axis(axes[0], minutiae1)
        axes[0].set_title(f'Fingerprint 1 ({len(minutiae1)} minutiae)')
        axes[0].axis('off')
        
        # Second fingerprint
        axes[1].imshow(image2, cmap='gray')
        self._plot_minutiae_on_axis(axes[1], minutiae2)
        axes[1].set_title(f'Fingerprint 2 ({len(minutiae2)} minutiae)')
        axes[1].axis('off')
        
        # Add match lines if provided
        if matches:
            for i, (idx1, idx2, score) in enumerate(matches[:5]):  # Show top 5 matches
                if idx1 < len(minutiae1) and idx2 < len(minutiae2):
                    # Highlight matched minutiae
                    m1, m2 = minutiae1[idx1], minutiae2[idx2]
                    
                    # Mark matched points differently
                    axes[0].plot(m1.x, m1.y, 'go', markersize=8, markerfacecolor='lime')
                    axes[1].plot(m2.x, m2.y, 'go', markersize=8, markerfacecolor='lime')
                    
                    # Add match score text
                    axes[0].text(m1.x + 10, m1.y + 10, f'M{i+1}:{score:.2f}',
                               fontsize=10, color='lime', weight='bold')
                    axes[1].text(m2.x + 10, m2.y + 10, f'M{i+1}:{score:.2f}',
                               fontsize=10, color='lime', weight='bold')
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def show_enhancement_pipeline(self, original: np.ndarray, segmented: np.ndarray,
                                gabor: np.ndarray, final: np.ndarray,
                                title: str = "Enhancement Pipeline"):
        """Visualize the complete enhancement pipeline"""
        if not self.show_config.get('image_enhanced', False):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(segmented, cmap='gray')
        axes[0, 1].set_title('2. After STFT Analysis')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(gabor, cmap='gray')
        axes[1, 0].set_title('3. After Gabor Filtering')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(final, cmap='gray')
        axes[1, 1].set_title('4. Final Enhanced (SMQT)')
        axes[1, 1].axis('off')
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()


class DatasetLoader(ABC):
    """Abstract base class for dataset loading"""
    
    @abstractmethod
    def load_fingerprint(self, path: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_image_pairs(self) -> List[Tuple[str, str]]:
        pass


class FVCDatasetLoader(DatasetLoader):
    """FVC dataset loader implementation"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
    
    def load_fingerprint(self, path: str) -> np.ndarray:
        """Load fingerprint image from path"""
        full_path = os.path.join(self.dataset_path, path) if not os.path.isabs(path) else path
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {full_path}")
        return image
    
    def get_image_pairs(self) -> List[Tuple[str, str]]:
        """Get pairs of fingerprint images for matching"""
        pairs = []
        files = [f for f in os.listdir(self.dataset_path) if f.endswith('.tif')]
        files.sort()
        
        # Create genuine pairs (same finger, different impressions)
        for i in range(1, 101):  # FVC format: 100 fingers
            finger_files = [f for f in files if f.startswith(f"{i}_")]
            for j in range(len(finger_files)):
                for k in range(j + 1, len(finger_files)):
                    pairs.append((finger_files[j], finger_files[k]))
        
        # Add some impostor pairs
        for i in range(0, min(100, len(files))):
            for j in range(i + 1, min(i + 10, len(files))):
                if files[i].split('_')[0] != files[j].split('_')[0]:
                    pairs.append((files[i], files[j]))
        
        return pairs


class ImageSegmentor:
    """Handles fingerprint image segmentation"""
    
    def __init__(self, block_size: int = 16, threshold: float = 100):
        self.block_size = block_size
        self.threshold = threshold
    
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment fingerprint from background using variance-based method"""
        h, w = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(0, h - self.block_size, self.block_size):
            for j in range(0, w - self.block_size, self.block_size):
                block = image[i:i + self.block_size, j:j + self.block_size]
                variance = np.var(block.astype(np.float32))
                
                if variance > self.threshold:
                    mask[i:i + self.block_size, j:j + self.block_size] = 255
        
        # Morphological operations to smooth mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return image, mask



import numpy as np
import cv2
from scipy import ndimage
from scipy.fft import fft2, ifft2

class STFTAnalyzer:
    """Enhanced STFT analyzer with configurable overlapping windows"""
    
    def __init__(self, window_size: int = 14, overlap: int = 10):
        self.window_size = window_size
        self.overlap = overlap
        self.step = window_size - overlap
        
        # Ensure minimum step size for heavy overlap
        if self.step < 1:
            self.step = 1
            self.overlap = window_size - 1
    
    def analyze(self, image: np.ndarray) -> tuple:
        """Enhanced STFT analysis with heavy overlapping and smooth blending"""
        # Normalize image first
        image = self._normalize_image(image)
        
        h, w = image.shape
        orientation = np.zeros((h, w))
        frequency = np.zeros((h, w))
        energy = np.zeros((h, w))
        weight_sum = np.zeros((h, w))  # For proper weighted averaging
        
        # Use heavy overlap (75-90%) for smoother results
        step = max(1, self.window_size // 8)  # 87.5% overlap
        
        # Create smooth blending window (Hann window for better blending)
        blend_window = self._create_hann_window(self.window_size)
        
        # Process with heavy overlap
        for i in range(0, h - self.window_size + 1, step):
            for j in range(0, w - self.window_size + 1, step):
                # Extract block with padding if needed
                block = self._extract_padded_block(image, i, j, self.window_size)
                
                # Calculate local features
                local_orientation = self._calculate_local_orientation(block)
                local_frequency = self._calculate_local_frequency(block, local_orientation)
                local_energy = self._calculate_local_energy(block)
                
                # Apply features with smooth weighting
                for di in range(self.window_size):
                    for dj in range(self.window_size):
                        if i + di < h and j + dj < w:
                            weight = blend_window[di, dj]
                            
                            orientation[i + di, j + dj] += local_orientation * weight
                            frequency[i + di, j + dj] += local_frequency * weight
                            energy[i + di, j + dj] += local_energy * weight
                            weight_sum[i + di, j + dj] += weight
        
        # Normalize by weight sum
        mask = weight_sum > 1e-10
        orientation[mask] /= weight_sum[mask]
        frequency[mask] /= weight_sum[mask]
        energy[mask] /= weight_sum[mask]
        
        # Handle boundaries with smooth extension
        orientation = self._smooth_boundaries(orientation, mask)
        frequency = self._smooth_boundaries(frequency, mask)
        energy = self._smooth_boundaries(energy, mask)
        
        # Apply final smoothing to remove any remaining artifacts
        orientation = self._smooth_orientation_field(orientation, sigma=1.5)
        frequency = self._smooth_field(frequency, sigma=1.0)
        energy = self._smooth_field(energy, sigma=1.0)
        
        # Normalize to proper ranges
        orientation = np.arctan2(np.sin(orientation), np.cos(orientation))
        frequency = self._normalize_to_01(frequency)
        energy = self._normalize_to_01(energy)
        
        return orientation, frequency, energy

    def _create_hann_window(self, size: int) -> np.ndarray:
        """Create 2D Hann window for smooth blending"""
        hann_1d = np.hanning(size)
        hann_2d = np.outer(hann_1d, hann_1d)
        return hann_2d

    def _extract_padded_block(self, image: np.ndarray, i: int, j: int, size: int) -> np.ndarray:
        """Extract block with reflection padding if needed"""
        h, w = image.shape
        
        # Calculate actual block boundaries
        i_start, i_end = i, min(i + size, h)
        j_start, j_end = j, min(j + size, w)
        
        # Extract the block
        block = image[i_start:i_end, j_start:j_end]
        
        # Pad if necessary using reflection
        if block.shape != (size, size):
            pad_h = size - block.shape[0]
            pad_w = size - block.shape[1]
            block = np.pad(block, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        return block

    def _smooth_boundaries(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Smooth boundary regions using inpainting"""
        if not np.any(~mask):
            return data
        
        # Convert to uint8 for inpainting
        data_norm = ((data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8) * 255).astype(np.uint8)
        mask_uint8 = (~mask).astype(np.uint8) * 255
        
        # Apply inpainting
        if np.any(mask_uint8):
            filled = cv2.inpaint(data_norm, mask_uint8, 5, cv2.INPAINT_TELEA)
            # Convert back to original range
            data = filled.astype(np.float64) / 255.0 * (np.max(data) - np.min(data)) + np.min(data)
        
        return data

    def _smooth_orientation_field(self, orientation: np.ndarray, sigma: float = 1.5) -> np.ndarray:
        """Smooth orientation field preserving discontinuities"""
        # Convert to complex representation for smooth averaging
        complex_orient = np.exp(2j * orientation)
        
        # Apply Gaussian smoothing to real and imaginary parts
        real_smooth = cv2.GaussianBlur(complex_orient.real, (0, 0), sigma)
        imag_smooth = cv2.GaussianBlur(complex_orient.imag, (0, 0), sigma)
        
        # Convert back to orientation
        smoothed_orient = np.angle(real_smooth + 1j * imag_smooth) / 2
        return smoothed_orient

    def _smooth_field(self, field: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing to remove artifacts"""
        return cv2.GaussianBlur(field, (0, 0), sigma)

    def _calculate_local_orientation(self, block: np.ndarray) -> float:
        """Calculate local ridge orientation using improved gradient method"""
        # Apply slight Gaussian blur to reduce noise
        block = cv2.GaussianBlur(block, (3, 3), 0.8)
        
        # Calculate gradients using Sobel operators
        gx = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=5)
        
        # Calculate orientation tensor components with center weighting
        center_weight = self._create_hann_window(block.shape[0])
        
        gxx = gx * gx * center_weight
        gyy = gy * gy * center_weight
        gxy = gx * gy * center_weight
        
        # Weighted averages
        avg_gxx = np.sum(gxx)
        avg_gyy = np.sum(gyy)
        avg_gxy = np.sum(gxy)
        
        # Calculate orientation (ridge direction is perpendicular to gradient)
        if abs(avg_gxx - avg_gyy) > 1e-10 or abs(avg_gxy) > 1e-10:
            orientation = 0.5 * np.arctan2(2 * avg_gxy, avg_gxx - avg_gyy)
        else:
            orientation = 0
        
        return orientation

    def _calculate_local_frequency(self, block: np.ndarray, orientation: float) -> float:
        """Calculate local ridge frequency using improved projection method"""
        h, w = block.shape
        center_x, center_y = w // 2, h // 2
        
        # Create projection perpendicular to ridge orientation
        projection_angle = orientation + np.pi / 2
        
        # Collect signature along multiple parallel lines for robustness
        signatures = []
        for offset in range(-3, 4):  # More parallel projections
            signature = []
            for t in range(-min(center_x, center_y) + 8, min(center_x, center_y) - 8):
                x = int(center_x + t * np.cos(projection_angle) + offset * np.cos(orientation))
                y = int(center_y + t * np.sin(projection_angle) + offset * np.sin(orientation))
                
                if 0 <= x < w and 0 <= y < h:
                    signature.append(block[y, x])
            
            if len(signature) > 15:
                signatures.append(signature)
        
        if not signatures:
            return 0.1
        
        # Average the signatures with length normalization
        min_len = min(len(sig) for sig in signatures)
        avg_signature = np.mean([sig[:min_len] for sig in signatures], axis=0)
        
        # Apply stronger smoothing
        if len(avg_signature) > 7:
            avg_signature = cv2.GaussianBlur(avg_signature.reshape(-1, 1), (7, 1), 1.5).flatten()
        
        # Estimate frequency using autocorrelation
        frequency = self._estimate_frequency_autocorr(avg_signature)
        return frequency

    def _calculate_local_energy(self, block: np.ndarray) -> float:
        """Calculate local energy using variance with center weighting"""
        center_weight = self._create_hann_window(block.shape[0])
        weighted_mean = np.sum(block * center_weight) / np.sum(center_weight)
        weighted_var = np.sum(((block - weighted_mean) ** 2) * center_weight) / np.sum(center_weight)
        return np.sqrt(weighted_var)  # Use standard deviation instead of variance

    def _estimate_frequency_autocorr(self, signature: np.ndarray) -> float:
        """Estimate frequency using autocorrelation method"""
        if len(signature) < 12:
            return 0.1
        
        # Remove DC component and apply window
        signature = signature - np.mean(signature)
        signature = signature * np.hanning(len(signature))
        
        # Compute autocorrelation
        autocorr = np.correlate(signature, signature, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        
        # Find first significant peak (skip lag 0)
        peaks = []
        for i in range(4, min(len(autocorr)//2, 25)):  # Look for peaks between 4-25 pixels
            if i < len(autocorr) - 1:
                if (autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and 
                    autocorr[i] > 0.2):  # Lower threshold for better detection
                    peaks.append((i, autocorr[i]))
        
        if peaks:
            # Use the strongest peak within reasonable range
            peaks.sort(key=lambda x: x[1], reverse=True)
            period = peaks[0][0]
            frequency = 1.0 / period
        else:
            frequency = 0.1  # Default frequency
        
        # Normalize frequency to reasonable range
        return np.clip(frequency, 0.04, 0.4)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        image = image.astype(np.float64)
        return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

    def _normalize_to_01(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] range"""
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)


def apply_enhanced_gabor_filter(image: np.ndarray, orientation: np.ndarray, 
                               window_size: int = 31, overlap_ratio: float = 0.75) -> np.ndarray:
    """
    Apply overlapping Gabor filtering with smooth blending to avoid artifacts
    
    Args:
        image: Input image
        orientation: Orientation field
        window_size: Size of processing windows (default 31)
        overlap_ratio: Overlap ratio between windows (default 0.75)
    """
    h, w = image.shape
    filtered = np.zeros_like(image, dtype=np.float32)
    weight_sum = np.zeros_like(image, dtype=np.float32)
    
    # Calculate step size based on overlap ratio
    step = max(1, int(window_size * (1 - overlap_ratio)))
    
    # Pre-compute Gabor kernels for different orientations
    sigma = window_size / 8.0
    lambd = window_size / 4.0
    gamma = 0.5
    
    # Quantize orientations to reduce computation
    num_orientations = 24  # Increased for better quality
    orientation_bins = np.linspace(-np.pi/2, np.pi/2, num_orientations, endpoint=False)
    
    # Pre-compute kernels
    kernels = {}
    for i, angle in enumerate(orientation_bins):
        kernels[i] = cv2.getGaborKernel((window_size, window_size), sigma, angle, 
                                      lambd, gamma, 0, cv2.CV_32F)
    
    # Create blending window
    blend_window = np.outer(np.hanning(window_size), np.hanning(window_size))
    
    # Process overlapping windows
    for i in range(0, h - window_size + step, step):
        for j in range(0, w - window_size + step, step):
            # Extract window
            i_end = min(i + window_size, h)
            j_end = min(j + window_size, w)
            
            window_img = image[i:i_end, j:j_end]
            window_orient = orientation[i:i_end, j:j_end]
            
            # Pad if necessary
            if window_img.shape != (window_size, window_size):
                pad_h = window_size - window_img.shape[0]
                pad_w = window_size - window_img.shape[1]
                window_img = np.pad(window_img, ((0, pad_h), (0, pad_w)), mode='reflect')
                window_orient = np.pad(window_orient, ((0, pad_h), (0, pad_w)), mode='reflect')
            
            # Determine dominant orientation in window
            avg_orientation = np.mean(window_orient)
            
            # Find closest orientation bin
            orientation_idx = np.argmin(np.abs(orientation_bins - avg_orientation))
            
            # Apply Gabor filter
            filtered_window = cv2.filter2D(window_img, cv2.CV_32F, kernels[orientation_idx])
            
            # Apply blending weights
            filtered_window = filtered_window * blend_window
            weight_window = blend_window.copy()
            
            # Add to result with proper bounds checking
            actual_h = min(window_size, h - i)
            actual_w = min(window_size, w - j)
            
            filtered[i:i+actual_h, j:j+actual_w] += filtered_window[:actual_h, :actual_w]
            weight_sum[i:i+actual_h, j:j+actual_w] += weight_window[:actual_h, :actual_w]
    
    # Normalize by weights
    mask = weight_sum > 1e-10
    filtered[mask] /= weight_sum[mask]
    
    # Handle areas with no coverage (shouldn't happen with proper overlap)
    if np.any(~mask):
        filtered[~mask] = image[~mask]
    
    # Final smoothing to remove any remaining artifacts
    filtered = cv2.GaussianBlur(filtered, (3, 3), 0.5)
    
    return np.clip(filtered, 0, 255).astype(np.uint8)


def apply_enhanced_smqt(image: np.ndarray, tile_size: int = 8, clip_limit: float = 3.0) -> np.ndarray:
    """
    Enhanced SMQT using adaptive CLAHE with overlapping tiles
    
    Args:
        image: Input image
        tile_size: Size of CLAHE tiles (default 8)
        clip_limit: Clipping limit for CLAHE (default 3.0)
    """
    # Apply CLAHE with smaller tiles for better local adaptation
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(image)
    
    # Apply additional smoothing to reduce tile artifacts
    enhanced = cv2.bilateralFilter(enhanced, 5, 20, 20)
    
    return enhanced


class FingerprintEnhancer:
    """Combines multiple enhancement techniques"""
    
    def __init__(self, visualizer: VisualizationManager = None):
        self.segmentor = ImageSegmentor()
        self.stft_analyzer = STFTAnalyzer()
        self.visualizer = visualizer
    
    def enhance(self, image: np.ndarray, image_name: str = "fingerprint", 
                                    stft_window_size: int = 16, stft_overlap: int = 12,
                                    gabor_window_size: int = 31, gabor_overlap_ratio: float = 0.8) -> dict:
        """
        Enhanced fingerprint enhancement pipeline with configurable overlapping windows
        
        Args:
            image: Input fingerprint image
            image_name: Name for visualization
            stft_window_size: Window size for STFT analysis
            stft_overlap: Overlap size for STFT windows
            gabor_window_size: Window size for Gabor filtering
            gabor_overlap_ratio: Overlap ratio for Gabor filtering (0.0 to 0.95)
        """
        # Normalize image to [0, 1] first
        normalized_image = self._normalize_image(image)
        
        # Show original image
        if self.visualizer:
            self.visualizer.show_image_loaded(image, f"Loaded Image: {image_name}")
        
        # Segmentation (use original for better thresholding)
        segmented, mask = self.segmentor.segment(image)
        
        # Show segmentation result
        if self.visualizer:
            self.visualizer.show_image_segmented(image, segmented, "Segmentation Result")
        
        # Enhanced STFT analysis with configurable windows
        stft_analyzer = STFTAnalyzer(window_size=stft_window_size, overlap=stft_overlap)
        orientation, frequency, energy = stft_analyzer.analyze(normalized_image)
        
        # Show masks and texture features
        if self.visualizer:
            self.visualizer.show_image_masks(image, mask, orientation, frequency, energy,
                                        "Segmentation Mask and Texture Features")
        
        # Enhanced Gabor filtering with overlapping windows
        gabor_enhanced = apply_enhanced_gabor_filter(segmented, orientation, 
                                                gabor_window_size, gabor_overlap_ratio)
        
        # Enhanced SMQT
        final_enhanced = apply_enhanced_smqt(gabor_enhanced)
        
        if self.visualizer:
            self.visualizer.show_enhancement_pipeline(image, segmented, gabor_enhanced, final_enhanced,
                                                    "Complete Enhancement Pipeline")
            self.visualizer.show_image_enhanced(image, final_enhanced, "Original vs Final Enhanced")
        
        return {
            'enhanced': final_enhanced,
            'mask': mask,
            'orientation': orientation,
            'frequency': frequency,
            'energy': energy,
            'gabor_filtered': gabor_enhanced
        }
        

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        image = image.astype(np.float32)
        return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

    def _apply_gabor_filter(self, image: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Apply pixel-wise adaptive Gabor filtering for smooth results"""
        filtered = np.zeros_like(image, dtype=np.float32)
        
        # Pre-compute Gabor kernels for different orientations
        kernel_size = 31
        sigma = 4.0
        lambd = 8.0
        gamma = 0.5
        
        # Quantize orientations to reduce computation
        num_orientations = 16
        orientation_bins = np.linspace(-np.pi/2, np.pi/2, num_orientations, endpoint=False)
        
        # Pre-compute kernels
        kernels = {}
        for i, angle in enumerate(orientation_bins):
            kernels[i] = cv2.getGaborKernel((kernel_size, kernel_size), sigma, angle, 
                                        lambd, gamma, 0, cv2.CV_32F)
        
        # Create orientation index map
        orientation_indices = np.zeros_like(orientation, dtype=int)
        for i, angle in enumerate(orientation_bins):
            if i == 0:
                mask = orientation <= (orientation_bins[0] + orientation_bins[1]) / 2
            elif i == len(orientation_bins) - 1:
                mask = orientation > (orientation_bins[-2] + orientation_bins[-1]) / 2
            else:
                mask = ((orientation > (orientation_bins[i-1] + orientation_bins[i]) / 2) & 
                    (orientation <= (orientation_bins[i] + orientation_bins[i+1]) / 2))
            orientation_indices[mask] = i
        
        # Apply filtering with interpolation between adjacent orientations
        pad_size = kernel_size // 2
        padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, 
                                        cv2.BORDER_REFLECT)
        
        for i in range(num_orientations):
            # Find pixels that should use this orientation
            mask = orientation_indices == i
            if not np.any(mask):
                continue
            
            # Apply Gabor filter
            kernel_response = cv2.filter2D(padded_image, cv2.CV_32F, kernels[i])
            kernel_response = kernel_response[pad_size:-pad_size, pad_size:-pad_size]
            
            # Assign to output
            filtered[mask] = kernel_response[mask]
        
        # Smooth transitions between different orientation regions
        filtered = cv2.GaussianBlur(filtered, (3, 3), 0.5)
        
        return np.clip(filtered, 0, 255).astype(np.uint8)
    
    def _apply_smqt(self, image: np.ndarray) -> np.ndarray:
        """Apply Successive Mean Quantize Transform (simplified as CLAHE)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)


class MinutiaExtractor:
    """Extracts minutiae from enhanced fingerprint images"""
    
    def __init__(self, quality_threshold: float = 0.3, visualizer: VisualizationManager = None):
        self.quality_threshold = quality_threshold
        self.visualizer = visualizer
    
    def extract(self, enhanced_image: np.ndarray, mask: np.ndarray) -> List[Minutia]:
        """Extract minutiae using ridge analysis"""
        # Binarize image
        _, binary = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Thin the ridges
        skeleton = self._thin_ridges(binary)
        
        # Show binary processing
        if self.visualizer:
            self.visualizer.show_binary_image(enhanced_image, binary, skeleton, "Binary Processing Steps")
        
        minutiae = []
        h, w = skeleton.shape
        
        # Find minutiae points
        for i in range(2, h-2):
            for j in range(2, w-2):
                if skeleton[i, j] == 255 and mask[i, j] == 255:
                    # Check 8-neighborhood
                    neighbors = skeleton[i-1:i+2, j-1:j+2]
                    neighbor_count = np.sum(neighbors == 255) - 1
                    
                    if neighbor_count == 1 or neighbor_count == 3:  # Ending or bifurcation
                        angle = self._calculate_ridge_direction(skeleton, i, j)
                        quality = self._calculate_quality(enhanced_image, i, j)
                        
                        if quality > self.quality_threshold:
                            minutiae.append(Minutia(j, i, angle, quality))
        
        # Show minutiae detection results
        if self.visualizer:
            self.visualizer.show_minutiae_plots(enhanced_image, minutiae, skeleton, 
                                              "Minutiae Detection Results")
        
        return minutiae
    
    def _thin_ridges(self, binary_image: np.ndarray) -> np.ndarray:
        """Thin ridges to skeleton"""
        # Simple thinning using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skeleton = np.zeros_like(binary_image)
        temp = binary_image.copy()
        
        while True:
            eroded = cv2.erode(temp, kernel)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
            subset = eroded - opened
            skeleton = cv2.bitwise_or(skeleton, subset)
            temp = eroded.copy()
            
            if cv2.countNonZero(temp) == 0:
                break
        
        return skeleton
    
    def _calculate_ridge_direction(self, skeleton: np.ndarray, y: int, x: int) -> float:
        """Calculate ridge direction at given point"""
        region = skeleton[y-2:y+3, x-2:x+3]
        if region.shape != (5, 5):
            return 0.0
        
        # Gradient-based direction
        gy = np.sum(region * np.array([[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1],
                                      [0, 0, 0, 0, 0],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1]]))
        gx = np.sum(region * np.array([[-1, -1, 0, 1, 1],
                                      [-1, -1, 0, 1, 1],
                                      [-1, -1, 0, 1, 1],
                                      [-1, -1, 0, 1, 1],
                                      [-1, -1, 0, 1, 1]]))
        
        return np.arctan2(gy, gx)
    
    def _calculate_quality(self, image: np.ndarray, y: int, x: int) -> float:
        """Calculate minutia quality"""
        region = image[max(0, y-8):y+9, max(0, x-8):x+9]
        if region.size == 0:
            return 0.0
        return np.std(region.astype(np.float32)) / 255.0


class CylinderCodeExtractor:
    """Extracts MCC/MTCC cylinder codes from minutiae"""
    
    def __init__(self, radius: int = 65, ns: int = 18, nd: int = 5):
        self.radius = radius
        self.ns = ns
        self.nd = nd
        self.delta_s = 2 * radius / ns
        self.delta_d = 2 * np.pi / nd
        self.sigma_s = 6.0
        self.sigma_d = 5.0 / 36.0 * np.pi
    
    def extract_mcc(self, minutiae: List[Minutia]) -> List[MTCCDescriptor]:
        """Extract traditional MCC descriptors"""
        descriptors = []
        for minutia in minutiae:
            cylinder = self._create_cylinder(minutia, minutiae, 'angle')
            descriptors.append(MTCCDescriptor(minutia, cylinder))
        return descriptors
    
    def extract_mtcc(self, minutiae: List[Minutia], texture_images: Dict[str, np.ndarray], 
                     feature_type: str) -> List[MTCCDescriptor]:
        """Extract MTCC descriptors using texture features"""
        descriptors = []
        if feature_type not in texture_images:
            return descriptors
            
        texture_image = texture_images[feature_type]
        
        for minutia in minutiae:
            cylinder = self._create_texture_cylinder(minutia, minutiae, texture_image)
            descriptors.append(MTCCDescriptor(minutia, cylinder))
        
        return descriptors
    
    def _create_cylinder(self, central_minutia: Minutia, all_minutiae: List[Minutia], 
                        mode: str) -> np.ndarray:
        """Create 3D cylinder for given minutia"""
        cylinder = np.zeros((self.ns, self.ns, self.nd))
        
        for i in range(self.ns):
            for j in range(self.ns):
                for k in range(self.nd):
                    cell_center = self._get_cell_center(central_minutia, i, j)
                    cell_angle = -np.pi + (k + 0.5) * self.delta_d
                    
                    neighbors = self._find_neighbors(cell_center, all_minutiae, central_minutia)
                    
                    contribution = 0.0
                    for neighbor in neighbors:
                        spatial_contrib = self._gaussian_spatial(cell_center, neighbor)
                        directional_contrib = self._gaussian_directional(cell_angle, neighbor.angle)
                        contribution += spatial_contrib * directional_contrib
                    
                    cylinder[i, j, k] = self._sigmoid(contribution)
        
        return cylinder
    
    def _create_texture_cylinder(self, central_minutia: Minutia, all_minutiae: List[Minutia],
                               texture_image: np.ndarray) -> np.ndarray:
        """Create texture-based cylinder"""
        cylinder = np.zeros((self.ns, self.ns, self.nd))
        h, w = texture_image.shape
        
        for i in range(self.ns):
            for j in range(self.ns):
                for k in range(self.nd):
                    cell_center = self._get_cell_center(central_minutia, i, j)
                    cell_angle = -np.pi + (k + 0.5) * self.delta_d
                    
                    # Get texture value at cell center
                    cy, cx = int(np.clip(cell_center[1], 0, h-1)), int(np.clip(cell_center[0], 0, w-1))
                    texture_value = texture_image[cy, cx]
                    
                    # Cell-centered approach: use texture value directly
                    directional_diff = self._angle_difference(cell_angle, texture_value)
                    contribution = np.exp(-directional_diff**2 / (2 * self.sigma_d**2))
                    
                    cylinder[i, j, k] = self._sigmoid(contribution)
        
        return cylinder
    
    def _get_cell_center(self, minutia: Minutia, i: int, j: int) -> np.ndarray:
        """Calculate cell center coordinates"""
        cos_theta = np.cos(minutia.angle)
        sin_theta = np.sin(minutia.angle)
        
        local_x = (i - (self.ns - 1) / 2) * self.delta_s
        local_y = (j - (self.ns - 1) / 2) * self.delta_s
        
        global_x = minutia.x + cos_theta * local_x - sin_theta * local_y
        global_y = minutia.y + sin_theta * local_x + cos_theta * local_y
        
        return np.array([global_x, global_y])
    
    def _find_neighbors(self, cell_center: np.ndarray, minutiae: List[Minutia], 
                       central_minutia: Minutia) -> List[Minutia]:
        """Find neighboring minutiae within 3*sigma_s radius"""
        neighbors = []
        threshold = 3 * self.sigma_s
        
        for minutia in minutiae:
            if minutia != central_minutia:
                distance = np.linalg.norm([minutia.x - cell_center[0], 
                                         minutia.y - cell_center[1]])
                if distance <= threshold:
                    neighbors.append(minutia)
        
        return neighbors
    
    def _gaussian_spatial(self, cell_center: np.ndarray, minutia: Minutia) -> float:
        """Calculate spatial Gaussian contribution"""
        distance = np.linalg.norm([minutia.x - cell_center[0], 
                                 minutia.y - cell_center[1]])
        return np.exp(-distance**2 / (2 * self.sigma_s**2))
    
    def _gaussian_directional(self, cell_angle: float, minutia_angle: float) -> float:
        """Calculate directional Gaussian contribution"""
        angle_diff = self._angle_difference(cell_angle, minutia_angle)
        return np.exp(-angle_diff**2 / (2 * self.sigma_d**2))
    
    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """Calculate normalized angle difference"""
        diff = angle1 - angle2
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function to normalize contributions"""
        mu = 5.0 / 1000.0
        tau = 400.0
        return 1.0 / (1.0 + np.exp(-tau * (x - mu)))


class DistanceCalculator(ABC):
    """Abstract base class for distance calculation"""
    
    @abstractmethod
    def calculate(self, desc1: MTCCDescriptor, desc2: MTCCDescriptor) -> float:
        pass


class EuclideanDistance(DistanceCalculator):
    """Euclidean distance for traditional MCC"""
    
    def calculate(self, desc1: MTCCDescriptor, desc2: MTCCDescriptor) -> float:
        if not (desc1.valid and desc2.valid):
            return 0.0
        
        diff = desc1.cylinder.flatten() - desc2.cylinder.flatten()
        norm1 = np.linalg.norm(desc1.cylinder.flatten())
        norm2 = np.linalg.norm(desc2.cylinder.flatten())
        
        if norm1 + norm2 == 0:
            return 0.0
        
        return 1.0 - np.linalg.norm(diff) / (norm1 + norm2)


class DoubleAngleDistance(DistanceCalculator):
    """Double angle distance for texture features"""
    
    def calculate(self, desc1: MTCCDescriptor, desc2: MTCCDescriptor) -> float:
        if not (desc1.valid and desc2.valid):
            return 0.0
        
        c1_flat = desc1.cylinder.flatten()
        c2_flat = desc2.cylinder.flatten()
        
        cos_diff = np.cos(2 * c1_flat) - np.cos(2 * c2_flat)
        sin_diff = np.sin(2 * c1_flat) - np.sin(2 * c2_flat)
        
        cos_norm = np.linalg.norm(cos_diff)
        sin_norm = np.linalg.norm(sin_diff)
        
        cos_denom = np.linalg.norm(np.cos(2 * c1_flat)) + np.linalg.norm(np.cos(2 * c2_flat))
        sin_denom = np.linalg.norm(np.sin(2 * c1_flat)) + np.linalg.norm(np.sin(2 * c2_flat))
        
        if cos_denom == 0 and sin_denom == 0:
            return 0.0
        
        cos_dist = 1.0 - cos_norm / (cos_denom + 1e-8)
        sin_dist = 1.0 - sin_norm / (sin_denom + 1e-8)
        
        return np.sqrt((cos_dist**2 + sin_dist**2) / 2)


class FingerprintMatcher:
    """Handles fingerprint matching using LSS with relaxation"""
    
    def __init__(self, distance_calculator: DistanceCalculator):
        self.distance_calculator = distance_calculator
    
    def match(self, descriptors1: List[MTCCDescriptor], 
              descriptors2: List[MTCCDescriptor]) -> float:
        """Match two sets of descriptors using LSS with relaxation"""
        if not descriptors1 or not descriptors2:
            return 0.0
        
        # Create local similarity matrix
        similarity_matrix = self._create_similarity_matrix(descriptors1, descriptors2)
        
        # Local similarity sort
        matched_pairs = self._local_similarity_sort(similarity_matrix)
        
        # Apply relaxation
        final_score = self._apply_relaxation(matched_pairs, descriptors1, descriptors2)
        
        return final_score
    
    def _create_similarity_matrix(self, desc1: List[MTCCDescriptor], 
                                desc2: List[MTCCDescriptor]) -> np.ndarray:
        """Create local similarity matrix"""
        matrix = np.zeros((len(desc1), len(desc2)))
        
        for i, d1 in enumerate(desc1):
            for j, d2 in enumerate(desc2):
                matrix[i, j] = self.distance_calculator.calculate(d1, d2)
        
        return matrix
    
    def _local_similarity_sort(self, matrix: np.ndarray, top_k: int = 30) -> List[Tuple[int, int, float]]:
        """Perform local similarity sort to find top matching pairs"""
        pairs = []
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                pairs.append((i, j, matrix[i, j]))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:min(top_k, len(pairs))]
    
    def _apply_relaxation(self, pairs: List[Tuple[int, int, float]], 
                         desc1: List[MTCCDescriptor], desc2: List[MTCCDescriptor]) -> float:
        """Apply relaxation to penalize incompatible pairs"""
        if not pairs:
            return 0.0
        
        total_score = 0.0
        valid_pairs = 0
        
        for i, (idx1, idx2, similarity) in enumerate(pairs[:10]):
            compatibility = 1.0
            
            for j, (other_idx1, other_idx2, _) in enumerate(pairs[:10]):
                if i != j:
                    # Distance consistency check
                    dist1 = np.linalg.norm([desc1[idx1].minutia.x - desc1[other_idx1].minutia.x,
                                          desc1[idx1].minutia.y - desc1[other_idx1].minutia.y])
                    dist2 = np.linalg.norm([desc2[idx2].minutia.x - desc2[other_idx2].minutia.x,
                                          desc2[idx2].minutia.y - desc2[other_idx2].minutia.y])
                    
                    dist_diff = abs(dist1 - dist2)
                    if dist_diff > 20:
                        compatibility *= 0.8
            
            total_score += similarity * compatibility
            valid_pairs += 1
        
        return total_score / valid_pairs if valid_pairs > 0 else 0.0


class MTCCSystem:
    """Main system orchestrating the MTCC fingerprint recognition pipeline"""
    
    def __init__(self, dataset_loader: DatasetLoader, visualizer: VisualizationManager = None):
        self.dataset_loader = dataset_loader
        self.visualizer = visualizer
        self.enhancer = FingerprintEnhancer(visualizer)
        self.minutia_extractor = MinutiaExtractor(visualizer=visualizer)
        self.cylinder_extractor = CylinderCodeExtractor()
        
        self.euclidean_calc = EuclideanDistance()
        self.double_angle_calc = DoubleAngleDistance()
    
    def process_fingerprint(self, image_path: str) -> Dict:
        """Process a single fingerprint through the complete pipeline"""
        # Load and enhance image
        image = self.dataset_loader.load_fingerprint(image_path)
        enhancement_results = self.enhancer.enhance(image, os.path.basename(image_path))
        
        # Extract minutiae
        minutiae = self.minutia_extractor.extract(
            enhancement_results['enhanced'], 
            enhancement_results['mask']
        )
        
        # Extract different types of descriptors
        mcc_descriptors = self.cylinder_extractor.extract_mcc(minutiae)
        
        mtcc_orientation = self.cylinder_extractor.extract_mtcc(
            minutiae, enhancement_results, 'orientation'
        )
        mtcc_frequency = self.cylinder_extractor.extract_mtcc(
            minutiae, enhancement_results, 'frequency'
        )
        mtcc_energy = self.cylinder_extractor.extract_mtcc(
            minutiae, enhancement_results, 'energy'
        )
        
        # Show cylinder visualization for first minutia if available
        if self.visualizer and mcc_descriptors:
            self.visualizer.show_cylinder_visualization(
                mcc_descriptors[0].minutia, mcc_descriptors[0].cylinder, 
                f"MCC Cylinder - {os.path.basename(image_path)}"
            )
        
        return {
            'minutiae': minutiae,
            'mcc': mcc_descriptors,
            'mtcc_orientation': mtcc_orientation,
            'mtcc_frequency': mtcc_frequency,
            'mtcc_energy': mtcc_energy,
            'enhancement_results': enhancement_results,
            'original_image': image
        }
    
    def match_fingerprints(self, features1: Dict, features2: Dict, 
                         show_matching: bool = False) -> Dict[str, float]:
        """Match two fingerprints using different descriptor types"""
        scores = {}
        
        # MCC matching
        mcc_matcher = FingerprintMatcher(self.euclidean_calc)
        scores['mcc'] = mcc_matcher.match(features1['mcc'], features2['mcc'])
        
        # MTCC matching
        mtcc_matcher = FingerprintMatcher(self.double_angle_calc)
        scores['mtcc_orientation'] = mtcc_matcher.match(
            features1['mtcc_orientation'], features2['mtcc_orientation']
        )
        scores['mtcc_frequency'] = mtcc_matcher.match(
            features1['mtcc_frequency'], features2['mtcc_frequency']
        )
        scores['mtcc_energy'] = mtcc_matcher.match(
            features1['mtcc_energy'], features2['mtcc_energy']
        )
        
        # Show matching visualization
        if show_matching and self.visualizer:
            # Get top matches for visualization
            similarity_matrix = mcc_matcher._create_similarity_matrix(features1['mcc'], features2['mcc'])
            top_matches = mcc_matcher._local_similarity_sort(similarity_matrix, top_k=10)
            
            self.visualizer.show_matching_visualization(
                features1['original_image'], features1['minutiae'],
                features2['original_image'], features2['minutiae'],
                top_matches, "Fingerprint Matching Results"
            )
        
        return scores
    
    def evaluate_on_dataset(self, max_pairs: Optional[int] = None) -> Dict:
        """Evaluate the system on the loaded dataset"""
        pairs = self.dataset_loader.get_image_pairs()
        if max_pairs:
            pairs = pairs[:max_pairs]
        
        results = []
        
        for i, (img1_path, img2_path) in enumerate(pairs):
            print(f"Processing pair {i+1}/{len(pairs)}: {img1_path} vs {img2_path}")
            
            try:
                features1 = self.process_fingerprint(img1_path)
                features2 = self.process_fingerprint(img2_path)
                
                scores = self.match_fingerprints(features1, features2)
                
                # Determine if genuine match
                finger1 = img1_path.split('_')[0]
                finger2 = img2_path.split('_')[0]
                is_genuine = finger1 == finger2
                
                results.append({
                    'pair': (img1_path, img2_path),
                    'is_genuine': is_genuine,
                    'scores': scores
                })
                
            except Exception as e:
                print(f"Error processing pair {img1_path}, {img2_path}: {e}")
                continue
        
        return {'results': results, 'summary': self._calculate_metrics(results)}
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate metrics from results"""
        genuine_scores = {'mcc': [], 'mtcc_orientation': [], 'mtcc_frequency': [], 'mtcc_energy': []}
        impostor_scores = {'mcc': [], 'mtcc_orientation': [], 'mtcc_frequency': [], 'mtcc_energy': []}
        
        for result in results:
            scores = result['scores']
            if result['is_genuine']:
                for key, score in scores.items():
                    genuine_scores[key].append(score)
            else:
                for key, score in scores.items():
                    impostor_scores[key].append(score)
        
        metrics = {}
        for feature_type in genuine_scores.keys():
            if genuine_scores[feature_type] and impostor_scores[feature_type]:
                metrics[feature_type] = {
                    'genuine_mean': np.mean(genuine_scores[feature_type]),
                    'genuine_std': np.std(genuine_scores[feature_type]),
                    'impostor_mean': np.mean(impostor_scores[feature_type]),
                    'impostor_std': np.std(impostor_scores[feature_type]),
                    'separation': np.mean(genuine_scores[feature_type]) - np.mean(impostor_scores[feature_type])
                }
        
        return metrics


class MTCCSystemFactory:
    """Factory for creating MTCC system instances"""
    
    @staticmethod
    def create_fvc_system(dataset_path: str, visualization_config: Dict[str, bool] = None) -> MTCCSystem:
        """Create MTCC system for FVC datasets with optional visualization"""
        loader = FVCDatasetLoader(dataset_path)
        visualizer = VisualizationManager(visualization_config) if visualization_config else None
        return MTCCSystem(loader, visualizer)
    
    @staticmethod
    def create_custom_system(dataset_loader: DatasetLoader, 
                           visualization_config: Dict[str, bool] = None) -> MTCCSystem:
        """Create MTCC system with custom dataset loader and visualization"""
        visualizer = VisualizationManager(visualization_config) if visualization_config else None
        return MTCCSystem(dataset_loader, visualizer)


def demo_usage():
    """Demonstrate how to use the MTCC system with visualizations"""
    
    # Configuration for visualizations
    viz_config = {
        'image_loaded': True,
        'image_enhanced': True,
        'image_segmented': True,
        'image_masks': True,
        'binary_image': True,
        'minutiae_plots': True
    }
    
    # Create system for FVC dataset with visualization
    dataset_path = R"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002\FVC2002\Db1_a"
    system = MTCCSystemFactory.create_fvc_system(dataset_path, viz_config)
    
    try:
        # Process a single fingerprint with full visualization
        print("Processing fingerprint with visualizations...")
        features = system.process_fingerprint("1_1.tif")
        print(f"Extracted {len(features['minutiae'])} minutiae")
        print(f"Generated {len(features['mcc'])} MCC descriptors")
        print(f"Generated {len(features['mtcc_orientation'])} MTCC orientation descriptors")
        
        # Match two fingerprints with matching visualization
        print("\nMatching two fingerprints...")
        features1 = system.process_fingerprint("1_1.tif")
        features2 = system.process_fingerprint("1_2.tif")
        scores = system.match_fingerprints(features1, features2, show_matching=True)
        
        print("Matching scores:")
        for descriptor_type, score in scores.items():
            print(f"  {descriptor_type}: {score:.4f}")
        
        # Evaluate on dataset (small subset for demo)
        print("\nEvaluating on dataset...")
        results = system.evaluate_on_dataset(max_pairs=3)
        print("\nEvaluation results:")
        for feature_type, metrics in results['summary'].items():
            print(f"{feature_type}:")
            print(f"  Genuine mean: {metrics['genuine_mean']:.4f}")
            print(f"  Impostor mean: {metrics['impostor_mean']:.4f}")
            print(f"  Separation: {metrics['separation']:.4f}")
            
    except Exception as e:
        print(f"Error in demo: {e}")
        print("Make sure to set the correct dataset path and have valid fingerprint images")


def demo_selective_visualization():
    """Demonstrate selective visualization control"""
    
    print("Demo 1: Only show image loading and minutiae detection")
    viz_config_minimal = {
        'image_loaded': True,
        'image_enhanced': False,
        'image_segmented': False,
        'image_masks': False,
        'binary_image': False,
        'minutiae_plots': True
    }
    
    system = MTCCSystemFactory.create_fvc_system("/path/to/dataset", viz_config_minimal)
    
    print("Demo 2: Only show enhancement pipeline")
    viz_config_enhancement = {
        'image_loaded': False,
        'image_enhanced': True,
        'image_segmented': True,
        'image_masks': True,
        'binary_image': True,
        'minutiae_plots': False
    }
    
    system = MTCCSystemFactory.create_fvc_system("/path/to/dataset", viz_config_enhancement)
    
    print("Demo 3: Show everything")
    viz_config_all = {
        'image_loaded': True,
        'image_enhanced': True,
        'image_segmented': True,
        'image_masks': True,
        'binary_image': True,
        'minutiae_plots': True
    }
    
    system = MTCCSystemFactory.create_fvc_system("/path/to/dataset", viz_config_all)


def main():
    """Main function for testing the system with visualizations"""
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        dataset_path = R"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002\FVC2002\Db1_a"
        print(f"Using dataset path: {dataset_path}")
        
        # Configure visualizations based on command line arguments
        show_viz = len(sys.argv) > 2 and sys.argv[2].lower() == 'viz'
        
        if show_viz:
            print("Running with full visualizations...")
            viz_config = {
                'image_loaded': True,
                'image_enhanced': True,
                'image_segmented': True,
                'image_masks': True,
                'binary_image': True,
                'minutiae_plots': True
            }
        else:
            print("Running without visualizations...")
            viz_config = None
        
        # Create system
        dataset_path = R"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002\FVC2002\Db1_a"
        system = MTCCSystemFactory.create_fvc_system(dataset_path, viz_config)
        
        # Run evaluation
        print("Starting evaluation...")
        results = system.evaluate_on_dataset(max_pairs=5 if show_viz else 20)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        for feature_type, metrics in results['summary'].items():
            print(f"\n{feature_type.upper()}:")
            print(f"  Genuine scores: {metrics['genuine_mean']:.4f} ± {metrics['genuine_std']:.4f}")
            print(f"  Impostor scores: {metrics['impostor_mean']:.4f} ± {metrics['impostor_std']:.4f}")
            print(f"  Separation: {metrics['separation']:.4f}")
        
        # Save results
        import json
        with open('mtcc_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to mtcc_results.json")
        
    else:
        print("Usage: python mtcc_system.py <dataset_path> [viz]")
        print("  dataset_path: Path to FVC dataset")
        print("  viz: Optional flag to enable visualizations")
        print("\nExample:")
        print("  python mtcc_system.py /path/to/fvc2002/DB1_A")
        print("  python mtcc_system.py /path/to/fvc2002/DB1_A viz")
        print("\nRunning demo with placeholder path...")
        demo_usage()


if __name__ == "__main__":
    main()