#!/usr/bin/env python3
"""
MTCC (Minutia Texture Cylinder Codes) Fingerprint Recognition System
Implementation based on:
- [11] Baig et al., "Minutia Texture Cylinder Codes for fingerprint matching", 2018
- [8] Gottschlich, "Curved Gabor Filters for Fingerprint Image Enhancement", 2014
- [9] Shimna & Neethu, "Fingerprint Image Enhancement Using STFT Analysis", 2015
- [10] Bazen & Gerez, "Segmentation of Fingerprint Images", 2001

Dependencies: OpenCV, NumPy, SciPy only
"""

import cv2
import numpy as np
import scipy.ndimage
from scipy import signal
from scipy.ndimage import uniform_filter, generic_filter
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Global parameters
BLOCK_SIZE = 16
GABOR_SIGMA_X = 4.0
GABOR_SIGMA_Y = 4.0
CYLINDER_RADIUS = 70
CYLINDER_HEIGHT = 16
ANGULAR_BINS = 16
RADIAL_BINS = 6


def load_image(path: str) -> np.ndarray:
    """Load fingerprint image and convert to grayscale float32."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    return img.astype(np.float32)


def normalize(img: np.ndarray) -> np.ndarray:
    """Zero-mean, unit variance normalization as per [11]."""
    # Robust normalization to handle varying contrast
    mean_val = np.mean(img)
    std_val = np.std(img)
    if std_val < 1e-6:  # Avoid division by zero
        return np.zeros_like(img)
    
    normalized = (img - mean_val) / std_val
    # Scale to 0-255 range for compatibility
    normalized = ((normalized - normalized.min()) / 
                 (normalized.max() - normalized.min()) * 255)
    return normalized.astype(np.float32)


def segment(img: np.ndarray, block_size: int = 16) -> np.ndarray:
    """
    Block-wise segmentation using variance and coherence as per [10].
    Returns binary mask where 1 = foreground (fingerprint), 0 = background.
    """
    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate block-wise statistics
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = img[i:i+block_size, j:j+block_size]
            
            # Variance-based segmentation
            variance = np.var(block)
            
            # Coherence calculation using gradients
            gx = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)
            
            # Coherence measure
            gxx = gx * gx
            gyy = gy * gy
            gxy = gx * gy
            
            coherence = ((gxx - gyy)**2 + 4*gxy**2) / ((gxx + gyy)**2 + 1e-6)
            coherence = np.mean(coherence)
            
            # Threshold based on variance and coherence
            if variance > 100 and coherence > 0.3:  # Empirically determined
                mask[i:i+block_size, j:j+block_size] = 1
    
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def estimate_orientation_frequency(img: np.ndarray, block_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate local ridge orientation and frequency maps.
    Based on gradient-based orientation estimation and spectral analysis.
    """
    h, w = img.shape
    orientation_map = np.zeros((h, w), dtype=np.float32)
    frequency_map = np.zeros((h, w), dtype=np.float32)
    
    # Gradient calculation
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    
    for i in range(block_size//2, h - block_size//2, block_size//2):
        for j in range(block_size//2, w - block_size//2, block_size//2):
            # Extract local block
            block_gx = gx[i-block_size//2:i+block_size//2, 
                         j-block_size//2:j+block_size//2]
            block_gy = gy[i-block_size//2:i+block_size//2, 
                         j-block_size//2:j+block_size//2]
            
            # Orientation estimation using least squares
            gxx = np.sum(block_gx * block_gx)
            gyy = np.sum(block_gy * block_gy)
            gxy = np.sum(block_gx * block_gy)
            
            # Avoid division by zero
            if gxx + gyy < 1e-6:
                orientation = 0
            else:
                orientation = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
            
            # Frequency estimation using spectral analysis
            block_img = img[i-block_size//2:i+block_size//2, 
                           j-block_size//2:j+block_size//2]
            
            # Project along orientation
            cos_o = np.cos(orientation)
            sin_o = np.sin(orientation)
            
            # Create projection profile
            profile = []
            for k in range(block_size):
                x_proj = int(k * cos_o)
                y_proj = int(k * sin_o)
                if 0 <= x_proj < block_size and 0 <= y_proj < block_size:
                    profile.append(np.mean(block_img[max(0, y_proj-1):min(block_size, y_proj+2),
                                                    max(0, x_proj-1):min(block_size, x_proj+2)]))
            
            if len(profile) > 4:
                # Find dominant frequency using FFT
                fft_profile = np.fft.fft(profile)
                freqs = np.fft.fftfreq(len(profile))
                dominant_freq_idx = np.argmax(np.abs(fft_profile[1:len(profile)//2])) + 1
                frequency = abs(freqs[dominant_freq_idx]) * len(profile)
                frequency = np.clip(frequency, 0.05, 0.25)  # Typical ridge frequency range
            else:
                frequency = 0.1  # Default frequency
            
            # Fill the maps
            orientation_map[i-block_size//4:i+block_size//4, 
                          j-block_size//4:j+block_size//4] = orientation
            frequency_map[i-block_size//4:i+block_size//4, 
                         j-block_size//4:j+block_size//4] = frequency
    
    # Smooth the maps
    orientation_map = scipy.ndimage.gaussian_filter(orientation_map, sigma=2.0)
    frequency_map = scipy.ndimage.gaussian_filter(frequency_map, sigma=2.0)
    
    return orientation_map, frequency_map


def gabor_enhance(img: np.ndarray, orientation_map: np.ndarray, freq_map: np.ndarray) -> np.ndarray:
    """
    Context-adaptive Gabor filtering as per [8][9].
    Uses curved/block-adapted Gabor filters based on local orientation and frequency.
    """
    h, w = img.shape
    enhanced = np.zeros_like(img)
    
    # Create Gabor filter bank
    filter_size = 31  # Should be odd
    
    for i in range(filter_size//2, h - filter_size//2):
        for j in range(filter_size//2, w - filter_size//2):
            # Local orientation and frequency
            local_orientation = orientation_map[i, j]
            local_frequency = freq_map[i, j]
            
            # Skip if frequency is too low
            if local_frequency < 0.01:
                enhanced[i, j] = img[i, j]
                continue
            
            # Create local Gabor filter
            gabor_filter = create_gabor_filter(filter_size, local_orientation, 
                                             local_frequency, GABOR_SIGMA_X, GABOR_SIGMA_Y)
            
            # Extract local region
            region = img[i-filter_size//2:i+filter_size//2+1, 
                        j-filter_size//2:j+filter_size//2+1]
            
            if region.shape == gabor_filter.shape:
                # Apply filter
                response = np.sum(region * gabor_filter)
                enhanced[i, j] = response
            else:
                enhanced[i, j] = img[i, j]
    
    # Normalize enhanced image
    enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-6) * 255
    return enhanced.astype(np.float32)


def create_gabor_filter(size: int, orientation: float, frequency: float, 
                       sigma_x: float, sigma_y: float) -> np.ndarray:
    """Create a Gabor filter with given parameters."""
    center = size // 2
    gabor = np.zeros((size, size), dtype=np.float32)
    
    cos_theta = np.cos(orientation)
    sin_theta = np.sin(orientation)
    
    for i in range(size):
        for j in range(size):
            x = (i - center) * cos_theta + (j - center) * sin_theta
            y = -(i - center) * sin_theta + (j - center) * cos_theta
            
            # Gabor function
            gaussian = np.exp(-(x*x/(2*sigma_x*sigma_x) + y*y/(2*sigma_y*sigma_y)))
            sinusoid = np.cos(2 * np.pi * frequency * x)
            gabor[i, j] = gaussian * sinusoid
    
    # Normalize
    gabor = gabor - np.mean(gabor)
    norm = np.sqrt(np.sum(gabor * gabor))
    if norm > 1e-6:
        gabor = gabor / norm
    
    return gabor


def smqt(img: np.ndarray, levels: int = 8) -> np.ndarray:
    """
    Successive Mean Quantization Transform as per [11].
    Enhances low-contrast ridges before STFT analysis.
    """
    enhanced = img.copy()
    
    for level in range(levels):
        # Calculate local mean using sliding window
        kernel_size = 3 + 2 * level  # Increasing window size
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(enhanced, -1, kernel)
        
        # Quantization step
        diff = enhanced - local_mean
        threshold = np.std(diff) / (2 ** level)
        
        # Apply quantization
        enhanced = np.where(diff > threshold, local_mean + threshold,
                           np.where(diff < -threshold, local_mean - threshold, enhanced))
    
    return enhanced


def stft_features(img: np.ndarray, window: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 3-channel STFT-based texture features as per [11][9].
    Returns orientation, frequency, and energy maps.
    """
    h, w = img.shape
    orientation_texture = np.zeros((h, w), dtype=np.float32)
    frequency_texture = np.zeros((h, w), dtype=np.float32)
    energy_texture = np.zeros((h, w), dtype=np.float32)
    
    # STFT parameters
    overlap = window // 2
    
    for i in range(window//2, h - window//2, overlap):
        for j in range(window//2, w - window//2, overlap):
            # Extract local window
            local_window = img[i-window//2:i+window//2, j-window//2:j+window//2]
            
            if local_window.shape[0] != window or local_window.shape[1] != window:
                continue
            
            # Apply window function
            hann_window = np.outer(np.hanning(window), np.hanning(window))
            windowed = local_window * hann_window
            
            # 2D FFT
            fft_result = np.fft.fft2(windowed)
            fft_shifted = np.fft.fftshift(fft_result)
            magnitude = np.abs(fft_shifted)
            phase = np.angle(fft_shifted)
            
            # Extract features
            # 1. Dominant orientation from spectral analysis
            center = window // 2
            y_coords, x_coords = np.ogrid[:window, :window]
            y_coords = y_coords - center
            x_coords = x_coords - center
            
            # Weighted orientation calculation
            angles = np.arctan2(y_coords, x_coords)
            weights = magnitude * magnitude  # Energy weighting
            
            # Circular mean for orientation
            cos_sum = np.sum(weights * np.cos(2 * angles))
            sin_sum = np.sum(weights * np.sin(2 * angles))
            dominant_orientation = 0.5 * np.arctan2(sin_sum, cos_sum)
            
            # 2. Dominant frequency
            distances = np.sqrt(x_coords*x_coords + y_coords*y_coords)
            freq_weights = magnitude * distances
            total_weight = np.sum(magnitude)
            if total_weight > 1e-6:
                dominant_frequency = np.sum(freq_weights) / (total_weight * window)
            else:
                dominant_frequency = 0.1
            
            # 3. Energy (spectral energy)
            spectral_energy = np.sum(magnitude * magnitude) / (window * window)
            
            # Fill texture maps
            orientation_texture[i-overlap//2:i+overlap//2, 
                              j-overlap//2:j+overlap//2] = dominant_orientation
            frequency_texture[i-overlap//2:i+overlap//2, 
                             j-overlap//2:j+overlap//2] = dominant_frequency
            energy_texture[i-overlap//2:i+overlap//2, 
                          j-overlap//2:j+overlap//2] = spectral_energy
    
    # Normalize texture maps
    orientation_texture = (orientation_texture + np.pi) / (2 * np.pi) * 255
    frequency_texture = (frequency_texture / np.max(frequency_texture + 1e-6)) * 255
    energy_texture = (energy_texture / np.max(energy_texture + 1e-6)) * 255
    
    return orientation_texture, frequency_texture, energy_texture


def binarize_thin(img: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding followed by Zhang-Suen thinning.
    """
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Invert so ridges are white
    binary = 255 - binary
    
    # Zhang-Suen thinning
    skeleton = zhang_suen_thinning(binary)
    
    return skeleton


def zhang_suen_thinning(img: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen thinning algorithm implementation.
    """
    img = img.copy() // 255  # Convert to 0/1
    changing = True
    
    while changing:
        changing = False
        
        # Step 1
        to_delete = []
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                if img[i, j] == 1:
                    # Get 8-neighbors
                    p = [img[i-1, j], img[i-1, j+1], img[i, j+1], img[i+1, j+1],
                         img[i+1, j], img[i+1, j-1], img[i, j-1], img[i-1, j-1]]
                    
                    # Conditions for Zhang-Suen
                    transitions = sum([1 for k in range(8) if p[k] == 0 and p[(k+1)%8] == 1])
                    neighbors = sum(p)
                    
                    if (2 <= neighbors <= 6 and transitions == 1 and 
                        p[0] * p[2] * p[4] == 0 and p[2] * p[4] * p[6] == 0):
                        to_delete.append((i, j))
        
        for i, j in to_delete:
            img[i, j] = 0
            changing = True
        
        # Step 2
        to_delete = []
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                if img[i, j] == 1:
                    # Get 8-neighbors
                    p = [img[i-1, j], img[i-1, j+1], img[i, j+1], img[i+1, j+1],
                         img[i+1, j], img[i+1, j-1], img[i, j-1], img[i-1, j-1]]
                    
                    # Conditions for Zhang-Suen
                    transitions = sum([1 for k in range(8) if p[k] == 0 and p[(k+1)%8] == 1])
                    neighbors = sum(p)
                    
                    if (2 <= neighbors <= 6 and transitions == 1 and 
                        p[0] * p[2] * p[6] == 0 and p[0] * p[4] * p[6] == 0):
                        to_delete.append((i, j))
        
        for i, j in to_delete:
            img[i, j] = 0
            changing = True
    
    return (img * 255).astype(np.uint8)


def extract_minutiae(skeleton: np.ndarray) -> List[Tuple[int, int, float, str]]:
    """
    Extract minutiae using Crossing Number (CN) algorithm.
    Returns list of (x, y, orientation, type) tuples.
    """
    minutiae = []
    h, w = skeleton.shape
    
    # Convert to binary
    binary_skeleton = (skeleton > 128).astype(np.uint8)
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if binary_skeleton[i, j] == 1:  # Ridge pixel
                # Get 8-neighbors in clockwise order
                neighbors = [
                    binary_skeleton[i-1, j], binary_skeleton[i-1, j+1],
                    binary_skeleton[i, j+1], binary_skeleton[i+1, j+1],
                    binary_skeleton[i+1, j], binary_skeleton[i+1, j-1],
                    binary_skeleton[i, j-1], binary_skeleton[i-1, j-1]
                ]
                
                # Calculate crossing number
                cn = 0
                for k in range(8):
                    cn += abs(neighbors[k] - neighbors[(k + 1) % 8])
                cn = cn // 2
                
                # Classify minutiae
                minutia_type = None
                if cn == 1:
                    minutia_type = "termination"
                elif cn == 3:
                    minutia_type = "bifurcation"
                
                if minutia_type:
                    # Estimate orientation
                    orientation = estimate_minutia_orientation(binary_skeleton, i, j)
                    minutiae.append((j, i, orientation, minutia_type))
    
    return minutiae


def estimate_minutia_orientation(skeleton: np.ndarray, y: int, x: int) -> float:
    """Estimate minutia orientation using local ridge direction."""
    window_size = 16
    h, w = skeleton.shape
    
    # Extract local window
    y_start = max(0, y - window_size // 2)
    y_end = min(h, y + window_size // 2)
    x_start = max(0, x - window_size // 2)
    x_end = min(w, x + window_size // 2)
    
    local_window = skeleton[y_start:y_end, x_start:x_end]
    
    # Calculate gradients
    gx = cv2.Sobel(local_window.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(local_window.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    
    # Orientation using gradient
    gxx = np.sum(gx * gx)
    gyy = np.sum(gy * gy)
    gxy = np.sum(gx * gy)
    
    if gxx + gyy < 1e-6:
        return 0.0
    
    orientation = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
    return orientation


def create_cylinders(minutiae: List[Tuple[int, int, float, str]], 
                    texture_maps: Tuple[np.ndarray, np.ndarray, np.ndarray], 
                    radius: int = 70) -> List[np.ndarray]:
    """
    Create MTCC descriptors - 3D cylinders with STFT texture features as per [11].
    """
    orientation_map, frequency_map, energy_map = texture_maps
    cylinders = []
    
    for x, y, angle, mtype in minutiae:
        # Create cylinder descriptor
        cylinder = np.zeros((CYLINDER_HEIGHT, ANGULAR_BINS, RADIAL_BINS, 3), dtype=np.float32)
        
        # Sample points in cylindrical coordinates
        for h_idx in range(CYLINDER_HEIGHT):
            height_offset = (h_idx - CYLINDER_HEIGHT // 2) * 2  # Height sampling
            
            for a_idx in range(ANGULAR_BINS):
                theta = 2 * np.pi * a_idx / ANGULAR_BINS
                
                for r_idx in range(RADIAL_BINS):
                    r = (r_idx + 1) * radius / RADIAL_BINS
                    
                    # Convert to Cartesian coordinates
                    sample_x = int(x + r * np.cos(theta + angle))
                    sample_y = int(y + r * np.sin(theta + angle) + height_offset)
                    
                    # Check bounds
                    if (0 <= sample_x < orientation_map.shape[1] and 
                        0 <= sample_y < orientation_map.shape[0]):
                        
                        # Sample STFT texture features
                        cylinder[h_idx, a_idx, r_idx, 0] = orientation_map[sample_y, sample_x]
                        cylinder[h_idx, a_idx, r_idx, 1] = frequency_map[sample_y, sample_x]
                        cylinder[h_idx, a_idx, r_idx, 2] = energy_map[sample_y, sample_x]
        
        cylinders.append(cylinder)
    
    return cylinders


def match(cylinders1: List[np.ndarray], cylinders2: List[np.ndarray]) -> float:
    """
    MTCC matching using Local Similarity Sort (LSS) as per [11].
    Returns similarity score between 0 and 1.
    """
    if not cylinders1 or not cylinders2:
        return 0.0
    
    # Calculate pairwise similarities
    similarities = []
    
    for cyl1 in cylinders1:
        best_sim = 0.0
        
        for cyl2 in cylinders2:
            # Calculate similarity between cylinders
            sim = cylinder_similarity(cyl1, cyl2)
            best_sim = max(best_sim, sim)
        
        similarities.append(best_sim)
    
    # Local Similarity Sort - use top matches
    similarities.sort(reverse=True)
    top_k = min(len(similarities), max(1, len(similarities) // 3))
    
    if top_k > 0:
        return np.mean(similarities[:top_k])
    else:
        return 0.0


def cylinder_similarity(cyl1: np.ndarray, cyl2: np.ndarray) -> float:
    """Calculate similarity between two MTCC cylinders."""
    # Normalize cylinders
    cyl1_norm = (cyl1 - np.mean(cyl1)) / (np.std(cyl1) + 1e-6)
    cyl2_norm = (cyl2 - np.mean(cyl2)) / (np.std(cyl2) + 1e-6)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(cyl1_norm.flatten(), cyl2_norm.flatten())[0, 1]
    
    # Handle NaN case
    if np.isnan(correlation):
        correlation = 0.0
    
    # Convert to similarity score [0, 1]
    similarity = (correlation + 1) / 2
    return similarity


def calculate_eer(genuine_scores: List[float], impostor_scores: List[float]) -> float:
    """Calculate Equal Error Rate (EER)."""
    if not genuine_scores or not impostor_scores:
        return 1.0
    
    # Create threshold range
    all_scores = genuine_scores + impostor_scores
    thresholds = np.linspace(min(all_scores), max(all_scores), 1000)
    
    min_diff = float('inf')
    eer = 1.0
    
    for threshold in thresholds:
        # False Accept Rate (impostor scores >= threshold)
        far = sum(1 for score in impostor_scores if score >= threshold) / len(impostor_scores)
        
        # False Reject Rate (genuine scores < threshold)
        frr = sum(1 for score in genuine_scores if score < threshold) / len(genuine_scores)
        
        # Find point where FAR ≈ FRR
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2
    
    return eer


def visualize_pipeline(original: np.ndarray, normalized: np.ndarray, 
                      segmented: np.ndarray, gabor: np.ndarray, 
                      smqt_img: np.ndarray, stft_orientation: np.ndarray,
                      binarized: np.ndarray, thinned: np.ndarray, 
                      minutiae_overlay: np.ndarray) -> None:
    """Visualize all pipeline stages in a 3x3 grid as per [11]."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    images = [
        (original, "Original"),
        (normalized, "Normalized"),
        (segmented * 255, "Segmented"),
        (gabor, "Gabor Enhanced"),
        (smqt_img, "SMQT"),
        (stft_orientation, "STFT Orientation"),
        (binarized, "Binarized"),
        (thinned, "Thinned"),
        (minutiae_overlay, "Minutiae Overlay")
    ]
    
    for idx, (img, title) in enumerate(images):
        row, col = idx // 3, idx % 3
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


def process_fingerprint(image_path: str, visualize: bool = False) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Complete MTCC pipeline for a single fingerprint image.
    Returns MTCC descriptors and minutiae overlay image.
    """
    # Load and preprocess
    img = load_image(image_path)
    normalized = normalize(img)
    
    # Segmentation
    mask = segment(normalized)
    
    # Apply mask
    masked_img = normalized * (mask / 255.0)
    
    # Orientation and frequency estimation
    orientation_map, frequency_map = estimate_orientation_frequency(masked_img)
    
    # Gabor enhancement
    gabor_enhanced = gabor_enhance(masked_img, orientation_map, frequency_map)
    
    # SMQT enhancement
    smqt_enhanced = smqt(gabor_enhanced)
    
    # STFT feature extraction
    stft_orientation, stft_frequency, stft_energy = stft_features(smqt_enhanced)
    
    # Binarization and thinning
    skeleton = binarize_thin(smqt_enhanced)
    
    # Minutiae extraction
    minutiae = extract_minutiae(skeleton)
    
    # Create minutiae overlay
    minutiae_overlay = img.copy()
    for x, y, angle, mtype in minutiae:
        color = 255 if mtype == "termination" else 128
        cv2.circle(minutiae_overlay, (x, y), 3, color, -1)
        # Draw orientation line
        end_x = int(x + 10 * np.cos(angle))
        end_y = int(y + 10 * np.sin(angle))
        cv2.line(minutiae_overlay, (x, y), (end_x, end_y), color, 1)
    
    # Create MTCC descriptors
    texture_maps = (stft_orientation, stft_frequency, stft_energy)
    cylinders = create_cylinders(minutiae, texture_maps)
    
    # Visualization
    if visualize:
        visualize_pipeline(img, normalized, mask, gabor_enhanced, smqt_enhanced,
                          stft_orientation, (skeleton > 0) * 255, skeleton, minutiae_overlay)
    
    return cylinders, minutiae_overlay


def test_two_images(image_path1: str, image_path2: str, visualize: bool = True) -> float:
    """Test matching between two fingerprint images."""
    print(f"Processing {image_path1}...")
    cylinders1, overlay1 = process_fingerprint(image_path1, visualize)
    
    print(f"Processing {image_path2}...")
    cylinders2, overlay2 = process_fingerprint(image_path2, visualize)
    
    # Match
    similarity = match(cylinders1, cylinders2)
    
    print(f"Similarity score: {similarity:.4f}")
    print(f"Number of minutiae in image 1: {len(cylinders1)}")
    print(f"Number of minutiae in image 2: {len(cylinders2)}")
    
    return similarity


def test_fvc_dataset(dataset_path: str) -> float:
    """
    Test on FVC dataset and calculate EER.
    Assumes FVC dataset structure with subdirectories for each finger.
    """
    genuine_scores = []
    impostor_scores = []
    
    # Get all image files
    image_files = []
    for ext in ['*.bmp', '*.tif', '*.png', '*.jpg']:
        image_files.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
    
    if len(image_files) < 2:
        print(f"Not enough images found in {dataset_path}")
        return 1.0
    
    print(f"Found {len(image_files)} images")
    
    # Process all images
    all_descriptors = {}
    for img_path in image_files[:20]:  # Limit for demo
        try:
            descriptors, _ = process_fingerprint(img_path, visualize=False)
            all_descriptors[img_path] = descriptors
            print(f"Processed {os.path.basename(img_path)}: {len(descriptors)} minutiae")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Calculate genuine and impostor scores
    processed_files = list(all_descriptors.keys())
    
    for i in range(len(processed_files)):
        for j in range(i + 1, len(processed_files)):
            file1, file2 = processed_files[i], processed_files[j]
            
            # Determine if genuine or impostor
            # Assume same finger if filenames share common prefix
            name1 = os.path.basename(file1).split('_')[0]
            name2 = os.path.basename(file2).split('_')[0]
            
            score = match(all_descriptors[file1], all_descriptors[file2])
            
            if name1 == name2:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)
    
    print(f"Genuine scores: {len(genuine_scores)}")
    print(f"Impostor scores: {len(impostor_scores)}")
    
    if genuine_scores and impostor_scores:
        eer = calculate_eer(genuine_scores, impostor_scores)
        print(f"Equal Error Rate (EER): {eer:.4f}")
        return eer
    else:
        print("Insufficient data for EER calculation")
        return 1.0


# Example usage and testing
if __name__ == "__main__":
    # Test with sample images
    print("MTCC Fingerprint Recognition System")
    print("===================================")
    
    # Example: Test two images
    # similarity = test_two_images("finger1.bmp", "finger2.bmp", visualize=True)
    
    # Example: Test FVC dataset
    # eer = test_fvc_dataset("/path/to/fvc/dataset")
    
    print("Implementation complete. Use test_two_images() or test_fvc_dataset() to evaluate.")

