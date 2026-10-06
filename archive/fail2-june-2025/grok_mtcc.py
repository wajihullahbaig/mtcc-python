import numpy as np
import cv2
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from pathlib import Path
import os
from typing import Tuple, List, Optional
import uuid

# Constants
R = 63  # Cylinder radius
NS = 18  # Spatial divisions
ND = 5   # Directional divisions
SIGMA_S = 6
SIGMA_D = np.pi / 6
WINDOW_SIZE = 14
OVERLAP = 6
DESIRED_MEAN = 100
DESIRED_VAR = 100

def morphological_thinning(img: np.ndarray) -> np.ndarray:
    """Perform morphological thinning to obtain skeleton of binary image."""
    img = img.copy()
    if img.max() > 1:
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    skeleton = np.zeros(img.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skeleton

class FingerprintProcessor:
    def __init__(self, visualize: bool = False):
        self.visualize = visualize
        self.fig = None
        if self.visualize:
            self.fig = plt.figure(figsize=(15, 10))

    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to desired mean and variance."""
        mean = np.mean(img)
        var = np.var(img)
        if var == 0:
            var = 1
        norm_img = DESIRED_MEAN + np.sqrt(DESIRED_VAR * (img - mean)**2 / var)
        norm_img = np.clip(norm_img, 0, 255).astype(np.uint8)
        if self.visualize:
            plt.subplot(231)
            plt.imshow(norm_img, cmap='gray')
            plt.title('Normalized Image')
        return norm_img

    def compute_orientation_field(self, img: np.ndarray, block_size: int = 16) -> np.ndarray:
        """Compute local orientation field using gradient-based method."""
        h, w = img.shape
        orientation = np.zeros((h, w))
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = img[i:i+block_size, j:j+block_size]
                # Compute gradients
                gx = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)
                # Compute local orientation
                v = np.sum(gx * gy)
                h = np.sum(gx * gx - gy * gy)
                theta = 0.5 * np.arctan2(v, h)
                orientation[i:i+block_size, j:j+block_size] = theta
        return orientation

    def gabor_filter_enhance(self, img: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Enhance fingerprint ridges using a bank of Gabor filters."""
        h, w = img.shape
        enhanced = np.zeros((h, w), dtype=np.uint8)
        kernel_size = 31
        num_orientations = 8  # Number of orientations in the filter bank
        orientations = np.linspace(0, np.pi, num_orientations, endpoint=False)

        # Apply Gabor filters for each orientation
        for theta in orientations:
            gabor_kernel = cv2.getGaborKernel(
                (kernel_size, kernel_size), sigma=6.0, theta=theta, 
                lambd=8.0, gamma=0.5, psi=0, ktype=cv2.CV_32F
            )
            gabor_kernel /= np.sum(gabor_kernel)  # Normalize kernel
            filtered = cv2.filter2D(img, cv2.CV_8U, gabor_kernel)
            # Combine with maximum response to enhance ridges
            enhanced = np.maximum(enhanced, filtered)

        # Normalize the enhanced image
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if self.visualize:
            plt.subplot(232)
            plt.imshow(enhanced, cmap='gray')
            plt.title('Gabor Enhanced')
        return enhanced

    def segment_image(self, img: np.ndarray, block_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """Segment fingerprint using block-wise variance."""
        h, w = img.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = img[i:i+block_size, j:j+block_size]
                if np.var(block) > 100:  # Variance threshold
                    mask[i:i+block_size, j:j+block_size] = 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        segmented = cv2.bitwise_and(img, img, mask=mask)
        if self.visualize:
            plt.subplot(233)
            plt.imshow(mask, cmap='gray')
            plt.title('Segmentation Mask')
        return segmented, mask

    def extract_minutiae(self, img: np.ndarray) -> List[Tuple[int, int, float]]:
        """Extract minutiae after binarizing and thinning to single-pixel width."""
        # Binarize using Otsu's method
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Thin to single-pixel width
        skeleton = morphological_thinning(binary)
        # Detect corners as minutiae
        corners = cv2.goodFeaturesToTrack(skeleton, maxCorners=100, qualityLevel=0.01, minDistance=10)
        minutiae = []
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                # Simulate angle (simplified, replace with actual computation if needed)
                angle = np.random.uniform(0, 2*np.pi)
                minutiae.append((int(x), int(y), angle))
        if self.visualize:
            vis_img = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
            for x, y, _ in minutiae:
                cv2.circle(vis_img, (x, y), 3, (0, 255, 0), -1)
            plt.subplot(234)
            plt.imshow(vis_img, cmap='gray')
            plt.title('Minutiae')
        return minutiae

    def compute_stft_features(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute orientation, frequency, and energy using STFT."""
        h, w = img.shape
        orientation = np.zeros((h, w))
        frequency = np.zeros((h, w))
        energy = np.zeros((h, w))
        
        window = signal.windows.hann(WINDOW_SIZE)
        window_2d = np.outer(window, window)
        
        for i in range(0, h - WINDOW_SIZE + 1, WINDOW_SIZE - OVERLAP):
            for j in range(0, w - WINDOW_SIZE + 1, WINDOW_SIZE - OVERLAP):
                block = img[i:i+WINDOW_SIZE, j:j+WINDOW_SIZE] * window_2d
                f = fftshift(fft2(block))
                mag = np.abs(f)
                angle = np.angle(f)
                
                # Dominant frequency
                max_idx = np.unravel_index(np.argmax(mag), mag.shape)
                freq_val = np.sqrt((max_idx[0] - WINDOW_SIZE//2)**2 + (max_idx[1] - WINDOW_SIZE//2)**2)
                freq_val = freq_val / WINDOW_SIZE if freq_val != 0 else 0.1
                
                # Orientation
                ori_val = angle[max_idx] if np.max(mag) > 0 else 0
                
                # Energy
                energy_val = np.log(np.sum(mag) + 1)
                
                orientation[i:i+WINDOW_SIZE, j:j+WINDOW_SIZE] = ori_val
                frequency[i:i+WINDOW_SIZE, j:j+WINDOW_SIZE] = freq_val
                energy[i:i+WINDOW_SIZE, j:j+WINDOW_SIZE] = energy_val
        
        if self.visualize:
            plt.subplot(235)
            plt.imshow(orientation, cmap='gray')
            plt.title('Orientation Map')
            plt.subplot(236)
            plt.imshow(frequency, cmap='gray')
            plt.title('Frequency Map')
        
        return orientation, frequency, energy

    def compute_mtcc_features(self, minutiae: List[Tuple[int, int, float]], 
                            orientation: np.ndarray, frequency: np.ndarray, 
                            energy: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
        """Compute Minutia Texture Cylinder Codes."""
        cylinders = []
        delta_s = 2 * R / NS
        delta_d = 2 * np.pi / ND
        
        for m in minutiae:
            x_m, y_m, theta_m = m
            cylinder = np.zeros((NS, NS, ND))
            
            for i in range(NS):
                for j in range(NS):
                    # Cell center
                    p_ij = np.array([
                        x_m + delta_s * (i - (NS+1)/2) * np.cos(theta_m) + delta_s * (j - (NS+1)/2) * np.sin(theta_m),
                        y_m - delta_s * (i - (NS+1)/2) * np.sin(theta_m) + delta_s * (j - (NS+1)/2) * np.cos(theta_m)
                    ])
                    
                    if not (0 <= p_ij[0] < mask.shape[1] and 0 <= p_ij[1] < mask.shape[0] and mask[int(p_ij[1]), int(p_ij[0])]):
                        continue
                        
                    for k in range(ND):
                        d_phi_k = -np.pi + (k - 0.5) * delta_d
                        
                        # Cell-centered orientation feature (MCC_co)
                        if 0 <= p_ij[1] < orientation.shape[0] and 0 <= p_ij[0] < orientation.shape[1]:
                            ori_val = orientation[int(p_ij[1]), int(p_ij[0])]
                            d_phi = self.normalize_angle(ori_val - d_phi_k)
                            contrib = np.exp(-d_phi**2 / (2 * SIGMA_D**2))
                            
                            # Spatial contribution
                            spatial_dist = np.linalg.norm(p_ij - np.array([x_m, y_m]))
                            spatial_contrib = np.exp(-spatial_dist**2 / (2 * SIGMA_S**2))
                            
                            cylinder[i, j, k] = contrib * spatial_contrib
            
            cylinders.append(cylinder)
        
        return cylinders

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle < -np.pi:
            angle += 2 * np.pi
        while angle >= np.pi:
            angle -= 2 * np.pi
        return angle

    def match_cylinders(self, cylinders1: List[np.ndarray], cylinders2: List[np.ndarray]) -> float:
        """Match two sets of cylinders using LSSR."""
        lsm = np.zeros((len(cylinders1), len(cylinders2)))
        
        for i, c1 in enumerate(cylinders1):
            for j, c2 in enumerate(cylinders2):
                valid_cells = (c1 != 0) & (c2 != 0)
                if np.sum(valid_cells) < 10:  # Minimum valid cells
                    continue
                    
                # Double angle distance for texture features
                cos_diff = np.abs(np.cos(2 * c1[valid_cells]) - np.cos(2 * c2[valid_cells]))
                sin_diff = np.abs(np.sin(2 * c1[valid_cells]) - np.sin(2 * c2[valid_cells]))
                dist = np.sqrt(cos_diff**2 + sin_diff**2) / 2
                lsm[i, j] = 1 - np.mean(dist) if len(dist) > 0 else 0
        
        # Local Similarity Sort with Relaxation
        top_n = 5
        sorted_indices = np.argsort(lsm.ravel())[::-1][:top_n]
        scores = lsm.ravel()[sorted_indices]
        
        # Simplified relaxation
        final_score = np.mean(scores) if len(scores) > 0 else 0
        
        return final_score

def load_fvc_image(fvc_path: str, finger_id: int, impression: int) -> np.ndarray:
    """Load image from FVC dataset."""
    filename = f"{finger_id}_{impression}.tif"
    path = Path(fvc_path) / filename
    if not path.exists():
        raise FileNotFoundError(f"Image {filename} not found in {fvc_path}")
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

def test_image_match(fvc_path: str, finger1: int, imp1: int, finger2: int, imp2: int, 
                    visualize: bool = False) -> float:
    """Test matching between two fingerprint images."""
    processor = FingerprintProcessor(visualize=visualize)
    
    # Load images
    img1 = load_fvc_image(fvc_path, finger1, imp1)
    img2 = load_fvc_image(fvc_path, finger2, imp2)
    
    # Process first image
    norm1 = processor.normalize_image(img1)
    ori1 = processor.compute_orientation_field(norm1)
    enhanced1 = processor.gabor_filter_enhance(norm1, ori1)
    seg1, mask1 = processor.segment_image(enhanced1)
    _, freq1, energy1 = processor.compute_stft_features(enhanced1)
    minutiae1 = processor.extract_minutiae(enhanced1)
    cylinders1 = processor.compute_mtcc_features(minutiae1, ori1, freq1, energy1, mask1)
    
    # Process second image
    norm2 = processor.normalize_image(img2)
    ori2 = processor.compute_orientation_field(norm2)
    enhanced2 = processor.gabor_filter_enhance(norm2, ori2)
    seg2, mask2 = processor.segment_image(enhanced2)
    _, freq2, energy2 = processor.compute_stft_features(enhanced2)
    minutiae2 = processor.extract_minutiae(enhanced2)
    cylinders2 = processor.compute_mtcc_features(minutiae2, ori2, freq2, energy2, mask2)
    
    # Match
    score = processor.match_cylinders(cylinders1, cylinders2)
    
    if visualize:
        plt.show()
    
    return score

if __name__ == "__main__":
    # Example usage
    fvc_path = r"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002\FVC2002\Db1_a"
    score = test_image_match(fvc_path, finger1=1, imp1=1, finger2=1, imp2=2, visualize=True)
    print(f"Matching score: {score}")