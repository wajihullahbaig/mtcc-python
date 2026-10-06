import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.spatial.distance import euclidean
from skimage.morphology import thin
import warnings
warnings.filterwarnings('ignore')

class MTCCFingerprintSystem:
    """
    Minutiae Texture Cylinder Codes (MTCC) Fingerprint Recognition System
    Based on Gabor Filter-Bank and STFT Analysis
    """
    
    def __init__(self):
        self.ridge_frequency = 0.1  # Average ridge frequency (1/K where K is inter-ridge distance)
        self.gabor_orientations = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]  # 8 orientations
        self.block_size = 16
        self.overlap = 6
        
    def load_fingerprint(self, image_path):
        """Load and convert fingerprint image to grayscale"""
        try:
            if isinstance(image_path, str):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                image = image_path
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if image is None:
                raise ValueError("Could not load image")
            return image.astype(np.float64)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def normalize_image(self, image, target_mean=100, target_variance=100):
        """Normalize image to fixed mean and variance"""
        image = image.astype(np.float64)
        current_mean = np.mean(image)
        current_var = np.var(image)
        
        if current_var == 0:
            return image
        
        normalized = target_mean + np.sqrt(target_variance) * (image - current_mean) / np.sqrt(current_var)
        return np.clip(normalized, 0, 255)
    
    def segment_fingerprint(self, image, block_size=16, threshold=0.1):
        """Segment fingerprint using variance-based method"""
        height, width = image.shape
        mask = np.zeros((height, width), dtype=bool)
        
        for i in range(0, height - block_size, block_size):
            for j in range(0, width - block_size, block_size):
                block = image[i:i+block_size, j:j+block_size]
                block_variance = np.var(block)
                
                if block_variance > threshold * 255 * 255:
                    mask[i:i+block_size, j:j+block_size] = True
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        
        return mask
    
    def gabor_filter(self, image, frequency, orientation, sigma_x=4, sigma_y=4):
        """Apply Gabor filter with specified frequency and orientation"""
        theta = np.radians(orientation)
        
        # Create Gabor kernel
        kernel_size = int(6 * max(sigma_x, sigma_y))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create coordinate grids
        x = np.arange(kernel_size) - kernel_size // 2
        y = np.arange(kernel_size) - kernel_size // 2
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        x_rot = X * np.cos(theta) + Y * np.sin(theta)
        y_rot = -X * np.sin(theta) + Y * np.cos(theta)
        
        # Create Gabor kernel
        gaussian = np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))
        sinusoid = np.cos(2 * np.pi * frequency * x_rot)
        gabor_kernel = gaussian * sinusoid
        
        # Apply filter
        filtered = cv2.filter2D(image, cv2.CV_32F, gabor_kernel)
        return filtered
    
    def gabor_filter_bank_enhancement(self, image, mask):
        """Enhance fingerprint using bank of 8 Gabor filters"""
        enhanced = np.zeros_like(image)
        
        for orientation in self.gabor_orientations:
            filtered = self.gabor_filter(image, self.ridge_frequency, orientation)
            enhanced += np.abs(filtered)
        
        enhanced = enhanced / len(self.gabor_orientations)
        enhanced[~mask] = 0
        return enhanced
    
    def smqt_normalization(self, image, iterations=3):
        """Successive Mean Quantization Transform for enhancement"""
        enhanced = image.copy()
        
        for _ in range(iterations):
            mean_val = np.mean(enhanced[enhanced > 0])
            enhanced[enhanced < mean_val] = enhanced[enhanced < mean_val] * 0.8
            enhanced[enhanced >= mean_val] = enhanced[enhanced >= mean_val] * 1.2
            enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced
    
    def stft_analysis(self, image, window_size=32, overlap=16):
        """STFT Analysis to extract orientation, frequency and energy features"""
        height, width = image.shape
        orientation_map = np.zeros((height, width))
        frequency_map = np.zeros((height, width))
        energy_map = np.zeros((height, width))
        
        # Create window
        window = np.outer(signal.windows.hann(window_size), signal.windows.hann(window_size))
        
        for i in range(0, height - window_size, overlap):
            for j in range(0, width - window_size, overlap):
                block = image[i:i+window_size, j:j+window_size] * window
                
                # Compute FFT
                fft_block = np.fft.fft2(block)
                fft_shifted = np.fft.fftshift(fft_block)
                magnitude = np.abs(fft_shifted)
                
                # Convert to polar coordinates
                center = window_size // 2
                y_indices, x_indices = np.ogrid[:window_size, :window_size]
                x_indices = x_indices - center
                y_indices = y_indices - center
                
                rho = np.sqrt(x_indices**2 + y_indices**2)
                theta = np.arctan2(y_indices, x_indices)
                
                # Find dominant orientation and frequency
                if np.max(magnitude) > 0:
                    # Weighted average for orientation
                    orientation = np.sum(magnitude * theta) / np.sum(magnitude)
                    
                    # Weighted average for frequency
                    frequency = np.sum(magnitude * rho) / np.sum(magnitude)
                    
                    # Energy is sum of magnitudes
                    energy = np.sum(magnitude)
                else:
                    orientation = 0
                    frequency = 0
                    energy = 0
                
                # Fill the region
                orientation_map[i:i+window_size, j:j+window_size] = orientation
                frequency_map[i:i+window_size, j:j+window_size] = frequency
                energy_map[i:i+window_size, j:j+window_size] = energy
        
        return orientation_map, frequency_map, energy_map
    
    def binarize_and_thin(self, image, mask):
        """Binarize image and apply thinning"""
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply mask
        binary[~mask] = 0
        
        # Thinning to get ridge skeleton
        thinned = thin(binary > 0).astype(np.uint8) * 255
        
        return binary, thinned
    
    def extract_minutiae(self, thinned_image, mask):
        """Extract minutiae points from thinned image"""
        minutiae = []
        height, width = thinned_image.shape
        
        # Define 8-connected neighborhood
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                if thinned_image[i, j] == 255 and mask[i, j]:
                    # Count ridge endings and bifurcations
                    ridge_pixels = 0
                    for di, dj in neighbors:
                        if thinned_image[i+di, j+dj] == 255:
                            ridge_pixels += 1
                    
                    # Ridge ending (1 neighbor) or bifurcation (3+ neighbors)
                    if ridge_pixels == 1 or ridge_pixels >= 3:
                        # Calculate orientation at this point
                        orientation = self.calculate_minutia_orientation(thinned_image, i, j)
                        minutia_type = 'ending' if ridge_pixels == 1 else 'bifurcation'
                        minutiae.append({
                            'x': j, 'y': i, 
                            'orientation': orientation, 
                            'type': minutia_type,
                            'quality': 1.0
                        })
        
        return minutiae
    
    def calculate_minutia_orientation(self, image, y, x, window_size=9):
        """Calculate orientation at minutia point"""
        half_size = window_size // 2
        
        # Extract local window
        y_start = max(0, y - half_size)
        y_end = min(image.shape[0], y + half_size + 1)
        x_start = max(0, x - half_size)
        x_end = min(image.shape[1], x + half_size + 1)
        
        window = image[y_start:y_end, x_start:x_end]
        
        if window.size == 0:
            return 0
        
        # Calculate gradients
        dy, dx = np.gradient(window.astype(float))
        
        # Calculate orientation using structure tensor
        Gxx = np.sum(dx * dx)
        Gyy = np.sum(dy * dy)
        Gxy = np.sum(dx * dy)
        
        # Calculate orientation
        orientation = 0.5 * np.arctan2(2 * Gxy, Gxx - Gyy)
        return orientation
    
    def create_cylinder_codes(self, minutiae, orientation_map, frequency_map, energy_map, 
                            radius=50, ns=16, nd=8):
        """Create MTCC cylinder codes for each minutia"""
        cylinder_codes = []
        
        for minutia in minutiae:
            x, y = minutia['x'], minutia['y']
            orientation = minutia['orientation']
            
            # Create cylinder
            cylinder = np.zeros((ns, ns, nd))
            
            # Sample points in circular region
            for i in range(ns):
                for j in range(ns):
                    # Convert to local coordinates relative to minutia
                    local_x = (i - ns//2) * (2 * radius / ns)
                    local_y = (j - ns//2) * (2 * radius / ns)
                    
                    # Rotate according to minutia orientation
                    rotated_x = local_x * np.cos(orientation) - local_y * np.sin(orientation)
                    rotated_y = local_x * np.sin(orientation) + local_y * np.cos(orientation)
                    
                    # Convert back to image coordinates
                    img_x = int(x + rotated_x)
                    img_y = int(y + rotated_y)
                    
                    # Check if within image bounds
                    if (0 <= img_x < orientation_map.shape[1] and 
                        0 <= img_y < orientation_map.shape[0]):
                        
                        # Extract texture features at this point
                        local_orientation = orientation_map[img_y, img_x]
                        local_frequency = frequency_map[img_y, img_x]
                        local_energy = energy_map[img_y, img_x]
                        
                        # Quantize into directional bins
                        for k in range(nd):
                            angle_bin = k * 2 * np.pi / nd
                            
                            # Use orientation difference for original MCC
                            orientation_diff = abs(local_orientation - angle_bin)
                            cylinder[i, j, k] = np.exp(-orientation_diff**2 / (2 * 0.5**2))
            
            cylinder_codes.append({
                'minutia': minutia,
                'cylinder': cylinder,
                'orientation_features': orientation_map[max(0, min(y, orientation_map.shape[0]-1)), 
                                                      max(0, min(x, orientation_map.shape[1]-1))],
                'frequency_features': frequency_map[max(0, min(y, frequency_map.shape[0]-1)), 
                                                   max(0, min(x, frequency_map.shape[1]-1))],
                'energy_features': energy_map[max(0, min(y, energy_map.shape[0]-1)), 
                                             max(0, min(x, energy_map.shape[1]-1))]
            })
        
        return cylinder_codes
    
    def match_cylinder_codes(self, codes1, codes2, threshold=0.7):
        """Match two sets of cylinder codes"""
        if not codes1 or not codes2:
            return 0.0
        
        scores = []
        
        for code1 in codes1:
            best_score = 0
            for code2 in codes2:
                # Calculate cylinder similarity
                cylinder_similarity = self.calculate_cylinder_similarity(
                    code1['cylinder'], code2['cylinder']
                )
                
                # Calculate spatial constraint
                spatial_distance = euclidean(
                    [code1['minutia']['x'], code1['minutia']['y']],
                    [code2['minutia']['x'], code2['minutia']['y']]
                )
                spatial_score = np.exp(-spatial_distance / 50.0)
                
                # Combined score
                combined_score = cylinder_similarity * spatial_score
                best_score = max(best_score, combined_score)
            
            scores.append(best_score)
        
        # Calculate overall matching score
        if scores:
            matching_score = np.mean(scores)
        else:
            matching_score = 0.0
        
        return matching_score
    
    def calculate_cylinder_similarity(self, cylinder1, cylinder2):
        """Calculate similarity between two cylinders"""
        # Normalize cylinders
        c1_norm = cylinder1 / (np.linalg.norm(cylinder1) + 1e-8)
        c2_norm = cylinder2 / (np.linalg.norm(cylinder2) + 1e-8)
        
        # Calculate correlation
        correlation = np.sum(c1_norm * c2_norm)
        return max(0, correlation)
    
    def calculate_eer(self, genuine_scores, impostor_scores):
        """Calculate Equal Error Rate"""
        # Combine and sort all scores
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        thresholds = np.sort(np.unique(all_scores))
        
        min_diff = float('inf')
        eer = 0
        
        for threshold in thresholds:
            far = np.sum(impostor_scores >= threshold) / len(impostor_scores)
            frr = np.sum(genuine_scores < threshold) / len(genuine_scores)
            
            diff = abs(far - frr)
            if diff < min_diff:
                min_diff = diff
                eer = (far + frr) / 2
        
        return eer
    
    def match_two_fingerprints(self, image1, image2):
        """Complete matching pipeline for two fingerprints"""
        # Process first fingerprint
        print("Processing first fingerprint...")
        processed1 = self.process_fingerprint(image1)
        
        # Process second fingerprint
        print("Processing second fingerprint...")
        processed2 = self.process_fingerprint(image2)
        
        if processed1 is None or processed2 is None:
            return 0.0
        
        # Match cylinder codes
        score = self.match_cylinder_codes(processed1['cylinder_codes'], 
                                        processed2['cylinder_codes'])
        
        return score
    
    def process_fingerprint(self, image):
        """Complete processing pipeline for a single fingerprint"""
        try:
            # 1. Load image
            if isinstance(image, str):
                img = self.load_fingerprint(image)
            else:
                img = image.copy()
            
            if img is None:
                return None
            
            # 2. Normalize
            normalized = self.normalize_image(img)
            
            # 3. Segment
            mask = self.segment_fingerprint(normalized)
            
            # 4. Gabor enhancement
            gabor_enhanced = self.gabor_filter_bank_enhancement(normalized, mask)
            
            # 5. SMQT enhancement
            smqt_enhanced = self.smqt_normalization(gabor_enhanced)
            
            # 6. STFT Analysis
            orientation_map, frequency_map, energy_map = self.stft_analysis(smqt_enhanced)
            
            # 7. Binarize and thin
            binary, thinned = self.binarize_and_thin(smqt_enhanced, mask)
            
            # 8. Extract minutiae
            minutiae = self.extract_minutiae(thinned, mask)
            
            # 9. Create cylinder codes
            cylinder_codes = self.create_cylinder_codes(minutiae, orientation_map, 
                                                      frequency_map, energy_map)
            
            return {
                'original': img,
                'normalized': normalized,
                'mask': mask,
                'gabor_enhanced': gabor_enhanced,
                'smqt_enhanced': smqt_enhanced,
                'orientation_map': orientation_map,
                'frequency_map': frequency_map,
                'energy_map': energy_map,
                'binary': binary,
                'thinned': thinned,
                'minutiae': minutiae,
                'cylinder_codes': cylinder_codes
            }
        
        except Exception as e:
            print(f"Error processing fingerprint: {e}")
            return None
    
    def visualize_processing_steps(self, processed_data, figsize=(15, 10)):
        """Visualize all processing steps in a single figure"""
        if processed_data is None:
            print("No processed data to visualize")
            return
        
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Original image
        axes[0].imshow(processed_data['original'], cmap='gray')
        axes[0].set_title('1. Original Image')
        axes[0].axis('off')
        
        # 2. Normalized image
        axes[1].imshow(processed_data['normalized'], cmap='gray')
        axes[1].set_title('2. Normalized')
        axes[1].axis('off')
        
        # 3. Segmentation mask
        axes[2].imshow(processed_data['mask'], cmap='gray')
        axes[2].set_title('3. Segmentation Mask')
        axes[2].axis('off')
        
        # 4. Gabor enhanced
        axes[3].imshow(processed_data['gabor_enhanced'], cmap='gray')
        axes[3].set_title('4. Gabor Enhanced')
        axes[3].axis('off')
        
        # 5. SMQT enhanced
        axes[4].imshow(processed_data['smqt_enhanced'], cmap='gray')
        axes[4].set_title('5. SMQT Enhanced')
        axes[4].axis('off')
        
        # 6. Orientation map
        axes[5].imshow(processed_data['orientation_map'], cmap='hsv')
        axes[5].set_title('6. Orientation Map')
        axes[5].axis('off')
        
        # 7. Frequency map
        axes[6].imshow(processed_data['frequency_map'], cmap='jet')
        axes[6].set_title('7. Frequency Map')
        axes[6].axis('off')
        
        # 8. Energy map
        axes[7].imshow(processed_data['energy_map'], cmap='hot')
        axes[7].set_title('8. Energy Map')
        axes[7].axis('off')
        
        # 9. Binary image
        axes[8].imshow(processed_data['binary'], cmap='gray')
        axes[8].set_title('9. Binary Image')
        axes[8].axis('off')
        
        # 10. Thinned image
        axes[9].imshow(processed_data['thinned'], cmap='gray')
        axes[9].set_title('10. Thinned Image')
        axes[9].axis('off')
        
        # 11. Minutiae overlay
        axes[10].imshow(processed_data['original'], cmap='gray')
        if processed_data['minutiae']:
            x_coords = [m['x'] for m in processed_data['minutiae']]
            y_coords = [m['y'] for m in processed_data['minutiae']]
            axes[10].scatter(x_coords, y_coords, c='red', s=20, marker='o')
        axes[10].set_title(f'11. Minutiae ({len(processed_data["minutiae"])} found)')
        axes[10].axis('off')
        
        # 12. Cylinder codes visualization (simplified)
        if processed_data['cylinder_codes']:
            # Show first cylinder as example
            cylinder = processed_data['cylinder_codes'][0]['cylinder']
            axes[11].imshow(np.sum(cylinder, axis=2), cmap='viridis')
            axes[11].set_title('12. Cylinder Code (sample)')
        else:
            axes[11].text(0.5, 0.5, 'No Cylinders', ha='center', va='center')
            axes[11].set_title('12. Cylinder Code')
        axes[11].axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage
def demo_mtcc_system():
    """Demonstration of the MTCC system"""
    
    # Create MTCC system
    mtcc = MTCCFingerprintSystem()
    
    # Create a synthetic fingerprint for demonstration
    def create_synthetic_fingerprint(size=(256, 256)):
        """Create a synthetic fingerprint pattern for testing"""
        h, w = size
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Create sinusoidal ridge pattern
        angle = np.pi / 6  # 30 degrees
        frequency = 0.1
        
        pattern = np.sin(2 * np.pi * frequency * (x * np.cos(angle) + y * np.sin(angle)))
        
        # Add some noise and normalize
        noise = np.random.normal(0, 0.1, size)
        fingerprint = (pattern + noise) * 127 + 128
        fingerprint = np.clip(fingerprint, 0, 255)
        
        return fingerprint.astype(np.uint8)
    
    # Create two slightly different synthetic fingerprints
    print("Creating synthetic fingerprints for demonstration...")
    fp1 = create_synthetic_fingerprint()
    fp2 = create_synthetic_fingerprint()
    
    # Process fingerprints
    print("\n=== Processing Fingerprint 1 ===")
    processed1 = mtcc.process_fingerprint(fp1)
    
    print("\n=== Processing Fingerprint 2 ===")
    processed2 = mtcc.process_fingerprint(fp2)
    
    # Visualize processing steps
    if processed1:
        print("\n=== Visualizing Processing Steps ===")
        mtcc.visualize_processing_steps(processed1)
    
    # Match fingerprints
    if processed1 and processed2:
        print("\n=== Matching Fingerprints ===")
        score = mtcc.match_cylinder_codes(processed1['cylinder_codes'], 
                                        processed2['cylinder_codes'])
        print(f"Matching Score: {score:.4f}")
        
        # Generate some sample scores for EER calculation
        genuine_scores = np.random.normal(0.7, 0.2, 100)
        impostor_scores = np.random.normal(0.3, 0.15, 100)
        eer = mtcc.calculate_eer(genuine_scores, impostor_scores)
        print(f"Sample EER: {eer:.4f}")
    
    return mtcc

if __name__ == "__main__":
    # Run demonstration
    system = demo_mtcc_system()