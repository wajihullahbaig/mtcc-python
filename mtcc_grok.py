import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.filters import gabor
import cv2
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class Minutia:
    x: float
    y: float
    theta: float
    quality: float

@dataclass
class Cylinder:
    minutia: Minutia
    cells: np.ndarray
    valid: bool

class VisualizationConfig:
    def __init__(self):
        self.show_steps = {
            'loaded_image': False,
            'enhanced_image': False,
            'segmented_image': False,
            'mask': False,
            'binary_image': False,
            'minutiae_plots': False
        }

    def set_show(self, step: str, show: bool) -> None:
        """Set visibility for a visualization step."""
        if step in self.show_steps:
            self.show_steps[step] = show

class Visualizer:
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range for display."""
        img = image.astype(np.float32)
        if img.size == 0:
            return img
        min_val, max_val = np.min(img), np.max(img)
        if max_val == min_val:
            return np.zeros_like(img)
        return (img - min_val) / (max_val - min_val + 1e-10)

    @staticmethod
    def plot_image(image: np.ndarray, title: str, show: bool = False) -> None:
        """Display a normalized image if show is True."""
        if show:
            plt.figure(figsize=(6, 6))
            plt.imshow(Visualizer.normalize_image(image), cmap='gray')
            plt.title(title)
            plt.axis('off')
            plt.show()

    @staticmethod
    def plot_minutiae(image: np.ndarray, minutiae: List[Minutia], title: str, show: bool = False) -> None:
        """Display minutiae on a normalized image if show is True."""
        if show:
            plt.figure(figsize=(6, 6))
            plt.imshow(Visualizer.normalize_image(image), cmap='gray')
            for m in minutiae:
                plt.plot(m.x, m.y, 'ro', markersize=5)
                plt.arrow(m.x, m.y, 10*np.cos(m.theta), 10*np.sin(m.theta), color='r', width=0.5)
            plt.title(title)
            plt.axis('off')
            plt.show()

class FingerprintDataset:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.images = []
        self.labels = []

    def load_fvc(self, dataset: str) -> None:
        """Load FVC 2002/2004 dataset images and labels."""
        for root, _, files in os.walk(os.path.join(self.data_dir, dataset)):
            for file in files:
                if file.endswith('.tif'):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    self.images.append(img)
                    self.labels.append(file.split('_')[0])

class ImageEnhancer:
    def __init__(self, window_size: int = 14, overlap: int = 6):
        self.window_size = window_size
        self.overlap = overlap

    def segment(self, image: np.ndarray, vis_config: VisualizationConfig) -> np.ndarray:
        """Segment fingerprint image using block-wise variance."""
        h, w = image.shape
        block_size = 16
        blocks = []
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = image[i:i+block_size, j:j+block_size]
                blocks.append(block)
        
        variance = np.var(blocks, axis=(1,2))
        variance_map = np.zeros((h//block_size, w//block_size))
        idx = 0
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                variance_map[i//block_size, j//block_size] = variance[idx]
                idx += 1
        
        mask = variance_map > np.mean(variance)
        mask = morphology.binary_dilation(mask)
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        Visualizer.plot_image(mask * image, "Segmented Image", vis_config.show_steps['segmented_image'])
        Visualizer.plot_image(mask, "Mask", vis_config.show_steps['mask'])
        return mask

    def stft_enhance(self, image: np.ndarray, vis_config: VisualizationConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply 2D FFT for orientation, frequency, and energy images."""
        h, w = image.shape
        step = self.window_size - self.overlap
        orientation = np.zeros((h, w))
        frequency = np.zeros((h, w))
        energy = np.zeros((h, w))
        
        for i in range(0, h - self.window_size + 1, step):
            for j in range(0, w - self.window_size + 1, step):
                block = image[i:i+self.window_size, j:j+self.window_size]
                # Apply windowing to reduce edge effects
                window = np.hanning(self.window_size)[:, None] * np.hanning(self.window_size)[None, :]
                block = block * window
                # Compute 2D FFT and shift to center DC component
                fft_block = np.fft.fft2(block)
                fft_shift = np.fft.fftshift(fft_block)
                magnitude = np.abs(fft_shift)
                
                # Mask out the DC component (center) to focus on ridge frequencies
                center = self.window_size // 2
                magnitude[center-1:center+2, center-1:center+2] = 0
                
                # Find dominant frequency and orientation
                freq_y, freq_x = np.unravel_index(np.argmax(magnitude), magnitude.shape)
                if magnitude[freq_y, freq_x] == 0:  # Skip if no valid peak
                    continue
                    
                # Compute frequency as distance from center
                dominant_freq = np.sqrt((freq_x - center)**2 + (freq_y - center)**2) / self.window_size
                # Compute orientation from frequency coordinates
                dominant_angle = np.arctan2(freq_y - center, freq_x - center)
                # Compute energy as log of total magnitude (excluding DC)
                block_energy = np.log(np.sum(magnitude) + 1e-10)
                
                # Assign to block region
                orientation[i:i+self.window_size, j:j+self.window_size] = dominant_angle
                frequency[i:i+self.window_size, j:j+self.window_size] = dominant_freq
                energy[i:i+self.window_size, j:j+self.window_size] = block_energy
        
        # Normalize for visualization
        Visualizer.plot_image(orientation, "Enhanced Image (Orientation)", vis_config.show_steps['enhanced_image'])
        Visualizer.plot_image(frequency, "Frequency Image", vis_config.show_steps['enhanced_image'])
        Visualizer.plot_image(energy, "Energy Image", vis_config.show_steps['enhanced_image'])
        return image, orientation, frequency, energy

    def gabor_filter(self, image: np.ndarray, orientation: np.ndarray, frequency: np.ndarray, vis_config: VisualizationConfig) -> np.ndarray:
        """Apply Gabor filtering to smooth ridges."""
        filtered = np.zeros_like(image, dtype=np.float32)
        for i in range(0, image.shape[0], 16):
            for j in range(0, image.shape[1], 16):
                theta = orientation[i, j]
                freq = frequency[i, j]
                if freq > 0:
                    filtered[i:i+16, j:j+16] = gabor(image[i:i+16, j:j+16], frequency=freq, theta=theta)[0]
        Visualizer.plot_image(filtered, "Gabor Filtered Image", vis_config.show_steps['enhanced_image'])
        return filtered

    def smqt_normalize(self, image: np.ndarray, vis_config: VisualizationConfig) -> np.ndarray:
        """Apply Successive Mean Quantization Transform."""
        def quantize(x, levels=8):
            min_val, max_val = np.min(x), np.max(x)
            return np.round((x - min_val) / (max_val - min_val + 1e-10) * (levels - 1))
        
        normalized = quantize(image)
        Visualizer.plot_image(normalized, "SMQT Normalized Image", vis_config.show_steps['enhanced_image'])
        return normalized

class MinutiaeExtractor:
    def extract(self, image: np.ndarray, vis_config: VisualizationConfig) -> List[Minutia]:
        """Extract minutiae using a simplified approach."""
        image_uint8 = (image * 255 / np.max(image)).astype(np.uint8)
        _, binary = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        Visualizer.plot_image(binary, "Binary Image", vis_config.show_steps['binary_image'])
        skeleton = morphology.skeletonize(binary // 255).astype(np.uint8)
        minutiae = []
        for i in range(1, skeleton.shape[0]-1):
            for j in range(1, skeleton.shape[1]-1):
                if skeleton[i, j]:
                    neighbors = np.sum(skeleton[i-1:i+2, j-1:j+2]) - 1
                    if neighbors in [1, 3]:
                        theta = np.arctan2(skeleton[i+1, j] - skeleton[i-1, j], skeleton[i, j+1] - skeleton[i, j-1])
                        minutiae.append(Minutia(x=j, y=i, theta=theta, quality=1.0))
        Visualizer.plot_minutiae(image, minutiae, "Minutiae Plots", vis_config.show_steps['minutiae_plots'])
        return minutiae

class CylinderFactory:
    def __init__(self, radius: int = 63, ns: int = 18, nd: int = 5):
        self.R = radius
        self.NS = ns
        self.ND = nd
        self.delta_s = 2 * radius / ns
        self.delta_d = 2 * np.pi / nd

    def create_cylinder(self, minutia: Minutia, minutiae: List[Minutia], 
                      orientation: np.ndarray, frequency: np.ndarray, energy: np.ndarray, 
                      mask: np.ndarray) -> Cylinder:
        """Create MTCC cylinder for a minutia."""
        cells = np.zeros((self.NS, self.NS, self.ND))
        valid_cells = 0
        
        for i in range(self.NS):
            for j in range(self.NS):
                for k in range(self.ND):
                    d_sk = -np.pi + (k - 0.5) * self.delta_d
                    p_ij = self._get_cell_center(minutia, i, j)
                    
                    # Ensure indices are within bounds
                    y, x = int(p_ij[1]), int(p_ij[0])
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                        contribution = self._compute_contribution(minutia, minutiae, p_ij, d_sk, 
                                                               orientation, frequency, energy)
                        cells[i, j, k] = contribution
                        if contribution > 0:
                            valid_cells += 1
        
        is_valid = valid_cells > 0.1 * self.NS * self.NS * self.ND
        return Cylinder(minutia, cells, is_valid)

    def _get_cell_center(self, minutia: Minutia, i: int, j: int) -> np.ndarray:
        """Compute cell center coordinates."""
        rotation = np.array([[np.cos(minutia.theta), np.sin(minutia.theta)],
                           [-np.sin(minutia.theta), np.cos(minutia.theta)]])
        offset = np.array([i - (self.NS + 1) / 2, j - (self.NS + 1) / 2])
        return np.array([minutia.x, minutia.y]) + self.delta_s * rotation @ offset

    def _compute_contribution(self, minutia: Minutia, minutiae: List[Minutia], 
                         p_ij: np.ndarray, d_sk: float, 
                         orientation: np.ndarray, frequency: np.ndarray, 
                         energy: np.ndarray) -> float:
        """Compute cell contribution using texture features."""
        spatial_sigma = self.R / 3
        directional_sigma = np.pi / 6
        neighbors = [m for m in minutiae if m != minutia and 
                    np.linalg.norm([m.x - p_ij[0], m.y - p_ij[1]]) <= 3 * spatial_sigma]
        
        if not neighbors:
            return 0.0
        
        total_contribution = 0.0
        for neighbor in neighbors:
            spatial_dist = np.linalg.norm([neighbor.x - p_ij[0], neighbor.y - p_ij[1]])
            spatial_contrib = np.exp(-spatial_dist**2 / (2 * spatial_sigma**2))
            
            y, x = int(minutia.y), int(minutia.x)
            py, px = int(p_ij[1]), int(p_ij[0])
            if 0 <= y < frequency.shape[0] and 0 <= x < frequency.shape[1] and \
            0 <= py < frequency.shape[0] and 0 <= px < frequency.shape[1]:
                freq_diff = frequency[y, x] - frequency[py, px]
                freq_contrib = np.exp(-freq_diff**2 / (2 * directional_sigma**2))
                total_contribution += spatial_contrib * freq_contrib
        
        return min(1.0, max(0.0, total_contribution / len(neighbors)))

class Matcher:
    def __init__(self, min_matchable_cells: int = 10):
        self.min_matchable_cells = min_matchable_cells

    def match_cylinders(self, cylinders_a: List[Cylinder], cylinders_b: List[Cylinder]) -> float:
        """Match two sets of cylinders using LSSR."""
        lsm = self._compute_local_similarity_matrix(cylinders_a, cylinders_b)
        top_pairs = self._local_similarity_sort(lsm)
        return self._relaxation(top_pairs, cylinders_a, cylinders_b)

    def _compute_local_similarity_matrix(self, cylinders_a: List[Cylinder], 
                                      cylinders_b: List[Cylinder]) -> np.ndarray:
        """Compute local similarity matrix."""
        lsm = np.zeros((len(cylinders_a), len(cylinders_b)))
        for i, cyl_a in enumerate(cylinders_a):
            for j, cyl_b in enumerate(cylinders_b):
                if cyl_a.valid and cyl_b.valid:
                    lsm[i, j] = self._cylinder_distance(cyl_a, cyl_b)
        return lsm

    def _cylinder_distance(self, cyl_a: Cylinder, cyl_b: Cylinder) -> float:
        """Compute distance between two cylinders."""
        valid_cells_a = np.sum(cyl_a.cells > 0)
        valid_cells_b = np.sum(cyl_b.cells > 0)
        if valid_cells_a < self.min_matchable_cells or valid_cells_b < self.min_matchable_cells:
            return 0.0
        
        diff = np.abs(cyl_a.cells - cyl_b.cells)
        norm = np.sum(cyl_a.cells) + np.sum(cyl_b.cells)
        return 1 - np.sum(diff) / (norm + 1e-10) if norm > 0 else 0.0

    def _local_similarity_sort(self, lsm: np.ndarray) -> List[Tuple[int, int, float]]:
        """Sort local similarity matrix to get top matching pairs."""
        pairs = [(i, j, lsm[i, j]) for i in range(lsm.shape[0]) for j in range(lsm.shape[1])]
        return sorted(pairs, key=lambda x: x[2], reverse=True)[:50]

    def _relaxation(self, pairs: List[Tuple[int, int, float]], 
                   cylinders_a: List[Cylinder], cylinders_b: List[Cylinder]) -> float:
        """Apply relaxation to penalize dissimilar pairs."""
        if not pairs:
            return 0.0
        
        scores = np.array([score for _, _, score in pairs])
        compatibilities = np.ones(len(pairs))
        
        for i, (idx_a1, idx_b1, _) in enumerate(pairs):
            for j, (idx_a2, idx_b2, _) in enumerate(pairs):
                if i != j:
                    dist_a = np.linalg.norm([cylinders_a[idx_a1].minutia.x - cylinders_a[idx_a2].minutia.x,
                                           cylinders_a[idx_a1].minutia.y - cylinders_a[idx_a2].minutia.y])
                    dist_b = np.linalg.norm([cylinders_b[idx_b1].minutia.x - cylinders_b[idx_b2].minutia.x,
                                           cylinders_b[idx_b1].minutia.y - cylinders_b[idx_b2].minutia.y])
                    compatibilities[i] *= np.exp(-abs(dist_a - dist_b) / 100)
        
        final_scores = scores * compatibilities
        return np.mean(final_scores[:10]) if final_scores.size > 0 else 0.0

class FingerprintMatcher:
    def __init__(self, data_dir: str):
        self.dataset = FingerprintDataset(data_dir)
        self.enhancer = ImageEnhancer()
        self.extractor = MinutiaeExtractor()
        self.cylinder_factory = CylinderFactory()
        self.matcher = Matcher()
        self.vis_config = VisualizationConfig()

    def set_visualization(self, step: str, show: bool) -> None:
        """Set visualization for a specific step."""
        self.vis_config.set_show(step, show)

    def process_image(self, image: np.ndarray) -> List[Cylinder]:
        """Process a single fingerprint image."""
        Visualizer.plot_image(image, "Loaded Image", self.vis_config.show_steps['loaded_image'])
        mask = self.enhancer.segment(image, self.vis_config)
        enhanced, orientation, frequency, energy = self.enhancer.stft_enhance(image, self.vis_config)
        enhanced = self.enhancer.gabor_filter(enhanced, orientation, frequency, self.vis_config)
        enhanced = self.enhancer.smqt_normalize(enhanced, self.vis_config)
        minutiae = self.extractor.extract(enhanced, self.vis_config)
        return [self.cylinder_factory.create_cylinder(m, minutiae, orientation, frequency, energy, mask)
                for m in minutiae]

    def match(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Match two fingerprint images."""
        cylinders1 = self.process_image(img1)
        cylinders2 = self.process_image(img2)
        return self.matcher.match_cylinders(cylinders1, cylinders2)

def main():
    matcher =  FingerprintMatcher(R"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002")
    matcher.dataset.load_fvc('FVC2002')
    
    for step in ['loaded_image', 'enhanced_image', 'segmented_image', 'mask', 'binary_image', 'minutiae_plots']:
        matcher.set_visualization(step, True)
    
    img1, img2 = matcher.dataset.images[:2]
    score = matcher.match(img1, img2)
    print(f"Matching score: {score}")

if __name__ == "__main__":
    main()