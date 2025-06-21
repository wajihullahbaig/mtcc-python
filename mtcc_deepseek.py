import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os
from enum import Enum
import math

# ==================== Data Structures ====================
@dataclass
class Minutia:
    x: float
    y: float
    angle: float  # in radians
    quality: float = 1.0

@dataclass
class Fingerprint:
    image: np.ndarray
    minutiae: List[Minutia]
    mask: Optional[np.ndarray] = None
    orientation_map: Optional[np.ndarray] = None
    frequency_map: Optional[np.ndarray] = None
    energy_map: Optional[np.ndarray] = None

class FeatureType(Enum):
    ORIENTATION = 1
    FREQUENCY = 2
    ENERGY = 3
    CELL_CENTERED_ORIENTATION = 4
    CELL_CENTERED_FREQUENCY = 5
    CELL_CENTERED_ENERGY = 6

# ==================== Interfaces ====================
class IDatasetLoader(ABC):
    @abstractmethod
    def load_dataset(self, path: str) -> Dict[str, List[Fingerprint]]:
        pass

class IImageEnhancer(ABC):
    @abstractmethod
    def enhance(self, fingerprint: Fingerprint) -> Fingerprint:
        pass

class IFeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, fingerprint: Fingerprint, feature_type: FeatureType) -> np.ndarray:
        pass

class IMatcher(ABC):
    @abstractmethod
    def match(self, template1: np.ndarray, template2: np.ndarray) -> float:
        pass

# ==================== Implementations ====================
class FVCDatasetLoader(IDatasetLoader):
    def __init__(self, db_name: str = "db1_a", competition_year: int = 2000):
        """
        Initialize the FVC dataset loader
        
        Args:
            db_name: Database name (e.g., "DB1_A", "DB2_B")
            competition_year: Year of FVC competition (2000, 2002, 2004)
        """
        self.db_name = db_name.upper()  # Normalize to uppercase
        self.competition_year = competition_year
        
    def load_dataset(self, path: str) -> Dict[str, List[Fingerprint]]:
        """
        Load FVC dataset from directory structure
        
        Args:
            path: Path to the dataset directory (e.g., "C:\...\FVC2000\Db1_a")
            
        Returns:
            Dictionary mapping person IDs to lists of Fingerprint objects
            
        Note:
            FVC2000 naming convention:
            - Images named like: {person_id}_{sample_num}.tif
            - Example: 1_1.tif, 1_2.tif, ..., 2_1.tif, etc.
            - Person ID is extracted from the first part before underscore
        """
        dataset = {}
        
        # Get all image files in directory
        try:
            image_files = [
                f for f in os.listdir(path) 
                if f.lower().endswith('.tif')  # FVC2000 uses TIFF format
            ]
        except FileNotFoundError:
            raise FileNotFoundError(f"Directory not found: {path}")
        
        # Group images by person ID
        for img_file in image_files:
            try:
                # Extract person ID from filename (e.g., "1" from "1_1.tif")
                person_id = img_file.split('_')[0]
                
                # Skip if person_id couldn't be extracted
                if not person_id.isdigit():
                    continue
                    
                img_path = os.path.join(path, img_file)
                
                # Read image (FVC2000 uses 8-bit grayscale TIFF)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"Warning: Could not read image {img_file}")
                    continue
                    
                # Create fingerprint object
                fp = Fingerprint(
                    image=image,
                    minutiae=[],
                    metadata={
                        "filename": img_file,
                        "source": f"FVC{self.competition_year}",
                        "database": self.db_name
                    }
                )
                
                # Add to dataset
                if person_id not in dataset:
                    dataset[person_id] = []
                dataset[person_id].append(fp)
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue
                
        if not dataset:
            raise ValueError(f"No valid fingerprint images found in {path}")
            
        print(f"Loaded {len(dataset)} subjects from {path}")
        return dataset

class FingerprintEnhancer(IImageEnhancer):
    def __init__(self, window_size: int = 14, overlap: int = 6):
        self.window_size = window_size
        self.overlap = overlap

    def enhance(self, fingerprint: Fingerprint) -> Fingerprint:
        """Apply STFT, Gabor filtering, and SMQT enhancement"""
        # 1. STFT enhancement
        stft_enhanced = self._stft_enhancement(fingerprint.image)
        
        # 2. Gabor filtering
        gabor_filtered = self._gabor_filtering(stft_enhanced)
        
        # 3. SMQT enhancement
        smqt_enhanced = self._smqt_enhancement(gabor_filtered)
        
        # 4. Generate texture maps
        orientation, frequency, energy = self._stft_analysis(fingerprint.image)
        
        return Fingerprint(
            image=smqt_enhanced,
            minutiae=fingerprint.minutiae,
            mask=self._segment(smqt_enhanced),
            orientation_map=orientation,
            frequency_map=frequency,
            energy_map=energy
        )

    def _stft_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply STFT-based enhancement"""
        # Implementation simplified for brevity
        enhanced = cv2.GaussianBlur(image, (5, 5), 0)
        return enhanced

    def _gabor_filtering(self, image: np.ndarray) -> np.ndarray:
        """Apply Gabor filtering"""
        ksize = 31
        sigma = 5
        theta = 0
        lambd = 10
        gamma = 0.5
        gabor = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        return cv2.filter2D(image, cv2.CV_8UC3, gabor)

    def _smqt_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply Successive Mean Quantization Transform"""
        # Implementation simplified for brevity
        return cv2.equalizeHist(image)

    def _stft_analysis(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate orientation, frequency and energy maps using STFT"""
        # Simplified implementation - in practice would use overlapping windows
        orientation = np.zeros_like(image, dtype=np.float32)
        frequency = np.zeros_like(image, dtype=np.float32)
        energy = np.zeros_like(image, dtype=np.float32)
        
        # For each block, calculate features
        for i in range(0, image.shape[0], self.window_size - self.overlap):
            for j in range(0, image.shape[1], self.window_size - self.overlap):
                block = image[i:i+self.window_size, j:j+self.window_size]
                if block.size == 0:
                    continue
                    
                # Calculate features (simplified)
                fft = np.fft.fft2(block)
                magnitude = np.abs(fft)
                
                # Dominant orientation
                orientation[i:i+self.window_size, j:j+self.window_size] = np.angle(fft[1, 0])
                
                # Dominant frequency
                frequency[i:i+self.window_size, j:j+self.window_size] = np.argmax(magnitude[1:]) + 1
                
                # Energy
                energy[i:i+self.window_size, j:j+self.window_size] = np.log(np.sum(magnitude**2) + 1e-6)
        
        return orientation, frequency, energy

    def _segment(self, image: np.ndarray) -> np.ndarray:
        """Segment fingerprint area using variance-based method"""
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

class MinutiaeExtractor:
    def extract(self, fingerprint: Fingerprint) -> List[Minutia]:
        """Extract minutiae from enhanced fingerprint image"""
        # Simplified implementation - in practice would use FingerJetFXOSE or similar
        # This is a placeholder that simulates minutiae detection
        minutiae = []
        for i in range(0, fingerprint.image.shape[0], 20):
            for j in range(0, fingerprint.image.shape[1], 20):
                if fingerprint.mask is not None and fingerprint.mask[i, j] == 0:
                    continue
                angle = fingerprint.orientation_map[i, j] if fingerprint.orientation_map is not None else 0
                minutiae.append(Minutia(x=j, y=i, angle=angle))
        return minutiae

class MTCCFeatureExtractor(IFeatureExtractor):
    def __init__(self, radius: float = 65, ns: int = 18, nd: int = 5):
        self.radius = radius
        self.ns = ns  # Spatial divisions
        self.nd = nd  # Angular divisions
        self.delta_s = 2 * radius / ns
        self.delta_d = 2 * np.pi / nd

    def extract_features(self, fingerprint: Fingerprint, feature_type: FeatureType) -> np.ndarray:
        """Extract MTCC features for all minutiae"""
        cylinders = []
        for minutia in fingerprint.minutiae:
            cylinder = self._create_cylinder(minutia, fingerprint, feature_type)
            cylinders.append(cylinder)
        return np.array(cylinders)

    def _create_cylinder(self, minutia: Minutia, fingerprint: Fingerprint, feature_type: FeatureType) -> np.ndarray:
        """Create a cylinder for a single minutia"""
        cylinder = np.zeros((self.ns, self.ns, self.nd))
        
        for i in range(self.ns):
            for j in range(self.ns):
                for k in range(self.nd):
                    cell_center = self._calculate_cell_center(minutia, i, j)
                    
                    if not self._is_valid_cell(cell_center, fingerprint):
                        continue
                        
                    if feature_type in [FeatureType.CELL_CENTERED_ORIENTATION, 
                                      FeatureType.CELL_CENTERED_FREQUENCY, 
                                      FeatureType.CELL_CENTERED_ENERGY]:
                        contribution = self._calculate_cell_centered_contribution(
                            minutia, cell_center, k, fingerprint, feature_type)
                    else:
                        contribution = self._calculate_minutiae_based_contribution(
                            minutia, cell_center, k, fingerprint, feature_type)
                    
                    cylinder[i, j, k] = contribution
                    
        return cylinder

    def _calculate_cell_center(self, minutia: Minutia, i: int, j: int) -> Tuple[float, float]:
        """Calculate cell center coordinates"""
        rel_i = i - (self.ns + 1) / 2
        rel_j = j - (self.ns + 1) / 2
        
        rot_matrix = np.array([
            [np.cos(minutia.angle), np.sin(minutia.angle)],
            [-np.sin(minutia.angle), np.cos(minutia.angle)]
        ])
        
        offset = np.dot(rot_matrix, np.array([rel_i, rel_j])) * self.delta_s
        return minutia.x + offset[0], minutia.y + offset[1]

    def _is_valid_cell(self, cell_center: Tuple[float, float], fingerprint: Fingerprint) -> bool:
        """Check if cell is within fingerprint area"""
        x, y = cell_center
        if x < 0 or y < 0 or x >= fingerprint.image.shape[1] or y >= fingerprint.image.shape[0]:
            return False
        if fingerprint.mask is not None and fingerprint.mask[int(y), int(x)] == 0:
            return False
        return True

    def _calculate_cell_centered_contribution(self, minutia: Minutia, cell_center: Tuple[float, float], 
                                            k: int, fingerprint: Fingerprint, feature_type: FeatureType) -> float:
        """Calculate contribution for cell-centered features"""
        d_phi_k = -np.pi + (k - 0.5) * self.delta_d
        
        if feature_type == FeatureType.CELL_CENTERED_ORIENTATION:
            center_value = fingerprint.orientation_map[int(cell_center[1]), int(cell_center[0])]
            minutia_value = minutia.angle
        elif feature_type == FeatureType.CELL_CENTERED_FREQUENCY:
            center_value = fingerprint.frequency_map[int(cell_center[1]), int(cell_center[0])]
            minutia_value = fingerprint.frequency_map[int(minutia.y), int(minutia.x)]
        else:  # CELL_CENTERED_ENERGY
            center_value = fingerprint.energy_map[int(cell_center[1]), int(cell_center[0])]
            minutia_value = fingerprint.energy_map[int(minutia.y), int(minutia.x)]
        
        angle_diff = self._normalize_angle(d_phi_k - self._normalize_angle(center_value - minutia_value))
        return self._gaussian(angle_diff, np.pi/36)  # sigma_d = pi/36 as in paper

    def _calculate_minutiae_based_contribution(self, minutia: Minutia, cell_center: Tuple[float, float], 
                                             k: int, fingerprint: Fingerprint, feature_type: FeatureType) -> float:
        """Calculate contribution for minutiae-based features"""
        d_phi_k = -np.pi + (k - 0.5) * self.delta_d
        neighbors = self._find_neighbors(minutia, cell_center, fingerprint)
        
        total_contribution = 0
        for neighbor in neighbors:
            spatial_contribution = self._gaussian(
                np.sqrt((neighbor.x - cell_center[0])**2 + (neighbor.y - cell_center[1])**2),
                6.0  # sigma_s = 6 as in paper
            )
            
            if feature_type == FeatureType.ORIENTATION:
                angle_diff = self._normalize_angle(d_phi_k - self._normalize_angle(neighbor.angle - minutia.angle))
                directional_contribution = self._gaussian(angle_diff, np.pi/36)
            elif feature_type == FeatureType.FREQUENCY:
                neighbor_freq = fingerprint.frequency_map[int(neighbor.y), int(neighbor.x)]
                minutia_freq = fingerprint.frequency_map[int(minutia.y), int(minutia.x)]
                angle_diff = self._normalize_angle(d_phi_k - self._normalize_angle(neighbor_freq - minutia_freq))
                directional_contribution = self._gaussian(angle_diff, np.pi/36)
            else:  # ENERGY
                neighbor_energy = fingerprint.energy_map[int(neighbor.y), int(neighbor.x)]
                minutia_energy = fingerprint.energy_map[int(minutia.y), int(minutia.x)]
                angle_diff = self._normalize_angle(d_phi_k - self._normalize_angle(neighbor_energy - minutia_energy))
                directional_contribution = self._gaussian(angle_diff, np.pi/36)
            
            total_contribution += spatial_contribution * directional_contribution
        
        return self._sigmoid(total_contribution, 0.005, 400)  # mu_psi=0.005, tau_psi=400 as in paper

    def _find_neighbors(self, central_minutia: Minutia, cell_center: Tuple[float, float], 
                       fingerprint: Fingerprint) -> List[Minutia]:
        """Find minutiae in neighborhood of cell center"""
        neighbors = []
        for minutia in fingerprint.minutiae:
            if minutia == central_minutia:
                continue
            distance = np.sqrt((minutia.x - cell_center[0])**2 + (minutia.y - cell_center[1])**2)
            if distance <= 18:  # 3*sigma_s = 3*6 = 18
                neighbors.append(minutia)
        return neighbors

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π] range"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _gaussian(self, value: float, sigma: float) -> float:
        """Gaussian function"""
        return np.exp(-0.5 * (value / sigma)**2)

    def _sigmoid(self, value: float, mu: float, tau: float) -> float:
        """Sigmoid function"""
        return 1 / (1 + np.exp(-tau * (value - mu)))

class LSSRMatcher(IMatcher):
    """Local Similarity Sort with Relaxation matcher"""
    def __init__(self, min_valid_cells: float = 0.2, min_matchable_cells: float = 0.2):
        self.min_valid_cells = min_valid_cells
        self.min_matchable_cells = min_matchable_cells

    def match(self, template1: np.ndarray, template2: np.ndarray) -> float:
        """Match two templates using LSSR"""
        # 1. Local similarity matrix
        lsm = self._calculate_local_similarity_matrix(template1, template2)
        
        # 2. Sort and select top matches
        top_matches = self._select_top_matches(lsm)
        
        # 3. Apply relaxation
        final_score = self._apply_relaxation(top_matches)
        
        return final_score

    def _calculate_local_similarity_matrix(self, template1: np.ndarray, template2: np.ndarray) -> np.ndarray:
        """Calculate local similarity matrix"""
        n1 = len(template1)
        n2 = len(template2)
        lsm = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                lsm[i, j] = self._compare_cylinders(template1[i], template2[j])
        
        return lsm

    def _compare_cylinders(self, cyl1: np.ndarray, cyl2: np.ndarray) -> float:
        """Compare two cylinders"""
        valid_cells1 = np.sum(cyl1 > 0)
        valid_cells2 = np.sum(cyl2 > 0)
        
        if valid_cells1 / cyl1.size < self.min_valid_cells or valid_cells2 / cyl2.size < self.min_valid_cells:
            return 0.0
        
        # Find best alignment by rotating cylinder2
        best_score = 0
        for rot in range(cyl2.shape[2]):
            rotated_cyl2 = np.roll(cyl2, rot, axis=2)
            score = self._euclidean_distance(cyl1, rotated_cyl2)
            if score > best_score:
                best_score = score
                
        return best_score

    def _euclidean_distance(self, cyl1: np.ndarray, cyl2: np.ndarray) -> float:
        """Calculate normalized Euclidean distance between cylinders"""
        mask = (cyl1 > 0) & (cyl2 > 0)
        if np.sum(mask) / cyl1.size < self.min_matchable_cells:
            return 0.0
            
        valid1 = cyl1[mask]
        valid2 = cyl2[mask]
        
        norm = np.linalg.norm(valid1) + np.linalg.norm(valid2)
        if norm == 0:
            return 0.0
            
        return 1 - np.linalg.norm(valid1 - valid2) / norm

    def _select_top_matches(self, lsm: np.ndarray, top_n: int = 30) -> List[Tuple[int, int, float]]:
        """Select top matches from LSM"""
        flat_indices = np.argsort(lsm, axis=None)[::-1]
        rows, cols = np.unravel_index(flat_indices, lsm.shape)
        
        top_matches = []
        for i in range(min(top_n, len(flat_indices))):
            if lsm[rows[i], cols[i]] > 0:
                top_matches.append((rows[i], cols[i], lsm[rows[i], cols[i]]))
                
        return top_matches

    def _apply_relaxation(self, matches: List[Tuple[int, int, float]], num_iter: int = 4) -> float:
        """Apply relaxation to penalize incompatible matches"""
        if not matches:
            return 0.0
            
        scores = np.array([m[2] for m in matches])
        
        for _ in range(num_iter):
            compatibilities = np.ones(len(matches))
            
            for i in range(len(matches)):
                for j in range(i+1, len(matches)):
                    # Simplified compatibility check
                    compat = 1 - abs(scores[i] - scores[j])
                    compatibilities[i] *= compat
                    compatibilities[j] *= compat
                    
            scores *= compatibilities
            
        # Select top matches after relaxation
        top_scores = sorted(scores, reverse=True)[:10]
        return np.mean(top_scores) if top_scores else 0.0

# ==================== Factory Classes ====================
class FeatureExtractorFactory:
    @staticmethod
    def create_extractor(feature_type: FeatureType) -> IFeatureExtractor:
        return MTCCFeatureExtractor()

class MatcherFactory:
    @staticmethod
    def create_matcher() -> IMatcher:
        return LSSRMatcher()

# ==================== Main System ====================
class FingerprintMatcher:
    def __init__(self):
        self.dataset_loader = FVCDatasetLoader()
        self.enhancer = FingerprintEnhancer()
        self.minutiae_extractor = MinutiaeExtractor()
        self.feature_extractor_factory = FeatureExtractorFactory()
        self.matcher_factory = MatcherFactory()

    def process_fingerprint(self, image: np.ndarray) -> Fingerprint:
        """Process a fingerprint image through the pipeline"""
        fingerprint = Fingerprint(image=image, minutiae=[])
        
        # 1. Enhancement
        enhanced = self.enhancer.enhance(fingerprint)
        
        # 2. Minutiae extraction
        minutiae = self.minutiae_extractor.extract(enhanced)
        enhanced.minutiae = minutiae
        
        return enhanced

    def extract_features(self, fingerprint: Fingerprint, feature_type: FeatureType) -> np.ndarray:
        """Extract features from processed fingerprint"""
        extractor = self.feature_extractor_factory.create_extractor(feature_type)
        return extractor.extract_features(fingerprint, feature_type)

    def match_fingerprints(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Match two sets of fingerprint features"""
        matcher = self.matcher_factory.create_matcher()
        return matcher.match(features1, features2)

# ==================== Example Usage ====================
if __name__ == "__main__":
    # Initialize system
    matcher = FingerprintMatcher()
    
    # Load dataset (in practice would point to FVC dataset)
    dataset = matcher.dataset_loader.load_dataset(R"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002\FVC2002\db1_a")
    
    # Process two fingerprints from the dataset
    fp1 = dataset["101"][0]
    fp2 = dataset["101"][1]  # Same finger, different impression
    
    processed_fp1 = matcher.process_fingerprint(fp1.image)
    processed_fp2 = matcher.process_fingerprint(fp2.image)
    
    # Extract features (using cell-centered orientation as in paper)
    features1 = matcher.extract_features(processed_fp1, FeatureType.CELL_CENTERED_ORIENTATION)
    features2 = matcher.extract_features(processed_fp2, FeatureType.CELL_CENTERED_ORIENTATION)
    
    # Match fingerprints
    score = matcher.match_fingerprints(features1, features2)
    print(f"Matching score: {score:.4f}")