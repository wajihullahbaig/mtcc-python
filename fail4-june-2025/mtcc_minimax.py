import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import morphology
from scipy.fftpack import fft2, fftshift
from scipy.spatial.distance import cdist

# 1. Image Loading
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found")
    return img

# 2. Normalization
def normalize_image(img):
    M = np.mean(img)
    VAR = np.var(img)
    M0, VAR0 = 100, 100
    normalized = np.zeros_like(img, dtype=np.float32)
    mask = img > M
    normalized[mask] = M0 + np.sqrt((VAR0 * (img[mask] - M)**2)/VAR)
    normalized[~mask] = M0 - np.sqrt((VAR0 * (img[~mask] - M)**2)/VAR)
    return np.clip(normalized, 0, 255).astype(np.uint8)

# 3. Segmentation (Otsu + Morphology)
def segment_image(img):
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 4. Gabor Filtering
def apply_gabor(img, ksize=31, sigma=4.0, theta=np.pi/4):
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, 10.0, 0.5, 0)
    return cv2.filter2D(img, cv2.CV_32F, kernel)

# 5. SMQT Implementation
def smqt(img, levels=3):
    img = img.astype(np.float32)
    for _ in range(levels):
        mean = np.mean(img)
        img = np.where(img > mean, img - mean, 0)
    return np.clip(img, 0, 255).astype(np.uint8)

# 6. STFT Analysis
def stft_analysis(img, window_size=14, overlap=6):
    h, w = img.shape
    stride = window_size - overlap
    orientation = np.zeros((h, w))
    frequency = np.zeros((h, w))
    energy = np.zeros((h, w))
    
    for i in range(0, h - window_size + 1, stride):
        for j in range(0, w - window_size + 1, stride):
            patch = img[i:i+window_size, j:j+window_size].astype(np.float32)
            patch -= np.mean(patch)  # Remove DC
            
            # FFT and shift
            f = np.log(np.abs(fftshift(fft2(patch))) + 1e-6)
            energy[i:i+window_size, j:j+window_size] += f
            
            # Orientation calculation using gradients
            gy, gx = np.gradient(patch)
            gy = cv2.GaussianBlur(gy, (5,5), 0)
            gx = cv2.GaussianBlur(gx, (5,5), 0)
            orientation_patch = 0.5 * np.arctan2(2*np.sum(gx*gy), np.sum(gx**2 - gy**2))
            orientation[i:i+window_size, j:j+window_size] += orientation_patch
            
            # Frequency from FFT peak
            fft_mag = np.abs(fft2(patch))
            cy, cx = np.unravel_index(np.argmax(fft_mag[1:window_size//2, 1:window_size//2]), 
                                     fft_mag[1:window_size//2, 1:window_size//2].shape)
            frequency[i:i+window_size, j:j+window_size] += cy/window_size
    
    # Normalize accumulated values
    orientation /= (energy > 0).sum()
    frequency /= (energy > 0).sum()
    energy = np.log(energy + 1e-6)
    energy_normalized = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX)
    return orientation, frequency, energy_normalized.astype(np.uint8)
    
# 7. Binarization & Thinning
def binarize_and_thin(img):
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return morphology.thin(binary / 255).astype(np.uint8) * 255

# 8. Minutiae Extraction (Crossing Number)
def extract_minutiae(thinned):
    thinned = thinned.astype(np.uint8)
    cn = np.zeros_like(thinned)
    rows, cols = thinned.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if thinned[i,j] == 0:
                continue
            vals = [
                thinned[i-1,j-1], thinned[i-1,j], thinned[i-1,j+1],
                thinned[i,j+1], thinned[i+1,j+1], thinned[i+1,j],
                thinned[i+1,j-1], thinned[i,j-1], thinned[i-1,j-1]
            ]
            cn_val = sum(np.abs(np.diff(vals)))
            if cn_val == 2:
                cn[i,j] = 255
    points = np.column_stack(np.where(cn > 0))
    return [(x,y) for y,x in points]

# 9. Cylinder Creation (Simplified MCC)
def create_cylinders(minutiae, img_shape, radius=65):
    cylinders = []
    for m in minutiae:
        x, y = m
        cylinder = {
            'position': (x,y),
            'cells': {},
            'features': {
                'orientation': np.zeros((18,5)),
                'frequency': np.zeros((18,5)),
                'energy': np.zeros((18,5))
            }
        }
        # Simplified cell creation
        for i in range(18):  # Angular divisions
            for j in range(5):  # Radial divisions
                # Calculate cell position (simplified)
                cylinder['cells'][(i,j)] = []
        cylinders.append(cylinder)
    return cylinders

# 10. Matching (Local Similarity Sort)
def match_cylinders(cylinders1, cylinders2):
    scores = []
    for c1 in cylinders1:
        for c2 in cylinders2:
            # Simplified distance calculation
            pos1 = np.array(c1['position'])
            pos2 = np.array(c2['position'])
            dist = np.linalg.norm(pos1 - pos2)
            if dist < 20:  # Threshold for match
                scores.append(1 - dist/20)
    return np.mean(scores) if scores else 0

# 11. EER Calculation
def calculate_eer(genuine_scores, impostor_scores):
    thresholds = np.linspace(0, 1, 100)
    far = np.zeros_like(thresholds)
    frr = np.zeros_like(thresholds)
    
    for i, t in enumerate(thresholds):
        far[i] = np.mean([s >= t for s in impostor_scores])
        frr[i] = 1 - np.mean([s >= t for s in genuine_scores])
    
    eer = thresholds[np.argmin(np.abs(far - frr))]
    return eer

# 12. Visualization
def visualize_steps(img_path):
    steps = [
        ("Original", lambda x: x),
        ("Normalized", normalize_image),
        ("Segmented", segment_image),
        ("Gabor", lambda x: apply_gabor(x, theta=0)),
        ("SMQT", smqt),
        ("STFT Energy", lambda x: stft_analysis(x)[2]),
        ("Binarized", lambda x: cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("Thinned", lambda x: morphology.thin(x/255).astype(np.uint8)*255)
    ]
    
    fig, axs = plt.subplots(2, 4, figsize=(15,8))
    img = load_image(img_path)
    
    for i, (name, func) in enumerate(steps):
        ax = axs[i//4, i%4]
        current = img.copy() if i == 0 else processed
        processed = func(current) if i > 0 else img
        ax.imshow(processed, cmap='gray')
        ax.set_title(name)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage with FVC dataset
if __name__ == "__main__":
    # Example path - replace with actual FVC image path
    img1_path = R'C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2000\FVC2000\Db1_a\1_1.tif'
    img1_path = R'C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2000\FVC2000\Db1_a\1_2.tif'
    
    visualize_steps(img1_path)
    
    # Example matching
    img1 = load_image(img1_path)
    img2 = load_image(img1_path)  # Different impression
    
    # Process both images
    norm1 = normalize_image(img1)
    norm2 = normalize_image(img2)
    seg1 = segment_image(norm1)
    seg2 = segment_image(norm2)
    gabor1 = apply_gabor(norm1)
    gabor2 = apply_gabor(norm2)
    stft1 = stft_analysis(gabor1)
    stft2 = stft_analysis(gabor2)
    minutiae1 = extract_minutiae(binarize_and_thin(gabor1))
    minutiae2 = extract_minutiae(binarize_and_thin(gabor2))
    cylinders1 = create_cylinders(minutiae1, img1.shape)
    cylinders2 = create_cylinders(minutiae2, img2.shape)
    score = match_cylinders(cylinders1, cylinders2)
    print(f"Matching Score: {score:.2f}")