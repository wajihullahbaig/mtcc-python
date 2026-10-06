import cv2
import numpy as np
from scipy import signal, ndimage
from scipy.fft import fft2, fftshift
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

def load_image(path):
    """Load grayscale image with OpenCV"""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def normalize(img):
    """Zero-mean, unit variance normalization"""
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / (std + 1e-8)

def segment(img, block_size=16):
    """Block-wise variance segmentation using Otsu thresholding"""
    h, w = img.shape
    mask = np.zeros_like(img)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            if block.size == 0:
                continue
            var = np.var(block)
            mask[i:i+block_size, j:j+block_size] = var
    
    _, binary_mask = cv2.threshold(mask.astype(np.uint8), 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask

def compute_orientation(img, block_size=16):
    """Gradient-based orientation estimation"""
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    h, w = img.shape
    orientation = np.zeros_like(img, dtype=np.float32)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            Gx = sobel_x[i:i+block_size, j:j+block_size]
            Gy = sobel_y[i:i+block_size, j:j+block_size]
            
            if Gx.size == 0 or Gy.size == 0:
                continue
                
            Vx = np.sum(2 * Gx * Gy)
            Vy = np.sum(Gx**2 - Gy**2)
            
            if Vx != 0 or Vy != 0:
                angle = 0.5 * np.arctan2(Vx, Vy)
                orientation[i:i+block_size, j:j+block_size] = angle
    
    return orientation

def gabor_enhance(img, orientation_map, freq=0.1, kernel_size=11):
    """Orientation-adaptive Gabor filtering"""
    enhanced = np.zeros_like(img)
    k = (kernel_size - 1) // 2
    h, w = img.shape
    
    for i in range(k, h-k, kernel_size):
        for j in range(k, w-k, kernel_size):
            theta = orientation_map[i, j]
            kernel = cv2.getGaborKernel(
                (kernel_size, kernel_size), 
                4.0,  # sigma
                theta, 
                1.0/freq,  # wavelength
                0.5,  # gamma
                0,    # psi
                ktype=cv2.CV_32F
            )
            kernel /= np.sum(np.abs(kernel))
            
            block = img[i-k:i+k+1, j-k:j+k+1]
            if block.shape != (kernel_size, kernel_size):
                continue
                
            filtered = cv2.filter2D(block, cv2.CV_32F, kernel)
            enhanced[i-k:i+k+1, j-k:j+k+1] = filtered
    
    return enhanced

def smqt(img, levels=8):
    """Successive Mean Quantization Transform"""
    current = img.astype(np.float32)
    result = np.zeros_like(img, dtype=np.uint8)
    
    for level in range(levels):
        mean_val = np.mean(current)
        bit_plane = (current > mean_val).astype(np.uint8)
        result = (result << 1) | bit_plane
        current = current - mean_val
    
    return result

def stft_features(img, window=16, overlap=8):
    """Compute STFT-based texture features (orientation, frequency, energy)"""
    h, w = img.shape
    step = window - overlap
    y_steps = (h - window) // step + 1
    x_steps = (w - window) // step + 1
    
    # Initialize feature grids
    orientation_grid = np.zeros((y_steps, x_steps))
    freq_grid = np.zeros((y_steps, x_steps))
    energy_grid = np.zeros((y_steps, x_steps))
    
    # Compute features per block
    for i in range(y_steps):
        for j in range(x_steps):
            y = i * step
            x = j * step
            block = img[y:y+window, x:x+window]
            
            # Compute 2D FFT
            F = fftshift(fft2(block))
            mag = np.abs(F)
            
            # Energy feature
            energy = np.sum(mag**2)
            energy_grid[i, j] = energy
            
            # Frequency and orientation
            if energy > 1e-8:
                # Find dominant frequency component
                max_idx = np.unravel_index(np.argmax(mag), mag.shape)
                cy, cx = max_idx[0] - window//2, max_idx[1] - window//2
                
                # Convert to polar coordinates
                freq = np.sqrt(cy**2 + cx**2) / window
                angle = np.arctan2(cy, cx) % np.pi
                
                freq_grid[i, j] = freq
                orientation_grid[i, j] = angle
    
    # Create interpolators
    y_coords = np.arange(0, h, step) + window//2
    x_coords = np.arange(0, w, step) + window//2
    y_coords = y_coords[:y_steps]
    x_coords = x_coords[:x_steps]
    
    interp_ori = RectBivariateSpline(y_coords, x_coords, orientation_grid)
    interp_freq = RectBivariateSpline(y_coords, x_coords, freq_grid)
    interp_energy = RectBivariateSpline(y_coords, x_coords, energy_grid)
    
    # Create full-resolution maps
    yy, xx = np.mgrid[0:h, 0:w]
    orientation_map = interp_ori(yy, xx, grid=False)
    freq_map = interp_freq(yy, xx, grid=False)
    energy_map = interp_energy(yy, xx, grid=False)
    
    return orientation_map, freq_map, energy_map

def binarize_thin(binary):
    from skimage.morphology import skeletonize

    # Ensure the input is a boolean array for skeletonize
    binary_bool = (binary > 0)
    skeleton = skeletonize(binary_bool)
    # Convert boolean skeleton back to uint8 (0 or 255)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    return skeleton_uint8, binary_bool

def extract_minutiae(skeleton):
    """Minutiae extraction using Crossing Number method"""
    skeleton = skeleton.astype(np.uint8) // 255
    h, w = skeleton.shape
    minutiae = []
    
    # Define 8-connectivity neighborhood
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y, x] == 0:
                continue
                
            # Calculate crossing number
            neighbors = skeleton[y-1:y+2, x-1:x+2]
            cn = np.sum(neighbors * kernel)
            
            # Filter valid minutiae types
            if cn == 1:   # Ridge ending
                minutiae.append((x, y, 0))
            elif cn == 3: # Bifurcation
                minutiae.append((x, y, 1))
    
    return minutiae

def create_cylinders(minutiae, texture_maps, radius=70, n_s=10, n_d=5):
    """Generate MTCC descriptors with texture features"""
    orientation_map, freq_map, energy_map = texture_maps
    cylinders = []
    
    for mx, my, mtype in minutiae:
        cylinder = np.zeros((n_s, n_s, n_d))
        delta_s = 2 * radius / n_s
        
        for i in range(n_s):
            for j in range(n_s):
                # Calculate relative coordinates
                rx = (i - n_s/2) * delta_s
                ry = (j - n_s/2) * delta_s
                
                # Rotate to minutia orientation
                rot_x = mx + rx
                rot_y = my + ry
                
                # Check bounds
                if (0 <= rot_x < orientation_map.shape[1] and 
                    0 <= rot_y < orientation_map.shape[0]):
                    # Use energy feature (Set 2 approach)
                    val = energy_map[int(rot_y), int(rot_x)]
                    
                    # Distribute across angular bins
                    for k in range(n_d):
                        cylinder[i, j, k] = val
        
        cylinders.append(cylinder.flatten())
    
    return cylinders

def match(cylinders1, cylinders2):
    """Similarity scoring between two sets of cylinders"""
    if len(cylinders1) == 0 or len(cylinders2) == 0:
        return 0.0
    
    scores = []
    for cyl1 in cylinders1:
        min_dist = float('inf')
        for cyl2 in cylinders2:
            dist = np.linalg.norm(cyl1 - cyl2)
            if dist < min_dist:
                min_dist = dist
        scores.append(min_dist)
    
    return 1.0 / (1.0 + np.mean(scores))

def calculate_eer(genuine_scores, impostor_scores):
    """Calculate Equal Error Rate"""
    thresholds = np.linspace(0, 1, 100)
    min_diff = float('inf')
    eer = 0.5
    
    for thresh in thresholds:
        far = np.mean(impostor_scores >= thresh)
        frr = np.mean(genuine_scores < thresh)
        diff = np.abs(far - frr)
        
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2
    
    return eer

def visualize_pipeline(original, *steps):
    """Create 3x3 grid visualization of processing stages"""
    titles = [
        "Original", "Normalized", "Segmented", 
        "Gabor Enhanced", "SMQT", "STFT Energy",
        "Binarized", "Thinned", "Minutiae"
    ]
    
    plt.figure(figsize=(15, 10))
    for i, (img, title) in enumerate(zip([original] + list(steps), titles)):
        plt.subplot(3, 3, i+1)
        
        if i == 5:  # STFT energy map
            plt.imshow(img, cmap='jet')
            plt.colorbar()
        elif i == 8:  # Minutiae plot
            plt.imshow(img[0], cmap='gray')
            for x, y, _ in img[1]:
                plt.plot(x, y, 'ro', markersize=3)
        else:
            plt.imshow(img, cmap='gray')
        
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example test harness for FVC2004
if __name__ == "__main__":
    # Load sample image
    img_path =R"C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2002\FVC2002\DB1_A\1_1.tif"
    img = load_image(img_path)
    
    # Pipeline execution
    norm_img = normalize(img)
    seg_mask = segment(norm_img)
    orientation = compute_orientation(norm_img)
    gabor_img = gabor_enhance(norm_img, orientation)
    smqt_img = smqt(gabor_img)
    stft_ori, stft_freq, stft_energy = stft_features(smqt_img)
    binary, thin_img = binarize_thin(smqt_img)
    minutiae = extract_minutiae(thin_img)
    
    # Create texture maps and cylinders
    texture_maps = (stft_ori, stft_freq, stft_energy)
    cylinders = create_cylinders(minutiae, texture_maps)
    
    # Visualization
    visualize_pipeline(
        img, norm_img, seg_mask, gabor_img, 
        smqt_img, stft_energy, binary, thin_img, (thin_img, minutiae)
    )
    
    # Matching simulation
    genuine_scores = []
    impostor_scores = []
    
    # In real usage, compare multiple images
    # For demonstration:
    match_score = match(cylinders, cylinders)  # Self-match
    print(f"Self-match score: {match_score:.4f}")
    
    # EER simulation
    print("Simulating EER calculation with random scores...")
    genuine = np.random.beta(8, 2, 100)  # Higher scores
    impostor = np.random.beta(2, 8, 100)  # Lower scores
    eer = calculate_eer(genuine, impostor)
    print(f"Simulated EER: {eer:.4f}")