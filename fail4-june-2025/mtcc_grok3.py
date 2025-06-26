import numpy as np
import cv2
from scipy import signal
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def load_fingerprint_image(image_path):
    """Load and convert fingerprint image to grayscale."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image from {image_path}")
    return img

def normalize_image(img, desired_mean=100, desired_variance=100):
    """Normalize image to a fixed mean and variance."""
    mean = np.mean(img)
    variance = np.var(img)
    if variance == 0:
        return img
    normalized = np.zeros_like(img, dtype=np.float32)
    condition = img > mean
    normalized[condition] = desired_mean + np.sqrt(desired_variance * (img[condition] - mean)**2 / variance)
    normalized[~condition] = desired_mean - np.sqrt(desired_variance * (img[~condition] - mean)**2 / variance)
    return normalized.astype(np.uint8)

def segment_image(img, block_size=16, threshold_factor=0.1):
    """Segment fingerprint image using block-wise variance."""
    h, w = img.shape
    mask = np.zeros_like(img, dtype=np.uint8)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                if np.var(block) > threshold_factor * np.var(img):
                    mask[i:i+block_size, j:j+block_size] = 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def gabor_filter_enhancement(img, orientations, frequencies, block_size=16):
    """Apply Gabor filters to enhance ridge structures."""
    h, w = img.shape
    enhanced = np.zeros_like(img, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                theta = orientations[i//block_size, j//block_size]
                freq = frequencies[i//block_size, j//block_size]
                if freq == 0:
                    freq = 0.1
                gabor_kernel = cv2.getGaborKernel((block_size, block_size), sigma=4.0, theta=theta,
                                                 lambd=1.0/freq, gamma=0.5, psi=0)
                filtered = cv2.filter2D(block, cv2.CV_32F, gabor_kernel)
                enhanced[i:i+block_size, j:j+block_size] = filtered
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def smqt_enhancement(img, L=8):
    """Apply Successive Mean Quantization Transform (SMQT) for enhancement."""
    def quantize(x, mean, L):
        return np.round((2**L - 1) * (x >= mean))
    
    img_float = img.astype(np.float32)
    for _ in range(L):
        mean = np.mean(img_float)
        img_float = quantize(img_float, mean, 1)
        img_float = img_float * (255.0 / (2**L - 1))
    return img_float.astype(np.uint8)

def stft_analysis(img, block_size=14, overlap=6):
    """Perform Short-Time Fourier Transform (STFT) to extract orientation, frequency, and energy."""
    h, w = img.shape
    step = block_size - overlap
    window = signal.windows.hann(block_size)
    window_2d = np.outer(window, window)
    
    orientations = np.zeros((h//step, w//step))
    frequencies = np.zeros((h//step, w//step))
    energies = np.zeros((h//step, w//step))
    
    for i in range(0, h-block_size, step):
        for j in range(0, w-block_size, step):
            block = img[i:i+block_size, j:j+block_size] * window_2d
            f = np.abs(fft2(block))
            energy = np.log(np.sum(f**2) + 1e-10)
            energies[i//step, j//step] = energy
            
            freqs = np.fft.fftfreq(block_size)
            max_freq_idx = np.argmax(np.sum(f, axis=0))
            frequencies[i//step, j//step] = np.abs(freqs[max_freq_idx])
            
            angles = np.arctan2(f.imag, f.real)
            dominant_angle = np.angle(np.sum(np.exp(1j * 2 * angles)))
            orientations[i//step, j//step] = dominant_angle / 2
    
    return orientations, frequencies, energies

def binarize_and_thin(img, threshold=128):
    """Binarize and thin the fingerprint image."""
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    thinned = cv2.ximgproc.thinning(binary)
    return thinned

def extract_minutiae(thinned_img):
    """Extract minutiae (terminations and bifurcations) using contour analysis."""
    contours, _ = cv2.findContours(thinned_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minutiae = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            neighborhood = thinned_img[max(0, y-1):y+2, max(0, x-1):x+2]
            if neighborhood.shape == (3, 3):
                count = np.sum(neighborhood == 255) - 1
                if count == 1 or count == 3:
                    minutiae.append((x, y, 0))  # Angle set to 0 for simplicity
    return minutiae

def create_cylinder(minutia, img_shape, orientation_img, frequency_img, energy_img, R=63, Ns=18, Nd=5):
    """Create Minutia Texture Cylinder Code (MTCC) for a single minutia."""
    x_m, y_m, _ = minutia
    cylinder = np.zeros((Ns, Ns, Nd))
    delta_s = 2 * R / Ns
    delta_d = 2 * np.pi / Nd
    sigma_s = R / 3
    
    for i in range(Ns):
        for j in range(Ns):
            for k in range(Nd):
                d_sk = -np.pi + (k - 0.5) * delta_d
                p_ij = np.array([x_m, y_m]) + delta_s * np.array([
                    np.cos(0) * (i - (Ns + 1) / 2) + np.sin(0) * (j - (Ns + 1) / 2),
                    -np.sin(0) * (i - (Ns + 1) / 2) + np.cos(0) * (j - (Ns + 1) / 2)
                ])
                if 0 <= p_ij[0] < img_shape[1] and 0 <= p_ij[1] < img_shape[0]:
                    spatial_contrib = np.exp(-np.sum((p_ij - [x_m, y_m])**2) / (2 * sigma_s**2))
                    orientation_contrib = np.exp(-((d_sk - orientation_img[int(p_ij[1]//(img_shape[0]//orientation_img.shape[0])), 
                                                                    int(p_ij[0]//(img_shape[1]//orientation_img.shape[1]))])**2) / (2 * (np.pi/6)**2))
                    frequency_contrib = np.exp(-((d_sk - frequency_img[int(p_ij[1]//(img_shape[0]//frequency_img.shape[0])), 
                                                                      int(p_ij[0]//(img_shape[1]//frequency_img.shape[1]))])**2) / (2 * (0.1)**2))
                    energy_contrib = np.exp(-((d_sk - energy_img[int(p_ij[1]//(img_shape[0]//energy_img.shape[0])), 
                                                                int(p_ij[0]//(img_shape[1]//energy_img.shape[1]))])**2) / (2 * (1.0)**2))
                    cylinder[i, j, k] = spatial_contrib * (orientation_contrib + frequency_contrib + energy_contrib) / 3
    return cylinder

def match_cylinders(cyl1, cyl2):
    """Match two cylinders using cosine and sine distance metrics."""
    valid_cells = (cyl1 > 0) & (cyl2 > 0)
    if np.sum(valid_cells) < 10:
        return 0
    cos_dist = 1 - np.abs(np.cos(2 * cyl1[valid_cells]) - np.cos(2 * cyl2[valid_cells])).sum() / (np.sum(cyl1[valid_cells]) + np.sum(cyl2[valid_cells]) + 1e-10)
    sin_dist = 1 - np.abs(np.sin(2 * cyl1[valid_cells]) - np.sin(2 * cyl2[valid_cells])).sum() / (np.sum(cyl1[valid_cells]) + np.sum(cyl2[valid_cells]) + 1e-10)
    return np.sqrt(cos_dist**2 + sin_dist**2) / 2

def compute_eer(scores, labels):
    """Compute Equal Error Rate (EER) from scores and labels."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    return fpr[eer_idx]

def match_fingerprints(img1_path, img2_path):
    """Match two fingerprint images and return a similarity score."""
    img1 = load_fingerprint_image(img1_path)
    img2 = load_fingerprint_image(img2_path)
    
    norm1 = normalize_image(img1)
    norm2 = normalize_image(img2)
    
    mask1 = segment_image(norm1)
    mask2 = segment_image(norm2)
    
    orientations1, frequencies1, energies1 = stft_analysis(norm1)
    orientations2, frequencies2, energies2 = stft_analysis(norm2)
    
    enhanced1 = gabor_filter_enhancement(norm1, orientations1, frequencies1)
    enhanced2 = gabor_filter_enhancement(norm2, orientations2, frequencies2)
    
    smqt1 = smqt_enhancement(enhanced1)
    smqt2 = smqt_enhancement(enhanced2)
    
    thin1 = binarize_and_thin(smqt1)
    thin2 = binarize_and_thin(smqt2)
    
    minutiae1 = extract_minutiae(thin1)
    minutiae2 = extract_minutiae(thin2)
    
    cylinders1 = [create_cylinder(m, img1.shape, orientations1, frequencies1, energies1) for m in minutiae1]
    cylinders2 = [create_cylinder(m, img2.shape, orientations2, frequencies2, energies2) for m in minutiae2]
    
    scores = []
    for cyl1 in cylinders1:
        max_score = max([match_cylinders(cyl1, cyl2) for cyl2 in cylinders2], default=0)
        scores.append(max_score)
    return np.mean(scores) if scores else 0

def visualize_steps(img_path):
    """Visualize intermediate steps of the fingerprint processing pipeline."""
    img = load_fingerprint_image(img_path)
    norm = normalize_image(img)
    mask = segment_image(norm)
    orientations, frequencies, energies = stft_analysis(norm)
    enhanced = gabor_filter_enhancement(norm, orientations, frequencies)
    smqt = smqt_enhancement(enhanced)
    thin = binarize_and_thin(smqt)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 1].imshow(norm, cmap='gray')
    axes[0, 1].set_title('Normalized Image')
    axes[0, 2].imshow(mask, cmap='gray')
    axes[0, 2].set_title('Segmentation Mask')
    axes[0, 3].imshow(orientations, cmap='hsv')
    axes[0, 3].set_title('Orientation Image')
    axes[1, 0].imshow(frequencies, cmap='viridis')
    axes[1, 0].set_title('Frequency Image')
    axes[1, 1].imshow(energies, cmap='plasma')
    axes[1, 1].set_title('Energy Image')
    axes[1, 2].imshow(smqt, cmap='gray')
    axes[1, 2].set_title('SMQT Enhanced')
    axes[1, 3].imshow(thin, cmap='gray')
    axes[1, 3].set_title('Thinned Image')
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to demonstrate MTCC pipeline."""
    img1_path = R'C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2000\FVC2000\Db1_a\1_1.tif'
    img2_path = R'C:\Users\Precision\Onus\Data\FVC-DataSets\DataSets\FVC2000\FVC2000\Db1_a\1_2.tif'
    
    
    # Visualize processing steps for first image
    visualize_steps(img1_path)
    
    # Match two fingerprints
    score = match_fingerprints(img1_path, img2_path)
    print(f"Matching score: {score:.4f}")
    
    # Simulate EER calculation (requires a dataset)
    scores = [score, 0.8, 0.3, 0.9, 0.2]  # Example scores
    labels = [1, 1, 0, 1, 0]  # Example labels (1 for genuine, 0 for impostor)
    eer = compute_eer(scores, labels)
    print(f"Equal Error Rate: {eer:.4f}")

if __name__ == "__main__":
    main()