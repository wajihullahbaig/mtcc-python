#!/usr/bin/env python3
"""
Create synthetic fingerprint-like test images for MTCC testing
"""

import numpy as np
import cv2
import os

def create_synthetic_fingerprint(width=256, height=256, ridge_freq=0.15, noise_level=0.05):
    """Create a more realistic synthetic fingerprint-like pattern"""
    
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Create a basic ridge pattern with some curvature
    center_x, center_y = width // 2, height // 2
    
    # Distance from center
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Angle from center
    angle = np.arctan2(y - center_y, x - center_x)
    
    # Create more realistic ridge flow field
    # Add some core and delta patterns
    core_x, core_y = center_x + 30, center_y - 20
    delta_x, delta_y = center_x - 40, center_y + 30
    
    # Distance to singular points
    dist_core = np.sqrt((x - core_x)**2 + (y - core_y)**2)
    dist_delta = np.sqrt((x - delta_x)**2 + (y - delta_y)**2)
    
    # Create orientation field influenced by singular points
    orientation = angle + 0.5 * np.sin(dist_core / 20) - 0.3 * np.cos(dist_delta / 25)
    
    # Create ridge frequency map
    freq_map = ridge_freq * (1 + 0.3 * np.sin(dist / 40))
    
    # Generate ridge pattern following orientation
    ridge_pattern = np.sin(2 * np.pi * freq_map * 
                          (x * np.cos(orientation) + y * np.sin(orientation)))
    
    # Add some ridge endings and bifurcations
    # Create random disruptions that look like minutiae
    np.random.seed(42)  # For reproducible results
    for _ in range(8):  # Add some ridge endings
        mx, my = np.random.randint(50, width-50), np.random.randint(50, height-50)
        mask_radius = np.random.randint(8, 15)
        mask_dist = np.sqrt((x - mx)**2 + (y - my)**2)
        ridge_pattern = np.where(mask_dist < mask_radius, 
                               ridge_pattern * (mask_dist / mask_radius), 
                               ridge_pattern)
    
    # Add noise
    noise = noise_level * np.random.randn(height, width)
    fingerprint = ridge_pattern + noise
    
    # Enhance contrast
    fingerprint = np.tanh(3 * fingerprint)  # Sigmoid-like enhancement
    
    # Normalize to 0-255
    fingerprint = (fingerprint - fingerprint.min()) / (fingerprint.max() - fingerprint.min())
    fingerprint = (fingerprint * 255).astype(np.uint8)
    
    # Add some realistic fingerprint characteristics
    # Create a circular mask to simulate finger boundary
    mask = dist < min(width, height) // 2 - 20
    fingerprint = fingerprint * mask + 128 * (1 - mask)
    
    # Add some pressure variations
    pressure_variation = 0.8 + 0.4 * np.sin(dist / 30) * np.cos(angle * 2)
    fingerprint = np.clip(fingerprint * pressure_variation, 0, 255).astype(np.uint8)
    
    return fingerprint

def create_test_images():
    """Create test fingerprint images"""
    
    os.makedirs('test_images', exist_ok=True)
    
    # Create first fingerprint
    print("Creating test fingerprint 1...")
    finger1 = create_synthetic_fingerprint(width=256, height=256, ridge_freq=0.12, noise_level=0.03)
    cv2.imwrite('test_images/finger1.bmp', finger1)
    
    # Create second fingerprint (slightly different - same finger, different impression)
    print("Creating test fingerprint 2...")
    np.random.seed(43)  # Different seed for variation
    finger2 = create_synthetic_fingerprint(width=256, height=256, ridge_freq=0.12, noise_level=0.04)
    cv2.imwrite('test_images/finger2.bmp', finger2)
    
    # Create third fingerprint (more different - impostor)
    print("Creating test fingerprint 3 (impostor)...")
    np.random.seed(100)  # Very different seed
    finger3 = create_synthetic_fingerprint(width=256, height=256, ridge_freq=0.18, noise_level=0.05)
    cv2.imwrite('test_images/finger3.bmp', finger3)
    
    print("Test images created successfully!")
    print("- finger1.bmp: Base fingerprint")
    print("- finger2.bmp: Similar fingerprint (should match)")
    print("- finger3.bmp: Different fingerprint (impostor)")

if __name__ == "__main__":
    create_test_images() 