#!/usr/bin/env python3
"""
Test script for MTCC implementation
"""

import numpy as np
import cv2
from mtcc_implementation import *
import os

def test_mtcc_matching():
    """Test MTCC matching with synthetic fingerprint images"""
    
    print("=== MTCC Fingerprint Recognition Test ===\n")
    
    # Check if test images exist
    if not os.path.exists('test_images/finger1.bmp'):
        print("Test images not found. Please run create_test_images.py first.")
        return
    
    # Load test images
    print("Loading test images...")
    img1 = load_image('test_images/finger1.bmp')
    img2 = load_image('test_images/finger2.bmp')
    img3 = load_image('test_images/finger3.bmp')
    
    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    print(f"Image 3 shape: {img3.shape}")
    
    # Process first image (template)
    print("\n--- Processing Template Image (finger1) ---")
    
    # Normalize
    norm1 = normalize(img1)
    print("✓ Normalization completed")
    
    # Segment
    seg1 = segment(norm1)
    print("✓ Segmentation completed")
    
    # Estimate orientation and frequency maps
    orientation_map1, freq_map1 = estimate_orientation_frequency(seg1)
    print("✓ Orientation and frequency estimation completed")
    
    # Gabor enhancement
    enhanced1 = gabor_enhance(seg1, orientation_map1, freq_map1)
    print("✓ Gabor enhancement completed")
    
    # SMQT
    smqt1 = smqt(enhanced1)
    print("✓ SMQT completed")
    
    # STFT features
    stft1 = stft_features(smqt1)
    print("✓ STFT features extracted")
    
    # Binarize and thin (use enhanced image instead of SMQT for better results)
    binary1 = binarize_thin(enhanced1)
    print("✓ Binarization and thinning completed")
    
    # Extract minutiae
    minutiae1 = extract_minutiae(binary1)
    print(f"✓ Minutiae extracted: {len(minutiae1)} points")
    
    # Create MTCC cylinders
    cylinders1 = create_cylinders(minutiae1, stft1)
    print(f"✓ MTCC cylinders created: {len(cylinders1)} cylinders")
    
    # Process second image (genuine match)
    print("\n--- Processing Query Image 1 (finger2 - should match) ---")
    
    norm2 = normalize(img2)
    seg2 = segment(norm2)
    orientation_map2, freq_map2 = estimate_orientation_frequency(seg2)
    enhanced2 = gabor_enhance(seg2, orientation_map2, freq_map2)
    smqt2 = smqt(enhanced2)
    stft2 = stft_features(smqt2)
    binary2 = binarize_thin(enhanced2)
    minutiae2 = extract_minutiae(binary2)
    cylinders2 = create_cylinders(minutiae2, stft2)
    
    print(f"✓ Processing completed: {len(minutiae2)} minutiae, {len(cylinders2)} cylinders")
    
    # Match cylinders
    score12 = match(cylinders1, cylinders2)
    print(f"✓ Matching score (finger1 vs finger2): {score12:.4f}")
    
    # Process third image (impostor)
    print("\n--- Processing Query Image 2 (finger3 - impostor) ---")
    
    norm3 = normalize(img3)
    seg3 = segment(norm3)
    orientation_map3, freq_map3 = estimate_orientation_frequency(seg3)
    enhanced3 = gabor_enhance(seg3, orientation_map3, freq_map3)
    smqt3 = smqt(enhanced3)
    stft3 = stft_features(smqt3)
    binary3 = binarize_thin(enhanced3)
    minutiae3 = extract_minutiae(binary3)
    cylinders3 = create_cylinders(minutiae3, stft3)
    
    print(f"✓ Processing completed: {len(minutiae3)} minutiae, {len(cylinders3)} cylinders")
    
    # Match cylinders
    score13 = match(cylinders1, cylinders3)
    print(f"✓ Matching score (finger1 vs finger3): {score13:.4f}")
    
    # Self-match test
    score11 = match(cylinders1, cylinders1)
    print(f"✓ Self-matching score (finger1 vs finger1): {score11:.4f}")
    
    # Results summary
    print("\n=== RESULTS SUMMARY ===")
    print(f"Template vs Similar (finger1 vs finger2):  {score12:.4f}")
    print(f"Template vs Impostor (finger1 vs finger3): {score13:.4f}")
    print(f"Template vs Self (finger1 vs finger1):     {score11:.4f}")
    
    # Interpretation
    print("\n=== INTERPRETATION ===")
    if score11 > score12 > score13:
        print("✓ CORRECT: Self-match > Genuine match > Impostor match")
        print("✓ The MTCC system is working correctly!")
    else:
        print("⚠ Results may need investigation:")
        if score12 <= score13:
            print("  - Genuine match score is not higher than impostor")
        if score11 <= score12:
            print("  - Self-match score is not the highest")
    
    # Visualize the pipeline for the first image
    print("\n--- Creating Visualization ---")
    try:
        # Use the built-in process_fingerprint function for visualization
        _, minutiae_overlay = process_fingerprint('test_images/finger1.bmp', visualize=False)
        cv2.imwrite('test_images/finger1_minutiae.png', minutiae_overlay)
        print("✓ Minutiae visualization saved as 'test_images/finger1_minutiae.png'")
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")
    
    print("\n=== TEST COMPLETED ===")

if __name__ == "__main__":
    test_mtcc_matching() 