# Task 5: Complete Image Segmentation and Description Pipeline
import os
import sys
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

# skimage imports
try:
    from skimage.feature import graycomatrix, graycoprops
    from skimage import img_as_ubyte
except Exception:
    print("Error: scikit-image is required. Install with `pip install scikit-image`.")
    sys.exit(1)


def complete_pipeline(image_path=None):
    if image_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, 'leaf.jpg')
    """Complete pipeline for leaf image segmentation and feature extraction"""
    
    print("=" * 60)
    print("COMPLETE LEAF IMAGE PROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Read and display image
    print("\nStep 1: Reading image...")
    if not os.path.exists(image_path):
        print(f"Error: image not found at '{image_path}'")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"  Image size: {image.shape}")
    
    # Step 2: Threshold segmentation
    print("\nStep 2: Threshold segmentation...")
    otsu_thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"  Otsu threshold value: {otsu_thresh}")
    
    # Step 3: Morphological filtering
    print("\nStep 3: Morphological filtering...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find largest connected component (leaf)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closing, connectivity=8)
    if num_labels <= 1:
        # no components found (only background)
        print("Warning: no connected components found; using morphological result as mask")
        leaf_mask = closing
    else:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        leaf_mask = np.uint8(labels == largest_label) * 255
        print(f"  Found {num_labels-1} regions")
        print(f"  Leaf area: {stats[largest_label, cv2.CC_STAT_AREA]} pixels")
    
    # Step 4: Extract leaf region
    leaf_region = cv2.bitwise_and(gray, gray, mask=leaf_mask)
    leaf_region_color = cv2.bitwise_and(image_rgb, image_rgb, mask=leaf_mask)
    
    # Step 5: Calculate features
    print("\nStep 4: Calculating features...")
    
    # Statistical features
    leaf_pixels = leaf_region[leaf_mask > 0]
    if leaf_pixels.size == 0:
        print("Error: no leaf pixels found in mask. Aborting feature calculation.")
        return

    mean_intensity = np.mean(leaf_pixels)
    std_intensity = np.std(leaf_pixels)
    
    # Texture features using GLCM
    leaf_region_uint8 = img_as_ubyte(leaf_region)
    glcm = graycomatrix(leaf_region_uint8, 
                        distances=[1], 
                        angles=[0],
                        levels=256, 
                        symmetric=True, 
                        normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    # Display all results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    titles = ['Original Image', 'Binary (BWI)', 'Leaf Mask',
              'Leaf Region', 'Histogram', 'Features']
    
    images = [image_rgb, binary, leaf_mask,
              leaf_region_color, None, None]
    
    for i in range(6):
        ax = axes[i//3, i%3]
        
        if i < 4:
            if i == 3:  # Leaf region color
                ax.imshow(images[i])
            else:
                ax.imshow(images[i], cmap='gray' if i in [1, 2] else None)
            ax.set_title(titles[i])
            ax.axis('off')
        
        elif i == 4:  # Histogram
            ax.hist(leaf_pixels, bins=50, color='green', alpha=0.7)
            ax.set_title('Leaf Region Histogram')
            ax.set_xlabel('Intensity')
            ax.set_ylabel('Frequency')
            ax.axvline(mean_intensity, color='red', linestyle='--', 
                      label=f'Mean: {mean_intensity:.1f}')
            ax.legend()
        
        else:  # Features text
            ax.axis('off')
            features_text = f"""STATISTICAL FEATURES:
Mean Intensity: {mean_intensity:.2f}
Std Intensity: {std_intensity:.2f}
Min/Max: {leaf_pixels.min():.0f}/{leaf_pixels.max():.0f}
Area: {len(leaf_pixels)} pixels

TEXTURE FEATURES:
Contrast: {contrast:.4f}
Homogeneity: {homogeneity:.4f}
Energy: {energy:.4f}

IMAGE INFO:
Size: {image.shape[1]}x{image.shape[0]}
Channels: {image.shape[2]}
Format: {image.dtype}
"""
            ax.text(0.1, 0.5, features_text, fontsize=10,
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('Leaf Image Segmentation and Feature Extraction', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Save results
    cv2.imwrite('results_binary.jpg', binary)
    cv2.imwrite('results_mask.jpg', leaf_mask)
    cv2.imwrite('results_leaf_region.jpg', cv2.cvtColor(leaf_region_color, cv2.COLOR_RGB2BGR))
    
    print("\nStep 5: Saving results...")
    print("  Binary image saved as: 'results_binary.jpg'")
    print("  Leaf mask saved as: 'results_mask.jpg'")
    print("  Leaf region saved as: 'results_leaf_region.jpg'")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Leaf area: {len(leaf_pixels)} pixels ({100*len(leaf_pixels)/(gray.size):.1f}% of image)")
    print(f"Mean leaf intensity: {mean_intensity:.1f}")
    print(f"Leaf texture contrast: {contrast:.4f}")
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run complete leaf processing pipeline')
    parser.add_argument('image', nargs='?', help='Path to input image (optional)')
    args = parser.parse_args()
    complete_pipeline(args.image)