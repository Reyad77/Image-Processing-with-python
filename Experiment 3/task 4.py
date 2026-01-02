
import os
import sys
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

# skimage imports may not be available in some environments
try:
    from skimage.feature import graycomatrix, graycoprops
    from skimage import img_as_ubyte
except Exception:
    print("Error: scikit-image is required. Install with `pip install scikit-image`.")
    sys.exit(1)


# Allow passing image and mask paths (optional)
parser = argparse.ArgumentParser(description='Calculate texture features for a leaf image')
parser.add_argument('-i', '--image', help='Path to the color image (default: ./leaf.jpg)')
parser.add_argument('-m', '--mask', help='Path to the leaf mask (default: ./leaf_mask.jpg)')
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
default_image = os.path.join(script_dir, 'leaf.jpg')
default_mask = os.path.join(script_dir, 'leaf_mask.jpg')

image_path = args.image if args.image else default_image
mask_path = args.mask if args.mask else default_mask

if not os.path.exists(image_path):
    print(f"Error: image not found at '{image_path}'")
    sys.exit(1)
if not os.path.exists(mask_path):
    print(f"Error: leaf mask not found at '{mask_path}'. Run Task 3 first or provide a mask via -m")
    sys.exit(1)

# Read original image and leaf mask
image = cv2.imread(image_path)
leaf_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: failed to read image at '{image_path}'")
    sys.exit(1)
if leaf_mask is None:
    print(f"Error: failed to read mask at '{mask_path}'")
    sys.exit(1)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply mask to get only leaf region
leaf_region = cv2.bitwise_and(gray, gray, mask=leaf_mask)

# Convert to uint8 for GLCM calculation
leaf_region_uint8 = img_as_ubyte(leaf_region)

# Calculate Gray Level Co-occurrence Matrix (GLCM)
# Parameters: distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]
glcm = graycomatrix(
    leaf_region_uint8,
    distances=[1, 2, 3],
    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
    levels=256,
    symmetric=True,
    normed=True,
)

# Calculate texture features
contrast = graycoprops(glcm, 'contrast').mean()
dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
homogeneity = graycoprops(glcm, 'homogeneity').mean()
energy = graycoprops(glcm, 'energy').mean()
correlation = graycoprops(glcm, 'correlation').mean()
asm = graycoprops(glcm, 'ASM').mean()  # Angular Second Moment

# Calculate statistical features
leaf_pixels = leaf_region[leaf_mask > 0]
if leaf_pixels.size == 0:
    print("Error: no leaf pixels found in mask. Check the mask or provide a correct mask file.")
    sys.exit(1)

mean_intensity = np.mean(leaf_pixels)
std_intensity = np.std(leaf_pixels)
median_intensity = np.median(leaf_pixels)
min_intensity = np.min(leaf_pixels)
max_intensity = np.max(leaf_pixels)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original image with mask overlay
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masked_image = image_rgb.copy()
masked_image[leaf_mask == 0] = [0, 0, 0]

axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(leaf_region, cmap='gray')
axes[0, 1].set_title('Leaf Region (Grayscale)')
axes[0, 1].axis('off')

axes[0, 2].imshow(masked_image)
axes[0, 2].set_title('Leaf Region (Color)')
axes[0, 2].axis('off')

# Histogram of leaf region
axes[1, 0].hist(leaf_pixels, bins=50, color='blue', alpha=0.7)
axes[1, 0].set_title('Leaf Region Histogram')
axes[1, 0].set_xlabel('Intensity')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(mean_intensity, color='red', linestyle='--', label=f'Mean: {mean_intensity:.1f}')
axes[1, 0].legend()

# Texture feature values (as bar plot)
features = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM']
values = [contrast, dissimilarity, homogeneity, energy, correlation, asm]

axes[1, 1].bar(features, values, color='green', alpha=0.7)
axes[1, 1].set_title('Texture Features')
axes[1, 1].set_ylabel('Value')
axes[1, 1].tick_params(axis='x', rotation=45)

# Statistical features (text display)
axes[1, 2].axis('off')
stats_text = f"""Statistical Features:
Mean Intensity: {mean_intensity:.2f}
Std Intensity: {std_intensity:.2f}
Median Intensity: {median_intensity:.2f}
Min Intensity: {min_intensity:.2f}
Max Intensity: {max_intensity:.2f}

Texture Features:
Contrast: {contrast:.4f}
Dissimilarity: {dissimilarity:.4f}
Homogeneity: {homogeneity:.4f}
Energy: {energy:.4f}
Correlation: {correlation:.4f}
ASM: {asm:.4f}
"""
axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, 
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Print all features
print("=" * 50)
print("LEAF TEXTURE AND STATISTICAL FEATURES")
print("=" * 50)
print("\nStatistical Features:")
print(f"  Mean Intensity: {mean_intensity:.2f}")
print(f"  Std Intensity: {std_intensity:.2f}")
print(f"  Median Intensity: {median_intensity:.2f}")
print(f"  Min Intensity: {min_intensity:.2f}")
print(f"  Max Intensity: {max_intensity:.2f}")
print(f"  Number of leaf pixels: {len(leaf_pixels)}")

print("\nGLCM-based Texture Features:")
print(f"  Contrast: {contrast:.4f}")
print(f"  Dissimilarity: {dissimilarity:.4f}")
print(f"  Homogeneity: {homogeneity:.4f}")
print(f"  Energy: {energy:.4f}")
print(f"  Correlation: {correlation:.4f}")
print(f"  ASM (Angular Second Moment): {asm:.4f}")
print("=" * 50)

# Save features to file
with open('leaf_features.txt', 'w') as f:
    f.write(stats_text)
print("\nFeatures saved to 'leaf_features.txt'")