
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse


# Allow passing an image path as an optional argument; otherwise use the default path
parser = argparse.ArgumentParser(description='Threshold segmentation for leaf image')
parser.add_argument('image', nargs='?', help='D:\Study\sem 5\Image processing\projects\Experiment 3\Experiment 3\leaf.jpg')
args = parser.parse_args()

default_path = r'D:\Study\sem 5\Image processing\projects\Experiment 3\Experiment 3\leaf.jpg'
image_path = args.image if args.image else default_path

if not os.path.exists(image_path):
    print(f"Error: image not found at {image_path}")
    sys.exit(1)

image = cv2.imread(image_path)
if image is None:
    print(f"Error: cv2 failed to read image at {image_path}")
    sys.exit(1)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Method 1: Otsu's automatic thresholding
otsu_thresh, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Method 2: Adaptive thresholding
# Use the correct OpenCV constant name: ADAPTIVE_THRESH_GAUSSIAN_C
binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

# Method 3: Manual threshold (adjust based on your image)
_, binary_manual = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Display results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Grayscale')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(binary_otsu, cmap='gray')
plt.title(f'Otsu Thresholding')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(binary_adaptive, cmap='gray')
plt.title('Adaptive Thresholding')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(binary_manual, cmap='gray')
plt.title('Manual Threshold (127)')
plt.axis('off')

# Compare histograms
plt.subplot(2, 3, 5)
plt.hist(gray.ravel(), 256, [0, 256])
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Show pixel value distributions
plt.subplot(2, 3, 6)
plt.boxplot([gray.ravel()])
plt.title('Pixel Value Distribution')
plt.ylabel('Intensity')

plt.tight_layout()
plt.show()

# Save the best binary image (using Otsu as default)
cv2.imwrite('BWI.jpg', binary_otsu)
print(f"Otsu threshold value used: {otsu_thresh}")
print(f"Binary image saved as 'BWI.jpg'")


binary = binary_otsu  