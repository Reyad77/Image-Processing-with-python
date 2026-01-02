
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load binary image (from previous step)
binary = cv2.imread('BWI.jpg', cv2.IMREAD_GRAYSCALE)
if binary is None:
    # If BWI.jpg doesn't exist, create it from leaf.jpg
    image = cv2.imread('leaf.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Define structuring elements (kernels)
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# 1. Opening (erosion followed by dilation) - removes small noise
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_medium, iterations=2)

# 2. Closing (dilation followed by erosion) - fills small holes
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_medium, iterations=2)

# 3. Alternative: Combine both operations
clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_medium, iterations=2)
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_medium, iterations=2)

# 4. Find connected components to get leaf region
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=8)

# Find the largest component (assuming it's the leaf)
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
leaf_mask = np.uint8(labels == largest_label) * 255

# Display results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(binary, cmap='gray')
plt.title('Original Binary (BWI)')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(opening, cmap='gray')
plt.title('After Opening')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(closing, cmap='gray')
plt.title('After Closing')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(clean, cmap='gray')
plt.title('Cleaned Binary')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(leaf_mask, cmap='gray')
plt.title(f'Leaf Mask (Largest Component)\nFound {num_labels-1} regions')
plt.axis('off')

# Show component statistics
plt.subplot(2, 3, 6)
areas = stats[1:, cv2.CC_STAT_AREA]
plt.bar(range(1, len(areas)+1), areas)
plt.title('Component Areas')
plt.xlabel('Component ID')
plt.ylabel('Area (pixels)')

plt.tight_layout()
plt.show()

# Save the leaf mask
cv2.imwrite('leaf_mask.jpg', leaf_mask)
print(f"Found {num_labels} connected components")
print(f"Largest component area: {stats[largest_label, cv2.CC_STAT_AREA]} pixels")
print(f"Leaf mask saved as 'leaf_mask.jpg'")