import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=== TASK 6: Show All Results ===")

# Create or load image
img = cv2.imread('bird.jpg')
if img is None:
    # Create a synthetic image
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue
    cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)  # Green
    cv2.rectangle(img, (200, 200), (300, 250), (0, 0, 255), -1)  # Red

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width = img.shape[:2]

# Apply all transformations (same as individual tasks)
# Task 2: Translation
tx, ty = 100, 50
M_translation = np.float32([[1, 0, tx], [0, 1, ty]])
translated_img = cv2.warpAffine(img_rgb, M_translation, (width, height))

# Task 3: Rotation
center = (width // 2, height // 2)
angle = 45
M_rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_img = cv2.warpAffine(img_rgb, M_rotation, (width, height))

# Task 4: Scaling
scale_factor = 0.6
scaled_img = cv2.resize(img_rgb, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# Task 5: Affine
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M_affine = cv2.getAffineTransform(pts1, pts2)
affine_img = cv2.warpAffine(img_rgb, M_affine, (width, height))

# Display all results in a grid
plt.figure(figsize=(16, 10))

# Task 1: Original Image
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('1. Original Image', fontweight='bold')
plt.axis('off')

# Task 2: Translated Image
plt.subplot(2, 3, 2)
plt.imshow(translated_img)
plt.title('2. Translated Image', fontweight='bold')
plt.axis('off')

# Task 3: Rotated Image
plt.subplot(2, 3, 3)
plt.imshow(rotated_img)
plt.title('3. Rotated Image', fontweight='bold')
plt.axis('off')

# Task 4: Scaled Image
plt.subplot(2, 3, 4)
plt.imshow(scaled_img)
plt.title('4. Scaled Image', fontweight='bold')
plt.axis('off')

# Task 5: Affine Image
plt.subplot(2, 3, 5)
plt.imshow(affine_img)
plt.title('5. Affine Image', fontweight='bold')
plt.axis('off')

# Empty subplot for clean layout
plt.subplot(2, 3, 6)
plt.axis('off')

plt.tight_layout()
plt.show()

print("All tasks displayed successfully!")
print("Task 6 completed successfully!")