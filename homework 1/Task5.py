import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=== TASK 5: Affine Transformation ===")

# Load the bird image
img = cv2.imread('bird.jpg')
if img is None:
    print("Error: bird.jpg not found!")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width = img.shape[:2]

# Affine transformation points
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

M_affine = cv2.getAffineTransform(pts1, pts2)

# Apply affine transformation
affine_img = cv2.warpAffine(img_rgb, M_affine, (width, height))

# Display results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original image', fontweight='bold')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(affine_img)
plt.title('Affine image', fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()

print("Affine transformation completed")
print("Task 5 completed successfully! ")