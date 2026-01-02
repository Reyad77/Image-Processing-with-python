import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=== TASK 3: Rotation Transformation ===")

# Create or load image
img = cv2.imread('bird.jpg')
if img is None:
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
    cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width = img.shape[:2]

# Rotation parameters
center = (width // 2, height // 2)  # Rotate around center
angle = 45  # 45 degrees counterclockwise
scale = 1.0  # No scaling

# Create rotation matrix
M_rotation = cv2.getRotationMatrix2D(center, angle, scale)

# Apply rotation
rotated_img = cv2.warpAffine(img_rgb, M_rotation, (width, height))

# Display both images
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image', fontweight='bold')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(rotated_img)
plt.title(f'Rotated Image\n({angle}° counterclockwise)', fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Rotation completed: {angle}° around center point {center}")
print("Task 3 completed successfully!")