import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=== TASK 2: Translation Transformation ===")

# Create or load image
img = cv2.imread('bird.jpg')
if img is None:
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
    cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width = img.shape[:2]

# Translation parameters: move 100 pixels right, 50 pixels down
tx, ty = 100, 50

# Create translation matrix
# [1, 0, tx]
# [0, 1, ty]
M_translation = np.float32([[1, 0, tx], [0, 1, ty]])

# Apply translation
translated_img = cv2.warpAffine(img_rgb, M_translation, (width, height))

# Display both images
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image', fontweight='bold')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(translated_img)
plt.title(f'Translated Image\n(x+{tx}, y+{ty})', fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Translation completed: moved {tx} pixels right, {ty} pixels down")
print("Task 2 completed successfully! ✅")