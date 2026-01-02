import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=== TASK 4: Scaling Transformation ===")

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

# Scaling parameters
scale_factor = 0.6  # Scale down to 60% of original size

# Apply scaling
scaled_img = cv2.resize(img_rgb, None, fx=scale_factor, fy=scale_factor, 
                       interpolation=cv2.INTER_LINEAR)

# Get new dimensions
new_height, new_width = scaled_img.shape[:2]

# Display both images side by side
plt.figure(figsize=(12, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original image', fontweight='bold')
plt.axis('off')

# Scaled Image
plt.subplot(1, 2, 2)
plt.imshow(scaled_img)
plt.title('Scaled image', fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()


print(f"Scaling completed: {scale_factor}x (from {width}x{height} to {new_width}x{new_height})")
print("Task 4 completed successfully! ")