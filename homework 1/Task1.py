import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=== TASK 1: Reading an Image ===")

# Try to read an image file
img = cv2.imread('bird.jpg')

if img is None:
    print("bird.jpg not found. Creating a synthetic image...")
    # Create a synthetic image
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    # Draw a blue rectangle
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
    # Draw a green circle
    cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)
    # Draw a red triangle
    pts = np.array([[200, 200], [300, 200], [250, 250]], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 255))
    # Add text
    cv2.putText(img, 'TEST IMAGE', (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
else:
    print("bird.jpg loaded successfully!")

# Convert from BGR to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get image dimensions
height, width = img.shape[:2]
print(f"Image size: {width} x {height} pixels")

# Display the image
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.title('Task 1: Original Image', fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

print("Task 1 completed successfully! ✅")