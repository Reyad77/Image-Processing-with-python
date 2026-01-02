import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read and preprocess image
img = cv2.imread("D:\\Study\\sem 5\\Image processing\\projects\\Experiment 4\\images\\shape.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Gaussian blur for noise reduction

# Step 2: Canny edge detection
canny_edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Step 3: Morphological filtering (close operation to repair edges)
kernel = np.ones((3, 3), np.uint8)
morph_closed = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

# Step 4: Region filling
contours, _ = cv2.findContours(morph_closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
fill_mask = np.zeros_like(canny_edges)
for cnt in contours:
    cv2.drawContours(fill_mask, [cnt], -1, 255, thickness=cv2.FILLED)

# Step 5: Post-process and extract segmented result
fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove small noise
segmented_img = cv2.bitwise_and(img_rgb, img_rgb, mask=fill_mask)

# Visualization layout
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1), plt.imshow(img_rgb), plt.title("Original Image"), plt.axis("off")
plt.subplot(2, 3, 2), plt.imshow(blurred), plt.title("Gaussian Blur"), plt.axis("off")
plt.subplot(2, 3, 3), plt.imshow(canny_edges, cmap="gray"), plt.title("Canny Edges"), plt.axis("off")
plt.subplot(2, 3, 4), plt.imshow(morph_closed, cmap="gray"), plt.title("Morphological Close"), plt.axis("off")
plt.subplot(2, 3, 5), plt.imshow(fill_mask, cmap="gray"), plt.title("Region Filled Mask"), plt.axis("off")
plt.subplot(2, 3, 6), plt.imshow(segmented_img), plt.title("Segmented Result"), plt.axis("off")
plt.tight_layout()
plt.show()

# Save results
cv2.imwrite("D:\\Study\\sem 5\\Image processing\\projects\\Experiment 4\\output\\canny_edges.png", canny_edges)
cv2.imwrite("D:\\Study\\sem 5\\Image processing\\projects\\Experiment 4\\output\\segmented_result.png", cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))
