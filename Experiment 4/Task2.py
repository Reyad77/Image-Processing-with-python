import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Internal segmentation (reused from Task 1)
img_path = "D:\\Study\\sem 5\\Image processing\\projects\\Experiment 4\\images\\shape.png"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Error: 'shape.png' not found at path: {img_path}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blurred = cv2.GaussianBlur(img, (5, 5), 0)
canny_edges = cv2.Canny(blurred, 50, 150)
kernel = np.ones((3, 3), np.uint8)
morph_closed = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

contours_segment, _ = cv2.findContours(morph_closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
fill_mask = np.zeros_like(canny_edges)
for cnt in contours_segment:
    cv2.drawContours(fill_mask, [cnt], -1, 255, cv2.FILLED)
fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Step 2: Extract features for each valid contour
contours, _ = cv2.findContours(fill_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
feature_list = []
min_contour_area = 100  # Filter tiny noise

for idx, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < min_contour_area:
        continue

    # Shape features
    perimeter = cv2.arcLength(cnt, closed=True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h) if h != 0 else 0
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

    # Boundary features (corners + vertex count)
    roi = img[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_roi, maxCorners=20, qualityLevel=0.01, minDistance=10)
    corner_count = len(corners) if corners is not None else 0
    approx = cv2.approxPolyDP(cnt, epsilon=0.02*perimeter, closed=True)
    approx_vertex_count = len(approx)

    # Color features (mean RGB)
    mask_roi = fill_mask[y:y+h, x:x+w]
    mean_r = np.mean(img_rgb[y:y+h, x:x+w, 0][mask_roi == 255])
    mean_g = np.mean(img_rgb[y:y+h, x:x+w, 1][mask_roi == 255])
    mean_b = np.mean(img_rgb[y:y+h, x:x+w, 2][mask_roi == 255])

    # Store features
    feature_list.append({
        "contour": cnt, "bounding_rect": (x, y, w, h), "area": area,
        "perimeter": perimeter, "aspect_ratio": aspect_ratio, "circularity": circularity,
        "corner_count": corner_count, "color_mean": (mean_r, mean_g, mean_b),
        "approx_vertices": approx_vertex_count
    })

    # Print features for analysis
    print(f"\n=== Shape {idx+1} Features ===")
    print(f"Area: {area:.2f} | Perimeter: {perimeter:.2f} | Aspect Ratio: {aspect_ratio:.2f}")
    print(f"Circularity: {circularity:.2f} | Corner Count: {corner_count} | Vertex Count: {approx_vertex_count}")
    print(f"Mean RGB: ({mean_r:.0f}, {mean_g:.0f}, {mean_b:.0f})")

# Step 3: Visualize features
vis_img = img_rgb.copy()
for feat in feature_list:
    x, y, w, h = feat["bounding_rect"]
    # Draw contour (green)
    cv2.drawContours(vis_img, [feat["contour"]], -1, (0, 255, 0), 2)
    # Draw bounding box (blue)
    cv2.rectangle(vis_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Draw corners (magenta)
    roi = img[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_roi, maxCorners=20, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        corners_rounded = corners.round().astype(int)
        for corner in corners_rounded:
            cx, cy = corner.ravel()
            cv2.circle(vis_img, (x+cx, y+cy), 5, (255, 0, 255), -1)

# Visualization layout
plt.figure(figsize=(16, 12))
plt.subplot(2, 2, 1), plt.imshow(img_rgb), plt.title("Original Image"), plt.axis("off")
plt.subplot(2, 2, 2), plt.imshow(canny_edges, cmap="gray"), plt.title("Canny Edges"), plt.axis("off")
plt.subplot(2, 2, 3), plt.imshow(fill_mask, cmap="gray"), plt.title("Segmented Mask"), plt.axis("off")
plt.subplot(2, 2, 4), plt.imshow(vis_img), plt.title("Feature Visualization (Contours: Green | Bounding Boxes: Blue | Corners: Magenta)"), plt.axis("off")
plt.tight_layout()
plt.show()

# Save visualization
cv2.imwrite("D:\\Study\\sem 5\\Image processing\\projects\\Experiment 4\\output\\task2_feature_visualization.png", cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
