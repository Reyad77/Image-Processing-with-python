import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Reuse segmentation and feature extraction (from Task 2)
img_path = "D:\\Study\\sem 5\\Image processing\\projects\\Experiment 4\\images\\shape.png"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Error: 'shape.png' not found at path: {img_path}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Segmentation (reused)
blurred = cv2.GaussianBlur(img, (5, 5), 0)
canny_edges = cv2.Canny(blurred, 50, 150)
kernel = np.ones((3, 3), np.uint8)
morph_closed = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
contours_segment, _ = cv2.findContours(morph_closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
fill_mask = np.zeros_like(canny_edges)
for cnt in contours_segment:
    cv2.drawContours(fill_mask, [cnt], -1, 255, cv2.FILLED)
fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Feature extraction (reused)
contours, _ = cv2.findContours(fill_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
feature_list = []
min_contour_area = 100
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < min_contour_area:
        continue
    perimeter = cv2.arcLength(cnt, closed=True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h) if h != 0 else 0
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    approx = cv2.approxPolyDP(cnt, epsilon=0.02*perimeter, closed=True)
    approx_vertex_count = len(approx)
    roi = img[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_roi, maxCorners=20, qualityLevel=0.01, minDistance=10)
    corner_count = len(corners) if corners is not None else 0
    mask_roi = fill_mask[y:y+h, x:x+w]
    mean_r = np.mean(img_rgb[y:y+h, x:x+w, 0][mask_roi == 255])
    mean_g = np.mean(img_rgb[y:y+h, x:x+w, 1][mask_roi == 255])
    mean_b = np.mean(img_rgb[y:y+h, x:x+w, 2][mask_roi == 255])
    feature_list.append({
        "contour": cnt, "bounding_rect": (x, y, w, h), "circularity": circularity,
        "aspect_ratio": aspect_ratio, "approx_vertices": approx_vertex_count,
        "corner_count": corner_count, "color_mean": (mean_r, mean_g, mean_b)
    })

# Step 2: Rule-based shape classification
def classify_shape(feat):
    if feat["circularity"] >= 0.85:  # High circularity = Circle
        return "Circle"
    elif feat["approx_vertices"] == 3:  # 3 vertices = Triangle
        return "Triangle"
    elif feat["approx_vertices"] == 4 and 0.9 <= feat["aspect_ratio"] <= 1.1:  # 4 vertices + square aspect ratio
        return "Square"
    elif feat["approx_vertices"] == 4:  # 4 vertices + non-square aspect ratio = Rectangle
        return "Rectangle"
    elif feat["approx_vertices"] == 5:  # 5 vertices = Pentagon
        return "Pentagon"
    elif feat["approx_vertices"] >= 6:  # ≥6 vertices = Polygon
        return "Polygon"
    else:
        return "Unknown"

# Color mapping for visualization
class_color_map = {
    "Circle": (255, 0, 0), "Triangle": (0, 255, 0), "Square": (0, 0, 255),
    "Rectangle": (255, 255, 0), "Pentagon": (255, 0, 255), "Polygon": (0, 255, 255),
    "Unknown": (128, 128, 128)
}

# Step 3: Visualize and output results
classification_results = []
vis_img = img_rgb.copy()
for idx, feat in enumerate(feature_list):
    shape_type = classify_shape(feat)
    classification_results.append(shape_type)
    x, y, w, h = feat["bounding_rect"]
    color = class_color_map[shape_type]
    # Draw contour and bounding box
    cv2.drawContours(vis_img, [feat["contour"]], -1, color, 3)
    cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 0), 1)
    # Draw labeled background and text
    label = f"{shape_type} ({idx+1})"
    cv2.rectangle(vis_img, (x, y-30), (x + len(label)*15, y), color, -1)
    cv2.putText(vis_img, label, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Print classification summary
print("=== Shape Classification Summary ===")
for idx, shape_type in enumerate(classification_results):
    print(f"Shape {idx+1}: {shape_type}")

# Visualization layout
plt.figure(figsize=(14, 10))
plt.subplot(1, 2, 1), plt.imshow(img_rgb), plt.title("Original Image"), plt.axis("off")
plt.subplot(1, 2, 2), plt.imshow(vis_img), plt.title("Shape Classification Result"), plt.axis("off")
plt.tight_layout()
plt.show()

# Save result
cv2.imwrite("D:\\Study\\sem 5\\Image processing\\projects\\Experiment 4\\output\\task3_shape_classification.png", cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
