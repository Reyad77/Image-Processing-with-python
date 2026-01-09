# Import libraries
from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
from pathlib import Path

def download_model_manually():
    """Download YOLOv8 model manually if automatic download fails"""
    print("Attempting to download YOLOv8 model manually...")
    try:
        import requests
        import urllib.request
        
        model_url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt"
        model_path = "yolov8s.pt"
        
        print(f"Downloading from: {model_url}")
        print("This may take a few moments...")
        
        # Using urllib as it's more reliable
        urllib.request.urlretrieve(model_url, model_path)
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✓ Model downloaded successfully: {file_size:.1f} MB")
            return True
        else:
            print("✗ Model download failed")
            return False
            
    except Exception as e:
        print(f"Manual download failed: {e}")
        print("\nPlease download the model manually:")
        print("1. Go to: https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt")
        print("2. Download 'yolov8s.pt'")
        print("3. Place it in the same folder as this script")
        return False

def load_yolo_model():
    """Load YOLO model with error handling"""
    model_path = "yolov8s.pt"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print("Model file not found. Attempting to download...")
        try:
            # Try automatic download first
            model = YOLO("yolov8s.pt")  # This should trigger download
            print("✓ Model downloaded and loaded successfully")
            return model
        except:
            print("Automatic download failed.")
            if download_model_manually():
                try:
                    model = YOLO(model_path)
                    print("✓ Model loaded from manual download")
                    return model
                except Exception as e:
                    print(f"Failed to load manually downloaded model: {e}")
                    return None
            else:
                return None
    else:
        # Model file exists, try to load it
        try:
            print("Loading existing model...")
            model = YOLO(model_path)
            print("✓ Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading existing model: {e}")
            print("The model file might be corrupted. Deleting and retrying...")
            
            try:
                os.remove(model_path)
                print("Corrupted file removed. Redownloading...")
                model = YOLO("yolov8s.pt")
                print("✓ Model redownloaded and loaded")
                return model
            except Exception as e2:
                print(f"Redownload failed: {e2}")
                return None

# Create output folder if it doesn't exist
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load YOLO model
model = load_yolo_model()
if model is None:
    print("\n❌ Could not load YOLO model.")
    print("\nTroubleshooting steps:")
    print("1. Check your internet connection")
    print("2. Manually download from: https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt")
    print("3. Run: pip install --upgrade ultralytics torch torchvision")
    print("4. Try using a smaller model by changing 'yolov8s.pt' to 'yolov8n.pt'")
    sys.exit(1)

# Load image
image_path = "BC.jpg"  # Change to your image name
if not os.path.exists(image_path):
    print(f"❌ Error: Image file '{image_path}' not found.")
    print(f"Please make sure '{image_path}' exists in the current directory.")
    sys.exit(1)

original_image = cv2.imread(image_path)

if original_image is None:
    print("❌ Error: Could not load image. The file might be corrupted or not an image.")
    sys.exit(1)

# Display image info
height, width, channels = original_image.shape
print(f"✓ Image loaded: {width}x{height}, {channels} channels")

image = original_image.copy()
image = cv2.resize(image, (640, 640))  # Optimal for YOLO

# Run detection
print("Running object detection...")
try:
    results = model(image)[0]
    print(f"✓ Detection completed. Found {len(results.boxes)} potential objects")
except Exception as e:
    print(f"❌ Detection failed: {e}")
    sys.exit(1)

# Define color ranges in HSV and corresponding labels/box colors
colors = {
    "Red":    ((0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255), (0, 0, 255)),      # BGR red box
    "Yellow": ((20, 100, 100), (40, 255, 255), None, None, (0, 255, 255)),                       # Yellow box
    "Green":  ((40, 100, 100), (80, 255, 255), None, None, (0, 255, 0)),                         # Green box
    "Blue":   ((100, 100, 100), (130, 255, 255), None, None, (255, 0, 0)),                       # Blue box
    "Black":  ((0, 0, 0), (180, 255, 50), None, None, (50, 50, 50)),                            # Dark gray box (low brightness)
    "White":  ((0, 0, 200), (180, 50, 255), None, None, (255, 255, 255))                        # White box (high brightness, low sat)
}

colored_count = 0

print("\nAnalyzing detected objects for colors...")
for i, box in enumerate(results.boxes):
    confidence = float(box.conf[0])
    if confidence < 0.4:
        continue
    
    class_id = int(box.cls[0])
    class_name = results.names[class_id]
    
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        continue
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    dominant_color = None
    max_percentage = 0
    
    for color_name, (lower1, upper1, lower2, upper2, box_color) in colors.items():
        if lower2 is None:  # Single range colors
            mask = cv2.inRange(hsv_roi, lower1, upper1)
        else:  # Red has two ranges
            mask1 = cv2.inRange(hsv_roi, lower1, upper1)
            mask2 = cv2.inRange(hsv_roi, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        
        percentage = (cv2.countNonZero(mask) / mask.size) * 100
        
        if percentage > max_percentage:
            max_percentage = percentage
            dominant_color = color_name
            current_box_color = box_color
    
    # Threshold for considering it a colored target
    if max_percentage > 18 and dominant_color:
        label = f"{dominant_color} {class_name} ({confidence:.2f}) - {max_percentage:.1f}%"
        cv2.rectangle(image, (x1, y1), (x2, y2), current_box_color, 3)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_box_color, 2)
        colored_count += 1
        print(f"  Found: {dominant_color} {class_name} (Confidence: {confidence:.2f}, Color match: {max_percentage:.1f}%)")

# Save results
try:
    processed_path = os.path.join(output_folder, "multi_color_targets.jpg")
    cv2.imwrite(processed_path, image)
    print(f"✓ Processed image saved to: {processed_path}")
    
    original_save_path = os.path.join(output_folder, "original.jpg")
    cv2.imwrite(original_save_path, original_image)
    print(f"✓ Original image saved to: {original_save_path}")
except Exception as e:
    print(f"❌ Error saving images: {e}")

# Display
cv2.imshow("Multi-Color Target Detection", image)
print("\nPress any key in the image window to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"\n{'='*50}")
print(f"SUCCESS! Found {colored_count} colored targets (Red, Blue, Yellow, Green, Black, White).")
print(f"Results saved in the '{output_folder}' folder.")
print(f"{'='*50}")