"""
Task 5: Change Yellow Rose Color & Grayscale Background
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_images, save_figure, show_image_grid

def detect_yellow_regions(image_bgr):
    """Detect yellow regions in an image using HSV color space"""
    # Convert to HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Define yellow range in HSV (adjust these values as needed)
    # Hue range: 15-35 for yellow in OpenCV (0-180 range)
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    
    # Create mask for yellow regions
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask, hsv

def recolor_rose(image_bgr, mask, new_color_bgr):
    """Recolor the rose while keeping background grayscale"""
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Create colored version
    colored = image_bgr.copy()
    colored[mask > 0] = new_color_bgr
    
    # Convert to RGB for display
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    # Convert background to grayscale
    gray_bg = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_bg_rgb = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2RGB)
    
    # Combine: colored rose + grayscale background
    final_result = np.where(mask[:, :, np.newaxis] > 0, colored_rgb, gray_bg_rgb)
    
    return final_result, colored_rgb, gray_bg_rgb

def main():
    print("=" * 50)
    print("TASK 5: RECOLOR YELLOW ROSE")
    print("=" * 50)
    
    # Load image
    ori_img, _, ori_rgb, _ = load_images()
    
    # Detect yellow regions
    print("\nDetecting yellow rose regions...")
    mask, hsv = detect_yellow_regions(ori_img)
    
    # Define different colors to try (in BGR format)
    color_options = {
        'Blue': [255, 0, 0],
        'Red': [0, 0, 255],
        'Green': [0, 255, 0],
        'Purple': [255, 0, 255],
        'Cyan': [255, 255, 0],
        'Pink': [203, 192, 255]
    }
    
    results = {}
    results['Original'] = ori_rgb
    results['Yellow Mask'] = mask
    
    print("\nRecoloring with different colors...")
    for color_name, bgr_value in color_options.items():
        recolored, colored_only, gray_bg = recolor_rose(ori_img, mask, bgr_value)
        results[f'{color_name} Rose'] = recolored
    
    # Display HSV channels
    hsv_channels = {
        'Hue': hsv[:, :, 0],
        'Saturation': hsv[:, :, 1],
        'Value': hsv[:, :, 2]
    }
    
    # Display results
    print("\nDisplaying recoloring results...")
    
    # Show HSV analysis
    hsv_images = [ori_rgb] + list(hsv_channels.values())
    hsv_titles = ['Original'] + list(hsv_channels.keys())
    
    fig1 = show_image_grid(hsv_images, hsv_titles, 2, 2, figsize=(12, 10))
    plt.suptitle('HSV Color Space Analysis', fontsize=16, y=1.02)
    save_figure('output_task5_hsv_analysis.png')
    plt.show()
    
    # Show recoloring results (first 6)
    recolored_items = list(results.items())[:8]  # Original, Mask, and 6 colors
    recolored_images = [v for k, v in recolored_items]
    recolored_titles = [k for k, v in recolored_items]
    
    fig2 = show_image_grid(recolored_images, recolored_titles, 2, 4, figsize=(16, 8))
    plt.suptitle('Rose Recoloring Results', fontsize=16, y=1.02)
    save_figure('output_task5_recoloring.png')
    plt.show()
    
    # Save mask and best result
    cv2.imwrite('output_yellow_mask.jpg', mask)
    
    # Save blue rose as example
    blue_result, _, _ = recolor_rose(ori_img, mask, [255, 0, 0])
    cv2.imwrite('output_blue_rose.jpg', cv2.cvtColor(blue_result, cv2.COLOR_RGB2BGR))
    
    # Calculate and display mask statistics
    yellow_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]
    yellow_percentage = (yellow_pixels / total_pixels) * 100
    
    print("\nMask Statistics:")
    print("-" * 30)
    print(f"Total pixels: {total_pixels:,}")
    print(f"Yellow pixels detected: {yellow_pixels:,}")
    print(f"Percentage of yellow: {yellow_percentage:.2f}%")
    
    print("\nTask 5 completed successfully!")

if __name__ == "__main__":
    main()