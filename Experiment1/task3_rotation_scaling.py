"""
Task 3: Rotation and Scaling with Interpolation
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_images, save_figure, show_image_grid

def rotate_image(image, angle, interpolation=cv2.INTER_LINEAR):
    """Rotate image by specified angle"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(image, M, (w, h), flags=interpolation)
    return rotated

def scale_image(image, scale_factor, interpolation=cv2.INTER_LINEAR):
    """Scale image by specified factor"""
    h, w = image.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    scaled = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return scaled

def main():
    print("=" * 50)
    print("TASK 3: ROTATION AND SCALING")
    print("=" * 50)
    
    # Load images
    _, _, ori_rgb, _ = load_images()
    h, w = ori_rgb.shape[:2]
    
    # Rotation examples
    print("\n1. Rotation Examples:")
    rotated_nearest = rotate_image(ori_rgb, 45, cv2.INTER_NEAREST)
    rotated_linear = rotate_image(ori_rgb, 45, cv2.INTER_LINEAR)
    rotated_cubic = rotate_image(ori_rgb, 45, cv2.INTER_CUBIC)
    rotated_90 = rotate_image(ori_rgb, 90, cv2.INTER_LINEAR)
    rotated_neg30 = rotate_image(ori_rgb, -30, cv2.INTER_LINEAR)
    
    # Scaling examples
    print("\n2. Scaling Examples:")
    scaled_150_nearest = scale_image(ori_rgb, 1.5, cv2.INTER_NEAREST)
    scaled_150_linear = scale_image(ori_rgb, 1.5, cv2.INTER_LINEAR)
    scaled_150_cubic = scale_image(ori_rgb, 1.5, cv2.INTER_CUBIC)
    scaled_50 = scale_image(ori_rgb, 0.5, cv2.INTER_LINEAR)
    scaled_200 = scale_image(ori_rgb, 2.0, cv2.INTER_LINEAR)
    
    # Prepare display for rotation
    rotation_images = [
        ori_rgb, rotated_nearest, rotated_linear, 
        rotated_cubic, rotated_90, rotated_neg30
    ]
    rotation_titles = [
        'Original', 'Rotate 45° (Nearest)', 'Rotate 45° (Linear)',
        'Rotate 45° (Cubic)', 'Rotate 90° (Linear)', 'Rotate -30° (Linear)'
    ]
    
    # Prepare display for scaling
    scaling_images = [
        ori_rgb, scaled_150_nearest, scaled_150_linear,
        scaled_150_cubic, scaled_50, scaled_200
    ]
    scaling_titles = [
        'Original', 'Scale 150% (Nearest)', 'Scale 150% (Linear)',
        'Scale 150% (Cubic)', 'Scale 50% (Linear)', 'Scale 200% (Linear)'
    ]
    
    # Display rotation results
    print("\nDisplaying rotation results...")
    fig1 = show_image_grid(rotation_images, rotation_titles, 2, 3, figsize=(15, 10))
    plt.suptitle('Image Rotation with Different Interpolation Methods', fontsize=16, y=1.02)
    save_figure('output_task3_rotation.png')
    plt.show()
    
    # Display scaling results
    print("\nDisplaying scaling results...")
    fig2 = show_image_grid(scaling_images, scaling_titles, 2, 3, figsize=(15, 10))
    plt.suptitle('Image Scaling with Different Interpolation Methods', fontsize=16, y=1.02)
    save_figure('output_task3_scaling.png')
    plt.show()
    
    # Save individual outputs
    cv2.imwrite('output_rotated_45_cubic.jpg', 
                cv2.cvtColor(rotated_cubic, cv2.COLOR_RGB2BGR))
    cv2.imwrite('output_scaled_150_cubic.jpg', 
                cv2.cvtColor(scaled_150_cubic, cv2.COLOR_RGB2BGR))
    
    print("\nTask 3 completed successfully!")

if __name__ == "__main__":
    main()