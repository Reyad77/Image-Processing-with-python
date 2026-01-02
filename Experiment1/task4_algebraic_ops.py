"""
Task 4: Algebraic Operations between Images
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_images, save_figure, show_image_grid

def algebraic_operations(img1, img2):
    """Perform algebraic operations between two images"""
    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Convert to float for operations
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    
    # Addition
    addition = cv2.add(img1, img2)
    
    # Subtraction (absolute difference)
    subtraction = cv2.subtract(img1, img2)
    
    # Multiplication (normalized)
    multiplication = np.multiply(img1_float / 255.0, img2_float / 255.0) * 255
    multiplication = multiplication.astype(np.uint8)
    
    # Division (with epsilon to avoid division by zero)
    epsilon = 1e-7
    division = np.divide(img1_float, img2_float + epsilon)
    division = np.clip(division, 0, 255).astype(np.uint8)
    
    # Blending (weighted addition)
    alpha = 0.6
    blending = cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)
    
    # Concatenation (side by side)
    concatenation = np.hstack((img1, img2))
    
    # Bitwise AND (for masks)
    bitwise_and = cv2.bitwise_and(img1, img2)
    
    # Bitwise OR
    bitwise_or = cv2.bitwise_or(img1, img2)
    
    return {
        'Image 1': img1,
        'Image 2': img2,
        'Addition': addition,
        'Subtraction': subtraction,
        'Multiplication': multiplication,
        'Division': division,
        'Blending (60% Im1 + 40% Im2)': blending,
        'Concatenation': concatenation,
        'Bitwise AND': bitwise_and,
        'Bitwise OR': bitwise_or
    }

def main():
    print("=" * 50)
    print("TASK 4: ALGEBRAIC OPERATIONS")
    print("=" * 50)
    
    # Load images
    _, _, ori_rgb, sec_rgb = load_images()
    
    # Perform algebraic operations
    results = algebraic_operations(ori_rgb, sec_rgb)
    
    # Display all results
    images = list(results.values())
    titles = list(results.keys())
    
    # Split into two displays for clarity
    basic_ops = list(results.items())[:8]  # First 8 operations
    advanced_ops = list(results.items())[8:]  # Remaining operations
    
    print("\nDisplaying basic algebraic operations...")
    fig1 = show_image_grid([v for k, v in basic_ops], 
                          [k for k, v in basic_ops], 
                          2, 4, figsize=(16, 8))
    plt.suptitle('Basic Algebraic Operations', fontsize=16, y=1.02)
    save_figure('output_task4_basic_operations.png')
    plt.show()
    
    print("\nDisplaying advanced operations...")
    fig2 = show_image_grid([v for k, v in advanced_ops], 
                          [k for k, v in advanced_ops], 
                          1, 3, figsize=(15, 5))
    plt.suptitle('Advanced Operations', fontsize=16, y=1.02)
    save_figure('output_task4_advanced_operations.png')
    plt.show()
    
    # Save individual outputs
    cv2.imwrite('output_addition.jpg', 
                cv2.cvtColor(results['Addition'], cv2.COLOR_RGB2BGR))
    cv2.imwrite('output_blending.jpg', 
                cv2.cvtColor(results['Blending (60% Im1 + 40% Im2)'], cv2.COLOR_RGB2BGR))
    
    print("\nTask 4 completed successfully!")

if __name__ == "__main__":
    main()