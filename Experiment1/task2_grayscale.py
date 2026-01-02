"""
Task 2: Convert to Grayscale using 3 Methods
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_images, save_figure, show_image_grid

def grayscale_methods(image):
    """Convert image to grayscale using 3 different methods"""
    
    # Method 1: Average
    gray_avg = np.mean(image, axis=2).astype(np.uint8)
    
    # Method 2: OpenCV built-in (Luminosity)
    # Convert BGR to RGB first
    gray_cv = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    
    # Method 3: Weighted method (ITU-R BT.601)
    weights = np.array([0.299, 0.587, 0.114])
    gray_weighted = np.dot(image[..., :3], weights).astype(np.uint8)
    
    # Method 4: Extract single channel (Red)
    gray_red = image[:, :, 0]
    
    # Method 5: Extract Green channel
    gray_green = image[:, :, 1]
    
    # Method 6: Extract Blue channel
    gray_blue = image[:, :, 2]
    
    return {
        'Original': image,
        'Average Method': gray_avg,
        'OpenCV (Luminosity)': gray_cv,
        'Weighted (0.299,0.587,0.114)': gray_weighted,
        'Red Channel': gray_red,
        'Green Channel': gray_green,
        'Blue Channel': gray_blue
    }

def calculate_statistics(grayscale_images):
    """Calculate basic statistics for each grayscale image"""
    stats = {}
    for name, img in grayscale_images.items():
        if name == 'Original':
            continue
        stats[name] = {
            'min': img.min(),
            'max': img.max(),
            'mean': img.mean(),
            'std': img.std()
        }
    return stats

def main():
    print("=" * 50)
    print("TASK 2: GRAYSCALE CONVERSION METHODS")
    print("=" * 50)
    
    # Load images
    _, _, ori_rgb, _ = load_images()
    
    # Convert to grayscale using different methods
    grayscale_results = grayscale_methods(ori_rgb)
    
    # Display results
    images = list(grayscale_results.values())
    titles = list(grayscale_results.keys())
    
    fig = show_image_grid(images, titles, 3, 3, figsize=(15, 12))
    plt.suptitle('Grayscale Conversion Methods', fontsize=16, y=1.02)
    
    # Calculate and print statistics
    print("\nGrayscale Image Statistics:")
    print("-" * 50)
    stats = calculate_statistics(grayscale_results)
    for name, stat in stats.items():
        print(f"\n{name}:")
        print(f"  Min: {stat['min']:3d}, Max: {stat['max']:3d}, "
              f"Mean: {stat['mean']:6.2f}, Std: {stat['std']:6.2f}")
    
    # Save outputs
    save_figure('output_task2_grayscale_methods.png')
    cv2.imwrite('output_grayscale_opencv.jpg', 
                list(grayscale_results.values())[2])  # Save OpenCV method
    
    plt.show()
    print("\nTask 2 completed successfully!")

if __name__ == "__main__":
    main()