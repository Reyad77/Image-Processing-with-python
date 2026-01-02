"""
Task 1: Swap Color Channels
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_images, save_figure, show_image_grid

def swap_channels(image):
    """Swap RGB channels in different ways"""
    r, g, b = cv2.split(image)
    
    # Different swapping methods
    swaps = {
        'R→G, G→B, B→R': cv2.merge([g, b, r]),
        'R→B, G→R, B→G': cv2.merge([b, r, g]),
        'Swap R and B': cv2.merge([b, g, r]),
        'Swap R and G': cv2.merge([g, r, b])
    }
    
    return swaps

def main():
    print("=" * 50)
    print("TASK 1: SWAP COLOR CHANNELS")
    print("=" * 50)
    
    # Load images
    _, _, ori_rgb, _ = load_images()
    
    # Perform channel swapping
    swapped_images = swap_channels(ori_rgb)
    
    # Prepare images and titles for display
    images = [ori_rgb] + list(swapped_images.values())
    titles = ['Original Image'] + list(swapped_images.keys())
    
    # Display results
    fig = show_image_grid(images, titles, 2, 3, figsize=(15, 8))
    plt.suptitle('Channel Swapping Results', fontsize=16, y=1.02)
    
    # Save outputs
    save_figure('output_task1_channel_swapping.png')
    
    # Also save one swapped image as JPEG
    cv2.imwrite('output_swapped_rgb.jpg', 
                cv2.cvtColor(list(swapped_images.values())[0], cv2.COLOR_RGB2BGR))
    
    plt.show()
    print("Task 1 completed successfully!")

if __name__ == "__main__":
    main()