"""
Task 1: Compute and Display RGB & Luminance Histograms
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_experiment2_images, save_figure, plot_histograms

def compute_statistics(image_rgb):
    """Compute statistical measures for each channel"""
    if len(image_rgb.shape) == 2:
        channels = {'Gray': image_rgb}
    else:
        channels = {
            'Red': image_rgb[:, :, 0],
            'Green': image_rgb[:, :, 1],
            'Blue': image_rgb[:, :, 2]
        }
    
    stats = {}
    for name, channel in channels.items():
        stats[name] = {
            'min': channel.min(),
            'max': channel.max(),
            'mean': channel.mean(),
            'std': channel.std(),
            'median': np.median(channel)
        }
    
    return stats

def main():
    print("=" * 60)
    print("TASK 1: RGB & LUMINANCE HISTOGRAMS")
    print("=" * 60)
    
    # Load first image (night.png)
    first_img, _, first_rgb, _ = load_experiment2_images()
    
    print(f"\nImage: night.png")
    print(f"Shape: {first_rgb.shape}")
    print(f"Data type: {first_rgb.dtype}")
    print(f"Value range: [{first_rgb.min()}, {first_rgb.max()}]")
    
    # Compute statistics
    stats = compute_statistics(first_rgb)
    
    print("\nChannel Statistics:")
    print("-" * 50)
    for channel, values in stats.items():
        print(f"\n{channel} Channel:")
        print(f"  Min: {values['min']:3d}")
        print(f"  Max: {values['max']:3d}")
        print(f"  Mean: {values['mean']:7.2f}")
        print(f"  Std Dev: {values['std']:7.2f}")
        print(f"  Median: {values['median']:7.2f}")
    
    # Plot histograms
    print("\nGenerating histogram plots...")
    fig_hist = plot_histograms(first_rgb, "Original - ")
    plt.suptitle('RGB and Luminance Histograms of night.png', fontsize=14, y=1.02)
    save_figure('output_task1_histograms.png')
    
    # Also show image alongside histograms
    fig_combo, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(first_rgb)
    axes[0].set_title('Original Image (night.png)')
    axes[0].axis('off')
    
    # Create combined RGB histogram
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        channel = first_rgb[:, :, i].flatten()
        axes[1].hist(channel, bins=256, color=color, alpha=0.5, 
                    density=True, label=color.capitalize())
    
    axes[1].set_title('Combined RGB Histogram')
    axes[1].set_xlabel('Pixel Value')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure('output_task1_image_and_histogram.png')
    
    plt.show()
    print("\n✅ Task 1 completed successfully!")

if __name__ == "__main__":
    main()