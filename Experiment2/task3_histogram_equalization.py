"""
Task 3: Histogram Equalization
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_experiment2_images, save_figure, show_image_grid

def apply_histogram_equalization(image_rgb):
    """Apply various histogram equalization techniques"""
    
    results = {'Original': image_rgb}
    
    # Convert to grayscale for basic equalization
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Global histogram equalization
    equalized_global = cv2.equalizeHist(gray)
    results['Global Equalization'] = cv2.cvtColor(equalized_global, cv2.COLOR_GRAY2RGB)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_clahe = clahe.apply(gray)
    results['CLAHE (Clip=2.0)'] = cv2.cvtColor(equalized_clahe, cv2.COLOR_GRAY2RGB)
    
    # 3. CLAHE with different parameters
    clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    equalized_clahe2 = clahe2.apply(gray)
    results['CLAHE (Clip=4.0)'] = cv2.cvtColor(equalized_clahe2, cv2.COLOR_GRAY2RGB)
    
    # 4. Color image equalization (per channel)
    if len(image_rgb.shape) == 3:
        # Method 1: Equalize each channel separately
        channels = cv2.split(image_rgb)
        eq_channels = []
        for ch in channels:
            eq_channels.append(cv2.equalizeHist(ch))
        equalized_color = cv2.merge(eq_channels)
        results['Color Equalization (per channel)'] = equalized_color
        
        # Method 2: Equalize in LAB color space
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.equalizeHist(l)
        lab_eq = cv2.merge([l_eq, a, b])
        equalized_lab = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        results['LAB Space Equalization'] = equalized_lab
        
        # Method 3: Equalize in YUV color space
        yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(yuv)
        y_eq = cv2.equalizeHist(y)
        yuv_eq = cv2.merge([y_eq, u, v])
        equalized_yuv = cv2.cvtColor(yuv_eq, cv2.COLOR_YUV2RGB)
        results['YUV Space Equalization'] = equalized_yuv
    
    # 5. Adaptive equalization with different grid sizes
    grid_sizes = [(4, 4), (8, 8), (16, 16)]
    for grid in grid_sizes:
        clahe_grid = cv2.createCLAHE(clipLimit=2.0, tileGridSize=grid)
        equalized_grid = clahe_grid.apply(gray)
        results[f'CLAHE Grid {grid}'] = cv2.cvtColor(equalized_grid, cv2.COLOR_GRAY2RGB)
    
    return results

def plot_cumulative_histograms(images_dict):
    """Plot cumulative distribution functions for comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    methods = list(images_dict.keys())[:6]  # Plot first 6 methods
    
    for idx, method in enumerate(methods):
        img = images_dict[method]
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Compute histogram and CDF
        hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()
        
        ax = axes[idx]
        ax.plot(cdf_normalized, color='blue')
        ax.set_title(f'{method}\nCDF')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Cumulative Probability')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 255])
    
    plt.tight_layout()
    return fig

def main():
    print("=" * 60)
    print("TASK 3: HISTOGRAM EQUALIZATION")
    print("=" * 60)
    
    # Load first image (night.png)
    first_img, _, first_rgb, _ = load_experiment2_images()
    
    print("\nApplying histogram equalization techniques...")
    
    # Apply equalization methods
    equalized = apply_histogram_equalization(first_rgb)
    
    # Display results
    images = list(equalized.values())
    titles = list(equalized.keys())
    
    fig1 = show_image_grid(images, titles, 3, 3, figsize=(15, 12))
    plt.suptitle('Histogram Equalization Methods', fontsize=16, y=1.02)
    save_figure('output_task3_equalization_methods.png')
    plt.show()
    
    # Plot cumulative histograms
    print("\nAnalyzing cumulative distribution functions...")
    fig2 = plot_cumulative_histograms(equalized)
    plt.suptitle('Cumulative Distribution Functions (CDFs)', fontsize=16, y=1.02)
    save_figure('output_task3_cumulative_histograms.png')
    
    # Compare before/after histograms for best methods
    print("\nComparing histogram distributions...")
    best_methods = ['Original', 'Global Equalization', 'CLAHE (Clip=2.0)', 
                   'LAB Space Equalization', 'CLAHE Grid (16, 16)']
    
    fig3, axes = plt.subplots(len(best_methods), 2, figsize=(12, 3*len(best_methods)))
    
    for idx, method in enumerate(best_methods):
        img = equalized[method]
        
        # Show image
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(method)
        axes[idx, 0].axis('off')
        
        # Show histogram
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        axes[idx, 1].hist(gray.flatten(), bins=256, color='gray', alpha=0.7, density=True)
        axes[idx, 1].set_title(f'Histogram\nMean: {gray.mean():.1f}, Std: {gray.std():.1f}')
        axes[idx, 1].set_xlabel('Pixel Value')
        axes[idx, 1].set_ylabel('Frequency')
        axes[idx, 1].set_xlim([0, 255])
        axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure('output_task3_histogram_comparison.png')
    
    # Save best results
    cv2.imwrite('output_global_equalized.jpg', 
                cv2.cvtColor(equalized['Global Equalization'], cv2.COLOR_RGB2BGR))
    cv2.imwrite('output_clahe_equalized.jpg', 
                cv2.cvtColor(equalized['CLAHE (Clip=2.0)'], cv2.COLOR_RGB2BGR))
    
    print("\n✅ Task 3 completed successfully!")
    print("\nSummary of findings:")
    print("- Global equalization can cause over-enhancement")
    print("- CLAHE preserves local contrast better")
    print("- Color space equalization (LAB/YUV) often gives better results for color images")
    
    plt.show()

if __name__ == "__main__":
    main()