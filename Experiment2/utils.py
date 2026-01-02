"""
Utility functions for Experiment 2: Image Enhancement
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import skimage.exposure as exposure

def load_experiment2_images():
    """Load images for Experiment 2"""
    first_img = cv2.imread('night.png')
    sec_img = cv2.imread('peppers.jpg')
    
    if first_img is None:
        raise FileNotFoundError("night.png not found")
    if sec_img is None:
        raise FileNotFoundError("peppers.jpg not found")
    
    # Convert BGR to RGB for display
    first_rgb = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
    sec_rgb = cv2.cvtColor(sec_img, cv2.COLOR_BGR2RGB)
    
    print(f"First Image (night.png): {first_rgb.shape}")
    print(f"Second Image (peppers.jpg): {sec_rgb.shape}")
    
    return first_img, sec_img, first_rgb, sec_rgb

def save_figure(filename, dpi=300):
    """Save current matplotlib figure"""
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {filename}")

def show_image_grid(images, titles, rows, cols, figsize=(15, 10)):
    """Display multiple images in a grid"""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows > 1 or cols > 1:
        axes = axes.flatten()
    
    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 2:  # Grayscale
            ax.imshow(img, cmap='gray')
        else:  # Color
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_histograms(image_rgb, title_prefix=""):
    """Plot RGB and luminance histograms"""
    if len(image_rgb.shape) == 2:  # Grayscale
        gray = image_rgb
        r = g = b = None
    else:  # Color
        r = image_rgb[:, :, 0].flatten()
        g = image_rgb[:, :, 1].flatten()
        b = image_rgb[:, :, 2].flatten()
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # RGB Histograms (if color)
    if r is not None:
        axes[0, 0].hist(r, bins=256, color='red', alpha=0.7, density=True)
        axes[0, 0].set_title(f'{title_prefix}Red Channel Histogram')
        axes[0, 0].set_xlabel('Pixel Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_xlim([0, 255])
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(g, bins=256, color='green', alpha=0.7, density=True)
        axes[0, 1].set_title(f'{title_prefix}Green Channel Histogram')
        axes[0, 1].set_xlabel('Pixel Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_xlim([0, 255])
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].hist(b, bins=256, color='blue', alpha=0.7, density=True)
        axes[1, 0].set_title(f'{title_prefix}Blue Channel Histogram')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_xlim([0, 255])
        axes[1, 0].grid(True, alpha=0.3)
    
    # Luminance Histogram
    axes[1, 1].hist(gray, bins=256, color='gray', alpha=0.7, density=True)
    axes[1, 1].set_title(f'{title_prefix}Luminance Histogram')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_xlim([0, 255])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig