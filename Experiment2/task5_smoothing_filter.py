"""
Task 5: Smoothing Filters for Noise Suppression
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from utils import load_experiment2_images, save_figure, show_image_grid

def apply_smoothing_filters(noisy_image, original_image=None):
    """Apply various smoothing filters to noisy image"""
    
    results = {'Noisy Image': noisy_image}
    
    if original_image is not None:
        results['Original'] = original_image
    
    # 1. Mean Filter (Box Filter)
    mean_3x3 = cv2.blur(noisy_image, (3, 3))
    results['Mean 3x3'] = mean_3x3
    
    mean_5x5 = cv2.blur(noisy_image, (5, 5))
    results['Mean 5x5'] = mean_5x5
    
    mean_7x7 = cv2.blur(noisy_image, (7, 7))
    results['Mean 7x7'] = mean_7x7
    
    # 2. Gaussian Filter
    gaussian_3x3 = cv2.GaussianBlur(noisy_image, (3, 3), sigmaX=1)
    results['Gaussian 3x3 (σ=1)'] = gaussian_3x3
    
    gaussian_5x5 = cv2.GaussianBlur(noisy_image, (5, 5), sigmaX=1.5)
    results['Gaussian 5x5 (σ=1.5)'] = gaussian_5x5
    
    gaussian_7x7 = cv2.GaussianBlur(noisy_image, (7, 7), sigmaX=2)
    results['Gaussian 7x7 (σ=2)'] = gaussian_7x7
    
    # 3. Median Filter (excellent for salt-and-pepper noise)
    median_3x3 = cv2.medianBlur(noisy_image, 3)
    results['Median 3x3'] = median_3x3
    
    median_5x5 = cv2.medianBlur(noisy_image, 5)
    results['Median 5x5'] = median_5x5
    
    # 4. Bilateral Filter (edge-preserving)
    bilateral = cv2.bilateralFilter(noisy_image, d=9, sigmaColor=75, sigmaSpace=75)
    results['Bilateral Filter'] = bilateral
    
    # 5. Wiener Filter (frequency domain)
    # Convert to grayscale for wiener filter
    if len(noisy_image.shape) == 3:
        noisy_gray = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2GRAY)
    else:
        noisy_gray = noisy_image
    
    wiener_filtered = ndimage.wiener(noisy_gray, (3, 3))
    wiener_filtered = np.clip(wiener_filtered, 0, 255).astype(np.uint8)
    if len(noisy_image.shape) == 3:
        results['Wiener Filter'] = cv2.cvtColor(wiener_filtered, cv2.COLOR_GRAY2RGB)
    else:
        results['Wiener Filter'] = wiener_filtered
    
    return results

def calculate_metrics(original, filtered, noisy):
    """Calculate PSNR, SSIM, and MSE for evaluation"""
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error as mse_c