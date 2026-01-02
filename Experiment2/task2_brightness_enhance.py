"""
Task 2: Linear and Nonlinear Brightness Enhancement
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_experiment2_images, save_figure, show_image_grid, plot_histograms

def linear_transformations(image_rgb):
    """Apply linear transformations for brightness enhancement"""
    
    # Original image
    original = image_rgb
    
    # 1. Simple brightness adjustment (add constant)
    brightness_add = np.clip(image_rgb.astype(np.float32) + 50, 0, 255).astype(np.uint8)
    
    # 2. Contrast stretching (linear scaling)
    contrast_stretched = np.clip((image_rgb.astype(np.float32) - 30) * 1.5, 0, 255).astype(np.uint8)
    
    # 3. Gamma correction (nonlinear)
    gamma = 0.5  # < 1 for brighter, > 1 for darker
    gamma_corrected = np.clip(255 * (image_rgb.astype(np.float32) / 255) ** gamma, 0, 255).astype(np.uint8)
    
    # 4. Log transformation (compress dynamic range)
    c = 255 / np.log(1 + np.max(image_rgb))
    log_transformed = np.clip(c * np.log1p(image_rgb.astype(np.float32)), 0, 255).astype(np.uint8)
    
    # 5. Histogram stretching to full range
    min_val = image_rgb.min()
    max_val = image_rgb.max()
    stretched = np.clip((image_rgb.astype(np.float32) - min_val) * 
                       (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)
    
    return {
        'Original': original,
        'Brightness +50': brightness_add,
        'Contrast Stretched': contrast_stretched,
        f'Gamma (γ={gamma})': gamma_corrected,
        'Log Transform': log_transformed,
        'Histogram Stretching': stretched
    }

def compare_methods(image_rgb):
    """Compare different enhancement methods with parameters"""
    
    results = {'Original': image_rgb}
    
    # Different gamma values
    gamma_values = [0.3, 0.7, 1.5, 2.0]
    for gamma in gamma_values:
        gamma_img = np.clip(255 * (image_rgb.astype(np.float32) / 255) ** gamma, 0, 255).astype(np.uint8)
        results[f'Gamma={gamma}'] = gamma_img
    
    # Different brightness adjustments
    brightness_values = [30, 80, -30]
    for bright in brightness_values:
        bright_img = np.clip(image_rgb.astype(np.float32) + bright, 0, 255).astype(np.uint8)
        sign = '+' if bright > 0 else ''
        results[f'Brightness {sign}{bright}'] = bright_img
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if len(image_rgb.shape) == 3:
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge([cl, a, b])
        clahe_result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        results['CLAHE'] = clahe_result
    
    return results

def main():
    print("=" * 60)
    print("TASK 2: BRIGHTNESS ENHANCEMENT")
    print("=" * 60)
    
    # Load first image (night.png)
    first_img, _, first_rgb, _ = load_experiment2_images()
    
    print("\n1. Applying linear and nonlinear transformations...")
    
    # Apply basic transformations
    enhanced = linear_transformations(first_rgb)
    
    # Display results
    images = list(enhanced.values())
    titles = list(enhanced.keys())
    
    fig1 = show_image_grid(images, titles, 2, 3, figsize=(15, 10))
    plt.suptitle('Brightness Enhancement Methods', fontsize=16, y=1.02)
    save_figure('output_task2_basic_enhancement.png')
    plt.show()
    
    # Compare methods with different parameters
    print("\n2. Comparing different parameters...")
    comparisons = compare_methods(first_rgb)
    
    # Select subset for display
    selected_keys = ['Original', 'Gamma=0.3', 'Gamma=1.5', 
                    'Brightness +80', 'CLAHE', 'Histogram Stretching']
    selected_images = [comparisons[k] for k in selected_keys]
    selected_titles = selected_keys
    
    fig2 = show_image_grid(selected_images, selected_titles, 2, 3, figsize=(15, 10))
    plt.suptitle('Parameter Comparison for Enhancement', fontsize=16, y=1.02)
    save_figure('output_task2_parameter_comparison.png')
    
    # Plot histograms for selected methods
    print("\n3. Analyzing histogram changes...")
    fig3, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    methods_to_analyze = ['Original', 'Brightness +50', 'Gamma=0.5', 
                         'Log Transform', 'Histogram Stretching', 'CLAHE']
    
    for idx, method in enumerate(methods_to_analyze):
        if method in comparisons:
            img = comparisons[method]
        else:
            img = enhanced.get(method, first_rgb)
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).flatten()
        else:
            gray = img.flatten()
        
        ax = axes.flatten()[idx]
        ax.hist(gray, bins=256, color='gray', alpha=0.7, density=True)
        ax.set_title(f'{method}\nMean: {gray.mean():.1f}, Std: {gray.std():.1f}')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.set_xlim([0, 255])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure('output_task2_histogram_analysis.png')
    
    # Save best results
    cv2.imwrite('output_gamma_enhanced.jpg', 
                cv2.cvtColor(comparisons['Gamma=0.3'], cv2.COLOR_RGB2BGR))
    cv2.imwrite('output_clahe_enhanced.jpg', 
                cv2.cvtColor(comparisons['CLAHE'], cv2.COLOR_RGB2BGR))
    
    plt.show()
    print("\n✅ Task 2 completed successfully!")

if __name__ == "__main__":
    main()