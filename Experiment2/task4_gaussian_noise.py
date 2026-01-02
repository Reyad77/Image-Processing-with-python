"""
Task 4: Add Gaussian Noise to Image
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_experiment2_images, save_figure, show_image_grid

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to image"""
    # Generate Gaussian noise
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    
    # Add noise to image
    noisy = image.astype(np.float32) + noise
    
    # Clip to valid range
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy, noise

def add_different_noise_levels(image_rgb):
    """Add Gaussian noise with different sigma values"""
    results = {'Original': image_rgb}
    
    sigma_values = [10, 25, 50, 75, 100]
    
    for sigma in sigma_values:
        noisy, _ = add_gaussian_noise(image_rgb, mean=0, sigma=sigma)
        results[f'Gaussian Noise (σ={sigma})'] = noisy
    
    # Also add salt-and-pepper noise for comparison
    if len(image_rgb.shape) == 3:
        noisy_sp = image_rgb.copy()
        amount = 0.05
        num_salt = np.ceil(amount * image_rgb.size * 0.5)
        num_pepper = np.ceil(amount * image_rgb.size * 0.5)
        
        # Add salt noise
        coords = [np.random.randint(0, i-1, int(num_salt)) for i in image_rgb.shape[:2]]
        noisy_sp[coords[0], coords[1], :] = 255
        
        # Add pepper noise
        coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image_rgb.shape[:2]]
        noisy_sp[coords[0], coords[1], :] = 0
        
        results['Salt & Pepper Noise (5%)'] = noisy_sp
    
    return results

def calculate_psnr(original, noisy):
    """Calculate PSNR between original and noisy images"""
    mse = np.mean((original.astype(np.float32) - noisy.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def main():
    print("=" * 60)
    print("TASK 4: ADD GAUSSIAN NOISE")
    print("=" * 60)
    
    # Load second image (peppers.jpg)
    _, sec_img, _, sec_rgb = load_experiment2_images()
    
    print(f"\nOriginal Image Statistics:")
    print(f"  Shape: {sec_rgb.shape}")
    print(f"  Min: {sec_rgb.min()}, Max: {sec_rgb.max()}")
    print(f"  Mean: {sec_rgb.mean():.2f}, Std: {sec_rgb.std():.2f}")
    
    # Add different levels of Gaussian noise
    print("\nAdding Gaussian noise with different sigma values...")
    noisy_results = add_different_noise_levels(sec_rgb)
    
    # Calculate PSNR for each noise level
    print("\nPSNR Values (higher is better):")
    print("-" * 40)
    
    psnr_values = {}
    for name, img in noisy_results.items():
        if name != 'Original':
            psnr = calculate_psnr(sec_rgb, img)
            psnr_values[name] = psnr
            print(f"{name}: {psnr:.2f} dB")
    
    # Display results
    images = list(noisy_results.values())[:6]  # First 6 images
    titles = list(noisy_results.keys())[:6]
    
    fig1 = show_image_grid(images, titles, 2, 3, figsize=(15, 10))
    plt.suptitle('Gaussian Noise Addition with Different Sigma Values', fontsize=16, y=1.02)
    save_figure('output_task4_gaussian_noise.png')
    plt.show()
    
    # Show noise patterns
    print("\nVisualizing noise patterns...")
    fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    noise_levels = [('Original', sec_rgb, None)]
    for sigma in [10, 25, 50, 75, 100]:
        noisy_img, noise_pattern = add_gaussian_noise(sec_rgb, sigma=sigma)
        noise_levels.append((f'σ={sigma}', noisy_img, noise_pattern))
    
    for idx, (title, img, noise) in enumerate(noise_levels):
        ax = axes.flatten()[idx]
        
        if idx == 0:
            ax.imshow(img)
            ax.set_title(title)
        else:
            # Show the noise pattern (amplified for visibility)
            if noise is not None:
                noise_display = np.clip(noise * 2 + 128, 0, 255).astype(np.uint8)
                ax.imshow(noise_display, cmap='gray')
                ax.set_title(f'{title}\nPSNR: {psnr_values.get(title, "N/A")} dB')
            else:
                ax.imshow(img)
                ax.set_title(title)
        
        ax.axis('off')
    
    plt.tight_layout()
    save_figure('output_task4_noise_patterns.png')
    
    # Plot PSNR vs Sigma
    print("\nPlotting PSNR vs Noise Level...")
    sigmas = [10, 25, 50, 75, 100]
    psnrs = [psnr_values.get(f'Gaussian Noise (σ={s})', 0) for s in sigmas]
    
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sigmas, psnrs, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Sigma (Noise Level)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('PSNR vs Gaussian Noise Level')
    ax.grid(True, alpha=0.3)
    
    for s, p in zip(sigmas, psnrs):
        ax.annotate(f'{p:.1f} dB', (s, p), textcoords="offset points", 
                   xytext=(0,10), ha='center')
    
    plt.tight_layout()
    save_figure('output_task4_psnr_vs_noise.png')
    
    # Save noisy image for next tasks (use σ=25 as example)
    noisy_25, _ = add_gaussian_noise(sec_rgb, sigma=25)
    cv2.imwrite('output_noisy_image.jpg', cv2.cvtColor(noisy_25, cv2.COLOR_RGB2BGR))
    cv2.imwrite('input_noisy_for_filtering.jpg', cv2.cvtColor(noisy_25, cv2.COLOR_RGB2BGR))
    
    print("\n✅ Task 4 completed successfully!")
    print(f"\nNoisy image saved as 'output_noisy_image.jpg' (σ=25)")
    print("This will be used as input for Task 5 (smoothing filters).")
    
    plt.show()

if __name__ == "__main__":
    main()