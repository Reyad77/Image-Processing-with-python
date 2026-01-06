import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from PIL import Image
import matplotlib.pyplot as plt

def load_image(filepath="C:\\Users\\latif\\OneDrive\\Desktop\\ImageDenoisingProject\\cameraman.png"):
    img = Image.open(filepath).convert('L')  # Convert to grayscale
    return np.array(img)

def save_image(image, filepath):
    img = Image.fromarray(image.astype(np.uint8))
    img.save(filepath)

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy = image.copy()
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1]] = 255
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

def calculate_psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def denoise_gaussian(image, sigma=1.0):
    """Apply Gaussian filter for denoising"""
    return gaussian_filter(image.astype(float), sigma=sigma).astype(np.uint8)

def denoise_median(image, size=3):
    """Apply median filter for denoising"""
    return median_filter(image, size=size)

def process_image(image_path, image_name):
    """Process a single image: add noise, denoise, and return results"""
    print(f"\nProcessing {image_name}...")
    
    # Load original image
    original = load_image(image_path)
    
    # Add Gaussian noise
    noisy_gaussian = add_gaussian_noise(original, mean=0, std=25)
    
    # Add salt and pepper noise
    noisy_salt_pepper = add_salt_pepper_noise(original, salt_prob=0.01, pepper_prob=0.01)
    
    # Denoise Gaussian noise with Gaussian filter
    denoised_gaussian_gf = denoise_gaussian(noisy_gaussian, sigma=1.5)
    
    # Denoise Gaussian noise with Median filter
    denoised_gaussian_mf = denoise_median(noisy_gaussian, size=3)
    
    # Denoise salt and pepper noise with Gaussian filter
    denoised_sp_gf = denoise_gaussian(noisy_salt_pepper, sigma=1.5)
    
    # Denoise salt and pepper noise with Median filter
    denoised_sp_mf = denoise_median(noisy_salt_pepper, size=3)
    
    # Calculate PSNR values
    psnr_gaussian_gf = calculate_psnr(original, denoised_gaussian_gf)
    psnr_gaussian_mf = calculate_psnr(original, denoised_gaussian_mf)
    psnr_sp_gf = calculate_psnr(original, denoised_sp_gf)
    psnr_sp_mf = calculate_psnr(original, denoised_sp_mf)
    
    return {
        'original': original,
        'noisy_gaussian': noisy_gaussian,
        'noisy_salt_pepper': noisy_salt_pepper,
        'denoised_gaussian_gf': denoised_gaussian_gf,
        'denoised_gaussian_mf': denoised_gaussian_mf,
        'denoised_sp_gf': denoised_sp_gf,
        'denoised_sp_mf': denoised_sp_mf,
        'psnr_gaussian_gf': psnr_gaussian_gf,
        'psnr_gaussian_mf': psnr_gaussian_mf,
        'psnr_sp_gf': psnr_sp_gf,
        'psnr_sp_mf': psnr_sp_mf,
        'name': image_name
    }

if __name__ == "__main__":
    # Process both images
    base_path = "C:\\Users\\latif\\OneDrive\\Desktop\\ImageDenoisingProject\\"
    
    cameraman_results = process_image(base_path + "cameraman.png", "Cameraman")
    lena_results = process_image(base_path + "lena.png", "Lena")
    
    # Create first window for Cameraman
    fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
    fig1.suptitle('Cameraman - Denoising Results', fontsize=16, fontweight='bold')
    
    # Cameraman - Row 1: Gaussian noise
    axes1[0, 0].imshow(cameraman_results['original'], cmap='gray')
    axes1[0, 0].set_title('Original')
    axes1[0, 0].axis('off')
    
    axes1[0, 1].imshow(cameraman_results['noisy_gaussian'], cmap='gray')
    axes1[0, 1].set_title('Gaussian Noise')
    axes1[0, 1].axis('off')
    
    axes1[0, 2].imshow(cameraman_results['denoised_gaussian_gf'], cmap='gray')
    axes1[0, 2].set_title(f'Gaussian Filter\nPSNR: {cameraman_results["psnr_gaussian_gf"]:.2f} dB')
    axes1[0, 2].axis('off')
    
    axes1[0, 3].imshow(cameraman_results['denoised_gaussian_mf'], cmap='gray')
    axes1[0, 3].set_title(f'Median Filter\nPSNR: {cameraman_results["psnr_gaussian_mf"]:.2f} dB')
    axes1[0, 3].axis('off')
    
    # Cameraman - Row 2: Salt and Pepper noise
    axes1[1, 0].imshow(cameraman_results['original'], cmap='gray')
    axes1[1, 0].set_title('Original')
    axes1[1, 0].axis('off')
    
    axes1[1, 1].imshow(cameraman_results['noisy_salt_pepper'], cmap='gray')
    axes1[1, 1].set_title('Salt & Pepper Noise')
    axes1[1, 1].axis('off')
    
    axes1[1, 2].imshow(cameraman_results['denoised_sp_gf'], cmap='gray')
    axes1[1, 2].set_title(f'Gaussian Filter\nPSNR: {cameraman_results["psnr_sp_gf"]:.2f} dB')
    axes1[1, 2].axis('off')
    
    axes1[1, 3].imshow(cameraman_results['denoised_sp_mf'], cmap='gray')
    axes1[1, 3].set_title(f'Median Filter\nPSNR: {cameraman_results["psnr_sp_mf"]:.2f} dB')
    axes1[1, 3].axis('off')
    
    plt.tight_layout()
    
    # Create second window for Lena
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    fig2.suptitle('Lena - Denoising Results', fontsize=16, fontweight='bold')
    
    # Lena - Row 1: Gaussian noise
    axes2[0, 0].imshow(lena_results['original'], cmap='gray')
    axes2[0, 0].set_title('Original')
    axes2[0, 0].axis('off')
    
    axes2[0, 1].imshow(lena_results['noisy_gaussian'], cmap='gray')
    axes2[0, 1].set_title('Gaussian Noise')
    axes2[0, 1].axis('off')
    
    axes2[0, 2].imshow(lena_results['denoised_gaussian_gf'], cmap='gray')
    axes2[0, 2].set_title(f'Gaussian Filter\nPSNR: {lena_results["psnr_gaussian_gf"]:.2f} dB')
    axes2[0, 2].axis('off')
    
    axes2[0, 3].imshow(lena_results['denoised_gaussian_mf'], cmap='gray')
    axes2[0, 3].set_title(f'Median Filter\nPSNR: {lena_results["psnr_gaussian_mf"]:.2f} dB')
    axes2[0, 3].axis('off')
    
    # Lena - Row 2: Salt and Pepper noise
    axes2[1, 0].imshow(lena_results['original'], cmap='gray')
    axes2[1, 0].set_title('Original')
    axes2[1, 0].axis('off')
    
    axes2[1, 1].imshow(lena_results['noisy_salt_pepper'], cmap='gray')
    axes2[1, 1].set_title('Salt & Pepper Noise')
    axes2[1, 1].axis('off')
    
    axes2[1, 2].imshow(lena_results['denoised_sp_gf'], cmap='gray')
    axes2[1, 2].set_title(f'Gaussian Filter\nPSNR: {lena_results["psnr_sp_gf"]:.2f} dB')
    axes2[1, 2].axis('off')
    
    axes2[1, 3].imshow(lena_results['denoised_sp_mf'], cmap='gray')
    axes2[1, 3].set_title(f'Median Filter\nPSNR: {lena_results["psnr_sp_mf"]:.2f} dB')
    axes2[1, 3].axis('off')
    
    plt.tight_layout()
    
    # Show both windows
    plt.show()
    
    # Print PSNR results for both images
    print("\n" + "="*60)
    print("PSNR Results (dB) - Higher is better")
    print("="*60)
    print("\nCAMERAMAN:")
    print(f"  Gaussian noise + Gaussian filter: {cameraman_results['psnr_gaussian_gf']:.2f} dB")
    print(f"  Gaussian noise + Median filter:  {cameraman_results['psnr_gaussian_mf']:.2f} dB")
    print(f"  Salt & Pepper + Gaussian filter: {cameraman_results['psnr_sp_gf']:.2f} dB")
    print(f"  Salt & Pepper + Median filter:   {cameraman_results['psnr_sp_mf']:.2f} dB")
    print("\nLENA:")
    print(f"  Gaussian noise + Gaussian filter: {lena_results['psnr_gaussian_gf']:.2f} dB")
    print(f"  Gaussian noise + Median filter:  {lena_results['psnr_gaussian_mf']:.2f} dB")
    print(f"  Salt & Pepper + Gaussian filter: {lena_results['psnr_sp_gf']:.2f} dB")
    print(f"  Salt & Pepper + Median filter:   {lena_results['psnr_sp_mf']:.2f} dB")
    print("="*60)