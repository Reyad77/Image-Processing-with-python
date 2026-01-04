# Final Working Version - Simple Image Display

import os
import sys
import traceback
import warnings
import torch
from PIL import Image
import numpy as np
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from deoldify import device as deoldify_device
from deoldify.device_id import DeviceId
from deoldify.visualize import *

# ==================== SET YOUR BASE DIRECTORY ====================
BASE_DIR = r"D:\Image Processing Projects\deblurred and colorization\DeOldify-master"
# ================================================================

os.chdir(BASE_DIR)

# Set up paths
INPUT_DIR = os.path.join(BASE_DIR, 'inputs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Check for input images
input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if len(input_files) == 0:
    print("ERROR: No images found in 'inputs' folder!")
    sys.exit(1)

print(f"Found {len(input_files)} images to process")

# Setup
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {torch_device}")
deoldify_device.set(device=DeviceId.GPU0 if torch.cuda.is_available() else DeviceId.CPU)

# Load models
print("\n" + "="*60)
print("LOADING MODELS...")
print("="*60)

# Load DeOldify
def load_deoldify():
    weights_path = os.path.join(MODELS_DIR, 'ColorizeArtistic_gen.pth')
    if not os.path.exists(weights_path):
        print("Downloading DeOldify weights...")
        torch.hub.download_url_to_file(
            'https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth', 
            weights_path
        )
    print("Loading colorizer...")
    return get_image_colorizer(artistic=True)

# Load Real-ESRGAN
def load_esrgan():
    print("Loading Real-ESRGAN...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        device=torch_device
    )

try:
    colorizer = load_deoldify()
    upsampler = load_esrgan()
    print("✓ All models loaded successfully!")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("PROCESSING IMAGES...")
print("(No windows will pop up during processing)")
print("="*60)

# Process each image
processed_results = []

for i, filename in enumerate(input_files, 1):
    try:
        print(f"\n[{i}/{len(input_files)}] Processing: {filename}")
        input_path = os.path.join(INPUT_DIR, filename)
        
        # 1. Load original
        original_img = Image.open(input_path).convert('RGB')
        original_np = np.array(original_img)
        
        # 2. Upscale with Real-ESRGAN
        print("  Upscaling...")
        with torch.no_grad():
            upscaled_np, _ = upsampler.enhance(original_np, outscale=4)
        
        # Save upscaled
        base_name = os.path.splitext(filename)[0]
        deblurred_path = os.path.join(RESULTS_DIR, f'deblurred_{base_name}.jpg')
        Image.fromarray(upscaled_np).save(deblurred_path, quality=95)
        print(f"  ✓ Upscaled saved")
        
        # 3. Colorize with DeOldify
        print("  Colorizing...")
        colorized_result = colorizer.plot_transformed_image(
            deblurred_path, 
            render_factor=35, 
            display_render_factor=False, 
            watermarked=False
        )
        
        # Handle result (could be Path or Image)
        if hasattr(colorized_result, 'save'):
            colorized_img = colorized_result
            colorized_path = os.path.join(RESULTS_DIR, f'colorized_{base_name}.jpg')
            colorized_img.save(colorized_path, quality=95)
        else:
            temp_path = str(colorized_result) if isinstance(colorized_result, Path) else colorized_result
            colorized_img = Image.open(temp_path)
            colorized_path = os.path.join(RESULTS_DIR, f'colorized_{base_name}.jpg')
            colorized_img.save(colorized_path, quality=95)
        
        print(f"  ✓ Colorized saved")
        
        # Store results
        processed_results.append({
            'name': filename,
            'original_path': input_path,
            'deblurred_path': deblurred_path,
            'colorized_path': colorized_path
        })
        
        print(f"  ✓ COMPLETE: {filename}")
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        traceback.print_exc()

print("\n" + "="*60)
print(f"PROCESSING COMPLETE!")
print(f"Successfully processed: {len(processed_results)}/{len(input_files)} images")
print("="*60)

# ================================================================
# SIMPLE DISPLAY USING MATPLOTLIB (More reliable than Tkinter)
# ================================================================

if processed_results:
    print("\nPreparing to display results...")
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        
        # Create a figure with subplots for each image
        num_images = len(processed_results)
        
        for i, result in enumerate(processed_results):
            print(f"\nDisplaying results for: {result['name']}")
            
            # Create a new figure for each image
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Image Restoration: {result['name']}", fontsize=16, fontweight='bold')
            
            # Load and display images
            original_img = mpimg.imread(result['original_path'])
            deblurred_img = mpimg.imread(result['deblurred_path'])
            colorized_img = mpimg.imread(result['colorized_path'])
            
            # Original image
            axes[0].imshow(original_img)
            axes[0].set_title("Original", fontweight='bold')
            axes[0].axis('off')
            
            # Deblurred/Upscaled image
            axes[1].imshow(deblurred_img)
            axes[1].set_title("Upscaled (4x)", fontweight='bold')
            axes[1].axis('off')
            
            # Colorized image
            axes[2].imshow(colorized_img)
            axes[2].set_title("Colorized", fontweight='bold')
            axes[2].axis('off')
            
            # Adjust layout and show
            plt.tight_layout()
            plt.show()
            
            # Ask if user wants to see next image
            if i < num_images - 1:
                input(f"\nPress Enter to see next image ({i+2}/{num_images})...")
        
        print("\n" + "="*60)
        print("ALL IMAGES DISPLAYED!")
        print("="*60)
        
    except ImportError:
        print("\nMatplotlib not found. Installing it automatically...")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
            
            print("Matplotlib installed successfully! Restarting display...")
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            
            # Now display images
            for i, result in enumerate(processed_results):
                print(f"\nDisplaying results for: {result['name']}")
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f"Image Restoration: {result['name']}", fontsize=16)
                
                original_img = mpimg.imread(result['original_path'])
                deblurred_img = mpimg.imread(result['deblurred_path'])
                colorized_img = mpimg.imread(result['colorized_path'])
                
                axes[0].imshow(original_img)
                axes[0].set_title("Original")
                axes[0].axis('off')
                
                axes[1].imshow(deblurred_img)
                axes[1].set_title("Upscaled (4x)")
                axes[1].axis('off')
                
                axes[2].imshow(colorized_img)
                axes[2].set_title("Colorized")
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.show()
                
                if i < len(processed_results) - 1:
                    input(f"\nPress Enter to see next image ({i+2}/{len(processed_results)})...")
            
            print("\n" + "="*60)
            print("ALL IMAGES DISPLAYED!")
            print("="*60)
            
        except Exception as e:
            print(f"Could not install matplotlib: {e}")
            print("\nShowing file locations instead:")
            for result in processed_results:
                print(f"\n{result['name']}:")
                print(f"  Original: {result['original_path']}")
                print(f"  Upscaled: {result['deblurred_path']}")
                print(f"  Colorized: {result['colorized_path']}")
    
    except Exception as e:
        print(f"Error displaying images: {e}")
        print("\nShowing file locations instead:")
        for result in processed_results:
            print(f"\n{result['name']}:")
            print(f"  Original: {result['original_path']}")
            print(f"  Upscaled: {result['deblurred_path']}")
            print(f"  Colorized: {result['colorized_path']}")

# ================================================================
# ALTERNATIVE: SIMPLE TKINTER DISPLAY (One image at a time)
# ================================================================

print("\n" + "="*60)
print("OPTION: Would you like to view images in a simple window?")
print("="*60)

view_in_window = input("View images in window? (y/n): ").strip().lower()

if view_in_window == 'y' and processed_results:
    try:
        import tkinter as tk
        from PIL import ImageTk
        
        class ImageViewer:
            def __init__(self, results):
                self.results = results
                self.current_index = 0
                
                # Create main window
                self.root = tk.Tk()
                self.root.title(f"Image Restoration Results (1/{len(results)})")
                
                # Load first image
                self.load_current_image()
                
                # Navigation buttons
                btn_frame = tk.Frame(self.root)
                btn_frame.pack(pady=10)
                
                if len(results) > 1:
                    prev_btn = tk.Button(btn_frame, text="← Previous", command=self.prev_image)
                    prev_btn.pack(side='left', padx=5)
                    
                    next_btn = tk.Button(btn_frame, text="Next →", command=self.next_image)
                    next_btn.pack(side='left', padx=5)
                
                close_btn = tk.Button(btn_frame, text="Close", command=self.root.destroy)
                close_btn.pack(side='left', padx=5)
                
                # Center window
                self.center_window()
                
                # Start main loop
                self.root.mainloop()
            
            def load_current_image(self):
                result = self.results[self.current_index]
                
                # Clear previous widgets
                for widget in self.root.winfo_children():
                    if widget.winfo_class() != 'Frame':  # Keep button frame
                        widget.destroy()
                
                # Load images
                original_img = Image.open(result['original_path'])
                deblurred_img = Image.open(result['deblurred_path'])
                colorized_img = Image.open(result['colorized_path'])
                
                # Resize for display
                def resize(img, max_size=300):
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    return img
                
                orig_disp = resize(original_img.copy())
                deblur_disp = resize(deblurred_img.copy())
                color_disp = resize(colorized_img.copy())
                
                # Convert to PhotoImage
                orig_photo = ImageTk.PhotoImage(orig_disp)
                deblur_photo = ImageTk.PhotoImage(deblur_disp)
                color_photo = ImageTk.PhotoImage(color_disp)
                
                # Title
                title = tk.Label(self.root, text=result['name'], font=("Arial", 14, "bold"))
                title.pack(pady=10)
                
                # Images frame
                img_frame = tk.Frame(self.root)
                img_frame.pack()
                
                # Display images
                tk.Label(img_frame, text="Original").pack()
                tk.Label(img_frame, image=orig_photo).pack(side='left', padx=10)
                tk.Label(img_frame, text="Upscaled").pack()
                tk.Label(img_frame, image=deblur_photo).pack(side='left', padx=10)
                tk.Label(img_frame, text="Colorized").pack()
                tk.Label(img_frame, image=color_photo).pack(side='left', padx=10)
                
                # Store references
                self.current_photos = [orig_photo, deblur_photo, color_photo]
                
                # Update window title
                self.root.title(f"Image Restoration Results ({self.current_index + 1}/{len(self.results)})")
            
            def next_image(self):
                if self.current_index < len(self.results) - 1:
                    self.current_index += 1
                    self.load_current_image()
            
            def prev_image(self):
                if self.current_index > 0:
                    self.current_index -= 1
                    self.load_current_image()
            
            def center_window(self):
                self.root.update_idletasks()
                width = self.root.winfo_width()
                height = self.root.winfo_height()
                x = (self.root.winfo_screenwidth() // 2) - (width // 2)
                y = (self.root.winfo_screenheight() // 2) - (height // 2)
                self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Start viewer
        print("\nOpening image viewer...")
        ImageViewer(processed_results)
        
    except Exception as e:
        print(f"Error opening image viewer: {e}")

print("\n" + "="*60)
print("Program finished successfully!")
print(f"Check the 'results' folder for your images:")
print(f"  {RESULTS_DIR}")
print("="*60)