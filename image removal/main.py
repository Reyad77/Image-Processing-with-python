import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import os
import sys
import argparse

def load_image(image_path):
   
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found!")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image. Check file format!")
        return None
    
    print(f"Image loaded successfully!")
    print(f"Image dimensions: {image.shape[1]} x {image.shape[0]} pixels")
    print(f"Color channels: {image.shape[2] if len(image.shape) == 3 else 1}")
    
    return image

def display_image_with_selection(image, title="Select Object to Remove"):
    """
    Display image and let user select region to remove
    
    Args:
        image: Input image
        title: Plot title
    
    Returns:
        Tuple of (start_point, end_point) or None if cancelled
    """
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title(title)
    ax.axis('off')
    
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("1. Click and drag to select the unwanted object")
    print("2. Adjust the rectangle if needed")
    print("3. Press 'Enter' to confirm selection")
    print("4. Press 'Esc' to cancel")
    print("="*60 + "\n")
    
    # Store selection coordinates
    selection_coords = {'start': None, 'end': None}
    
    def onselect(eclick, erelease):
        """Handle rectangle selection"""
        # Store coordinates
        selection_coords['start'] = (int(eclick.xdata), int(eclick.ydata))
        selection_coords['end'] = (int(erelease.xdata), int(erelease.ydata))
        
        # Print selection info
        x1, y1 = selection_coords['start']
        x2, y2 = selection_coords['end']
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        print(f"\nSelected area: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"Selection size: {width} x {height} pixels")
    
    def onpress(event):
        """Handle key press events"""
        if event.key == 'enter':
            print("\nSelection confirmed!")
            plt.close(fig)
        elif event.key == 'escape':
            print("\nSelection cancelled!")
            selection_coords['start'] = None
            selection_coords['end'] = None
            plt.close(fig)
    
    # Create rectangle selector
    rect_selector = RectangleSelector(
        ax, onselect,
        useblit=True,
        button=[1],  # Left mouse button
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True
    )
    
    # Connect key press event
    fig.canvas.mpl_connect('key_press_event', onpress)
    
    plt.show()
    
    if selection_coords['start'] and selection_coords['end']:
        return selection_coords['start'], selection_coords['end']
    return None

def create_mask(image_shape, start_point, end_point):
    """
    Create a binary mask for the selected region
    
    Args:
        image_shape: Shape of the original image
        start_point: (x1, y1) coordinates
        end_point: (x2, y2) coordinates
    
    Returns:
        Binary mask (white=object to remove, black=keep)
    """
    # Ensure coordinates are within bounds
    height, width = image_shape[:2]
    
    x1 = max(0, min(start_point[0], end_point[0]))
    y1 = max(0, min(start_point[1], end_point[1]))
    x2 = min(width, max(start_point[0], end_point[0]))
    y2 = min(height, max(start_point[1], end_point[1]))
    
    # Create empty mask (black)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw white rectangle for selected area
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    # Add border to mask for better blending
    border_size = 5
    kernel = np.ones((border_size, border_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    return mask, (x1, y1, x2, y2)

def remove_object_simple(image, mask):
    """
    Remove object using OpenCV's inpainting (fast method)
    
    Args:
        image: Input image
        mask: Binary mask indicating region to remove
    
    Returns:
        Image with object removed
    """
    print("\nRemoving object using simple inpainting...")
    
    # Apply inpainting
    result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return result

def remove_object_advanced(image, mask):
    """
    Remove object using advanced inpainting (better quality)
    
    Args:
        image: Input image
        mask: Binary mask indicating region to remove
    
    Returns:
        Image with object removed
    """
    print("\nRemoving object using advanced inpainting...")
    
    # Apply inpainting with Navier-Stokes method (better for textures)
    result = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    
    return result

def remove_object_clone(image, mask, bbox):
    """
    Remove object using cloning from surrounding areas
    
    Args:
        image: Input image
        mask: Binary mask
        bbox: (x1, y1, x2, y2) bounding box
    
    Returns:
        Image with object removed
    """
    print("\nRemoving object using cloning method...")
    
    x1, y1, x2, y2 = bbox
    height, width = image.shape[:2]
    
    # Create a larger region for sampling
    expand = 30
    sample_x1 = max(0, x1 - expand)
    sample_y1 = max(0, y1 - expand)
    sample_x2 = min(width, x2 + expand)
    sample_y2 = min(height, y2 + expand)
    
    # Create destination region (where to place cloned content)
    dest_region = image[y1:y2, x1:x2].copy()
    
    # Get source region from surrounding area
    source_region = image[sample_y1:sample_y2, sample_x1:sample_x2].copy()
    
    # Create a mask for the source region
    source_mask = np.zeros((sample_y2-sample_y1, sample_x2-sample_x1), dtype=np.uint8)
    
    # Center of source region
    center_x = source_mask.shape[1] // 2
    center_y = source_mask.shape[0] // 2
    
    # Size of destination
    dest_width = x2 - x1
    dest_height = y2 - y1
    
    # Create elliptical mask
    cv2.ellipse(source_mask, 
                (center_x, center_y),
                (dest_width//2, dest_height//2),
                0, 0, 360, 255, -1)
    
    # Clone the region
    clone_result = cv2.seamlessClone(
        source_region,
        image,
        source_mask,
        (x1 + dest_width//2, y1 + dest_height//2),
        cv2.NORMAL_CLONE
    )
    
    return clone_result

def smooth_transition(original, modified, mask, feather=30, blur_ksize=21):
    """
    Smoothly blend `modified` into `original` inside `mask` using a
    distance-based feather (soft alpha) to hide seams.

    Args:
        original: Original BGR image (H,W,3)
        modified: Image after removal/inpainting (H,W,3)
        mask: Binary mask (H,W) with 255 for removed region
        feather: Distance (px) over which to blend from edge inward
        blur_ksize: Gaussian blur kernel size for alpha smoothing (odd)

    Returns:
        Blended uint8 image
    """
    # Ensure mask is single-channel uint8 with 0/255
    mask_bin = (mask > 0).astype(np.uint8) * 255

    # If mask empty, return modified
    if mask_bin.sum() == 0:
        return modified

    # Distance transform inside mask
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5).astype(np.float32)

    # Normalize by feather distance to get alpha in [0,1]
    if feather <= 0:
        feather = 1
    alpha = np.clip(dist / float(feather), 0.0, 1.0)

    # Smooth alpha to avoid hard transitions
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    alpha = cv2.GaussianBlur(alpha, (blur_ksize, blur_ksize), 0)

    # Expand to 3 channels
    alpha_3 = cv2.merge([alpha, alpha, alpha])

    orig_f = original.astype(np.float32)
    mod_f = modified.astype(np.float32)

    blended = (mod_f * alpha_3) + (orig_f * (1.0 - alpha_3))
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended

def compare_results(original, mask, results_dict):
    """
    Display comparison of different removal methods
    
    Args:
        original: Original image
        mask: Binary mask
        results_dict: Dictionary of {method_name: result_image}
    """
    num_methods = len(results_dict)
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Plot original image
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Plot mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Selection Mask")
    axes[1].axis('off')
    
    # Plot results from different methods
    for i, (method_name, result) in enumerate(results_dict.items()):
        axes[i+2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[i+2].set_title(f"{method_name.capitalize()} Method")
        axes[i+2].axis('off')
    
    # Hide empty subplots
    for i in range(num_methods + 2, 6):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_results(image, mask, results_dict, base_filename):
    """
    Save all results to files
    
    Args:
        image: Original image
        mask: Binary mask
        results_dict: Dictionary of results
        base_filename: Base name for output files
    """
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save original
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_original.jpg"), image)

    # Save mask (use PNG to preserve binary mask cleanly)
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_mask.png"), mask)
    
    # Save all results
    for method_name, result in results_dict.items():
        filename = f"{output_dir}/{base_filename}_{method_name}.jpg"
        cv2.imwrite(filename, result)
        print(f"Saved: {filename}")
    
    print(f"\nAll results saved in '{output_dir}/' directory!")

def demo_with_sample_image():
    """
    Run a demo with a sample image (if no image is provided)
    """
    print("\n" + "="*60)
    print("RUNNING DEMO WITH SAMPLE IMAGE")
    print("="*60)
    
    # Create a sample image
    sample_image = create_sample_image()
    
    # Define "object" to remove (a red circle)
    start_point = (120, 100)
    end_point = (180, 160)
    
    # Create mask
    mask, bbox = create_mask(sample_image.shape, start_point, end_point)
    
    # Apply different removal methods
    results = {}
    results['simple'] = remove_object_simple(sample_image, mask)
    results['advanced'] = remove_object_advanced(sample_image, mask)
    results['cloning'] = remove_object_clone(sample_image, mask, bbox)

    # Apply smoothing blend to each result to soften seams
    for k in list(results.keys()):
        results[k] = smooth_transition(sample_image, results[k], mask)
    
    # Display results
    compare_results(sample_image, mask, results)
    
    # Save results
    save_results(sample_image, mask, results, "demo")
    
    return sample_image, mask, results

def create_sample_image():
    """
    Create a sample image for demonstration
    """
    # Create a 300x400 image with gradient background
    height, width = 300, 400
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient background
    for i in range(height):
        for j in range(width):
            # Blue-green gradient
            image[i, j] = [j//2, i//2, 100]
    
    # Add some objects
    # Red circle (our "unwanted object")
    cv2.circle(image, (150, 130), 30, (0, 0, 255), -1)
    
    # Some other objects
    cv2.rectangle(image, (50, 50), (100, 100), (0, 255, 0), -1)  # Green square
    cv2.rectangle(image, (250, 200), (300, 250), (255, 255, 0), -1)  # Cyan square
    cv2.line(image, (200, 50), (350, 150), (255, 0, 255), 3)  # Purple line
    
    # Add some text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Sample Image', (100, 280), font, 0.7, (255, 255, 255), 2)
    
    return image

def get_user_choice():
    """
    Get user's choice for removal method
    """
    print("\n" + "="*60)
    print("CHOOSE REMOVAL METHOD:")
    print("1. Simple Inpainting (Fastest)")
    print("2. Advanced Inpainting (Better quality)")
    print("3. Cloning (Best for complex backgrounds)")
    print("4. Try ALL methods and compare")
    print("="*60)
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-4): "))
            if 1 <= choice <= 4:
                return choice
            else:
                print("Please enter a number between 1 and 4")
        except ValueError:
            print("Please enter a valid number")

def main():
    """
    Main function - runs the object removal application
    """
    print("\n" + "="*60)
    print("UNWANTED OBJECT REMOVAL TOOL")
    print("Digital Image Processing Project")
    print("="*60)
    
    parser = argparse.ArgumentParser(description='Unwanted object removal')
    parser.add_argument('image', nargs='?', default=None, help='Path to input image')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample image')
    args = parser.parse_args()

    # If path provided on command line, use it. If --demo set, run demo.
    if args.demo:
        demo_with_sample_image()
        return

    image_path = args.image

    # If no CLI arg and stdin is interactive, prompt the user; otherwise run demo
    if image_path is None:
        if sys.stdin is None or not sys.stdin.isatty():
            demo_with_sample_image()
            return
        print("\nEnter the path to your image file.")
        print("Example: 'images/myphoto.jpg' or 'C:/Photos/picture.jpg'")
        print("Or press Enter to run a demo with a sample image.")
        image_path = input("\nImage path: ").strip()
        if image_path == "":
            demo_with_sample_image()
            return
    
    # Load image
    image = load_image(image_path)
    if image is None:
        print("\nRunning demo instead...")
        demo_with_sample_image()
        return
    
    # Display image and get selection
    coords = display_image_with_selection(image)
    if coords is None:
        print("No region selected. Exiting...")
        return
    
    start_point, end_point = coords
    
    # Create mask
    mask, bbox = create_mask(image.shape, start_point, end_point)
    
    # Get user's choice of method
    choice = get_user_choice()
    
    # Apply selected method(s)
    results = {}
    
    if choice == 1:
        results['simple'] = remove_object_simple(image, mask)
    elif choice == 2:
        results['advanced'] = remove_object_advanced(image, mask)
    elif choice == 3:
        results['cloning'] = remove_object_clone(image, mask, bbox)
    elif choice == 4:
        results['simple'] = remove_object_simple(image, mask)
        results['advanced'] = remove_object_advanced(image, mask)
        results['cloning'] = remove_object_clone(image, mask, bbox)

    # Post-process: smooth blend each result to hide seams
    for method_name in list(results.keys()):
        results[method_name] = smooth_transition(image, results[method_name], mask)
    
    # Display comparison
    compare_results(image, mask, results)
    
    # Save results
    save_option = input("\nSave results? (y/n): ").lower()
    if save_option == 'y':
        # Extract base filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_results(image, mask, results, base_name)
    
    print("\n" + "="*60)
    print("PROGRAM COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Ask if user wants to process another image
    another = input("\nProcess another image? (y/n): ").lower()
    if another == 'y':
        main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please check your input and try again.")