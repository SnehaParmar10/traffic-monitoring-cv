"""
TASK 1: EDGE DETECTION
Applies Sobel and Canny edge detectors and compares their quality
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys

# Image path
IMAGE_PATH = r"C:\Users\gtcam\OneDrive\Pictures\Camera Roll\OIP (2).webp"

def load_image(path):
    """Load image and convert to grayscale"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_rgb, img_gray

def apply_sobel(img_gray):
    """
    Apply Sobel edge detector
    Computes gradients in X and Y directions
    """
    print("[*] Applying Sobel operator...")
    
    # Compute gradients in X and Y directions
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    
    # Calculate magnitude (combine X and Y gradients)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))
    
    # Calculate direction
    sobel_direction = np.arctan2(sobely, sobelx)
    
    print(f"    ✓ Sobel magnitude shape: {sobel_magnitude.shape}")
    print(f"    ✓ Edge pixels detected: {np.count_nonzero(sobel_magnitude > 50)}")
    
    return sobel_magnitude, sobel_direction, sobelx, sobely

def apply_canny(img_gray, threshold1=50, threshold2=150):
    """
    Apply Canny edge detector
    Multi-stage algorithm for edge detection with better localization
    """
    print("[*] Applying Canny edge detector...")
    
    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1.5)
    
    # Apply Canny edge detection
    canny_edges = cv2.Canny(img_blur, threshold1, threshold2)
    
    print(f"    ✓ Canny edges shape: {canny_edges.shape}")
    print(f"    ✓ Edge pixels detected: {np.count_nonzero(canny_edges)}")
    
    return canny_edges

def compare_edge_quality(sobel_mag, canny_edges):
    """
    Compare quality of edge detection methods
    Analyzes edge density and properties
    """
    print("\n[*] Comparing edge detection quality...")
    
    # Normalize Sobel for comparison
    sobel_norm = np.uint8(255 * sobel_mag / np.max(sobel_mag))
    
    # Count edge pixels
    sobel_edge_count = np.count_nonzero(sobel_norm > 50)
    canny_edge_count = np.count_nonzero(canny_edges)
    
    # Calculate densities
    total_pixels = sobel_norm.size
    sobel_density = sobel_edge_count / total_pixels
    canny_density = canny_edge_count / total_pixels
    
    # Calculate edge connectivity (for Canny)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    canny_dilated = cv2.dilate(canny_edges, kernel, iterations=1)
    
    comparison = {
        'Sobel_edge_pixels': sobel_edge_count,
        'Canny_edge_pixels': canny_edge_count,
        'Sobel_density': sobel_density,
        'Canny_density': canny_density,
        'edge_ratio': canny_edge_count / sobel_edge_count if sobel_edge_count > 0 else 0,
        'total_pixels': total_pixels
    }
    
    print(f"\n    ═══ EDGE DETECTION COMPARISON ═══")
    print(f"    Sobel Statistics:")
    print(f"      • Edge pixels: {sobel_edge_count}")
    print(f"      • Edge density: {sobel_density:.6f}")
    print(f"      • Characteristics: Continuous gradients, sensitive to noise")
    
    print(f"\n    Canny Statistics:")
    print(f"      • Edge pixels: {canny_edge_count}")
    print(f"      • Edge density: {canny_density:.6f}")
    print(f"      • Characteristics: Thin, well-localized edges")
    
    print(f"\n    Quality Assessment:")
    if canny_density > sobel_density * 1.5:
        print(f"      ✓ Canny: Better for precise edge localization (density ratio: {comparison['edge_ratio']:.2f})")
    else:
        print(f"      ✓ Sobel: Captures more gradient information (density ratio: {comparison['edge_ratio']:.2f})")
    
    return comparison

def visualize_edge_detection(img_rgb, img_gray, sobel_mag, canny_edges, sobelx, sobely):
    """Visualize edge detection results"""
    print("\n[*] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('TASK 1: EDGE DETECTION - Sobel vs Canny', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Grayscale
    axes[0, 1].imshow(img_gray, cmap='gray')
    axes[0, 1].set_title('Grayscale Image', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Sobel magnitude
    axes[0, 2].imshow(sobel_mag, cmap='hot')
    axes[0, 2].set_title('Sobel Magnitude', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Sobel X (horizontal edges)
    sobelx_norm = np.uint8(255 * np.abs(sobelx) / np.max(np.abs(sobelx)))
    axes[1, 0].imshow(sobelx_norm, cmap='gray')
    axes[1, 0].set_title('Sobel X (Horizontal Edges)', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Sobel Y (vertical edges)
    sobely_norm = np.uint8(255 * np.abs(sobely) / np.max(np.abs(sobely)))
    axes[1, 1].imshow(sobely_norm, cmap='gray')
    axes[1, 1].set_title('Sobel Y (Vertical Edges)', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Canny edges
    axes[1, 2].imshow(canny_edges, cmap='gray')
    axes[1, 2].set_title('Canny Edges', fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = r'c:\Users\gtcam\OneDrive\Desktop\Assignment-4\task1_edge_detection_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ Visualization saved to: {output_path}")
    
    plt.show()

def main():
    print("="*70)
    print("TASK 1: EDGE DETECTION")
    print("="*70)
    
    # Load image
    print("\n[*] Loading image...")
    img_rgb, img_gray = load_image(IMAGE_PATH)
    print(f"    ✓ Image shape: {img_rgb.shape}")
    
    # Apply Sobel
    sobel_mag, sobel_dir, sobelx, sobely = apply_sobel(img_gray)
    
    # Apply Canny
    canny_edges = apply_canny(img_gray, threshold1=50, threshold2=150)
    
    # Compare quality
    comparison = compare_edge_quality(sobel_mag, canny_edges)
    
    # Visualize
    visualize_edge_detection(img_rgb, img_gray, sobel_mag, canny_edges, sobelx, sobely)
    
    print("\n" + "="*70)
    print("✓ TASK 1 COMPLETED")
    print("="*70)

if __name__ == "__main__":
    main()
