"""
TASK 3: FEATURE EXTRACTION
Applies ORB, SIFT, and SURF feature detectors
Visualizes keypoints and descriptors
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
    """Load image"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_rgb, img_gray

def apply_orb(img_gray, n_features=500):
    """
    Apply ORB (Oriented FAST and Rotated BRIEF) feature detector
    
    Characteristics:
    - Binary descriptors (fast comparison)
    - Rotation invariant
    - Scale invariant
    - Fast and efficient for real-time applications
    """
    print("[*] Applying ORB (Oriented FAST and Rotated BRIEF)...")
    
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)
    
    print(f"    ✓ ORB keypoints detected: {len(keypoints)}")
    if descriptors is not None:
        print(f"    ✓ Descriptor shape: {descriptors.shape}")
        print(f"    ✓ Descriptor type: Binary")
    
    return keypoints, descriptors, 'ORB'

def apply_sift(img_gray):
    """
    Apply SIFT (Scale-Invariant Feature Transform) feature detector
    
    Characteristics:
    - Scale invariant
    - Rotation invariant
    - Highly distinctive descriptors (Float-128)
    - Computationally more expensive
    - Patent-free since 2020
    """
    print("[*] Applying SIFT (Scale-Invariant Feature Transform)...")
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    
    print(f"    ✓ SIFT keypoints detected: {len(keypoints)}")
    if descriptors is not None:
        print(f"    ✓ Descriptor shape: {descriptors.shape}")
        print(f"    ✓ Descriptor type: Float (128-dimensional)")
    
    return keypoints, descriptors, 'SIFT'

def apply_surf(img_gray, hessian_threshold=400):
    """
    Apply SURF (Speeded Up Robust Features) feature detector
    
    Characteristics:
    - Similar to SIFT but faster
    - Scale and rotation invariant
    - Uses Hessian matrix for detection
    - Good balance between speed and quality
    """
    print("[*] Applying SURF (Speeded Up Robust Features)...")
    
    try:
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
        keypoints, descriptors = surf.detectAndCompute(img_gray, None)
        
        print(f"    ✓ SURF keypoints detected: {len(keypoints)}")
        if descriptors is not None:
            print(f"    ✓ Descriptor shape: {descriptors.shape}")
            print(f"    ✓ Descriptor type: Float")
        
        return keypoints, descriptors, 'SURF'
    
    except Exception as e:
        print(f"    ✗ SURF not available: {str(e)}")
        return [], None, 'SURF'

def visualize_keypoints(img_rgb, keypoints, detector_name):
    """
    Visualize keypoints on the image
    """
    img_with_kp = cv2.drawKeypoints(
        img_rgb, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return img_with_kp

def analyze_keypoint_distribution(keypoints, img_shape):
    """
    Analyze the spatial distribution of keypoints
    """
    if not keypoints:
        return None
    
    # Extract positions
    positions = np.array([kp.pt for kp in keypoints])
    
    # Calculate statistics
    height, width = img_shape[:2]
    
    distribution = {
        'total_keypoints': len(keypoints),
        'mean_x': np.mean(positions[:, 0]),
        'mean_y': np.mean(positions[:, 1]),
        'std_x': np.std(positions[:, 0]),
        'std_y': np.std(positions[:, 1]),
        'spread_x': np.max(positions[:, 0]) - np.min(positions[:, 0]),
        'spread_y': np.max(positions[:, 1]) - np.min(positions[:, 1]),
        'coverage_ratio': (distribution['spread_x'] * distribution['spread_y']) / (width * height) if 'spread_x' in locals() else 0
    }
    
    return distribution, positions

def analyze_keypoint_scales(keypoints):
    """
    Analyze the scale distribution of keypoints
    """
    if not keypoints:
        return None
    
    # Extract scales (sizes)
    scales = np.array([kp.size for kp in keypoints])
    
    scale_analysis = {
        'mean_scale': np.mean(scales),
        'median_scale': np.median(scales),
        'min_scale': np.min(scales),
        'max_scale': np.max(scales),
        'scale_range': np.max(scales) - np.min(scales),
        'scale_std': np.std(scales)
    }
    
    return scale_analysis

def print_feature_statistics(kp_orb, kp_sift, kp_surf):
    """Print detailed statistics about feature extraction"""
    print("\n" + "="*70)
    print("FEATURE EXTRACTION STATISTICS")
    print("="*70)
    
    print(f"\n    ═══ KEYPOINT COUNTS ═══")
    print(f"    ORB:  {len(kp_orb)} keypoints")
    print(f"    SIFT: {len(kp_sift)} keypoints")
    print(f"    SURF: {len(kp_surf)} keypoints")
    
    print(f"\n    ═══ ORB ANALYSIS ═══")
    if kp_orb:
        scales = np.array([kp.size for kp in kp_orb])
        angles = np.array([kp.angle for kp in kp_orb])
        print(f"      • Keypoints: {len(kp_orb)}")
        print(f"      • Mean scale: {np.mean(scales):.2f}")
        print(f"      • Scale range: {np.min(scales):.2f} - {np.max(scales):.2f}")
        print(f"      • Mean orientation: {np.mean(angles):.2f}°")
    
    print(f"\n    ═══ SIFT ANALYSIS ═══")
    if kp_sift:
        scales = np.array([kp.size for kp in kp_sift])
        angles = np.array([kp.angle for kp in kp_sift])
        print(f"      • Keypoints: {len(kp_sift)}")
        print(f"      • Mean scale: {np.mean(scales):.2f}")
        print(f"      • Scale range: {np.min(scales):.2f} - {np.max(scales):.2f}")
        print(f"      • Mean orientation: {np.mean(angles):.2f}°")
    
    print(f"\n    ═══ SURF ANALYSIS ═══")
    if kp_surf:
        scales = np.array([kp.size for kp in kp_surf])
        angles = np.array([kp.angle for kp in kp_surf])
        print(f"      • Keypoints: {len(kp_surf)}")
        print(f"      • Mean scale: {np.mean(scales):.2f}")
        print(f"      • Scale range: {np.min(scales):.2f} - {np.max(scales):.2f}")
        print(f"      • Mean orientation: {np.mean(angles):.2f}°")

def visualize_features(img_rgb, kp_orb, kp_sift, kp_surf, desc_orb, desc_sift, desc_surf):
    """Visualize feature extraction results"""
    print("\n[*] Creating visualizations...")
    
    # Draw keypoints
    img_orb_kp = visualize_keypoints(img_rgb, kp_orb, 'ORB')
    img_sift_kp = visualize_keypoints(img_rgb, kp_sift, 'SIFT')
    img_surf_kp = visualize_keypoints(img_rgb, kp_surf, 'SURF') if kp_surf else None
    
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Keypoint visualizations
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_orb_kp)
    ax1.set_title(f'ORB Keypoints ({len(kp_orb)} detected)', fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_sift_kp)
    ax2.set_title(f'SIFT Keypoints ({len(kp_sift)} detected)', fontweight='bold')
    ax2.axis('off')
    
    if img_surf_kp is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(img_surf_kp)
        ax3.set_title(f'SURF Keypoints ({len(kp_surf)} detected)', fontweight='bold')
        ax3.axis('off')
    
    # Row 2: Scale distributions
    ax4 = fig.add_subplot(gs[1, 0])
    if kp_orb:
        scales_orb = [kp.size for kp in kp_orb]
        ax4.hist(scales_orb, bins=15, color='#FF6B6B', edgecolor='black', alpha=0.7)
        ax4.set_title('ORB Scale Distribution', fontweight='bold')
        ax4.set_xlabel('Scale')
        ax4.set_ylabel('Count')
    
    ax5 = fig.add_subplot(gs[1, 1])
    if kp_sift:
        scales_sift = [kp.size for kp in kp_sift]
        ax5.hist(scales_sift, bins=15, color='#4ECDC4', edgecolor='black', alpha=0.7)
        ax5.set_title('SIFT Scale Distribution', fontweight='bold')
        ax5.set_xlabel('Scale')
        ax5.set_ylabel('Count')
    
    ax6 = fig.add_subplot(gs[1, 2])
    if kp_surf:
        scales_surf = [kp.size for kp in kp_surf]
        ax6.hist(scales_surf, bins=15, color='#45B7D1', edgecolor='black', alpha=0.7)
        ax6.set_title('SURF Scale Distribution', fontweight='bold')
        ax6.set_xlabel('Scale')
        ax6.set_ylabel('Count')
    
    # Row 3: Comparative bar charts
    ax7 = fig.add_subplot(gs[2, 0])
    detectors = ['ORB', 'SIFT', 'SURF']
    keypoints_count = [len(kp_orb), len(kp_sift), len(kp_surf)]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax7.bar(detectors, keypoints_count, color=colors, edgecolor='black', alpha=0.7, linewidth=2)
    ax7.set_ylabel('Number of Keypoints', fontweight='bold')
    ax7.set_title('Keypoint Detection Comparison', fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Descriptor information
    ax8 = fig.add_subplot(gs[2, 1])
    descriptor_info = ['Binary' if desc_orb is not None else 'None',
                      'Float-128' if desc_sift is not None else 'None',
                      'Float-64' if desc_surf is not None else 'None']
    y_pos = np.arange(len(detectors))
    ax8.barh(y_pos, [1, 1, 1] if kp_surf else [1, 1, 0], color=colors[:2+bool(kp_surf)], 
             alpha=0.7, edgecolor='black', linewidth=2)
    ax8.set_yticks(y_pos)
    ax8.set_yticklabels(detectors)
    ax8.set_xticks([])
    ax8.set_title('Descriptor Types', fontweight='bold')
    for i, (det, desc) in enumerate(zip(detectors, descriptor_info)):
        ax8.text(0.5, i, desc, va='center', ha='center', fontweight='bold', color='white')
    
    # Feature characteristics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    characteristics = [
        "ORB: Fast, Binary, Rotation-Invariant",
        "SIFT: Scale-Invariant, Distinctive",
        "SURF: Fast SIFT Alternative"
    ]
    y_start = 0.9
    for i, char in enumerate(characteristics):
        ax9.text(0.1, y_start - i*0.25, f"• {char}", fontsize=10, fontweight='bold',
                transform=ax9.transAxes, wrap=True)
    ax9.set_title('Feature Properties', fontweight='bold')
    
    fig.suptitle('TASK 3: FEATURE EXTRACTION - ORB, SIFT, SURF', 
                 fontsize=16, fontweight='bold')
    
    output_path = r'c:\Users\gtcam\OneDrive\Desktop\Assignment-4\task3_feature_extraction_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ Visualization saved to: {output_path}")
    
    plt.show()

def main():
    print("="*70)
    print("TASK 3: FEATURE EXTRACTION")
    print("="*70)
    
    # Load image
    print("\n[*] Loading image...")
    img_rgb, img_gray = load_image(IMAGE_PATH)
    print(f"    ✓ Image shape: {img_rgb.shape}")
    
    # Apply feature detectors
    print("\n" + "─"*70)
    kp_orb, desc_orb, _ = apply_orb(img_gray, n_features=500)
    kp_sift, desc_sift, _ = apply_sift(img_gray)
    kp_surf, desc_surf, _ = apply_surf(img_gray)
    
    # Print statistics
    print_feature_statistics(kp_orb, kp_sift, kp_surf)
    
    # Visualize
    visualize_features(img_rgb, kp_orb, kp_sift, kp_surf, desc_orb, desc_sift, desc_surf)
    
    print("\n" + "="*70)
    print("✓ TASK 3 COMPLETED")
    print("="*70)

if __name__ == "__main__":
    main()
