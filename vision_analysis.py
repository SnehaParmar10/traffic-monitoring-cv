"""
Computer Vision Analysis Script
Performs edge detection, object representation, feature extraction, and comparative analysis
on traffic/scene images for object detection and tracking purposes.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Image path
IMAGE_PATH = r"C:\Users\gtcam\OneDrive\Pictures\Camera Roll\OIP (2).webp"

def load_image(path):
    """Load image and convert to RGB"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_rgb, img_gray

# ============================================================================
# TASK 1: EDGE DETECTION
# ============================================================================

def apply_sobel(img_gray):
    """Apply Sobel edge detector"""
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    
    # Calculate magnitude
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))
    
    # Calculate direction
    sobel_direction = np.arctan2(sobely, sobelx)
    
    return sobel_magnitude, sobel_direction, sobelx, sobely

def apply_canny(img_gray, threshold1=50, threshold2=150):
    """Apply Canny edge detector"""
    canny_edges = cv2.Canny(img_gray, threshold1, threshold2)
    return canny_edges

def compare_edge_quality(sobel_mag, canny_edges):
    """Compare edge detection quality"""
    # Normalize for comparison
    sobel_norm = np.uint8(255 * sobel_mag / np.max(sobel_mag))
    
    # Calculate edge density and clarity
    sobel_edge_count = np.count_nonzero(sobel_norm > 50)
    canny_edge_count = np.count_nonzero(canny_edges)
    
    comparison = {
        'Sobel_edge_pixels': sobel_edge_count,
        'Canny_edge_pixels': canny_edge_count,
        'Sobel_density': sobel_edge_count / sobel_norm.size,
        'Canny_density': canny_edge_count / canny_edges.size,
    }
    return comparison

# ============================================================================
# TASK 2: OBJECT REPRESENTATION
# ============================================================================

def detect_contours(img_gray):
    """Detect contours in the image"""
    # Apply morphological operations to enhance contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_morph = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
    img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel)
    
    # Threshold
    _, thresh = cv2.threshold(img_morph, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, thresh

def draw_bounding_boxes(img_rgb, contours, min_area=100):
    """Draw bounding boxes around detected objects"""
    img_bbox = img_rgb.copy()
    object_data = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter small contours
        if area < min_area:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Get convex hull
        hull = cv2.convexHull(contour)
        
        # Draw bounding box (green)
        cv2.rectangle(img_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw convex hull (blue)
        cv2.drawContours(img_bbox, [hull], 0, (255, 0, 0), 2)
        
        # Calculate properties
        perimeter = cv2.arcLength(contour, True)
        
        object_data.append({
            'coordinates': (x, y, w, h),
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': w / h if h > 0 else 0
        })
    
    return img_bbox, object_data

def compute_object_properties(contours, min_area=100):
    """Compute area and perimeter for all detected objects"""
    properties = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate circularity (shape descriptor)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        properties.append({
            'object_id': i,
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity
        })
    
    return properties

# ============================================================================
# TASK 3: FEATURE EXTRACTION
# ============================================================================

def apply_orb(img_gray, n_features=500):
    """Apply ORB feature detector"""
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)
    return keypoints, descriptors, 'ORB'

def apply_sift(img_gray):
    """Apply SIFT feature detector"""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    return keypoints, descriptors, 'SIFT'

def apply_surf(img_gray, hessian_threshold=400):
    """Apply SURF feature detector"""
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
    keypoints, descriptors = surf.detectAndCompute(img_gray, None)
    return keypoints, descriptors, 'SURF'

def visualize_keypoints(img_rgb, keypoints, detector_name):
    """Visualize keypoints on the image"""
    img_with_kp = cv2.drawKeypoints(
        img_rgb, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return img_with_kp

def compare_feature_extractors(kp_orb, desc_orb, kp_sift, desc_sift, kp_surf, desc_surf):
    """Compare feature extractors"""
    comparison = {
        'ORB': {
            'keypoints': len(kp_orb),
            'descriptor_size': desc_orb.shape if desc_orb is not None else 0,
            'descriptor_type': 'Binary'
        },
        'SIFT': {
            'keypoints': len(kp_sift),
            'descriptor_size': desc_sift.shape if desc_sift is not None else 0,
            'descriptor_type': 'Float'
        },
        'SURF': {
            'keypoints': len(kp_surf),
            'descriptor_size': desc_surf.shape if desc_surf is not None else 0,
            'descriptor_type': 'Float'
        }
    }
    return comparison

# ============================================================================
# TASK 4: COMPARATIVE ANALYSIS & TRAFFIC MONITORING APPLICATION
# ============================================================================

def traffic_monitoring_analysis(edge_comparison, feature_comparison):
    """
    Analyze how features help in traffic monitoring
    """
    analysis = {
        'edge_detection': {
            'Sobel': {
                'use_case': 'Lane detection, Road outline detection',
                'advantages': 'Fast, good for directional edges',
                'disadvantages': 'Sensitive to noise'
            },
            'Canny': {
                'use_case': 'Vehicle boundary detection, Lane markings',
                'advantages': 'Better edge localization, noise resistant',
                'disadvantages': 'Slower than Sobel'
            }
        },
        'feature_extraction': {
            'ORB': {
                'use_case': 'Real-time vehicle tracking, License plate recognition',
                'advantages': 'Fast, rotation invariant, low memory',
                'disadvantages': 'Less distinctive features',
                'suitability_traffic': 'High (Real-time processing)'
            },
            'SIFT': {
                'use_case': 'Vehicle matching, Traffic incident database lookup',
                'advantages': 'Highly distinctive, scale invariant',
                'disadvantages': 'Computationally expensive, patented',
                'suitability_traffic': 'Medium (Archive/offline analysis)'
            },
            'SURF': {
                'use_case': 'Vehicle re-identification, Multi-camera tracking',
                'advantages': 'Fast SIFT alternative, scale invariant',
                'disadvantages': 'Patent issues in some regions',
                'suitability_traffic': 'Medium-High (Balance of speed/quality)'
            }
        }
    }
    return analysis

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("=" * 70)
    print("COMPUTER VISION ANALYSIS FOR TRAFFIC MONITORING")
    print("=" * 70)
    
    # Load image
    print("\n[*] Loading image...")
    img_bgr, img_rgb, img_gray = load_image(IMAGE_PATH)
    print(f"    Image shape: {img_rgb.shape}")
    print(f"    Image size: {img_rgb.nbytes / 1024 / 1024:.2f} MB")
    
    # ========== TASK 1: EDGE DETECTION ==========
    print("\n" + "=" * 70)
    print("TASK 1: EDGE DETECTION")
    print("=" * 70)
    
    print("\n[*] Applying Sobel edge detector...")
    sobel_mag, sobel_dir, sobelx, sobely = apply_sobel(img_gray)
    
    print("[*] Applying Canny edge detector...")
    canny_edges = apply_canny(img_gray, threshold1=50, threshold2=150)
    
    print("[*] Comparing edge quality...")
    edge_comparison = compare_edge_quality(sobel_mag, canny_edges)
    
    print("\n    Edge Detection Comparison:")
    print(f"    Sobel edge pixels: {edge_comparison['Sobel_edge_pixels']}")
    print(f"    Canny edge pixels: {edge_comparison['Canny_edge_pixels']}")
    print(f"    Sobel edge density: {edge_comparison['Sobel_density']:.4f}")
    print(f"    Canny edge density: {edge_comparison['Canny_density']:.4f}")
    
    if edge_comparison['Canny_density'] > edge_comparison['Sobel_density']:
        print("    → Canny provides better edge localization with higher density")
    else:
        print("    → Sobel captures more diverse edges")
    
    # ========== TASK 2: OBJECT REPRESENTATION ==========
    print("\n" + "=" * 70)
    print("TASK 2: OBJECT REPRESENTATION")
    print("=" * 70)
    
    print("\n[*] Detecting contours...")
    contours, thresh = detect_contours(img_gray)
    print(f"    Total contours detected: {len(contours)}")
    
    print("[*] Drawing bounding boxes...")
    img_bbox, object_data = draw_bounding_boxes(img_rgb, contours, min_area=100)
    print(f"    Objects with area > 100 pixels: {len(object_data)}")
    
    print("[*] Computing object properties...")
    obj_properties = compute_object_properties(contours, min_area=100)
    
    print("\n    Top 5 objects by area:")
    sorted_objs = sorted(obj_properties, key=lambda x: x['area'], reverse=True)[:5]
    for i, obj in enumerate(sorted_objs, 1):
        print(f"    Object {i}:")
        print(f"      Area: {obj['area']:.2f} pixels²")
        print(f"      Perimeter: {obj['perimeter']:.2f} pixels")
        print(f"      Circularity: {obj['circularity']:.4f}")
    
    # ========== TASK 3: FEATURE EXTRACTION ==========
    print("\n" + "=" * 70)
    print("TASK 3: FEATURE EXTRACTION")
    print("=" * 70)
    
    print("\n[*] Applying ORB feature detector...")
    kp_orb, desc_orb, _ = apply_orb(img_gray, n_features=500)
    print(f"    ORB keypoints: {len(kp_orb)}")
    
    print("[*] Applying SIFT feature detector...")
    kp_sift, desc_sift, _ = apply_sift(img_gray)
    print(f"    SIFT keypoints: {len(kp_sift)}")
    
    print("[*] Applying SURF feature detector...")
    try:
        kp_surf, desc_surf, _ = apply_surf(img_gray)
        print(f"    SURF keypoints: {len(kp_surf)}")
    except Exception as e:
        print(f"    SURF: Not available or error occurred - {str(e)}")
        kp_surf, desc_surf = [], None
    
    print("[*] Visualizing keypoints...")
    img_orb_kp = visualize_keypoints(img_rgb, kp_orb, 'ORB')
    img_sift_kp = visualize_keypoints(img_rgb, kp_sift, 'SIFT')
    img_surf_kp = visualize_keypoints(img_rgb, kp_surf, 'SURF') if kp_surf else None
    
    # ========== TASK 4: COMPARATIVE ANALYSIS ==========
    print("\n" + "=" * 70)
    print("TASK 4: COMPARATIVE ANALYSIS & TRAFFIC MONITORING INSIGHTS")
    print("=" * 70)
    
    print("\n[*] Comparing feature extractors...")
    feature_comparison = compare_feature_extractors(
        kp_orb, desc_orb, kp_sift, desc_sift, kp_surf, desc_surf
    )
    
    print("\n    Feature Extractor Comparison:")
    print(f"    ORB - Keypoints: {feature_comparison['ORB']['keypoints']}, "
          f"Type: {feature_comparison['ORB']['descriptor_type']}")
    print(f"    SIFT - Keypoints: {feature_comparison['SIFT']['keypoints']}, "
          f"Type: {feature_comparison['SIFT']['descriptor_type']}")
    print(f"    SURF - Keypoints: {feature_comparison['SURF']['keypoints'] if kp_surf else 'N/A'}, "
          f"Type: {feature_comparison['SURF']['descriptor_type']}")
    
    print("\n[*] Analyzing traffic monitoring applications...")
    traffic_analysis = traffic_monitoring_analysis(edge_comparison, feature_comparison)
    
    print("\n    EDGE DETECTORS IN TRAFFIC MONITORING:")
    print("\n    ► Sobel Operator:")
    analysis = traffic_analysis['edge_detection']['Sobel']
    print(f"      Use case: {analysis['use_case']}")
    print(f"      Advantages: {analysis['advantages']}")
    print(f"      Disadvantages: {analysis['disadvantages']}")
    
    print("\n    ► Canny Edge Detector:")
    analysis = traffic_analysis['edge_detection']['Canny']
    print(f"      Use case: {analysis['use_case']}")
    print(f"      Advantages: {analysis['advantages']}")
    print(f"      Disadvantages: {analysis['disadvantages']}")
    
    print("\n    FEATURE EXTRACTORS IN TRAFFIC MONITORING:")
    for method in ['ORB', 'SIFT', 'SURF']:
        analysis = traffic_analysis['feature_extraction'][method]
        print(f"\n    ► {method}:")
        print(f"      Use case: {analysis['use_case']}")
        print(f"      Advantages: {analysis['advantages']}")
        print(f"      Disadvantages: {analysis['disadvantages']}")
        if 'suitability_traffic' in analysis:
            print(f"      Traffic Monitoring Suitability: {analysis['suitability_traffic']}")
    
    # ========== VISUALIZATION ==========
    print("\n[*] Creating visualizations...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Original image
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_rgb)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Sobel edges
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(sobel_mag, cmap='gray')
    ax.set_title(f'Sobel Edges (Density: {edge_comparison["Sobel_density"]:.4f})', 
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Canny edges
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(canny_edges, cmap='gray')
    ax.set_title(f'Canny Edges (Density: {edge_comparison["Canny_density"]:.4f})', 
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Sobel-X
    ax = fig.add_subplot(gs[1, 0])
    sobelx_norm = np.uint8(255 * np.abs(sobelx) / np.max(np.abs(sobelx)))
    ax.imshow(sobelx_norm, cmap='gray')
    ax.set_title('Sobel-X (Horizontal Edges)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Sobel-Y
    ax = fig.add_subplot(gs[1, 1])
    sobely_norm = np.uint8(255 * np.abs(sobely) / np.max(np.abs(sobely)))
    ax.imshow(sobely_norm, cmap='gray')
    ax.set_title('Sobel-Y (Vertical Edges)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Threshold for contours
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(thresh, cmap='gray')
    ax.set_title(f'Binary Threshold ({len(contours)} contours)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Bounding boxes
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(img_bbox)
    ax.set_title(f'Bounding Boxes ({len(object_data)} objects)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # ORB keypoints
    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(img_orb_kp)
    ax.set_title(f'ORB Keypoints ({len(kp_orb)} detected)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # SIFT keypoints
    ax = fig.add_subplot(gs[2, 2])
    ax.imshow(img_sift_kp)
    ax.set_title(f'SIFT Keypoints ({len(kp_sift)} detected)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # SURF keypoints
    if img_surf_kp is not None:
        ax = fig.add_subplot(gs[3, 0])
        ax.imshow(img_surf_kp)
        ax.set_title(f'SURF Keypoints ({len(kp_surf)} detected)', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Comparison chart - Feature count
    ax = fig.add_subplot(gs[3, 1])
    methods = ['ORB', 'SIFT', 'SURF']
    kp_counts = [len(kp_orb), len(kp_sift), len(kp_surf) if kp_surf else 0]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(methods, kp_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Number of Keypoints', fontsize=11, fontweight='bold')
    ax.set_title('Feature Detector Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(kp_counts) * 1.2)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Edge density comparison
    ax = fig.add_subplot(gs[3, 2])
    edge_methods = ['Sobel', 'Canny']
    densities = [edge_comparison['Sobel_density'], edge_comparison['Canny_density']]
    colors = ['#FFA500', '#DC143C']
    bars = ax.bar(edge_methods, densities, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Edge Density', fontsize=11, fontweight='bold')
    ax.set_title('Edge Detector Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(densities) * 1.2)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Computer Vision Analysis: Edge Detection, Object Representation & Feature Extraction', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save and show
    output_path = r'c:\Users\gtcam\OneDrive\Desktop\Assignment-4\vision_analysis_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Visualization saved to: {output_path}")
    
    plt.show()
    
    # ========== SUMMARY REPORT ==========
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    print(f"\n✓ Image processed: {IMAGE_PATH}")
    print(f"✓ Edge detectors: Sobel, Canny")
    print(f"✓ Objects detected: {len(object_data)} objects (area > 100 px)")
    print(f"✓ Feature extractors: ORB ({len(kp_orb)} KP), SIFT ({len(kp_sift)} KP), "
          f"SURF ({len(kp_surf) if kp_surf else 'N/A'} KP)")
    print(f"✓ Visualizations: 12 subplots with comparative analysis")
    print(f"✓ Traffic monitoring insights: Provided for all methods")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
