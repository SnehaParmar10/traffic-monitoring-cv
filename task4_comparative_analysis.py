"""
TASK 4: COMPARATIVE ANALYSIS
Compares edge detectors and feature extractors
Explains how features help in traffic monitoring
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
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

def apply_sobel(img_gray):
    """Apply Sobel edge detector"""
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))
    return sobel_magnitude

def apply_canny(img_gray):
    """Apply Canny edge detector"""
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1.5)
    canny_edges = cv2.Canny(img_blur, 50, 150)
    return canny_edges

def apply_orb(img_gray):
    """Apply ORB feature detector"""
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)
    return keypoints, descriptors

def apply_sift(img_gray):
    """Apply SIFT feature detector"""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    return keypoints, descriptors

def apply_surf(img_gray):
    """Apply SURF feature detector"""
    try:
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        keypoints, descriptors = surf.detectAndCompute(img_gray, None)
        return keypoints, descriptors
    except:
        return [], None

def compare_edge_detectors(sobel_mag, canny_edges):
    """Compare Sobel and Canny edge detectors"""
    sobel_edge_count = np.count_nonzero(sobel_mag > 50)
    canny_edge_count = np.count_nonzero(canny_edges)
    
    total_pixels = sobel_mag.size
    sobel_density = sobel_edge_count / total_pixels
    canny_density = canny_edge_count / total_pixels
    
    comparison = {
        'Sobel': {
            'edge_pixels': sobel_edge_count,
            'density': sobel_density,
            'edge_image': sobel_mag
        },
        'Canny': {
            'edge_pixels': canny_edge_count,
            'density': canny_density,
            'edge_image': canny_edges
        }
    }
    
    return comparison

def compare_feature_extractors(kp_orb, desc_orb, kp_sift, desc_sift, kp_surf, desc_surf):
    """Compare ORB, SIFT, and SURF feature extractors"""
    comparison = {
        'ORB': {
            'keypoints': len(kp_orb),
            'descriptor_type': 'Binary',
            'descriptor_dim': 32 if desc_orb is not None else 0,
            'speed': 'Very Fast'
        },
        'SIFT': {
            'keypoints': len(kp_sift),
            'descriptor_type': 'Float',
            'descriptor_dim': 128 if desc_sift is not None else 0,
            'speed': 'Moderate'
        },
        'SURF': {
            'keypoints': len(kp_surf),
            'descriptor_type': 'Float',
            'descriptor_dim': 64 if desc_surf is not None else 0,
            'speed': 'Fast'
        }
    }
    
    return comparison

def print_edge_detector_analysis():
    """Print detailed analysis of edge detectors"""
    print("\n" + "="*70)
    print("EDGE DETECTOR COMPARATIVE ANALYSIS")
    print("="*70)
    
    analysis = {
        'Sobel': {
            'description': 'Computes gradients using Sobel operators',
            'kernel_size': '3x3 or larger',
            'detection_method': 'Gradient-based',
            'characteristics': ['Continuous gradient output', 'Sensitive to noise', 'Fast computation'],
            'advantages': [
                '✓ Fast and efficient',
                '✓ Directional information (X, Y gradients)',
                '✓ Good for detecting strong edges',
                '✓ Low memory requirement'
            ],
            'disadvantages': [
                '✗ Thick edges (not well-localized)',
                '✗ Sensitive to noise without preprocessing',
                '✗ May produce double edges',
                '✗ Less accurate edge localization'
            ],
            'traffic_applications': [
                '• Lane detection and marking',
                '• Road boundary detection',
                '• Vehicle silhouette extraction',
                '• Real-time edge processing for fast feedback'
            ]
        },
        'Canny': {
            'description': 'Multi-stage algorithm with Gaussian smoothing and non-maximum suppression',
            'kernel_size': 'Adaptive',
            'detection_method': 'Multi-stage (gradient, suppression, hysteresis)',
            'characteristics': ['Thin, well-localized edges', 'Noise-resistant', 'Hysteresis thresholding'],
            'advantages': [
                '✓ Thin, well-localized edges',
                '✓ Better edge connectivity',
                '✓ Less noise-sensitive',
                '✓ Hysteresis provides robustness',
                '✓ Single response per edge'
            ],
            'disadvantages': [
                '✗ Slower than Sobel',
                '✗ Requires threshold tuning',
                '✗ May miss weak edges',
                '✗ More computationally intensive'
            ],
            'traffic_applications': [
                '• Precise vehicle boundary detection',
                '• Lane marker detection',
                '• Road sign detection',
                '• Traffic light detection'
            ]
        }
    }
    
    for method, details in analysis.items():
        print(f"\n    ╔═══ {method.upper()} ═══╗")
        print(f"    ║ {details['description']}")
        print(f"    ║ Method: {details['detection_method']}")
        
        print(f"\n    Characteristics:")
        for char in details['characteristics']:
            print(f"      • {char}")
        
        print(f"\n    Advantages:")
        for adv in details['advantages']:
            print(f"      {adv}")
        
        print(f"\n    Disadvantages:")
        for dis in details['disadvantages']:
            print(f"      {dis}")
        
        print(f"\n    Traffic Monitoring Applications:")
        for app in details['traffic_applications']:
            print(f"      {app}")

def print_feature_extractor_analysis():
    """Print detailed analysis of feature extractors"""
    print("\n" + "="*70)
    print("FEATURE EXTRACTOR COMPARATIVE ANALYSIS")
    print("="*70)
    
    analysis = {
        'ORB': {
            'full_name': 'Oriented FAST and Rotated BRIEF',
            'descriptor_type': 'Binary (256 bits)',
            'detection_method': 'FAST corners + BRIEF descriptors',
            'characteristics': ['Rotation-invariant', 'Scale-invariant', 'Binary descriptors'],
            'advantages': [
                '✓ Extremely fast computation',
                '✓ Binary descriptors (fast matching)',
                '✓ Low memory requirement',
                '✓ Rotation and scale invariant',
                '✓ Patent-free',
                '✓ Best for real-time applications'
            ],
            'disadvantages': [
                '✗ Less distinctive features',
                '✗ Fewer reliable matches',
                '✗ Sensitive to scale changes',
                '✗ Less robust to viewpoint changes'
            ],
            'processing_speed': '~10-50ms per image',
            'memory_per_descriptor': '32 bytes',
            'traffic_applications': [
                '• Real-time vehicle tracking',
                '• License plate recognition',
                '• Traffic flow monitoring',
                '• Pedestrian detection',
                '• Mobile/embedded systems'
            ],
            'suitability_score': '★★★★★ (Real-time systems)',
            'use_case': 'Real-time traffic monitoring on edge devices'
        },
        'SIFT': {
            'full_name': 'Scale-Invariant Feature Transform',
            'descriptor_type': 'Float (128 dimensions)',
            'detection_method': 'Difference of Gaussians + gradient directions',
            'characteristics': ['Scale-invariant', 'Rotation-invariant', 'Distinctive features'],
            'advantages': [
                '✓ Highly distinctive features',
                '✓ Excellent for matching',
                '✓ Scale and rotation invariant',
                '✓ Robust to perspective changes',
                '✓ Industry standard',
                '✓ Patent-free since 2020'
            ],
            'disadvantages': [
                '✗ Computationally expensive',
                '✗ Slow for real-time applications',
                '✗ Large descriptor size',
                '✗ Memory intensive',
                '✗ Slow matching speed'
            ],
            'processing_speed': '~500-2000ms per image',
            'memory_per_descriptor': '512 bytes',
            'traffic_applications': [
                '• Vehicle re-identification',
                '• Traffic incident database matching',
                '• Offline video analysis',
                '• Archive searching'
            ],
            'suitability_score': '★★★☆☆ (Archive/Offline analysis)',
            'use_case': 'Post-incident analysis and archive searching'
        },
        'SURF': {
            'full_name': 'Speeded Up Robust Features',
            'descriptor_type': 'Float (64 dimensions)',
            'detection_method': 'Hessian matrix + BRIEF-like descriptors',
            'characteristics': ['Scale-invariant', 'Fast SIFT', 'Moderate complexity'],
            'advantages': [
                '✓ 3-5x faster than SIFT',
                '✓ Good feature distinctiveness',
                '✓ Scale and rotation invariant',
                '✓ Balance between speed and quality',
                '✓ Smaller descriptors than SIFT',
                '✓ Good for embedded systems'
            ],
            'disadvantages': [
                '✗ Still slower than ORB',
                '✗ Patent concerns in some regions',
                '✗ Parameter tuning needed',
                '✗ Descriptor size larger than ORB'
            ],
            'processing_speed': '~100-500ms per image',
            'memory_per_descriptor': '256 bytes',
            'traffic_applications': [
                '• Multi-camera vehicle tracking',
                '• Vehicle re-identification',
                '• Traffic pattern analysis',
                '• Lane change detection'
            ],
            'suitability_score': '★★★★☆ (Balanced systems)',
            'use_case': 'Moderate real-time systems with good accuracy'
        }
    }
    
    for method, details in analysis.items():
        print(f"\n    ╔═══ {method.upper()} ═══╗")
        print(f"    ║ {details['full_name']}")
        print(f"    ║ Descriptor: {details['descriptor_type']}")
        print(f"    ║ Method: {details['detection_method']}")
        
        print(f"\n    Characteristics:")
        for char in details['characteristics']:
            print(f"      • {char}")
        
        print(f"\n    Advantages:")
        for adv in details['advantages']:
            print(f"      {adv}")
        
        print(f"\n    Disadvantages:")
        for dis in details['disadvantages']:
            print(f"      {dis}")
        
        print(f"\n    Performance Metrics:")
        print(f"      • Processing Speed: {details['processing_speed']}")
        print(f"      • Memory per Descriptor: {details['memory_per_descriptor']}")
        print(f"      • Suitability: {details['suitability_score']}")
        
        print(f"\n    Traffic Monitoring Applications:")
        for app in details['traffic_applications']:
            print(f"      {app}")
        
        print(f"\n    Primary Use Case:")
        print(f"      → {details['use_case']}")

def print_traffic_monitoring_insights():
    """Print comprehensive traffic monitoring insights"""
    print("\n" + "="*70)
    print("TRAFFIC MONITORING: HOW FEATURES HELP")
    print("="*70)
    
    insights = {
        'edge_detection': {
            'title': 'Edge Detection in Traffic Monitoring',
            'overview': 'Provides structural information about scenes and objects',
            'applications': [
                '1. Lane Detection:',
                '   • Road boundaries detected using edge maps',
                '   • Lane markings identified as strong edges',
                '   • Helps in autonomous driving and lane-keeping assistance',
                '',
                '2. Vehicle Detection:',
                '   • Vehicle silhouettes extracted from edge maps',
                '   • Bounding box generation from edge contours',
                '   • Combined with other features for robust detection',
                '',
                '3. Traffic Sign Recognition:',
                '   • Signs have distinct boundaries (edges)',
                '   • Stop signs, speed limit signs have clear edges',
                '   • Used as preprocessing for sign classification',
                '',
                '4. Road Scene Understanding:',
                '   • Curbs and road borders detected',
                '   • Obstacles identified by edge patterns',
                '   • Context information for decision making'
            ],
            'processing_chain': 'Image → Edge Detection → Contour Analysis → Object Extraction'
        },
        'feature_extraction': {
            'title': 'Feature Extraction in Traffic Monitoring',
            'overview': 'Provides distinctive patterns for object matching and tracking',
            'applications': [
                '1. Vehicle Re-identification (ReID):',
                '   • SIFT/SURF for distinctive patterns on vehicles',
                '   • Matches vehicles across multiple camera views',
                '   • Tracks vehicles through traffic network',
                '',
                '2. License Plate Recognition:',
                '   • ORB for fast character detection',
                '   • SIFT for accurate character matching',
                '   • Enables automated toll collection',
                '',
                '3. Pedestrian Tracking:',
                '   • Keypoints on pedestrian clothing/appearance',
                '   • ORB for real-time tracking',
                '   • Safety monitoring and behavior analysis',
                '',
                '4. Multi-Object Tracking (MOT):',
                '   • Features maintain object identity across frames',
                '   • Prevents ID switches in crowded scenes',
                '   • Enables accurate traffic flow counting',
                '',
                '5. Anomaly Detection:',
                '   • Unusual feature patterns indicate violations',
                '   • Wrong-way driving detection',
                '   • Collision avoidance systems'
            ],
            'processing_chain': 'Image → Feature Detection → Feature Matching → Object Tracking'
        }
    }
    
    print(f"\n    {insights['edge_detection']['title']}")
    print(f"    Overview: {insights['edge_detection']['overview']}")
    print(f"\n    Applications:")
    for app in insights['edge_detection']['applications']:
        print(f"    {app}")
    print(f"\n    Processing Pipeline:")
    print(f"    {insights['edge_detection']['processing_chain']}")
    
    print(f"\n\n    {insights['feature_extraction']['title']}")
    print(f"    Overview: {insights['feature_extraction']['overview']}")
    print(f"\n    Applications:")
    for app in insights['feature_extraction']['applications']:
        print(f"    {app}")
    print(f"\n    Processing Pipeline:")
    print(f"    {insights['feature_extraction']['processing_chain']}")

def print_recommendation():
    """Print recommendations for traffic monitoring systems"""
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR TRAFFIC MONITORING SYSTEMS")
    print("="*70)
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║ OPTIMAL COMBINATIONS FOR DIFFERENT SCENARIOS                      ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    1. REAL-TIME TRAFFIC FLOW MONITORING (Mobile/Embedded)
       ├─ Edge Detection: Canny (balanced quality)
       ├─ Feature Extraction: ORB (speed critical)
       ├─ Processing Speed: 30-60 FPS
       └─ Justification: Speed + accuracy for embedded systems
    
    2. MULTI-CAMERA VEHICLE TRACKING
       ├─ Edge Detection: Sobel + Canny combined
       ├─ Feature Extraction: SURF (speed + accuracy balance)
       ├─ Processing Speed: 10-30 FPS per camera
       └─ Justification: Good match distinctiveness, reasonable speed
    
    3. TRAFFIC INCIDENT ANALYSIS (Offline)
       ├─ Edge Detection: Canny (precise edges)
       ├─ Feature Extraction: SIFT (best matching)
       ├─ Processing Speed: No constraint
       └─ Justification: Maximum accuracy for archival analysis
    
    4. AUTONOMOUS VEHICLE NAVIGATION
       ├─ Edge Detection: Canny (lane detection)
       ├─ Feature Extraction: ORB (real-time requirements)
       ├─ Processing Speed: 60+ FPS
       └─ Justification: Safety critical, requires maximum speed
    
    5. PARKING LOT MONITORING
       ├─ Edge Detection: Sobel (quick scene understanding)
       ├─ Feature Extraction: ORB (fast detection)
       ├─ Processing Speed: 10-30 FPS
       └─ Justification: Large areas covered, speed important
    
    ╔════════════════════════════════════════════════════════════════════╗
    ║ DECISION MATRIX: SPEED vs ACCURACY                               ║
    ╚════════════════════════════════════════════════════════════════════╝
    
                      FASTEST  MODERATE           MOST ACCURATE
                        ↓        ↓                     ↓
    Edge Detection:   Sobel → Canny          (Canny with preprocessing)
    Feature Extract:  ORB → SURF → SIFT
                      ↑     ↑                        ↑
                   Best for   Balance          Archive/Analysis
                   Real-time              
    """)

def visualize_comparative_analysis(sobel_edges, canny_edges, kp_orb, kp_sift, kp_surf, 
                                   img_rgb, sobel_canny_comp, feature_comp):
    """Create comprehensive comparative visualization"""
    print("\n[*] Creating comparative analysis visualizations...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Edge detection comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image', fontweight='bold', fontsize=12)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(sobel_edges, cmap='hot')
    ax2.set_title(f'Sobel Edges ({sobel_canny_comp["Sobel"]["edge_pixels"]} pixels)', 
                  fontweight='bold', fontsize=12)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(canny_edges, cmap='gray')
    ax3.set_title(f'Canny Edges ({sobel_canny_comp["Canny"]["edge_pixels"]} pixels)', 
                  fontweight='bold', fontsize=12)
    ax3.axis('off')
    
    # Edge density comparison
    ax4 = fig.add_subplot(gs[1, 0])
    edge_methods = ['Sobel', 'Canny']
    densities = [sobel_canny_comp['Sobel']['density'], sobel_canny_comp['Canny']['density']]
    colors_edge = ['#FF6B6B', '#DC143C']
    bars = ax4.bar(edge_methods, densities, color=colors_edge, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Edge Density', fontweight='bold', fontsize=11)
    ax4.set_title('Edge Density Comparison', fontweight='bold', fontsize=12)
    for bar, val in zip(bars, densities):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Feature detection comparison
    ax5 = fig.add_subplot(gs[1, 1])
    feature_methods = ['ORB', 'SIFT', 'SURF']
    keypoint_counts = [len(kp_orb), len(kp_sift), len(kp_surf)]
    colors_feat = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax5.bar(feature_methods, keypoint_counts, color=colors_feat, alpha=0.7, 
                   edgecolor='black', linewidth=2)
    ax5.set_ylabel('Number of Keypoints', fontweight='bold', fontsize=11)
    ax5.set_title('Feature Detector Comparison', fontweight='bold', fontsize=12)
    for bar, val in zip(bars, keypoint_counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Data characteristics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    table_data = [
        ['Method', 'Keypoints', 'Descriptor'],
        ['ORB', f'{len(kp_orb)}', 'Binary'],
        ['SIFT', f'{len(kp_sift)}', 'Float-128'],
        ['SURF', f'{len(kp_surf)}', 'Float-64']
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(table_data)):
        for j in range(len(table_data[i])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#34495E')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
    
    ax6.set_title('Feature Descriptor Comparison', fontweight='bold', fontsize=12, pad=20)
    
    # Speed comparison
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.axis('off')
    
    speed_data = [
        ['Detector', 'Speed', 'Suitability'],
        ['Sobel', 'Very Fast', 'Real-time'],
        ['Canny', 'Moderate', 'General'],
        ['ORB', 'V. Fast', 'RT Edge '],
        ['SIFT', 'Slow', 'Archive'],
        ['SURF', 'Fast', 'Balanced']
    ]
    
    table = ax7.table(cellText=speed_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    for i in range(len(speed_data)):
        for j in range(len(speed_data[i])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#34495E')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
    
    ax7.set_title('Speed & Suitability Review', fontweight='bold', fontsize=12, pad=20)
    
    # Applications in traffic monitoring
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    traffic_text = """
    KEY APPLICATIONS IN TRAFFIC MONITORING
    
    EDGE DETECTION:                                    FEATURE EXTRACTION:
    • Lane Detection & Lane Keeping Assistance        • Vehicle Re-identification Across Cameras
    • Vehicle Boundary Detection                      • License Plate Recognition & Matching
    • Road Sign & Signal Detection                    • Pedestrian Tracking in Crowds
    • Obstacle Detection & Avoidance                  • Multi-Camera Scene Understanding
    • Parking Space Detection                         • Anomaly & Violation Detection
    • Traffic Flow Visualization                      • Vehicle Speed Estimation
    """
    
    ax8.text(0.05, 0.95, traffic_text, transform=ax8.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))
    
    fig.suptitle('TASK 4: COMPARATIVE ANALYSIS - Edge Detectors & Feature Extractors',
                fontsize=16, fontweight='bold')
    
    output_path = r'c:\Users\gtcam\OneDrive\Desktop\Assignment-4\task4_comparative_analysis_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ Visualization saved to: {output_path}")
    
    plt.show()

def main():
    print("="*70)
    print("TASK 4: COMPARATIVE ANALYSIS & TRAFFIC MONITORING INSIGHTS")
    print("="*70)
    
    # Load image
    print("\n[*] Loading image...")
    img_rgb, img_gray = load_image(IMAGE_PATH)
    print(f"    ✓ Image shape: {img_rgb.shape}")
    
    # Apply all methods
    print("\n[*] Applying all detection and extraction methods...")
    
    sobel_edges = apply_sobel(img_gray)
    canny_edges = apply_canny(img_gray)
    
    kp_orb, desc_orb = apply_orb(img_gray)
    kp_sift, desc_sift = apply_sift(img_gray)
    kp_surf, desc_surf = apply_surf(img_gray)
    
    # Compare methods
    print("\n[*] Comparing methods...")
    sobel_canny_comp = compare_edge_detectors(sobel_edges, canny_edges)
    feature_comp = compare_feature_extractors(kp_orb, desc_orb, kp_sift, desc_sift, kp_surf, desc_surf)
    
    # Print analyses
    print_edge_detector_analysis()
    print_feature_extractor_analysis()
    print_traffic_monitoring_insights()
    print_recommendation()
    
    # Visualize
    visualize_comparative_analysis(sobel_edges, canny_edges, kp_orb, kp_sift, kp_surf,
                                   img_rgb, sobel_canny_comp, feature_comp)
    
    print("\n" + "="*70)
    print("✓ TASK 4 COMPLETED")
    print("="*70)

if __name__ == "__main__":
    main()
