"""
TASK 2: OBJECT REPRESENTATION
Detects contours, draws bounding boxes, and computes area and perimeter
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
    """Load image and convert to RGB and grayscale"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_rgb, img_gray

def detect_contours(img_gray, min_area=50):
    """
    Detect contours in the image
    Uses morphological operations to enhance object boundaries
    """
    print("[*] Detecting contours...")
    
    # Apply morphological operations to enhance boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_morph = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
    img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel)
    
    # Apply Gaussian blur for smoothing
    img_blur = cv2.GaussianBlur(img_morph, (5, 5), 1)
    
    # Threshold to create binary image
    _, thresh = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by minimum area
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            filtered_contours.append(contour)
    
    print(f"    ✓ Total contours detected: {len(contours)}")
    print(f"    ✓ Contours after filtering (area >= {min_area}): {len(filtered_contours)}")
    
    return filtered_contours, thresh

def draw_bounding_boxes(img_rgb, contours, min_area=50):
    """
    Draw bounding boxes and convex hulls around detected objects
    """
    print("\n[*] Drawing bounding boxes...")
    
    img_bbox = img_rgb.copy()
    object_data = []
    
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Filter small contours
        if area < min_area:
            continue
        
        # Get bounding rectangle (axis-aligned)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Get minimum area rectangle (rotated)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int_(box)
        
        # Get convex hull
        hull = cv2.convexHull(contour)
        
        # Draw axis-aligned bounding box (GREEN)
        cv2.rectangle(img_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw rotated rectangle (BLUE)
        cv2.polylines(img_bbox, [box], True, (255, 0, 0), 2)
        
        # Draw convex hull (RED)
        cv2.drawContours(img_bbox, [hull], 0, (0, 0, 255), 1)
        
        # Calculate properties
        perimeter = cv2.arcLength(contour, True)
        aspect_ratio = w / h if h > 0 else 0
        
        # Store object data
        object_data.append({
            'object_id': idx,
            'coordinates': (x, y, w, h),
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'contour': contour
        })
    
    print(f"    ✓ Objects with bounding boxes drawn: {len(object_data)}")
    
    return img_bbox, object_data

def compute_object_properties(contours, min_area=50):
    """
    Compute detailed properties for all detected objects:
    - Area
    - Perimeter
    - Circularity
    - Solidity
    - Eccentricity
    """
    print("\n[*] Computing object properties...")
    
    properties = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity (1 = circle, < 1 = more elongated)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        # Convex hull properties
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Solidity (how "solid" the object is)
        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 0
        
        # Fit ellipse for eccentricity
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, (major_axis, minor_axis), angle) = ellipse
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                eccentricity = 0
        else:
            eccentricity = 0
        
        # Bounding box info
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        fill_ratio = area / bbox_area if bbox_area > 0 else 0
        
        properties.append({
            'object_id': i,
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'solidity': solidity,
            'eccentricity': eccentricity,
            'fill_ratio': fill_ratio,
            'bbox_dimensions': (w, h)
        })
    
    print(f"    ✓ Properties computed for {len(properties)} objects")
    
    return properties

def print_object_statistics(object_properties):
    """Print detailed statistics about detected objects"""
    print("\n" + "="*70)
    print("OBJECT STATISTICS")
    print("="*70)
    
    if not object_properties:
        print("    No objects detected!")
        return
    
    # Sort by area
    sorted_by_area = sorted(object_properties, key=lambda x: x['area'], reverse=True)
    
    print(f"\n    Total objects detected: {len(object_properties)}")
    
    print(f"\n    ═══ TOP 5 LARGEST OBJECTS ═══")
    for i, obj in enumerate(sorted_by_area[:5], 1):
        print(f"\n    Object #{i}:")
        print(f"      Area: {obj['area']:.2f} pixels²")
        print(f"      Perimeter: {obj['perimeter']:.2f} pixels")
        print(f"      Circularity: {obj['circularity']:.4f} (1=perfect circle)")
        print(f"      Solidity: {obj['solidity']:.4f} (1=solid)")
        print(f"      Eccentricity: {obj['eccentricity']:.4f} (1=line)")
        print(f"      Fill Ratio: {obj['fill_ratio']:.4f}")
        print(f"      Bbox Dimensions: {obj['bbox_dimensions'][0]} x {obj['bbox_dimensions'][1]} pixels")
    
    # Statistics
    areas = [obj['area'] for obj in object_properties]
    perimeters = [obj['perimeter'] for obj in object_properties]
    
    print(f"\n    ═══ AGGREGATE STATISTICS ═══")
    print(f"      Total area covered: {sum(areas):.2f} pixels²")
    print(f"      Average object area: {np.mean(areas):.2f} pixels²")
    print(f"      Median object area: {np.median(areas):.2f} pixels²")
    print(f"      Min area: {min(areas):.2f} pixels²")
    print(f"      Max area: {max(areas):.2f} pixels²")
    print(f"      Area std dev: {np.std(areas):.2f}")
    
    print(f"\n      Average perimeter: {np.mean(perimeters):.2f} pixels")
    print(f"      Total perimeter: {sum(perimeters):.2f} pixels")

def visualize_objects(img_rgb, img_bbox, thresh, object_properties):
    """Visualize object representation results"""
    print("\n[*] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TASK 2: OBJECT REPRESENTATION - Contours & Bounding Boxes', 
                 fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Binary threshold
    axes[0, 1].imshow(thresh, cmap='gray')
    axes[0, 1].set_title(f'Binary Threshold ({len(object_properties)} objects)', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Bounding boxes
    axes[1, 0].imshow(img_bbox)
    axes[1, 0].set_title(f'Bounding Boxes & Convex Hulls', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Area histogram
    if object_properties:
        areas = [obj['area'] for obj in object_properties]
        axes[1, 1].hist(areas, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Object Area (pixels²)', fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontweight='bold')
        axes[1, 1].set_title('Distribution of Object Areas', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = r'c:\Users\gtcam\OneDrive\Desktop\Assignment-4\task2_object_representation_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ Visualization saved to: {output_path}")
    
    plt.show()

def main():
    print("="*70)
    print("TASK 2: OBJECT REPRESENTATION")
    print("="*70)
    
    # Load image
    print("\n[*] Loading image...")
    img_rgb, img_gray = load_image(IMAGE_PATH)
    print(f"    ✓ Image shape: {img_rgb.shape}")
    
    # Detect contours
    contours, thresh = detect_contours(img_gray, min_area=50)
    
    # Draw bounding boxes
    img_bbox, object_data = draw_bounding_boxes(img_rgb, contours, min_area=50)
    
    # Compute properties
    object_properties = compute_object_properties(contours, min_area=50)
    
    # Print statistics
    print_object_statistics(object_properties)
    
    # Visualize
    visualize_objects(img_rgb, img_bbox, thresh, object_properties)
    
    print("\n" + "="*70)
    print("✓ TASK 2 COMPLETED")
    print("="*70)

if __name__ == "__main__":
    main()
