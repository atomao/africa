"""
Functions for filtering and refining pseudo-annotations using Tier 2 data.

This module provides utilities to:
1. Remove false positive polygons that don't intersect with any Tier 2 annotations
2. Extract crops from images/masks where Tier 2 annotations didn't intersect with pseudo-annotations
3. Keep real annotations when IoU with pseudo-annotations is high (> 0.7)
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
from shapely.geometry import Polygon


def calculate_iou(polygon1: Polygon, polygon2: Polygon) -> float:
    """
    Calculate Intersection over Union (IoU) between two polygons.
    
    Parameters
    ----------
    polygon1 : Polygon
        First polygon (e.g., pseudo-annotation)
    polygon2 : Polygon
        Second polygon (e.g., Tier 2 annotation)
    
    Returns
    -------
    float
        IoU score between 0 and 1
    """
    if not polygon1.is_valid or not polygon2.is_valid:
        return 0.0
    
    try:
        intersection = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        
        if union == 0:
            return 0.0
        
        return intersection / union
    except Exception:
        return 0.0


def mask_to_polygons(mask: np.ndarray, min_area: int = 50) -> List[Polygon]:
    """
    Convert binary mask to list of polygons.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask of shape (H, W) with values 0 or 1
    min_area : int
        Minimum area threshold for polygons
    
    Returns
    -------
    List[Polygon]
        List of Shapely Polygon objects
    """
    # Ensure mask is binary uint8
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Skip small contours
        if cv2.contourArea(contour) < min_area:
            continue
        
        # Convert contour to polygon
        if len(contour) >= 3:
            points = contour.squeeze()
            if len(points.shape) == 1:
                continue
            try:
                polygon = Polygon(points)
                if polygon.is_valid and not polygon.is_empty:
                    polygons.append(polygon)
            except Exception:
                continue
    
    return polygons


def polygons_to_mask(polygons: List[Polygon], shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    
    for polygon in polygons:
        if not polygon.is_valid or polygon.is_empty:
            continue
        
        # Get exterior coordinates
        coords = np.array(polygon.exterior.coords, dtype=np.int32)
        cv2.fillPoly(mask, [coords], 1)
    
    return mask.astype(np.float32)


def filter_false_positive_polygons(
    pseudo_mask: np.ndarray,
    tier2_mask: np.ndarray,
    min_area: int = 50
) -> np.ndarray:
    # Convert masks to polygons
    pseudo_polygons = mask_to_polygons(pseudo_mask, min_area=min_area)
    tier2_polygons = mask_to_polygons(tier2_mask, min_area=min_area)
    
    # If no Tier 2 annotations, all pseudo-annotations are false positives
    if len(tier2_polygons) == 0:
        return np.zeros_like(pseudo_mask, dtype=np.float32)
    
    # Filter pseudo-annotations that intersect with at least one Tier 2 annotation
    valid_pseudo_polygons = []
    
    for pseudo_poly in pseudo_polygons:
        intersects = False
        for tier2_poly in tier2_polygons:
            if pseudo_poly.intersects(tier2_poly):
                intersects = True
                break
        
        if intersects:
            valid_pseudo_polygons.append(pseudo_poly)
    
    # Convert filtered polygons back to mask
    filtered_mask = polygons_to_mask(valid_pseudo_polygons, pseudo_mask.shape)
    
    return filtered_mask


def extract_non_intersecting_tier2_regions(
    image: np.ndarray,
    tier2_mask: np.ndarray,
    pseudo_mask: np.ndarray,
    padding: int = 50,
    min_area: int = 50
) -> List[Dict[str, np.ndarray]]:
    # Convert masks to polygons
    tier2_polygons = mask_to_polygons(tier2_mask, min_area=min_area)
    pseudo_polygons = mask_to_polygons(pseudo_mask, min_area=min_area)
    
    # Find Tier 2 polygons that don't intersect with any pseudo-annotations
    non_intersecting_polygons = []
    
    for tier2_poly in tier2_polygons:
        intersects = False
        for pseudo_poly in pseudo_polygons:
            if tier2_poly.intersects(pseudo_poly):
                intersects = True
                break
        
        if not intersects:
            non_intersecting_polygons.append(tier2_poly)
    
    # Extract crops for each non-intersecting polygon
    crops = []
    h, w = tier2_mask.shape
    
    for polygon in non_intersecting_polygons:
        # Get bounding box
        minx, miny, maxx, maxy = polygon.bounds
        
        # Add padding
        x1 = max(0, int(minx) - padding)
        y1 = max(0, int(miny) - padding)
        x2 = min(w, int(maxx) + padding)
        y2 = min(h, int(maxy) + padding)
        
        # Extract crops
        image_crop = image[y1:y2, x1:x2].copy()
        
        # Create mask crop with only this polygon
        mask_crop = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        
        # Translate polygon coordinates to crop space
        coords = np.array(polygon.exterior.coords, dtype=np.int32)
        coords[:, 0] -= x1
        coords[:, 1] -= y1
        
        cv2.fillPoly(mask_crop, [coords], 1)
        
        crops.append({
            'image': image_crop,
            'mask': mask_crop.astype(np.float32),
            'bbox': (x1, y1, x2, y2)
        })
    
    return crops


def refine_annotations_with_high_iou(
    pseudo_mask: np.ndarray,
    tier2_mask: np.ndarray,
    iou_threshold: float = 0.7,
    min_area: int = 50
) -> np.ndarray:
    # Convert masks to polygons
    pseudo_polygons = mask_to_polygons(pseudo_mask, min_area=min_area)
    tier2_polygons = mask_to_polygons(tier2_mask, min_area=min_area)
    
    refined_polygons = []
    used_pseudo_indices = set()
    
    # For each Tier 2 polygon, check IoU with pseudo-annotations
    for tier2_poly in tier2_polygons:
        max_iou = 0.0
        best_pseudo_idx = -1
        
        for idx, pseudo_poly in enumerate(pseudo_polygons):
            iou = calculate_iou(tier2_poly, pseudo_poly)
            if iou > max_iou:
                max_iou = iou
                best_pseudo_idx = idx
        
        # If high IoU, keep Tier 2 annotation
        if max_iou > iou_threshold:
            refined_polygons.append(tier2_poly)
            if best_pseudo_idx >= 0:
                used_pseudo_indices.add(best_pseudo_idx)
        # Otherwise, the corresponding pseudo-annotation will be kept later
    
    # Add pseudo-annotations that weren't matched with high IoU
    for idx, pseudo_poly in enumerate(pseudo_polygons):
        if idx not in used_pseudo_indices:
            refined_polygons.append(pseudo_poly)
    
    # Convert refined polygons back to mask
    refined_mask = polygons_to_mask(refined_polygons, pseudo_mask.shape)
    
    return refined_mask


def process_annotation_pair(
    image_path: Path,
    pseudo_mask_path: Path,
    tier2_mask_path: Path,
    output_dir: Path,
    iou_threshold: float = 0.7,
    extract_crops: bool = True,
    min_area: int = 50
) -> Dict[str, any]:
    """
    Complete processing pipeline for a single image-mask pair.
    
    Parameters
    ----------
    image_path : Path
        Path to input RGB image
    pseudo_mask_path : Path
        Path to pseudo-annotation mask
    tier2_mask_path : Path
        Path to Tier 2 annotation mask
    output_dir : Path
        Directory to save outputs
    iou_threshold : float
        IoU threshold for keeping Tier 2 annotations
    extract_crops : bool
        Whether to extract crops from non-intersecting regions
    min_area : int
        Minimum area threshold for polygons
    
    Returns
    -------
    Dict[str, any]
        Dictionary with processing results and statistics
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pseudo_mask = cv2.imread(str(pseudo_mask_path), cv2.IMREAD_GRAYSCALE)
    if pseudo_mask is None:
        raise FileNotFoundError(f"Pseudo mask not found: {pseudo_mask_path}")
    pseudo_mask = (pseudo_mask > 0).astype(np.float32)
    
    tier2_mask = cv2.imread(str(tier2_mask_path), cv2.IMREAD_GRAYSCALE)
    if tier2_mask is None:
        raise FileNotFoundError(f"Tier 2 mask not found: {tier2_mask_path}")
    tier2_mask = (tier2_mask > 0).astype(np.float32)
    
    # Step 1: Filter false positive polygons
    filtered_pseudo_mask = filter_false_positive_polygons(
        pseudo_mask, tier2_mask, min_area=min_area
    )
    
    # Step 2: Refine with high IoU
    refined_mask = refine_annotations_with_high_iou(
        filtered_pseudo_mask, tier2_mask, iou_threshold=iou_threshold, min_area=min_area
    )
    
    # Step 3: Extract crops (optional)
    crops = []
    if extract_crops:
        crops = extract_non_intersecting_tier2_regions(
            image, tier2_mask, pseudo_mask, min_area=min_area
        )
    
    # Save refined mask
    output_mask_path = output_dir / f"{image_path.stem}_refined_mask.png"
    cv2.imwrite(str(output_mask_path), (refined_mask * 255).astype(np.uint8))
    
    # Save crops
    crop_paths = []
    for i, crop in enumerate(crops):
        crop_image_path = output_dir / f"{image_path.stem}_crop_{i}_image.png"
        crop_mask_path = output_dir / f"{image_path.stem}_crop_{i}_mask.png"
        
        cv2.imwrite(str(crop_image_path), cv2.cvtColor(crop['image'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(crop_mask_path), (crop['mask'] * 255).astype(np.uint8))
        
        crop_paths.append({
            'image': crop_image_path,
            'mask': crop_mask_path,
            'bbox': crop['bbox']
        })
    
    stats = {
        'original_pseudo_polygons': len(mask_to_polygons(pseudo_mask, min_area=min_area)),
        'filtered_pseudo_polygons': len(mask_to_polygons(filtered_pseudo_mask, min_area=min_area)),
        'tier2_polygons': len(mask_to_polygons(tier2_mask, min_area=min_area)),
        'refined_polygons': len(mask_to_polygons(refined_mask, min_area=min_area)),
        'extracted_crops': len(crops),
        'output_mask': output_mask_path,
        'crop_paths': crop_paths
    }
    
    return stats


def batch_process_tier2_data(
    images_dir: Path,
    predicted_masks_dir: Path,
    gt_masks_dir: Path,
    output_dir: Path,
    iou_threshold: float = 0.7,
    extract_crops: bool = True,
    min_area: int = 50,
    image_extension: str = ".png",
    verbose: bool = True
) -> Dict[str, any]:
    images_dir = Path(images_dir)
    predicted_masks_dir = Path(predicted_masks_dir)
    gt_masks_dir = Path(gt_masks_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = sorted(images_dir.glob(f"*{image_extension}"))
    
    if not image_files:
        raise ValueError(f"No images found in {images_dir} with extension {image_extension}")
    
    if verbose:
        print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    total_stats = {
        'total_images': len(image_files),
        'successful': 0,
        'failed': 0,
        'total_original_polygons': 0,
        'total_filtered_polygons': 0,
        'total_refined_polygons': 0,
        'total_crops_extracted': 0
    }
    
    for idx, image_path in enumerate(image_files, 1):
        try:
            # Find corresponding masks
            mask_name = image_path.stem
            predicted_mask_path = predicted_masks_dir / f"{mask_name}{image_extension}"
            gt_mask_path = gt_masks_dir / f"{mask_name}{image_extension}"
            
            # Check if masks exist
            if not predicted_mask_path.exists():
                if verbose:
                    print(f"[{idx}/{len(image_files)}] Warning: Predicted mask not found for {image_path.name}")
                continue
            
            if not gt_mask_path.exists():
                if verbose:
                    print(f"[{idx}/{len(image_files)}] Warning: GT mask not found for {image_path.name}")
                continue
            
            if verbose:
                print(f"[{idx}/{len(image_files)}] Processing {image_path.name}...")
            
            # Process the image-mask pair
            stats = process_annotation_pair(
                image_path=image_path,
                pseudo_mask_path=predicted_mask_path,
                tier2_mask_path=gt_mask_path,
                output_dir=output_dir,
                iou_threshold=iou_threshold,
                extract_crops=extract_crops,
                min_area=min_area
            )
            
            # Update total statistics
            total_stats['successful'] += 1
            total_stats['total_original_polygons'] += stats['original_pseudo_polygons']
            total_stats['total_filtered_polygons'] += stats['filtered_pseudo_polygons']
            total_stats['total_refined_polygons'] += stats['refined_polygons']
            total_stats['total_crops_extracted'] += stats['extracted_crops']
            
            # Store results
            results.append({
                'image_name': image_path.name,
                'stats': stats
            })
            
            if verbose:
                print(f"  - Original polygons: {stats['original_pseudo_polygons']}")
                print(f"  - Filtered polygons: {stats['filtered_pseudo_polygons']}")
                print(f"  - Refined polygons: {stats['refined_polygons']}")
                print(f"  - Crops extracted: {stats['extracted_crops']}")
        
        except Exception as e:
            total_stats['failed'] += 1
            if verbose:
                print(f"[{idx}/{len(image_files)}] Error processing {image_path.name}: {str(e)}")
            results.append({
                'image_name': image_path.name,
                'error': str(e)
            })
    
    if verbose:
        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"  - Total images: {total_stats['total_images']}")
        print(f"  - Successful: {total_stats['successful']}")
        print(f"  - Failed: {total_stats['failed']}")
        print(f"  - Total original polygons: {total_stats['total_original_polygons']}")
        print(f"  - Total filtered polygons: {total_stats['total_filtered_polygons']}")
        print(f"  - Total refined polygons: {total_stats['total_refined_polygons']}")
        print(f"  - Total crops extracted: {total_stats['total_crops_extracted']}")
        print(f"{'='*60}")
    
    return {
        'summary': total_stats,
        'results': results
    }