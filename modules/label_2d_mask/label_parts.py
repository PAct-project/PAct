"""
Image Part Segmentation and Labeling Tool

This script segments images into meaningful parts using the Segment Anything Model (SAM)
and optionally removes backgrounds using BriaRMBG. It identifies, visualizes, and merges
different parts of objects in images.

Key features:
- Background removal with alpha channel preservation
- Automatic part segmentation with SAM
- Intelligent part merging for logical grouping
- Detection of parts that SAM might miss
- Splitting of disconnected parts into separate components
- Edge cleaning and smoothing of segmentations
- Visualization of segmented parts with clear labeling
"""

import os
import argparse
import numpy as np
import cv2
import torch
from PIL import Image

from torchvision.transforms import functional as F
from torchvision import transforms
import torch.nn.functional as F_nn
from segment_anything import SamAutomaticMaskGenerator, build_sam
from modules.label_2d_mask.visualizer import Visualizer

# Minimum size threshold for considering a segment (in pixels)
size_th = 2000

def get_mask(group_ids, image, ids=None, img_name=None, save_dir=None):
    """
    Creates and saves a colored visualization of mask segments.
    
    Args:
        group_ids: Array of segment IDs for each pixel
        image: Input image
        ids: Identifier to append to output filename
        img_name: Base name of the image for saving
        
    Returns:
        Array of segment IDs (unchanged, just for convenience)
    """
    colored_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    colored_mask[group_ids == -1] = [255, 255, 255]
    
    unique_ids = np.unique(group_ids)
    unique_ids = unique_ids[unique_ids >= 0]
    
    for i, unique_id in enumerate(unique_ids):
        color_r = (i * 50 + 80) % 256
        color_g = (i * 120 + 40) % 256
        color_b = (i * 180 + 20) % 256
        
        mask = (group_ids == unique_id)
        colored_mask[mask] = [color_r, color_g, color_b]
    
    mask_path = os.path.join(save_dir, f"{img_name}_mask_segments_{ids}.png")
    cv2.imwrite(mask_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    print(f"Saved mask segments visualization to {mask_path}")
    
    return group_ids


def clean_segment_edges(group_ids):
    """
    Clean up segment edges by applying morphological operations to each segment.
    
    Args:
        group_ids: Array of segment IDs for each pixel
        
    Returns:
        Cleaned array of segment IDs with smoother boundaries
    """
    # Get unique segment IDs (excluding background -1)
    unique_ids = np.unique(group_ids)
    unique_ids = unique_ids[unique_ids >= 0]
    
    # Create a clean group_ids array
    cleaned_group_ids = np.full_like(group_ids, -1)  # Start with all background
    
    # Define kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)
    
    # Process each segment individually
    for segment_id in unique_ids:
        # Extract the mask for this segment
        segment_mask = (group_ids == segment_id).astype(np.uint8)
        
        # Apply morphological closing to smooth edges
        smoothed_mask = cv2.morphologyEx(segment_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Apply morphological opening to remove small isolated pixels
        smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Add this segment back to the cleaned result
        cleaned_group_ids[smoothed_mask > 0] = segment_id
    
    print(f"Cleaned edges for {len(unique_ids)} segments")
    return cleaned_group_ids


def prepare_image(image, bg_color=None, rmbg_net=None):
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_images = transform_image(image).unsqueeze(0).to('cuda')

    # Prediction
    with torch.no_grad():
        preds = rmbg_net(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)

    return image


def resize_and_pad_to_square(image, target_size=518):
    """
    Resize image to have longest side equal to target_size and pad shorter side 
    to create a square image.
    
    Args:
        image: PIL image or numpy array
        target_size: Target square size, defaults to 518
    
    Returns:
        PIL Image resized and padded to square (target_size x target_size)
    """
    # Ensure image is a PIL Image object
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image)
    
    # Get original dimensions
    width, height = image.size
    
    # Determine which dimension is longer
    if width > height:
        # Width is longer
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        # Height is longer
        new_height = target_size
        new_width = int(width * (target_size / height))
    
    # Resize image while maintaining aspect ratio
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create new square image with proper mode (with or without alpha channel)
    mode = "RGBA" if image.mode == "RGBA" else "RGB"
    background_color = (255, 255, 255, 0) if mode == "RGBA" else (255, 255, 255)
    square_image = Image.new(mode, (target_size, target_size), background_color)
    
    # Calculate position to paste resized image (centered)
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    
    # Paste resized image onto square background
    if mode == "RGBA":
        square_image.paste(resized_image, (paste_x, paste_y), resized_image)
    else:
        square_image.paste(resized_image, (paste_x, paste_y))
    
    return square_image


def split_disconnected_parts(group_ids, size_threshold=None):
    """
    Split each part into separate parts if they contain disconnected regions.
    
    Args:
        group_ids: Array of segment IDs for each pixel
        size_threshold: Minimum size threshold for considering a segment (in pixels).
                       If None, uses the global size_th variable.
        
    Returns:
        Updated array with each connected component having a unique ID
    """
    # Use provided threshold or fall back to global variable
    if size_threshold is None:
        size_threshold = size_th
    # Create a copy to hold the result
    new_group_ids = np.full_like(group_ids, -1)  # Start with all background
    
    # Get unique part IDs (excluding background -1)
    unique_ids = np.unique(group_ids)
    unique_ids = unique_ids[unique_ids >= 0]
    
    # Track the next available ID
    next_id = 0
    total_split_regions = 0
    
    # For each existing part ID
    for part_id in unique_ids:
        # Extract the mask for this part
        part_mask = (group_ids == part_id).astype(np.uint8)
        
        # Find connected components within this part
        num_labels, labels = cv2.connectedComponents(part_mask, connectivity=8)
        
        if num_labels == 1:  # Just background (0), no regions found
            continue
            
        if num_labels == 2:  # One connected component (background + 1 region)
            # Assign the original part's area to the next available ID
            new_group_ids[labels == 1] = next_id
            next_id += 1
        else:  # Multiple disconnected components
            split_count = 0
            print(f"Part {part_id} has {num_labels-1} disconnected regions, splitting...")
            
            # For each connected component (skipping background label 0)
            for label in range(1, num_labels):
                region_mask = labels == label
                region_size = np.sum(region_mask)
                
                # Only include regions that are large enough
                if region_size >= size_threshold / 5:  # Using size threshold to avoid tiny fragments
                    new_group_ids[region_mask] = next_id
                    split_count += 1
                    next_id += 1
                else:
                    print(f"  Skipping small disconnected region ({region_size} pixels)")
            
            total_split_regions += split_count
            
    if total_split_regions > 0:
        print(f"Split disconnected parts: original {len(unique_ids)} parts -> {next_id} connected parts")
    else:
        print("No parts needed splitting - all parts are already connected")
    
    return new_group_ids

# -------------------------------------------------------
# MAIN SEGMENTATION FUNCTION
# -------------------------------------------------------

def get_sam_mask(image, mask_generator, visual, merge_groups=None, existing_group_ids=None, 
                check_undetected=True, rgba_image=None, img_name=None, skip_split=False, save_dir=None, size_threshold=None):
    """
    Generate and process SAM masks for the image, with optional merging and undetected region detection.
    
    Args:
        size_threshold: Minimum size threshold for considering a segment (in pixels). 
                       If None, uses the global size_th variable.
    """
    # Use provided threshold or fall back to global variable
    if size_threshold is None:
        size_threshold = size_th
    label_mode = '1'
    anno_mode = ['Mask', 'Mark']
    
    exist_group = False

    # Use existing group IDs if provided, otherwise generate new ones with SAM
    if existing_group_ids is not None:
        group_ids = existing_group_ids.copy()
        group_counter = np.max(group_ids) + 1
        exist_group = True
    else:
        # Generate masks using SAM
        masks = mask_generator.generate(image)
        group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
        num_masks = len(masks)
        group_counter = 0

        # Sort masks by area (largest first)
        area_sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        
        # Create background mask if we have RGBA image
        background_mask = None
        if rgba_image is not None:
            rgba_array = np.array(rgba_image)
            if rgba_array.shape[2] == 4:
                # Use alpha channel to create foreground/background mask
                background_mask = rgba_array[:, :, 3] <= 10  # Areas with very low alpha are background

        # First pass: assign original group IDs
        for i in range(0, num_masks):
            if area_sorted_masks[i]["area"] < size_threshold:
                print(f"Skipping mask {i}, area too small: {area_sorted_masks[i]['area']} < {size_threshold}")
                continue
            
            mask = area_sorted_masks[i]["segmentation"]
            
            # Check proportion of background pixels in this mask
            if background_mask is not None:
                # Calculate how many pixels in this mask are background
                background_pixels_in_mask = np.sum(mask & background_mask)
                mask_area = np.sum(mask)
                background_ratio = background_pixels_in_mask / mask_area
                
                # Skip mask if background proportion is too high (>10%)
                if background_ratio > 0.1:
                    print(f"  Skipping mask {i}, background ratio: {background_ratio:.2f}")
                    continue
            
            # Assign group ID to this mask's pixels
            group_ids[mask] = group_counter
            print(f"Assigned mask {i} with area {area_sorted_masks[i]['area']} to group {group_counter}")
            group_counter += 1
        
        # Split disconnected parts immediately after SAM segmentation
        print("Splitting disconnected parts in initial segmentation...")
        group_ids = split_disconnected_parts(group_ids, size_threshold)
        
        # Update group counter after splitting
        if np.max(group_ids) >= 0:
            group_counter = np.max(group_ids) + 1
        print(f"After early splitting, now have {len(np.unique(group_ids))-1} regions (excluding background)")
    
    # Check for undetected parts using RGBA information
    if check_undetected and rgba_image is not None:
        print("Checking for undetected parts using RGBA image...")
        # Create a foreground mask from the alpha channel
        rgba_array = np.array(rgba_image)
        
        # Check if the image has an alpha channel
        if rgba_array.shape[2] == 4:
            print(f"Image has alpha channel, checking for undetected parts...")
            # Use alpha channel to identify non-transparent pixels (foreground)
            alpha_mask = rgba_array[:, :, 3] > 0  
            
            # Create existing parts mask and dilate it
            existing_parts_mask = (group_ids != -1)
            kernel = np.ones((4, 4), np.uint8)
            
            # Use larger kernel for faster dilation
            large_kernel = np.ones((4, 4), np.uint8)
            dilated_parts = cv2.dilate(existing_parts_mask.astype(np.uint8), large_kernel)
            
            # Find undetected areas (foreground but not detected by SAM)
            undetected_mask = alpha_mask & (~dilated_parts.astype(bool))
            
            # Process only if there are enough undetected pixels
            if np.sum(undetected_mask) > size_threshold:
                print(f"Found undetected parts with {np.sum(undetected_mask)} pixels")
                
                # Find connected components in undetected regions
                num_labels, labels = cv2.connectedComponents(
                    undetected_mask.astype(np.uint8), 
                    connectivity=8
                )
                
                print(f"  Found {num_labels-1} initial regions")
                
                # Use Union-Find data structure for efficient region merging
                parent = list(range(num_labels))
                
                # Find with path compression
                def find(x):
                    """Find with path compression for Union-Find"""
                    if parent[x] != x:
                        parent[x] = find(parent[x])
                    return parent[x]
                
                # Union by rank/size
                def union(x, y):
                    """Union operation for Union-Find"""
                    root_x = find(x)
                    root_y = find(y)
                    if root_x != root_y:
                        # Use smaller ID as parent
                        if root_x < root_y:
                            parent[root_y] = root_x
                        else:
                            parent[root_x] = root_y
                
                # Calculate areas for all regions at once
                areas = np.bincount(labels.flatten())[1:] if num_labels > 1 else []
                
                # Filter regions by minimum size
                valid_regions = np.where(areas >= size_threshold/5)[0] + 1
                
                # Barrier mask for connectivity checks
                barrier_mask = existing_parts_mask
                
                # Pre-compute dilated regions for all valid regions
                dilated_regions = {}
                for i in valid_regions:
                    region_mask = (labels == i).astype(np.uint8)
                    dilated_regions[i] = cv2.dilate(region_mask, kernel, iterations=2)
                
                # Check for region merges based on proximity and overlap
                for idx, i in enumerate(valid_regions[:-1]):
                    for j in valid_regions[idx+1:]:
                        # Check overlap between dilated regions
                        overlap = dilated_regions[i] & dilated_regions[j]
                        overlap_size = np.sum(overlap)
                        
                        # Merge if significant overlap and not separated by existing parts
                        if overlap_size > 40 and not np.any(overlap & barrier_mask):
                            # Calculate overlap ratios
                            overlap_ratio_i = overlap_size / areas[i-1]
                            overlap_ratio_j = overlap_size / areas[j-1]
                            
                            if max(overlap_ratio_i, overlap_ratio_j) > 0.03:
                                union(i, j)
                                print(f"  Merging regions {i} and {j} (overlap: {overlap_size} px)")
                
                # Apply the merging results to create merged labels
                merged_labels = np.zeros_like(labels)
                for label in range(1, num_labels):
                    merged_labels[labels == label] = find(label)
                
                # Get unique merged regions
                unique_merged_regions = np.unique(merged_labels[merged_labels > 0])
                print(f"  After merging: {len(unique_merged_regions)} connected regions")
                
                # Add regions to group_ids if they're large enough
                group_counter_start = group_counter
                for label in unique_merged_regions:
                    region_mask = merged_labels == label
                    region_size = np.sum(region_mask)
                    
                    if region_size > size_threshold:
                        print(f"  Adding region with ID {label} ({region_size} pixels) as group {group_counter}")
                        group_ids[region_mask] = group_counter
                        group_counter += 1
                    else:
                        print(f"  Skipping small region with ID {label} ({region_size} pixels < {size_threshold})")
                
                print(f"  Added {group_counter - group_counter_start} regions that weren't detected by SAM")

                # Process edges for all new parts at once
                if group_counter > group_counter_start:
                    print("Processing edges for newly detected parts...")
                    
                    # Create combined mask for all new parts
                    new_parts_mask = np.zeros_like(group_ids, dtype=bool)
                    for part_id in range(group_counter_start, group_counter):
                        new_parts_mask |= (group_ids == part_id)
                    
                    # Compute edges for all new parts at once
                    all_new_dilated = cv2.dilate(new_parts_mask.astype(np.uint8), kernel, iterations=1)
                    all_new_eroded = cv2.erode(new_parts_mask.astype(np.uint8), kernel, iterations=1)
                    all_new_edges = all_new_dilated.astype(bool) & (~all_new_eroded.astype(bool))
                    
                    print(f"Edge processing completed for {group_counter - group_counter_start} new parts")

    # Save debug visualization of initial segmentation
    if not exist_group:
        get_mask(group_ids, image, ids=2, img_name=img_name, save_dir=save_dir)

    # Merge groups if specified
    if merge_groups is not None:
        # Start with current group_ids
        merged_group_ids = group_ids
        
        # Preserve background regions
        merged_group_ids[group_ids == -1] = -1
        
        # For each merge group, assign all pixels to the first ID in that group
        for new_id, group in enumerate(merge_groups):
            # Create a mask to include all original IDs in this group
            group_mask = np.zeros_like(group_ids, dtype=bool)

            orig_ids_first = group[0]
            # Process each original ID
            for orig_id in group:
                # Get mask for this original ID
                mask = (group_ids == orig_id)
                pixels = np.sum(mask)
                if pixels > 0:
                    print(f"  Including original ID {orig_id} ({pixels} pixels)")
                    group_mask = group_mask | mask
                else:
                    print(f"  Warning: Original ID {orig_id} does not exist")
            
            # Set all pixels in this group to the first ID in the group
            if np.any(group_mask):
                print(f"  Merging {np.sum(group_mask)} pixels to ID {orig_ids_first}")
                merged_group_ids[group_mask] = orig_ids_first

        # Reassign IDs to be continuous from 0
        unique_ids = np.unique(merged_group_ids)
        unique_ids = unique_ids[unique_ids != -1]  # Exclude background
        id_reassignment = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}

        # Create new array with reassigned IDs
        new_group_ids = np.full_like(merged_group_ids, -1)  # Start with all background
        for old_id, new_id in id_reassignment.items():
            new_group_ids[merged_group_ids == old_id] = new_id

        # Update merged_group_ids with continuous IDs
        merged_group_ids = new_group_ids

        print(f"ID reassignment complete: {len(id_reassignment)} groups now have sequential IDs from 0 to {len(id_reassignment)-1}")

        # Replace original group IDs with merged result
        group_ids = merged_group_ids
        print(f"Merging complete, now have {len(np.unique(group_ids))-1} regions (excluding background)")
        
        # Skip splitting disconnected parts if requested
        if not skip_split:
            # Split disconnected parts into separate parts
            group_ids = split_disconnected_parts(group_ids, size_threshold)
            print(f"After splitting disconnected parts, now have {len(np.unique(group_ids))-1} regions (excluding background)")
    else:
        # Always split disconnected parts for initial segmentation
        group_ids = split_disconnected_parts(group_ids, size_threshold)
        print(f"After splitting disconnected parts, now have {len(np.unique(group_ids))-1} regions (excluding background)")

    # Create visualization with clear labeling
    vis_mask = visual
    # First draw background areas (ID -1)
    background_mask = (group_ids == -1)
    if np.any(background_mask):
        vis_mask = visual.draw_binary_mask(background_mask, color=[1.0, 1.0, 1.0], alpha=0.0)

    # Then draw each segment with unique colors and labels
    for unique_id in np.unique(group_ids):
        if unique_id == -1:  # Skip background
            continue
        mask = (group_ids == unique_id)
        
        # Calculate center point and area of this region
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            area = len(y_indices)  # Calculate region area
            
            print(f"Labeling region {unique_id}, area: {area} pixels")
            if area < 30:  # Skip very small regions
                continue
            
            # Use different colors for different IDs to enhance visual distinction
            color_r = (unique_id * 50 + 80) % 200 / 255.0 + 0.2
            color_g = (unique_id * 120 + 40) % 200 / 255.0 + 0.2
            color_b = (unique_id * 180 + 20) % 200 / 255.0 + 0.2
            color = [color_r, color_g, color_b]
            
            # Adjust transparency based on area size
            adaptive_alpha = min(0.3, max(0.1, 0.1 + area / 100000))
            
            # Extract edges of this region
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
            edge = dilated.astype(bool) & (~eroded.astype(bool))
            
            # Build label text
            label = f"{unique_id}"
            
            # First draw the main body of the region
            vis_mask = visual.draw_binary_mask_with_number(
                mask, 
                text=label,
                label_mode=label_mode,
                alpha=adaptive_alpha,
                anno_mode=anno_mode,
                color=color,
                font_size=20
            )
            
            # Enhance edges (add border effect for all parts)
            edge_color = [min(c*1.3, 1.0) for c in color]  # Slightly brighter edge color
            vis_mask = visual.draw_binary_mask(
                edge,
                alpha=0.8,  # Lower transparency for edges to make them more visible
                color=edge_color
            )
            
    im = vis_mask.get_image()
    
    return group_ids, im