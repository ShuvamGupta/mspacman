# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:34:20 2025

@author: gupta46
"""

import numpy as np
import pandas as pd
from skimage import measure
from tqdm import tqdm
from joblib import Parallel, delayed
from skimage import morphology
from skimage.measure import regionprops
from skimage.measure import marching_cubes, mesh_surface_area
from scipy.spatial import ConvexHull, QhullError
import gc

def calculate_properties(labelled_image, ct_image, Properties, voxel_size, step_size):
    
    """
    Calculate 3D morphological and surface features for labeled particles in a CT image.
    
    Parameters:
    labelled_image : np.ndarray
        3D array with integer labels for each segmented particle.
    
    ct_image : np.ndarray
        3D grayscale CT image.
    
    Properties : list
        List of region properties to calculate using skimage.
    
    
    voxel_size : float
        Size of a voxel edge in microns or other physical unit.
    
    step_size : int
        Step size used in mesh generation for surface area estimation.
    
    Returns:
    merged : pd.DataFrame
        DataFrame containing calculated properties indexed by particle label.
    """

    unique_labels = np.unique(labelled_image)
    unique_labels = unique_labels[unique_labels != 0]

    # Surface Area Calculation 
    def calculate_surface_area(label, labelled_image, step_size):
        non_zero_indices = np.argwhere(labelled_image == label)
        if non_zero_indices.shape[0] < 2:
            return 0.0
        
        
        # Define a slightly extended bounding box
        min_idx = np.maximum(non_zero_indices.min(axis=0) - 2,  0)
        max_idx = np.minimum(non_zero_indices.max(axis=0) + 2, np.array(labelled_image.shape) - 1)
        
        # Crop the region and extract binary mask for current label
        region = labelled_image[min_idx[0]:max_idx[0]+1,
                                min_idx[1]:max_idx[1]+1,
                                min_idx[2]:max_idx[2]+1]

        mask = (region == label).astype(np.uint8)
        
        # Skip invalid shapes
        if np.any(np.array(mask.shape) < 2) or mask.sum() == 0:
            return 0.0
        
        
        # Compute surface mesh and calculate surface area
        verts, faces, *_ = marching_cubes(mask, level=0.5, spacing=(voxel_size,) * 3, step_size=step_size)
        return mesh_surface_area(verts, faces)

    def calculate_surface_areas(labelled_image, step_size, labels_to_process):
        
        """Compute surface area for all labels in parallel."""
        return pd.DataFrame({
            'label': labels_to_process,
            'Surface Area': Parallel(n_jobs=-1)(
                delayed(calculate_surface_area)(label, labelled_image, step_size)
                for label in tqdm(labels_to_process, desc='Calculating Surface Areas')
            )
        })

    surface_areas_df = calculate_surface_areas(labelled_image, step_size, unique_labels)

    # Region Properties 
    feret_keys = {'min_feret_diameter', 'max_feret_diameter'}
    feret_requested = feret_keys.intersection(Properties)
    Properties = [p for p in Properties if p not in feret_keys]
    
    # Compute skimage regionprops
    regionprops_df = pd.DataFrame(
        measure.regionprops_table(labelled_image, intensity_image=ct_image, properties=Properties)
    ).astype({'label': int}).set_index('label')

    # Rename area-related features
    rename_map = {}
    if 'area' in regionprops_df.columns:
        rename_map['area'] = 'Volume'
    if 'bbox_area' in regionprops_df.columns:
        rename_map['bbox_area'] = 'Bounding_Box_Volume'
    if 'filled_area' in regionprops_df.columns:
        rename_map['filled_area'] = 'Filled_Volume'

    regionprops_df.rename(columns=rename_map, inplace=True)

    # Merge Surface Area 
    surface_areas_df['label'] = surface_areas_df['label'].astype(int)
    merged = surface_areas_df.merge(regionprops_df, how='left', on='label')

    # Unit Conversion 
    voxel_volume = voxel_size ** 3

    # Properties that should be scaled by voxel size (length) or voxel volume (volume)
    length_keys = {
        'equivalent_diameter', 'major_axis_length', 'minor_axis_length'
    }
    volume_keys = {
        'Volume', 'Bounding_Box_Volume', 'Filled_Volume'
    }

    for col in merged.columns:
        if col in length_keys:
            merged[col] *= voxel_size
        elif col in volume_keys:
            merged[col] *= voxel_volume

    # Sphericity 
    merged['Sphericity'] = np.nan
    valid_area = merged['Surface Area'] > 0
    merged.loc[valid_area, 'Sphericity'] = ((np.pi ** (1/3)) *
        (6 * merged.loc[valid_area, 'Volume']) ** (2/3)) / (merged.loc[valid_area, 'Surface Area'])

    # Feret Diameters 
    if feret_requested:

        def fibonacci_sphere_samples(n=64800):  # 180x360 grid for ~0.5° spacing
            """Generate uniformly distributed directions on a sphere with ~0.5° spacing."""
            indices = np.arange(n, dtype=float) + 0.5
            phi = np.arccos(1 - 2 * indices / n)  # Polar angle (0 to π)
            theta = np.pi * (1 + 5**0.5) * indices  # Azimuthal angle (golden spiral)
            return np.stack([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ], axis=1)

        def exact_feret_diameters(coords):
            """Compute min/max Feret diameters with guaranteed ±0.5° accuracy."""
            if len(coords) < 2:
                return 0.0, 0.0
            
            # Use convex hull for efficiency (exact for Feret)
            try:
                hull = ConvexHull(coords)
                pts = coords[hull.vertices]
            except (QhullError, ValueError):
                pts = coords
            
            # Pre-compute all direction vectors
            directions = fibonacci_sphere_samples()
            
            # Vectorized projection (fastest method)
            projections = pts @ directions.T
            spans = np.max(projections, axis=0) - np.min(projections, axis=0)
            
            return np.max(spans), np.min(spans)

        def _process_object(prop):
            """Extract object coords and compute Feret diameters."""
            mask = prop.image
            if mask.sum() == 0:
                return prop.label, 0.0, 0.0
            
            # Get all object voxels (more accurate than surface-only for convex hull)
            coords = np.argwhere(mask)
            min_z, min_y, min_x, _, _, _ = prop.bbox
            global_coords = coords + np.array([min_z, min_y, min_x])
            
            max_feret, min_feret = exact_feret_diameters(global_coords)
            return prop.label, max_feret, min_feret

        def compute_feret_diameters(label_img, n_jobs=-1):
            """Main function: compute Feret diameters for all objects in a 3D label image."""
            props = regionprops(label_img)
            results = Parallel(n_jobs=n_jobs)(
                delayed(_process_object)(prop)
                for prop in tqdm(props, desc="Computing Feret diameters")
            )
            return pd.DataFrame(results, columns=['label', 'Max_Feret', 'Min_Feret']).set_index('label')

        # Usage
        feret_df = compute_feret_diameters(labelled_image)

        merged = merged.merge(feret_df, how='left', on='label')
        
        # Convert Feret diameters to physical units
        
        merged['Max_Feret'] *= voxel_size
        merged['Min_Feret'] *= voxel_size
        merged['Feret_ratio'] = merged['Min_Feret'] / merged['Max_Feret']

    # Extra ratio
    if 'Bounding_Box_Volume' in merged.columns and 'Volume' in merged.columns:
        merged['Volume/bbox'] = merged['Volume'] / merged['Bounding_Box_Volume']

    # Save and Return 

    
    merged = merged.rename(columns={'label': 'Label'})
    merged = merged.set_index('Label')


    return merged



def pve_gradient(labelled_image, ct_image, Background_mean):
    
    """
    Calculate the Partial Volume Effect (PVE) gradient for labeled particles.

    This function measures the mean CT intensity of the surface layer voxels at multiple erosion levels
    and normalizes the gradient against the maximum mean intensity to identify PVE artifacts.

    Parameters:
    labelled_image : np.ndarray
        3D array with labeled segmented particles.
        
    ct_image : np.ndarray
        3D grayscale CT image.
        
    Background_mean : float
        Mean intensity value of the background (used for normalization).

    Returns:
    surface_properties : pd.DataFrame
        DataFrame indexed by particle label with normalized gradients (Gradient_1 to Gradient_6).
    """
    
    # Create binary mask from labeled image (1 = particle, 0 = background)
    binary = np.where(labelled_image>0,1,0)
    surface_properties_list = []
    eroded_image = binary.astype(int)

    # Iterate through 6 erosion levels to extract surface layers
    
    for i in range(1, 7):
        
        # Erode the binary mask to get interior structure
        
        eroded_image = morphology.binary_erosion(eroded_image).astype(np.uint8)
        
        # Subtract to isolate the surface layer voxels for this level

        surface_diff = binary - eroded_image
        
        surface_diff = surface_diff.astype(np.uint8)
        
        # Map surface voxels back to their label values

        surface_labels = labelled_image*surface_diff
        del surface_diff
  
        print(f"surface_labels{i}_done")
        
        print (len(np.unique(surface_labels)))
        
        # Compute mean intensity for surface voxels per particle

        surface_mesh_properties = pd.DataFrame(
            measure.regionprops_table(
                surface_labels, ct_image,
                properties=['label', 'mean_intensity']
            )
        ).set_index('label')
        surface_mesh_properties.index.name = 'Label'
        
        del surface_labels
        
        # Rename the mean intensity column for this erosion depth

        surface_mesh_properties = surface_mesh_properties.rename(columns={
            "mean_intensity": f"mean_intensity{i}"
        })
        
        surface_properties_list.append(surface_mesh_properties)
        print(f"surface_properties_list{i}_done")

        del surface_mesh_properties
        gc.collect()

    del eroded_image, binary
    
    # Combine all mean intensity tables
    
    surface_properties = pd.concat(surface_properties_list, axis=1)
    cols = [f"mean_intensity{i}" for i in range(1, 7)]
    surface_properties = surface_properties[cols]
    
    # Compute normalized gradient at each erosion level

    gradient_columns = []
    max_mean_intensity = surface_properties.max(axis=1)
    for i in range(1, 7):
        gradient_col_name = f"Gradient_{i}"
        surface_properties[gradient_col_name] = (surface_properties[f"mean_intensity{i}"] - Background_mean) / (max_mean_intensity - Background_mean)
        gradient_columns.append(gradient_col_name)

    return surface_properties
