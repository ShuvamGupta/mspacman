# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:05:02 2025

@author: gupta46
"""

import numpy as np
from skimage import morphology
from joblib import Parallel, delayed
from tqdm import tqdm


def delete_small_particles(labelled_image, Size_threshold):
    
    """
    Removes labeled particles smaller than a specified size threshold from a 3D labeled image.
    
    Parameters:

    labelled_image : np.ndarray
        A 3D labeled image where each connected region has a unique integer label (background = 0).
    
    Size_threshold : int
        The minimum number of voxels a labeled object must have to be retained.
    
    Returns:

    labelled_image : np.ndarray
        The filtered labeled image with small objects removed.
    """
    
    # Step 1: Convert to binary (1 for any labeled object, 0 for background)
    
    binary = (labelled_image > 0).astype(np.uint8)
    
    binary = np.array(binary, dtype=bool)
    
    # Step 2: Remove small objects from the binary mask
    binary = morphology.remove_small_objects(binary, Size_threshold, connectivity=1)
    
    # Step 3: Convert back to int to use as mask
    binary = binary.astype(int)
    
    # Step 4: Mask the original labeled image to zero out small objects
    labelled_image = labelled_image * binary
    
    del binary
    

    return labelled_image



def get_border_labels(labelled_image):
    
    """
   Identify all unique labels that touch the border of a 3D labeled image.
   
   Parameters:
   labelled_image : np.ndarray
       A 3D labeled image (e.g., from segmentation) with each object assigned a unique label.

   Returns:
   np.ndarray
       Array of labels that do NOT touch any border (excluding background).
   """
   # All unique labels in the volume
    unique = np.unique(labelled_image)
    
    # Collect all unique labels touching each face of the 3D volume
    unique1 = np.unique(labelled_image[0])
    unique2 = np.unique(labelled_image[-1])
    unique3 = np.unique(labelled_image[:,0,:])
    unique4 = np.unique(labelled_image[:,-1,:])
    unique5 = np.unique(labelled_image[:,:,0])
    unique6 = np.unique(labelled_image[:,:,-1])
    
    # Combine border labels
    particles_to_delete = np.concatenate([unique1, unique2, unique3, unique4, unique5, unique6])
    
    # Labels to keep = all labels minus those on the border
    unique_filtered = np.array([i for i in unique if i not in particles_to_delete])
    
    # Exclude background (label 0)
    return np.delete(unique_filtered, np.where(unique_filtered == 0))

def process_slice(slice_data, unique_labels):
    
    """
    Keep only the specified valid labels in a 2D slice, remove all others.
    
    Parameters:
    slice_data : np.ndarray
        A 2D slice of the labeled image.
    
    valid_labels : set or np.ndarray
        Set of labels to keep.
    
    Returns:
    np.ndarray
        Modified slice with invalid labels set to 0.
    """
    slice_data = slice_data.copy()  
    for label in np.unique(slice_data):
        if label not in unique_labels:
            slice_data[slice_data == label] = 0  # Set label to background
    return slice_data

# parallelization of get_border_labels and process_slice
def delete_border_labels(labelled_image):
    border_labels = get_border_labels(labelled_image)
    result = Parallel(n_jobs=-1)(delayed(process_slice)(labelled_image[:, :, i], border_labels) for i in tqdm(range(labelled_image.shape[2]), desc='Deleting border labels'))
    return np.stack(result, axis=2)



def get_label_slices(labels1, labels_per_chunk):
    
    """
    Determine start and end Z-slices for processing chunks of labels in a 3D image.
    
    Parameters:
        labels1 (ndarray): 3D labeled image.
        labels_per_chunk (int): Number of labels to include in each chunk.
    
    Returns:
        start_slices (ndarray): Start slice indices for each chunk.
        end_slices (ndarray): End slice indices for each chunk.
    """
    
    start_slices = []
    end_slices = []
    
    max_label = labels1.max()
    max_slices = labels1.shape[0]  # Total number of slices
    label_ranges = np.arange(1, max_label + 1, labels_per_chunk)
    
    for start_label in tqdm(label_ranges, desc='Processing Labels'):
        end_label = min(start_label + labels_per_chunk - 1, max_label)
        
        # Create a mask to locate where any label in the chunk exists
        mask = (labels1 >= start_label) & (labels1 <= end_label)
        slice_indices = np.any(mask, axis=(1, 2)).nonzero()[0]
        
        del mask  # Free memory
        
        if len(slice_indices) > 0:
            start_slice = slice_indices[0]
            end_slice = slice_indices[-1]
            
            # Adjust start slice
            if start_slice > 1:
                start_slice -= 1
            
            # Adjust end slice
            if end_slice + 1 < max_slices:
                end_slice += 1
            
            start_slices.append(start_slice)
            end_slices.append(end_slice)
    
    return np.array(start_slices), np.array(end_slices)
