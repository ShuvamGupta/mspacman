# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:08:07 2025

@author: gupta46
"""

import numpy as np
from skimage.restoration import denoise_tv_chambolle
from pykuwahara import kuwahara
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from joblib import Parallel, delayed
from tqdm import tqdm


def tv_chambolle_3d(CT_image, labelled_image, weight, max_num_iter):
    """
    Denoise the CT image within labeled regions using Chambolle total variation.

    Parameters:
        CT_image (ndarray): Raw CT image.
        labelled_image (ndarray): Labeled image.
        weight (float): Denoising weight.
        max_num_iter (int): Number of iterations for denoising.

    Returns:
        denoised (ndarray): Denoised 3D image in original scale.
    """
    
    # Mask CT image to focus denoising only on labeled regions
    image = np.where(labelled_image > 0, CT_image, 0).astype(np.float32)
    
    # Normalize the image to [0, 1] range for stable TV denoising
    image /= np.max(image)
    
    # Apply TV denoising slice by slice in parallel
    denoised_slices = Parallel(n_jobs=-1)(
        delayed(denoise_tv_chambolle)(image[i], weight=weight, max_num_iter=max_num_iter)
        for i in tqdm(range(image.shape[0]), desc="Denoising 3D Image")
    )
    del image  # Free memory
    
    # Stack all 2D slices back into a 3D volume
    denoised = np.stack(denoised_slices, axis=0)
    del denoised_slices
    
    # Rescale denoised image back to original intensity range and convert to 16-bit
    denoised = (denoised * np.max(CT_image)).astype(np.uint16)
    return denoised


def kuwahara_3d(image, radius, method='gaussian'):
    """
    Apply Kuwahara filtering slice-by-slice on a 3D image.

    Parameters:
        image (ndarray): 3D input image (e.g., CT or grayscale stack).
        radius (int): Radius of the filtering kernel (must be > 0).
        method (str): 'gaussian' or 'mean' (default is 'gaussian').

    Returns:
        ndarray: 3D Kuwahara-filtered image (same dtype as input).
    """

    # Store the original max value for later rescaling
    original_max = np.max(image)

    # Convert image to float32 and normalize to [0, 1]
    image = image.astype(np.float32)
    image /= original_max

    # Apply Kuwahara filter to each 2D slice in parallel
    filtered_slices = Parallel(n_jobs=-1)(
        delayed(kuwahara)(image[i], method=method, radius=radius)
        for i in tqdm(range(image.shape[0]), desc="Applying Kuwahara Filter")
    )

    # Stack the filtered slices back into a 3D image
    filtered_image = np.stack(filtered_slices, axis=0)

    # Rescale the filtered image back to original intensity range and cast to uint16
    filtered_image = (filtered_image * original_max).astype(np.uint16)

    return filtered_image




def gaussian_3d(CT_image, labelled_image, sigma):
    """
    Apply Gaussian smoothing within labeled regions of a 3D image and return uint16 result.

    Parameters:
        CT_image (ndarray): Raw CT image.
        labelled_image (ndarray): Label mask.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        ndarray: Smoothed 3D image in uint16.
    """

    # Mask the CT image: keep only regions where labels exist
    masked_image = np.where(labelled_image > 0, CT_image, 0).astype(np.float32)

    # Normalize the masked image to [0, 1] range
    masked_image /= np.max(masked_image)

    # Apply 3D Gaussian filter to the normalized image
    smoothed = gaussian_filter(masked_image, sigma=sigma)

    # Rescale back to original intensity range and convert to uint16
    smoothed = (smoothed * np.max(CT_image)).astype(np.uint16)

    return smoothed



def sobel_cleanup(denoised, labelled_image, edge_thresh):
    """
    Apply Sobel filter to detect edges and remove edge regions from the labels.

    Parameters:
        denoised (ndarray): Denoised 3D image.
        labelled_image (ndarray): Labeled 3D image.
        edge_thresh (float): Threshold for Sobel edge removal.

    Returns:
        labels_cleaned (ndarray): Labels after edge cleanup.
    """

    # Normalize the denoised image to [0, 1] range for edge detection
    image = denoised.astype(np.float32) / np.max(denoised)

    # Apply Sobel filter to each 2D slice in parallel to detect edges
    sobel_slices = Parallel(n_jobs=-1)(
        delayed(sobel)(image[i]) for i in tqdm(range(image.shape[0]), desc="Applying Sobel")
    )
    del image  # Free up memory

    # Stack individual 2D Sobel results back into a 3D edge mask
    edge_mask = np.stack(sobel_slices, axis=0)
    del sobel_slices

    # Threshold the edge mask to get binary mask of strong edges
    edge_mask = (edge_mask * np.max(denoised)) > edge_thresh

    # Remove edge regions from the original labels
    labels_cleaned = labelled_image * (~edge_mask)

    # Return the cleaned labels with the original data type
    return labels_cleaned.astype(labelled_image.dtype)

