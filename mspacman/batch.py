# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:39:19 2025

@author: gupta46
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from .morphology import delete_border_labels, delete_small_particles, get_label_slices
from .properties import calculate_properties, pve_gradient
from .histogram import (
    Bulk_particle_histograms,
    inner_particle_histograms,
    surface_particle_histograms
)

def run_batch_processing(
    labelled_image,
    CT_image,
    Properties,
    Background_mean,
    labels_per_chunk,
    Size_threshold,
    voxel_size,
    step_size,
    calculate_properties_bulk=True,
    compute_bulk_histogram=True,
    compute_inner_volume_histogram=True,
    compute_surface_mesh_histogram=True,
    Gradients=True,
    Processed_labelled_image=None,
    smoothed_ct_image=None,
    compute_smoothed_image_histogram=False
):
    
    """
    Efficiently process large labeled 3D particle images in chunks.
    
    This function splits a large labeled image into chunks of labels and processes
    them one batch at a time. Each batch includes property calculation, histogram
    generation, and partial volume estimation.
    
    Parameters:
        labelled_image : ndarray
            3D labeled particle image.
        CT_image : ndarray
            Corresponding grayscale CT image.
        Properties : list of str
            Properties to calculate using regionprops (e.g., 'area', 'eccentricity').
        Background_mean : float
            Background intensity used for PVE gradient normalization.
        labels_per_chunk : int
            Number of labels to process in each batch.
        Size_threshold : int
            Minimum size (in voxels) for particles to be kept.
        voxel_size : float
            Physical size of a voxel (for unit conversion).
        step_size : int
            Step size used for marching cubes in surface area calculation.
        calculate_properties_bulk : bool
            Whether to calculate geometric properties for each batch.
        compute_bulk_histogram : bool
            Whether to compute histograms over full particle volumes.
        compute_inner_volume_histogram : bool
            Whether to compute histograms on eroded inner volumes.
        compute_surface_mesh_histogram : bool
            Whether to compute histograms on surface voxels.
        Gradients : bool
            Whether to compute gradient-based surface erosion info.
        Processed_labelled_image : ndarray, optional
            Cleaned label image (required for smoothed histogram).
        smoothed_ct_image : ndarray, optional
            Smoothed CT image (e.g., Kuwahara/Gaussian).
        compute_smoothed_image_histogram : bool
            Whether to compute histograms from smoothed CT data.
    """

    # Enforce dependency
    if compute_inner_volume_histogram and not Gradients:
        raise ValueError("To compute `inner_volume_histogram`, `Gradients` must also be True.")
    
    if compute_smoothed_image_histogram:
        if Processed_labelled_image is None or smoothed_ct_image is None:
            raise ValueError("Both `Processed_labelled_image` and `smoothed_ct_image` are required to compute `Smoothed_image_histograms`.")
    
    # Get chunk boundaries based on labels
    start_slices, end_slices = get_label_slices(labelled_image, labels_per_chunk)

    all_properties_bulk = []
    all_bulk_histograms = []
    all_inner_volume_histograms = []
    all_surface_mesh_histograms = []
    all_surface_properties = []
    all_smoothed_histograms = []
    
    # Loop through each chunk based on slice boundaries
    for batch_num, (start_slice, end_slice) in enumerate(zip(start_slices, end_slices), start=1):
        print(f"\nProcessing batch {batch_num} ({start_slice}-{end_slice})")
        
        # Extract chunk from full image
        chunk_labels = labelled_image[start_slice:end_slice + 1]
        chunk_nonbinary = CT_image[start_slice:end_slice + 1]
        
        # Determine label range for this chunk
        start_label = (batch_num - 1) * labels_per_chunk + 1
        end_label = batch_num * labels_per_chunk
        
        # Clean border-touching labels
        chunk_labels = delete_border_labels(chunk_labels)
        
        # Retain only labels in the target range
        unique_labels = np.unique(chunk_labels)
        unique_labels = unique_labels[(unique_labels != 0) & (unique_labels >= start_label) & (unique_labels <= end_label)]
        chunk_labels *= np.isin(chunk_labels, unique_labels)
        
        # Remove small particles
        chunk_labels = delete_small_particles(chunk_labels, Size_threshold)
        
        # Mask the CT image with valid labels
        binary = np.where(chunk_labels > 0, 1, 0)
        chunk_nonbinary = binary * chunk_nonbinary
        chunk_nonbinary = chunk_nonbinary.astype(np.uint16)
        
        # Mask the CT image with valid labels
        unique_labels = np.unique(chunk_labels)
        unique_labels = unique_labels[(unique_labels != 0) & (unique_labels >= start_label) & (unique_labels <= end_label)]
        chunk_labels *= np.isin(chunk_labels, unique_labels)
        
        # Adjust dtype based on label range
        if labels_per_chunk < 65535:
            chunk_nonbinary = chunk_nonbinary.astype(np.uint16)
        else:
            chunk_nonbinary = chunk_nonbinary.astype(np.uint32)

        if calculate_properties_bulk and Properties is not None:
            props = calculate_properties(
                chunk_labels, chunk_nonbinary, Properties,voxel_size, step_size
            )
            all_properties_bulk.append(props)

        print("properties calculated")

        if compute_bulk_histogram:
            bulk_hist = Bulk_particle_histograms(chunk_labels, chunk_nonbinary)
            all_bulk_histograms.append(bulk_hist)

        print("Bulk histograms extracted")

        if Gradients:
            surface_props = pve_gradient(chunk_labels, chunk_nonbinary, Background_mean)
            all_surface_properties.append(surface_props)
        else:
            surface_props = None
        print("Gradients extracted")

        if compute_inner_volume_histogram and surface_props is not None:
            inner_hist = inner_particle_histograms(chunk_labels, chunk_nonbinary, surface_props)
            all_inner_volume_histograms.append(inner_hist)
        print("inner histograms extracted")

        if compute_surface_mesh_histogram:
            surface_mesh = surface_particle_histograms(chunk_labels, chunk_nonbinary)
            all_surface_mesh_histograms.append(surface_mesh)
        print("surface mesh histograms extracted")

        # Handle Smoothed histogram extraction
        if compute_smoothed_image_histogram:

            chunk_processed_labels = Processed_labelled_image[start_slice:end_slice + 1] * binary
            chunk_smoothed_ct = smoothed_ct_image[start_slice:end_slice + 1]

            
            unique_processed_labels = np.unique(chunk_processed_labels)

            chunk_processed_labels *= np.isin(chunk_processed_labels, unique_processed_labels)

            chunk_smoothed_ct = chunk_smoothed_ct.astype(np.uint16)

            smoothed_hist = Bulk_particle_histograms(chunk_processed_labels, chunk_smoothed_ct)
            all_smoothed_histograms.append(smoothed_hist)
            print("Smoothed image histograms extracted")

        del chunk_labels, chunk_nonbinary

    results = {}

    if all_properties_bulk:
        results["properties_bulk"] = pd.concat(all_properties_bulk, ignore_index=False)

    if all_bulk_histograms:
        results["bulk_histogram"] = pd.concat(all_bulk_histograms, ignore_index=False)

    if all_inner_volume_histograms:
        results["inner_volume_histogram"] = pd.concat(all_inner_volume_histograms, ignore_index=False)

    if "bulk_histogram" in results and "inner_volume_histogram" in results:
        results["outer_volume_histogram"] = results["bulk_histogram"] - results["inner_volume_histogram"]

    if all_surface_mesh_histograms:
        results["surface_mesh_histogram"] = pd.concat(all_surface_mesh_histograms, ignore_index=False)

    if all_surface_properties:
        results["Gradients"] = pd.concat(all_surface_properties, ignore_index=False)

    if all_smoothed_histograms:
        results["Smoothed_image_histogram"] = pd.concat(all_smoothed_histograms, ignore_index=False)

    return results
