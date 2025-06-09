# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 15:33:27 2025

@author: gupta46
"""

from .morphology import (
    delete_small_particles,
    delete_border_labels,
    get_border_labels,
    process_slice,
    get_label_slices
)

from .io_utils import (
    upload_images
)

from .denoise import (
    tv_chambolle_3d,
    kuwahara_3d,
    gaussian_3d,
    sobel_cleanup
)

from .histogram import (
    Bulk_particle_histograms,
    bin_histograms,
    smooth_histograms,
    find_peaks_and_arrange,
    map_peaks_on_bulk_histograms,
    bias_dense_minerals,
    inner_particle_histograms,
    surface_particle_histograms
)

from .properties import (
    calculate_properties,
    pve_gradient
)

from .batch import run_batch_processing

from .visualization import view_particle

from .quantification import (
    Liberated_Particles,
    Binary_Particles,
    Ternary_Particles,
    Quaternary_Particles,
    Quinary_Particles,
    Concatenate,
    calculate_bootstrapping_error_bulk,
    calculate_bootstrapping_error_surface,
    compute_particle_quantification_percentages,
    compute_particle_surface_percentages,
    compute_particle_outer_percentages
)

from .utils import (
    convert_and_save_as_h5ad,
    extract_peaks_and_thresholds,
    standardize_index,
    upload_histograms_h5ad
)

from .plot import (
    plot_all_rows,
    plot_histograms_with_peaks
)

__all__ = [
    # Morphology
    "delete_small_particles",
    "delete_border_labels",
    "get_border_labels",
    "process_slice",
    "get_label_slices",
    # IO
    "upload_images",
    # Denoise
    "tv_chambolle_3d",
    "kuwahara_3d",
    "gaussian_3d",
    "sobel_cleanup",
    # Histogram and Plots
    "Bulk_particle_histograms",
    "bin_histograms",
    "smooth_histograms",
    "find_peaks_and_arrange",
    "map_peaks_on_bulk_histograms",
    "bias_dense_minerals",
    "inner_particle_histograms",
    "surface_particle_histograms",
    # Properties
    "calculate_properties",
    "pve_gradient",
    # Batch
    "run_batch_processing",
    # Plot
    "plot_all_rows",
    "plot_histograms_with_peaks",
    # Visualization
    "view_particle",
    # Quantification
    "Liberated_Particles",
    "Binary_Particles",
    "Ternary_Particles",
    "Quaternary_Particles",
    "Quinary_Particles",
    "Concatenate",
    "calculate_bootstrapping_error_bulk",
    "calculate_bootstrapping_error_surface",
    "compute_particle_quantification_percentages",
    "compute_particle_surface_percentages",
    "compute_particle_outer_percentages",
    # Utils
    "extract_peaks_and_thresholds",
    "standardize_index",
    "upload_histograms_h5ad",
    "convert_and_save_as_h5ad"
]
