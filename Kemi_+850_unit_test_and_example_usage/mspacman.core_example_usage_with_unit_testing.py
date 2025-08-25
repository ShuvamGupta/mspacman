# -*- coding: utf-8 -*-
"""
Created on Wed May 14 20:45:20 2025

@author: gupta46

This script serves a dual purpose:
1. **Example usage** – Demonstrates the full MSPaCMAn processing pipeline 
   using the Kemi +850 µm micro-CT dataset.
2. **Unit/regression test** – Verifies that each step of the pipeline produces
   identical results to pre-computed "golden" reference outputs stored in the
   Kemi_+850_unit_test_and_example_usage directory.

If the outputs match, the implementation is correct. If they differ,
it indicates changes in the algorithm, parameters, or input data.

"""

import mspacman.core
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# Step 1: Define paths to raw CT data and labeled particle masks
# -------------------------------------------------------------------------
# Download the sample dataset from Figshare:
# https://figshare.com/articles/dataset/Kemi_850_m_CT_data_and_Particle_Labels/29836160
#
# Non_binary_path: directory containing the raw (non-binary) CT image slices
# Labelled_path : directory containing the corresponding labeled segmentation
# -------------------------------------------------------------------------

Non_binary_path = r"C:\Users\gupta46\Downloads\29836160"

Labelled_path = r"C:\Users\gupta46\Downloads\29836160\Labels"

# -------------------------------------------------------------------------
# Step 2: Load pre-computed reference datasets for testing
# -------------------------------------------------------------------------
# The reference outputs were generated previously using the same inputs and
# parameters as in this script. They are stored in:
#   https://github.com/ShuvamGupta/mspacman/tree/main/Kemi_%2B850_unit_test_and_example_usage
#
# To download only that folder without cloning the whole repo:
#   1. Go to https://download-directory.github.io/
#   2. Paste the above GitHub folder URL
#
# These files will be compared against freshly generated results to verify
# correctness (regression testing).
# -------------------------------------------------------------------------


# Upload data using mspacman.core.upload_histograms_h5ad(Path_of_histogram) 

Bulk_histograms_test = mspacman.core.upload_histograms_h5ad(r"C:\Users\gupta46\Downloads\ShuvamGupta mspacman main Kemi_+850_unit_test_and_example_usage\Bulk_histograms.h5ad")
Inner_histograms_test = mspacman.core.upload_histograms_h5ad(r"C:\Users\gupta46\Downloads\ShuvamGupta mspacman main Kemi_+850_unit_test_and_example_usage\Inner_histograms.h5ad")
Outer_histograms_test = mspacman.core.upload_histograms_h5ad(r"C:\Users\gupta46\Downloads\ShuvamGupta mspacman main Kemi_+850_unit_test_and_example_usage\Outer_histograms.h5ad")
Smoothed_histograms_test = mspacman.core.upload_histograms_h5ad(r"C:\Users\gupta46\Downloads\ShuvamGupta mspacman main Kemi_+850_unit_test_and_example_usage\Smoothed_histogram.h5ad")
Surface_histograms_test = mspacman.core.upload_histograms_h5ad(r"C:\Users\gupta46\Downloads\ShuvamGupta mspacman main Kemi_+850_unit_test_and_example_usage\Surface_histograms.h5ad")
Properties_test = pd.read_csv(r"C:\Users\gupta46\Downloads\ShuvamGupta mspacman main Kemi_+850_unit_test_and_example_usage\Properties.csv", index_col = 0)
Gradient_test = pd.read_csv(r"C:\Users\gupta46\Downloads\ShuvamGupta mspacman main Kemi_+850_unit_test_and_example_usage\Gradients.csv", index_col = 0)

Smoothed_binned_test = pd.read_csv(r"C:\Users\gupta46\Downloads\ShuvamGupta mspacman main Kemi_+850_unit_test_and_example_usage\Smoothed_binned.csv", index_col = 0)
Smoothed_binned_svgol_test = pd.read_csv(r"C:\Users\gupta46\Downloads\ShuvamGupta mspacman main Kemi_+850_unit_test_and_example_usage\Smoothed_binned_svgol.csv", index_col = 0)
Peaks_test = pd.read_csv(r"C:\Users\gupta46\Downloads\ShuvamGupta mspacman main Kemi_+850_unit_test_and_example_usage\Peaks.csv", index_col = 0)
Peaks_mapped_test = pd.read_csv(r"C:\Users\gupta46\Downloads\ShuvamGupta mspacman main Kemi_+850_unit_test_and_example_usage\Peaks_mapped.csv", index_col = 0)
Peaks_dense_bias_test = pd.read_csv(r"C:\Users\gupta46\Downloads\ShuvamGupta mspacman main Kemi_+850_unit_test_and_example_usage\Peaks_dense_bias.csv", index_col = 0)
Quantification_test = pd.read_csv(r"C:\Users\gupta46\Downloads\ShuvamGupta mspacman main Kemi_+850_unit_test_and_example_usage\Quantification.csv", index_col = 0)


# -------------------------------------------------------------------------
# Function: data_upload_unit_test
#
# Purpose:
#   Verify that all pre-loaded reference datasets have the correct shape
#   and indexing before running further comparisons.
#
# What it checks:
#   - Shape matches the expected (rows, columns)
#   - Index name is exactly 'Label'
#
# Expected:
#   - Histograms (Bulk, Inner, Outer, Smoothed, Surface):
#       Shape: 17 rows × 65536 columns, index name: 'Label'
#   - Properties:
#       Shape: 17 rows × 13 columns, index name: 'Label'
#   - Gradient:
#       Shape: 17 rows × 12 columns, index name: 'Label'
#
# Output:
#   - Prints "Data upload unit test PASS" if all conditions hold
#   - Otherwise prints "FAIL" and lists the reasons
# -------------------------------------------------------------------------

def data_upload_unit_test(
    Bulk_histograms_test,
    Inner_histograms_test,
    Outer_histograms_test,
    Smoothed_histograms_test,
    Surface_histograms_test,
    Properties_test,
    Gradient_test,
):

    errors = []

    def check_df(name, df, exp_rows, exp_cols):
        # Shape
        if df.shape != (exp_rows, exp_cols):
            errors.append(f"{name}: shape {df.shape} != ({exp_rows}, {exp_cols})")
        # Index name
        if getattr(df.index, "name", None) != "Label":
            errors.append(f"{name}: index name {repr(df.index.name)} != 'Label'")

    # Histograms
    check_df("Bulk_histograms_test",     Bulk_histograms_test,     17, 65536)
    check_df("Inner_histograms_test",    Inner_histograms_test,    17, 65536)
    check_df("Outer_histograms_test",    Outer_histograms_test,    17, 65536)
    check_df("Smoothed_histograms_test", Smoothed_histograms_test, 17, 65536)
    check_df("Surface_histograms_test",  Surface_histograms_test,  17, 65536)

    # Properties and Gradient
    check_df("Properties_test", Properties_test, 17, 13)
    check_df("Gradient_test",   Gradient_test,   17, 12)

    if errors:
        print("Data upload unit test FAIL")
        print("Reasons:")
        for e in errors:
            print("-", e)
    else:
        print("Data upload unit test PASS")

data_upload_unit_test(
    Bulk_histograms_test,
    Inner_histograms_test,
    Outer_histograms_test,
    Smoothed_histograms_test,
    Surface_histograms_test,
    Properties_test,
    Gradient_test,
)

# Upload images 
# mspacman.core.upload_images(image_path)
Non_binary = mspacman.core.upload_images(Non_binary_path)
Labels = mspacman.core.upload_images(Labelled_path)

print (Non_binary.shape)
# Should give output: (561, 1427, 1427)

print (Labels.shape)
# Should give output: (561, 1427, 1427)(561, 1427, 1427)


# Can also use napari to check
import napari
viewer = napari.Viewer()

# Add raw as a grayscale
viewer.add_image(Non_binary, name='Raw CT', contrast_limits=[Non_binary.min(), Non_binary.max()])

# Add labels as a segmentation
viewer.add_labels(Labels, name='Labeled Particles')

napari.run()


"""
Smoothing is necessary to reduce small greyscale variations in CT images.
This helps in revealing clearer and more distinct histogram peaks, which are
essential for accurate phase segmentation and peak detection.

Users should decide whether smoothing is needed and choose the appropriate filter 
based on the sample characteristics and analysis goals.

Step 1: Apply 3D Total Variation (TV) denoising to reduce noise while preserving edges.

Usage:
    denoised = mspacman.core.tv_chambolle_3d(image, mask, weight, max_iter)

Parameters:
    image     : 3D CT image to be denoised (e.g., Non_binary)
    mask      : Binary or labeled mask to restrict the denoising region (e.g., Labels)
    weight    : Smoothing factor (typical range: 0.01–0.1). Higher values apply stronger smoothing.
    max_iter  : Number of iterations to run the TV algorithm
"""

denoised = mspacman.core.tv_chambolle_3d(Non_binary, Labels, 0.03, 100)

"""
Step 2: Refine the labeled particle boundaries using Sobel edge detection.

This step helps identify and clean boundary voxels that may be affected by partial volume effects.
By applying a Sobel filter to the greyscale image, edges are detected, and low-gradient boundary
voxels (below the specified threshold) can be marked as uncertain or removed.

Usage:
    cleaned_labels = mspacman.core.sobel_cleanup(image, labels, threshold)

Parameters:
    image    : 3D denoised CT image (usually output from TV denoising)
    labels   : Labeled 3D mask containing particle IDs
    threshold: Gradient magnitude threshold. Voxels with Sobel edge values below this
                are considered affected by partial volume and can be excluded or corrected.
"""

Labels_cleaned = mspacman.core.sobel_cleanup(denoised, Labels, 1400)

"""
Step 3: Apply Kuwahara 3D filtering for edge-aware noise reduction.

Kuwahara filtering smooths the image while preserving edges by adaptively selecting
the least-variant region in a voxel's neighborhood. This helps reduce noise without
blurring important structural boundaries.

Usage:
    filtered = mspacman.core.kuwahara_3d(image, radius, method)

Parameters:
    image  : 3D denoised image (typically from previous TV denoising step)
    radius : Neighborhood radius used for local filtering (typical: 1–3).
             Larger radius increases smoothing but may oversmooth fine features.
    method : Method for variance computation inside subregions. Options:
             - 'gaussian' (recommended): uses weighted local smoothing
             - 'box': uses uniform local averaging
"""

kuwahara_denoised = mspacman.core.kuwahara_3d(denoised, radius=2, method='gaussian')

"""
Step 4: Apply Gaussian 3D smoothing (basic blurring) inside labeled regions.

Gaussian filtering performs isotropic blurring by averaging voxel intensities with a 
Gaussian-weighted kernel. This reduces high-frequency noise uniformly across the labeled regions.

Usage:
    smoothed = mspacman.core.gaussian_3d(image, mask, sigma)

Parameters:
    image : 3D CT image (can be original or previously denoised)
    mask  : Labeled mask to restrict smoothing only within specific particles or regions
    sigma : Standard deviation of the Gaussian kernel. Controls the amount of smoothing.
            Typical values range from 0.5 to 2. Higher values apply more blur.

Notes:
    - Unlike Kuwahara or TV filters, Gaussian filtering does not preserve edges.
    - Best used for low-pass smoothing or as a quick denoise when edge preservation isn't critical.
"""

denoised_gausssian = mspacman.core.gaussian_3d(Non_binary, Labels, 1)


# Delete the particle labels that are cut off by the borders
Labels = mspacman.core.delete_border_labels(Labels)
print (np.unique(Labels))
# Should give output: [ 0  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 43 45 46 47 48 51]

# Delete small particles
# mspacman.core.delete_small_particles(labelled_image, Size_threshold)
# Size_threshold is the volume threshold of the particle as the number of voxels 

Labels = mspacman.core.delete_small_particles(Labels, 500000)
print (np.unique(Labels))
# Should give output: [ 0  7  8  9 10 15 18 20 21 25 26 34 36 37 38 43 45 46]




# List the properties you want to extract as an array. The complete list can be seen on skimage.regioprops
Properties = [
    'label', 'area', 'min_intensity', 'max_intensity', 'equivalent_diameter',
    'mean_intensity', 'bbox_area', 'filled_area',
    'min_feret_diameter', 'max_feret_diameter']


"""
Run batch processing on large 3D CT datasets with labeled particles.

This function processes properties and greyscale histograms in chunks, which is especially useful
when the data is too large to handle in a single pass — even on high-performance systems.

Parameters:
-----------
labelled_image : ndarray
    3D labeled image where each particle has a unique label (e.g., from segmentation).

CT_image : ndarray
    3D greyscale CT image used for calculating intensity-based properties and histograms.

Properties : dict
    A dictionary that stores particle-wise geometric and intensity-based properties.

Background_mean : int or float
    Grey value representing the peak/background matrix. Used in PVE gradient estimation.

labels_per_chunk : int
    Number of labeled particles to process at a time. Helps manage memory usage.
    Example: 3000 means process 3000 particles per batch.

Size_threshold : int
    Particles smaller than this volume (in voxels) will be excluded from processing.

voxel_size : float
    The real-world size of each voxel (e.g., 5 means 5 µm if units are in microns).
    Used to convert raw voxel-based measurements to physical units.

step_size : float
    Resolution step for marching cubes surface mesh. Lower step size = smoother surface but higher computation time.

calculate_properties_bulk : bool
    If True, computes volume, surface area, shape descriptors, and Feret diameters.

compute_bulk_histogram : bool
    If True, computes full greyscale histograms from all voxels of each particle.

compute_inner_volume_histogram : bool
    If True, computes histograms from the inner volume of particles.

compute_surface_mesh_histogram : bool
    If True, computes histograms based on surface mesh of particles.

Gradients : bool
    If True, calculates mean grey value gradients from particle surface to center.

Processed_labelled_image : ndarray, optional
    A refined label image (e.g., after Sobel cleanup).Required if compute_smoothed_image_histogram is set to True.

smoothed_ct_image : ndarray, optional
    A denoised version of the CT image (e.g., using Kuwahara filter) used for extracting clearer histograms.
    Required if compute_smoothed_image_histogram is set to True.

compute_smoothed_image_histogram : bool
    If True, computes greyscale histograms from the smoothed CT image.
    WARNING: This must be set to False if smoothed_ct_image or Processed_labelled_image is not provided.

"""

results = mspacman.core.run_batch_processing(
    labelled_image=Labels,
    CT_image=Non_binary,
    Properties=Properties,
    Background_mean=1000,
    labels_per_chunk=20,
    Size_threshold=500000,
    voxel_size=5,
    step_size=1,
    calculate_properties_bulk=True,
    compute_bulk_histogram=True,
    compute_inner_volume_histogram=True,
    compute_surface_mesh_histogram=True,
    Gradients=True,
    Processed_labelled_image=Labels_cleaned,
    smoothed_ct_image=kuwahara_denoised,
    compute_smoothed_image_histogram=True
)

Properties_batching = results.get("properties_bulk")
Bulk_histograms_batching = results.get("bulk_histogram")
Inner_histograms_batching = results.get("inner_volume_histogram")
Outer_histograms_batching = results.get("outer_volume_histogram")
Surface_histograms_batching = results.get("surface_mesh_histogram")
Gradients_batching = results.get("Gradients")
Smoothed_histogram_batching = results.get("Smoothed_image_histogram")

# -------------------------------------------------------------------------
# Function: test_datasets_equal
#
# Purpose:
#   Compare two DataFrames to ensure they are identical in content
#   to a pre-computed reference dataset.
#
# Context:
#   In this script, each newly generated dataset is compared against
#   its corresponding "golden" dataset stored in:
#   https://github.com/ShuvamGupta/mspacman/tree/main/Kemi_%2B850_unit_test_and_example_usage
#
# Pass criteria:
#   - All values must match exactly (ignoring dtype differences)
#   - Column names must match exactly (order and spelling)
#   - Index type and name differences are ignored for value comparison
#
# Output:
#   - Prints "Unit test passed" if DataFrames match
#   - Otherwise prints "Datasets are not the same" with the reason
# -------------------------------------------------------------------------
def test_datasets_equal(df1, df2):
    try:
        # Convert column names to strings for both DataFrames
        df1 = df1.copy()
        df2 = df2.copy()
        df1.columns = df1.columns.astype(str)
        df2.columns = df2.columns.astype(str)

        pd.testing.assert_frame_equal(
            df1, df2,
            check_dtype=False,      
            check_index_type=False,  
            check_like=False,
            check_names=False
        )

        # After equality check, ensure column names match exactly
        if not df1.columns.equals(df2.columns):
            raise AssertionError("Column names are different")
        
        print("Unit test passed")
    except AssertionError as e:
        print("Datasets are not same")
        print("Reason:", e)


# Unit test for Properties batching dataframe
test_datasets_equal(Properties_batching, Properties_test)
# Unit test for Bulk_histograms_batching dataframe
test_datasets_equal(Bulk_histograms_batching, Bulk_histograms_test)
# Unit test for Inner_histograms_batching
test_datasets_equal(Inner_histograms_batching, Inner_histograms_test)
# Unit test for Properties batching dataframe
test_datasets_equal(Outer_histograms_batching, Outer_histograms_test)
# Unit test for Surface_histograms_batching dataframe
test_datasets_equal(Surface_histograms_batching, Surface_histograms_test)
# Unit test for Gradients_batching dataframe
test_datasets_equal(Gradients_batching, Gradient_test)
# Unit test for Smoothed_histogram_batching dataframe
test_datasets_equal(Smoothed_histogram_batching, Smoothed_histograms_test)


"""
mspacman.core.calculate_properties(labelled_img, ct_img, props,  voxel_size, step_size)
Calculates particle properties from a labeled 3D image and CT scan.

Parameters
----------
labelled_image : ndarray
    3D labeled image where each particle has a unique label (integer > 0).

ct_image : ndarray
    3D greyscale CT image used to extract intensity-based properties.

Properties : dict
    Dictionary where computed properties will be saved.

voxel_size : float
    Real-world size of each voxel (e.g., in micrometres or millimetres).
    Used to convert volume and surface area from voxel units to real units.

step_size : float
    Step size used in the marching cubes algorithm to generate a surface mesh.
    Smaller step sizes yield more accurate surface areas but require more computation.

"""

Properties = mspacman.core.calculate_properties(Labels, Non_binary, Properties, 5,1)



# Unit test for Properties dataframe
test_datasets_equal(Properties, Properties_test)

# Extract histogram of the whole particles
# usage: mspacman.core.Bulk_particle_histograms(labelled_image, CT_image)
Bulk_histograms = mspacman.core.Bulk_particle_histograms(Labels, Non_binary)

# Unit test for Bulk_histograms dataframe
test_datasets_equal(Bulk_histograms, Bulk_histograms_test)

""" Extracts gradient of mean grey values from surface to 6th voxel layer toward the center of each particle.
    Useful for analyzing how the Partial Volume Effect (PVE) behaves near particle boundaries.

    Usage:
         mspacman.core.pve_gradient(labelled_image, CT_image, Background_peak)
    
     Parameters:
     - labelled_image : 3D labeled array where each particle has a unique integer label.
     - CT_image : Corresponding 3D greyscale CT scan used for intensity measurements.
     - Background_peak : Integer value representing the grey level peak of the background material.
    
     Description:
     This function calculates the average grey value at each inward voxel layer (up to 6 layers) 
     starting from the particle surface. This helps in understanding how strongly the 
     surrounding background influences the outer voxels of particles due to the partial volume effect.
    
     Returns:
     A DataFrame or dictionary containing average greyscale values for layers 1 through 6 per particle."""

Gradients = mspacman.core.pve_gradient(Labels, Non_binary, 1000)
# Unit test for Gradients dataframe
test_datasets_equal(Gradients, Gradient_test)

"""Extracts histograms of the surface of the particle
    Usage:
        mspacman.core.surface_particle_histograms(labelled_image, CT_image)"""

Surface_histograms = mspacman.core.surface_particle_histograms(Labels, Non_binary)
# Unit test for Surface_histograms dataframe
test_datasets_equal(Surface_histograms, Surface_histograms_test)

"""Extracts histograms of the Inner volume of the particle
    Usage:
        mspacman.core.surface_particle_histograms(labelled_image, CT_image, Gradients_dataframe)"""
Inner_histograms =  mspacman.core.inner_particle_histograms(Labels, Non_binary, Gradients)

# Unit test for Inner_histograms dataframe
test_datasets_equal(Inner_histograms, Inner_histograms_test)

# Outer volume can simply be obtained by subtracting the Inner volume histograms from the Bulk histograms
Outer_histograms = Bulk_histograms - Inner_histograms

# Unit test for Outer_histograms dataframe
test_datasets_equal(Outer_histograms, Outer_histograms_test)

# Deleting the labels that are smaller than the size threshold from Labels_cleaned
Labels_cleaned = Labels_cleaned*(np.where(Labels >0,1,0))
# Extracting histograms ofa  smoothed image
Smoothed_histogram = mspacman.core.Bulk_particle_histograms(Labels_cleaned, kuwahara_denoised)
# Unit test for Smoothed_histogram dataframe
test_datasets_equal(Smoothed_histogram, Smoothed_histograms_test)


# -------------------------------------------------------------------------
# Test: convert_and_save_as_h5ad round-trip
#
# Purpose:
#   Validate that MSPaCMAn’s .h5ad save and load functions are lossless.
#   We:
#     1. Save histogram DataFrames to .h5ad using convert_and_save_as_h5ad
#     2. Save Gradients and Properties to CSV
#     3. Reload the saved files
#     4. Compare the reloaded data to the reference datasets
#
# Why:
#   Ensures that saving and reloading data does not change its content,
#   which is critical for reproducibility and downstream analysis.
# -------------------------------------------------------------------------


"""
Convert a histogram DataFrame to `.h5ad` format and save it.

This function is used to export particle-wise histogram data to the
`.h5ad` format (AnnData), which is compatible with single-cell data tools 
like Scanpy and useful for advanced multivariate analysis and dimensionality reduction.

Usage:
    convert_and_save_as_h5ad(histograms_df, Path_to_save_histograms)

Parameters:
-----------
histograms_df 

Path_to_save_histograms : str or Path
    Full path where the `.h5ad` file should be saved, including the filename.
    Example: "output_folder/bulk_histograms.h5ad"

"""

Smoothed_histogram = mspacman.core.convert_and_save_as_h5ad(Smoothed_histogram, r"C:\Users\gupta46\Downloads\Test\Smoothed_histogram.h5ad")
Bulk_histograms = mspacman.core.convert_and_save_as_h5ad(Bulk_histograms, r"C:\Users\gupta46\Downloads\Test\Bulk_histograms.h5ad")
Inner_histograms = mspacman.core.convert_and_save_as_h5ad(Inner_histograms, r"C:\Users\gupta46\Downloads\Test\Inner_histograms.h5ad")
Outer_histograms = mspacman.core.convert_and_save_as_h5ad(Outer_histograms, r"C:\Users\gupta46\Downloads\Test\Outer_histograms.h5ad")
Surface_histograms = mspacman.core.convert_and_save_as_h5ad(Surface_histograms, r"C:\Users\gupta46\Downloads\Test\Surface_histograms.h5ad")
Gradients.to_csv(r"C:\Users\gupta46\Downloads\Test\Gradients.csv")
Properties.to_csv(r"C:\Users\gupta46\Downloads\Test\Properties.csv")


# Upload saved data again
Smoothed_histogram = mspacman.core.upload_histograms_h5ad(r"C:\Users\gupta46\Downloads\Test\Smoothed_histogram.h5ad")
Bulk_histograms = mspacman.core.upload_histograms_h5ad(r"C:\Users\gupta46\Downloads\Test\Bulk_histograms.h5ad")
Inner_histograms = mspacman.core.upload_histograms_h5ad(r"C:\Users\gupta46\Downloads\Test\Inner_histograms.h5ad")
Outer_histograms = mspacman.core.upload_histograms_h5ad(r"C:\Users\gupta46\Downloads\Test\Outer_histograms.h5ad")
Surface_histograms = mspacman.core.upload_histograms_h5ad(r"C:\Users\gupta46\Downloads\Test\Surface_histograms.h5ad")
Gradients = pd.read_csv(r"C:\Users\gupta46\Downloads\Test\Gradients.csv",index_col=0)
Properties = pd.read_csv(r"C:\Users\gupta46\Downloads\Test\Properties.csv",index_col=0)


# Unit test for the datasets
test_datasets_equal(Smoothed_histogram, Smoothed_histograms_test)
test_datasets_equal(Bulk_histograms, Bulk_histograms_test)
test_datasets_equal(Inner_histograms, Inner_histograms_test)
test_datasets_equal(Outer_histograms, Outer_histograms_test)
test_datasets_equal(Surface_histograms, Surface_histograms_test)
test_datasets_equal(Gradients, Gradient_test)
test_datasets_equal(Properties, Properties_test)

"""
Visualize a single particle across multiple images with optional padding.

This function allows you to visually inspect a labeled particle in context,
along with related data such as cleaned labels or CT greyscale images.

Usage:
    mspacman.core.view_particle(labelled_image, particle_id, image_stack,
                                 image_names=None, pad=0)

Parameters:
-----------
labelled_image : ndarray
    3D labeled image where each particle has a unique ID (e.g., Labels).
    This defines which particle to extract and visualize.

particle_id : int
    ID of the particle to visualize (must exist in labelled_image).
    Example: 1769

image_stack : list of ndarrays
    List of additional 3D images to visualize alongside the particle.
    These can be cleaned labels, CT images, phase masks, etc.

image_names : list of str, optional
    Names corresponding to each image in image_stack. These will be used as subplot titles.
    Must be the same length as image_stack.
    Example: ["Labels_cleaned", "Non_binary"]

pad : int, optional (default=0)
    Padding (in voxels) around the particle's bounding box for visualization.
    Example: pad=10 will show 10 voxels of context in all directions.

Returns:
--------
Displays 2D slice plots of the selected particle across all input images for visual comparison in napari.

"""
mspacman.core.view_particle(Labels,18, [Labels_cleaned, kuwahara_denoised], image_names=["Labels_cleaned", "Non_binary"], pad=10)

"""
Bin the smoothed greyscale histograms into a fixed number of bins.

This function reduces the number of intensity bins in the histogram from the original 
(high resolution, e.g., 65536 grey levels) to a lower number (e.g., 200 bins), 
making it easier to visualize, compare, or use in downstream analysis.

Usage:
    binned = mspacman.core.bin_histograms(histogram_df, number_of_bins)
"""

binned = mspacman.core.bin_histograms(Smoothed_histogram, 200)
test_datasets_equal(binned, Smoothed_binned_test)

"""
Smooth the binned histograms using a Savitzky-Golay filter.

This step helps reduce noise and sharp fluctuations in the histogram curves,
especially after binning. The Savitzky-Golay filter preserves peak shapes better
than basic averaging, making it ideal for downstream tasks like peak detection
or dimensionality reduction.

Usage:
    smoothed = mspacman.core.smooth_histograms(binned, window, polyorder)

Parameters:
-----------
binned : pd.DataFrame
    Binned histogram data (e.g., from bin_histograms), where each row represents a particle
    and each column represents a grey value bin.

window : int
    Size of the moving window used for smoothing (must be odd).
    A larger window increases smoothing but may flatten fine details.

polyorder : int
    Degree of the polynomial used for fitting within each window.
    Must be less than window size. Higher values preserve curvature better.
"""
smoothed = mspacman.core.smooth_histograms(binned, window=5, polyorder=3)
# unit test
test_datasets_equal(smoothed, Smoothed_binned_svgol_test)

#Plot histograms of all particle. Better to use binned histograms
mspacman.core.plot_all_rows(smoothed)

"""
Detects intensity peaks in particle histograms and assigns the strongest peaks to specific phases
based on predefined greyscale intensity thresholds. Used for phase classification in 3D CT data.

Parameters:
file : pd.DataFrame
    Input histogram DataFrame (each row is a particle, each column a greyscale bin).
distance : int
    Minimum horizontal distance (in bins) between peaks.
height_ratio : float
    Ratio to determine peak height threshold; used to ignore low peaks. Height ratio 10 means in a particular 
    histogram, peaks up to 10th of height of the peak having maximum height will be detected. Note that it keeps
    on changing with the histogram based on the peak having maximum height in the particle.
Binning : int
    The binning factor used during histogram preparation (to scale positions).
Voxel_size : float
    Voxel size in µm (used for scaling positions; not directly used here but retained for completeness).
Background_peak : float
    Threshold below which any detected peak is considered background.
Height_threshold : float
    Minimum height a peak must have to be retained.
*thresholds : float
    Threshold values separating different phases (at least one required for Phase 1).
"""

Peaks = mspacman.core.find_peaks_and_arrange(smoothed,  20, 100, 200, 5, 1000,0,10000, 17000, 30000)
# unit test
test_datasets_equal(Peaks, Peaks_test)

"""
Often we calculate peaks on processed binned histograms for better accuracy but in
processed histograms peaks shifts a bit. So we have to map the peaks back on bulk_histograms to 
adjust the shifts.
    
Computes peaks per particle using intensity thresholds.
Then, for each particle, if original Peak_position_n == Background_peak,
set position to background and height to 0 in the result.

Automatically detects which columns in histograms are intensity bins.

Usage:
    mspacman.core.map_peaks_on_bulk_histograms(histograms, Peaks)

Args:
    histograms (pd.DataFrame): Bulk_histogram
    Peaks (pd.DataFrame): rows = particles, with original peak metadata

Returns:
    pd.DataFrame: Computed peak positions and heights for each particle
"""

Peaks = mspacman.core.map_peaks_on_bulk_histograms(Bulk_histograms, Peaks)
# unit test
test_datasets_equal(Peaks, Peaks_mapped_test)
"""
Apply bias correction for detecting dense minerals in Peak data by replacing 
missed peaks with maximum intensity values from region properties. 
    
Motivation:
- Some peaks (especially from small inclusions) may be missed by automatic peak detection.
- Since high-density (high-attenuation) minerals like gold or REEs are more critical,
  this function biases the correction towards denser phases.
- For example, missing a small quartz grain is less important than missing a small gold inclusion.

Usage:
  Peaks = mspacman.core.bias_dense_minerals(Peaks, Properties, phases_to_correct = [Phase list]) 
    
Parameters:
  Peak_data (pd.DataFrame): Peak positions and phase thresholds per label (index = 'Label').
  Properties_Bulk (pd.DataFrame): Region properties including max intensity per label.
  phases_to_correct (list of int, optional): Phase numbers to apply correction to (e.g., [2, 4]).
    
Returns:
  pd.DataFrame: Corrected Peak_data with adjusted peak positions for selected phases.
"""

Peaks = mspacman.core.bias_dense_minerals(Peaks, Properties, phases_to_correct = [3])
# unit test
test_datasets_equal(Peaks, Peaks_dense_bias_test)

"""
Plot greyscale histograms of selected particles with overlaid detected peaks.

This function visualizes the histogram curves for selected particles and overlays
the detected peak positions as red dots. It's useful for visually validating
phase classification, peak detection accuracy, and smoothing effects.

Usage:
    mspacman.core.plot_histograms_with_peaks(
        histogram_df,
        Peaks,
        selected_particles,
        dpi=300,
        font_size=12,
        Grid=False
    )

Parameters:
-----------
histogram_df : pd.DataFrame
    Smoothed or raw histogram DataFrame where rows represent particles
    and columns represent grey value bins.

Peaks : pd.DataFrame
    DataFrame containing detected peak positions and heights
    (e.g., from `find_peaks_and_arrange` or `bias_dense_minerals`).

selected_particles : list of int
    List of particle indices (row IDs) to plot.
    These must exist in both `histogram_df` and `Peaks`.

dpi : int, optional (default=300)
    Resolution of the output plot in dots per inch.

font_size : int, optional (default=12)
    Font size used in labels and legends.

Grid : bool, optional (default=False)
    Whether to display gridlines on the plot.

Returns:
--------
A matplotlib figure with one histogram per particle and red points for peak locations.

"""

selected_particles = [18]
mspacman.core.plot_histograms_with_peaks(smoothed, Peaks, selected_particles, dpi=500, font_size=14, Grid = True)

"""
Liberated_Particles:
Quantifies particles containing a single dominant phase Peaks.

Binary_Particles:
Quantifies particles containing two phases.

Ternary_Particles:
Estimates three-phase contributions within particles.

Quaternary_Particles:
Performs four-phase decomposition of complex particles.

Quinary_Particles:
Quantifies up to five coexisting phases in highly mixed particles.

Note* MSPaCMan doesnt consider particles having more than 5 phases as we believe 
quntification unceratinity is vey high in such particles.
"""


Liberated_quantification = mspacman.core.Liberated_Particles(Peaks, Bulk_histograms, Surface_histograms)

Binary_quantification = mspacman.core.Binary_Particles(Peaks, Bulk_histograms, Inner_histograms, Outer_histograms, Surface_histograms, Gradients, 0.7)

Ternary_quantification = mspacman.core.Ternary_Particles(Peaks, Bulk_histograms, Inner_histograms, Outer_histograms, Surface_histograms, Gradients, 0.7)

Quaternary_quantification = mspacman.core.Quaternary_Particles(Peaks, Bulk_histograms, Inner_histograms, Outer_histograms, Surface_histograms, Gradients, 0.7)

Quinary_quantification = mspacman.core.Quinary_Particles(Peaks, Bulk_histograms, Inner_histograms, Outer_histograms, Surface_histograms, Gradients, 0.7)

# Concate nates all in single dataframe
Quantification = mspacman.core.Concatenate(Liberated_quantification, Binary_quantification,Ternary_quantification,
                             Quaternary_quantification,Quinary_quantification)

"""
This function performs bootstrapping to estimate uncertainty in phase quantification.
    
It resamples the input dataset 'fractions' times with replacement, splits it into
'fractions' number of sub-datasets, and calculates the mineral composition for each.
    
The function then computes the 2.5th and 97.5th percentiles to estimate the 95% confidence interval
for each phase, along with the actual percentage from the original dataset.

Usage:
     Bootstrapping_error_bulk = mspacman.core.calculate_bootstrapping_error_bulk(Quantification,fractions)
Parameters:
   dataset (pd.DataFrame): DataFrame containing columns 'Phase_1_quantification' to 'Phase_9_quantification'.
   fractions (int): Number of resamples/subsets to create.
    
Returns:
   pd.DataFrame: A DataFrame containing the min, max, and actual percentages for each phase.
"""

Bootstrapping_error_bulk = mspacman.core.calculate_bootstrapping_error_bulk(Quantification,1000)
print (Bootstrapping_error_bulk)

# Calculates bootstrapping errors for surface mineral compositions
Bootstrapping_error_surface = mspacman.core.calculate_bootstrapping_error_surface(Quantification,1000)
print (Bootstrapping_error_surface)

# Calculates the percentage contribution of each mineral phase within individual particles, 
#treating each particle as 100%. Percentages are computed separately for bulk, outer, and surface quantifications.

Quantification = mspacman.core.compute_particle_quantification_percentages(Quantification)
Quantification = mspacman.core.compute_particle_surface_percentages(Quantification)
Quantification = mspacman.core.compute_particle_outer_percentages(Quantification)
Quantification.to_csv(r"C:\Users\gupta46\Downloads\ShuvamGupta mspacman main Kemi_+850_unit_test_and_example_usage\Quantification.csv")
# unit test
test_datasets_equal(Quantification, Quantification_test)
