import numpy as np
from scipy import ndimage 
import pandas as pd
import gc
import glob
import tifffile
from joblib import Parallel, delayed
import anndata
from tqdm import tqdm
from skimage import (io, measure, morphology)
from skimage.measure import marching_cubes, mesh_surface_area
import os
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import sobel
import re
from pykuwahara import kuwahara
from scipy.ndimage import gaussian_filter




def upload_images(image_path):


    """
    Load 3D image data from a directory containing either:
    - A stack of 2D .tif/.tiff files
    - A single 3D .tif/.tiff file

    Parameters:
    image_path : str
        Path to the folder containing TIFF image(s).

    Returns:
    images : np.ndarray
        3D NumPy array representing the stacked image volume.
    """
    
    # Normalize and finalize path with separator
    
    image_path = os.path.normpath(image_path) + os.sep

    # Natural sort helper
    def natural_sort_key(s):
        """
        Helper for human-friendly sorting of numbered filenames.
        E.g., ['img1', 'img2', ..., 'img10'] instead of ['img1', 'img10', 'img2']
        """
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]

    # Collect and sort all TIFF files
    tiff_files = sorted(glob.glob(image_path + "*.tif") + glob.glob(image_path + "*.tiff"),
                        key=natural_sort_key)

    # Case 1: multiple 2D TIFF images
    if len(tiff_files) > 1:
        def process_tiff(img_path):
            return io.imread(img_path, as_gray=True)

        cv_img = Parallel(n_jobs=-1)(
            delayed(process_tiff)(img) for img in tqdm(tiff_files, desc="Uploading TIFF images")
        )
        images = np.dstack(cv_img)
        images = np.rollaxis(images, -1)
        del cv_img

    # Case 2: a single 3D TIFF file
    elif len(tiff_files) == 1:
        tiff_file = tiff_files[0]
        with tifffile.TiffFile(tiff_file) as tif:
            num_slices = len(tif.pages)

        def process_slice(slice_idx):
            with tifffile.TiffFile(tiff_file) as tif:
                return tif.pages[slice_idx].asarray()

        slice_indices = range(num_slices)
        images = Parallel(n_jobs=-1)(
            delayed(process_slice)(i) for i in tqdm(slice_indices, desc="Uploading slices from 3D TIFF")
        )
        images = np.array(images)

    else:
        raise ValueError("No compatible image files found in the specified directory.")



    # Standardize image dtype
    max_grey_value = np.max(images)
    if max_grey_value <= 255:
        images = images.astype(np.uint8)
    elif max_grey_value <= 65535:
        images = images.astype(np.uint16)
    else:
        images = images.astype(np.uint32)

    print("Image dtype:", images.dtype)
    gc.collect()

    return images



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





def Bulk_particle_histograms(labelled_image, image):
    """
   Compute intensity histograms for individual labeled particles in a 3D image.

   Parameters:
   labelled_image : np.ndarray
       3D labeled image where each particle has a unique integer label.
   
   image : np.ndarray
       3D grayscale image corresponding to the labeled image.

   Returns:
   histograms_df : pd.DataFrame
       DataFrame where each row is the histogram of a particle, indexed by particle label.
   """
    def get_histogram_for_label(label):
        """Extract the 3D cropped region for the label and calculate its histogram."""
        # Create a mask for the current label
        mask = labelled_image == label
        if not np.any(mask):  # Skip if label does not exist
            return None
        
        # Find the bounding box for the current label
        coords = np.argwhere(mask)
        del mask
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0) + 1  # +1 because slices are exclusive on the upper end

        # Extract cropped regions from both label and grayscale images
        cropped_labels = labelled_image[min_coords[0]:max_coords[0],
                                min_coords[1]:max_coords[1],
                                min_coords[2]:max_coords[2]]
        cropped_image = image[min_coords[0]:max_coords[0],
                                       min_coords[1]:max_coords[1],
                                       min_coords[2]:max_coords[2]]
        
        del min_coords, max_coords

        # Apply label mask inside the bounding box to extract the particle voxels
        label_mask = cropped_labels == label
        filtered_region = cropped_image[label_mask]
        del cropped_labels, cropped_image, label_mask

        # Compute histogram for the filtered region
        hist, bins = np.histogram(filtered_region, bins=range(65537))
        del filtered_region
        return hist, label

    # Get unique labels, excluding the background (label 0)
    unique_labels = np.unique(labelled_image)
    unique_labels = unique_labels[unique_labels != 0]

    # Parallelize the histogram computation
    results = Parallel(n_jobs=-1)(
        delayed(get_histogram_for_label)(label) for label in tqdm(unique_labels)
    )

    # Filter out None results
    results = [result for result in results if result is not None]

    # Construct DataFrame
    histograms = [result[0] for result in results]
    index = [result[1] for result in results]
    
    histograms_df = pd.DataFrame(histograms, index=index)
    del histograms, results, index, unique_labels

    return histograms_df


def calculate_properties(labelled_image, ct_image, Properties, angle_spacing, voxel_size, step_size):
    
    """
    Calculate 3D morphological and surface features for labeled particles in a CT image.
    
    Parameters:
    labelled_image : np.ndarray
        3D array with integer labels for each segmented particle.
    
    ct_image : np.ndarray
        3D grayscale CT image.
    
    Properties : list
        List of region properties to calculate using skimage.
    
    angle_spacing : int
        Angular step (in degrees) for Feret diameter calculation.
    
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
        def rotate_coords(coords, a1, a2, a3):
            
            """Apply 3D rotation to coordinates using Euler angles."""
            
            Rz = np.array([[np.cos(np.radians(a1)), -np.sin(np.radians(a1)), 0],
                           [np.sin(np.radians(a1)),  np.cos(np.radians(a1)), 0],
                           [0, 0, 1]])
            Ry = np.array([[np.cos(np.radians(a2)), 0, -np.sin(np.radians(a2))],
                           [0, 1, 0],
                           [np.sin(np.radians(a2)), 0, np.cos(np.radians(a2))]])
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(np.radians(a3)), -np.sin(np.radians(a3))],
                           [0, np.sin(np.radians(a3)),  np.cos(np.radians(a3))]])
            return coords @ Rz @ Ry @ Rx

        def compute_feret(label, coords):
            
            """Calculate min and max Feret diameters by rotating the particle."""
            
            min_feret = np.inf
            max_feret = 0
            for a1 in range(0, 180, angle_spacing):
                for a2 in range(0, 180, angle_spacing):
                    for a3 in range(0, 180, angle_spacing):
                        rotated = rotate_coords(coords, a1, a2, a3)
                        max_dim = rotated.ptp(axis=0)
                        max_feret = max(max_feret, max_dim[0])
                        min_feret = min(min_feret, max_dim[1], max_dim[2])
            return label, max_feret, min_feret
        
        # Extract coordinates for Feret diameter calculation

        regions = measure.regionprops(labelled_image, intensity_image=ct_image)
        coords_list = [(r.label, r.coords) for r in regions if r.label in unique_labels]

        feret_results = Parallel(n_jobs=-1)(
            delayed(compute_feret)(label, coords)
            for label, coords in tqdm(coords_list, desc="Feret Diameter")
        )
        
        # Assemble Feret diameter results

        feret_df = pd.DataFrame(feret_results, columns=['label', 'Max_Feret_Diameter', 'Min_Feret_Diameter']).set_index('label')
        merged = merged.merge(feret_df, how='left', on='label')
        
        # Convert Feret diameters to physical units
        
        merged['Max_Feret_Diameter'] *= voxel_size
        merged['Min_Feret_Diameter'] *= voxel_size
        merged['Feret_ratio'] = merged['Min_Feret_Diameter'] / merged['Max_Feret_Diameter']

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



def standardize_index(df):
    import inspect
    """
    Ensures that the DataFrame has an index named 'Label'.
    If not, it looks for a suitable column to use as the index and renames it.

    This is useful for standardizing particle-based dataframes where the index or label
    column may be inconsistently named across different sources.

    Parameters:
    df : pd.DataFrame
        The input DataFrame that should be indexed by 'Label'.

    Returns:
    df : pd.DataFrame
        The same DataFrame but with the index set and named as 'Label'.

    Raises:
    ValueError
        If no suitable label column or index is found.
    """
    # Try to get the variable name from the calling frame
    df_name = "DataFrame"
    try:
        frame = inspect.currentframe().f_back
        df_name = [name for name, val in frame.f_locals.items() if val is df][0]
    except Exception:
        pass  # Fallback to generic name if inspection fails

    candidates = [
        'label', 'labels',
        'particle label', 'particle labels',
        'particle_label', 'particle_labels'
    ]
    candidates_lower = [c.lower() for c in candidates]

    if df.index.name and df.index.name.lower() in candidates_lower:
        df.index.name = 'Label'
    else:
        matched = None
        for col in df.columns:
            if col.lower() in candidates_lower:
                matched = col
                break
        if matched:
            df = df.rename(columns={matched: 'Label'})
            df.set_index('Label', inplace=True)
        else:
            raise ValueError(f"No Labels found in {df_name}")

    return df



def inner_particle_histograms(labelled_image, image, pve_gradient):
    
    pve_gradient = standardize_index(pve_gradient)

    # Step 1: Count erosions
    def count_erosions(row):
        gradient_cols = row.filter(like='Gradient')
        return sum(1 for value in gradient_cols if value < 0.9 or value > 1)

    pve_gradient = pve_gradient.copy()
    pve_gradient['no_of_erosions'] = pve_gradient.apply(count_erosions, axis=1)

    # Step 2: Erode per particle group
    eroded_images = []
    unique_erosions = pve_gradient['no_of_erosions'].unique()

    for erosion_count in unique_erosions:
        labels_with_count = pve_gradient.index[
            pve_gradient['no_of_erosions'] == erosion_count
        ].values

        group_mask = np.isin(labelled_image, labels_with_count)
        eroded_mask = ndimage.binary_erosion(group_mask, iterations=erosion_count)

        eroded_images.append(eroded_mask.astype(np.uint8))

        # Clean up
        del labels_with_count, group_mask, eroded_mask

        print(f"Erosion level {erosion_count} processed.")

    final_eroded_mask = np.sum(eroded_images, axis=0) > 0
    del eroded_images

    Inner_volume_labels = final_eroded_mask.astype(np.int8) * labelled_image
    del final_eroded_mask

    # Step 3: Compute histograms
    histograms_df = Bulk_particle_histograms(Inner_volume_labels, image)
    del Inner_volume_labels

    # Step 4: Fill missing labels with zeros
    all_labels = np.unique(labelled_image)
    all_labels = all_labels[all_labels != 0]
    all_labels_index = pd.Index(all_labels)

    missing_indices = all_labels_index.difference(histograms_df.index)
    if len(missing_indices) > 0:
        missing_rows = pd.DataFrame(0, index=missing_indices, columns=histograms_df.columns)
        histograms_df = pd.concat([histograms_df, missing_rows])

    histograms_df = histograms_df.sort_index()
    del all_labels, missing_indices

    return histograms_df


def surface_particle_histograms(labelled_image, image):
    
    """
    Compute inner histograms of particles based on erosion levels determined from PVE gradient.
    
    Parameters:
    labelled_image : 3D numpy array
        Labelled segmentation image where each particle has a unique label.
    
    image : 3D numpy array
        Original grayscale CT image.
    
    pve_gradient : pd.DataFrame
        DataFrame with gradient information for each label. Must include 'Gradient_1' to 'Gradient_n'.
    
    Returns:
    histograms_df : pd.DataFrame
        Histogram of greyscale values inside each eroded particle.
    """

    # Step 1: Binary mask
    binary = (labelled_image > 0).astype(np.uint8)

    # Step 2: Erosion 1
    eroded_1 = ndimage.binary_erosion(binary, iterations=1)

    # Step 3: Erosion 2
    eroded_2 = ndimage.binary_erosion(eroded_1, iterations=1)

    # Step 4: Subtract erosion 2 from erosion 1 to get the surface shell
    surface_mask = eroded_1.astype(np.uint8) - eroded_2.astype(np.uint8)
    del binary, eroded_1, eroded_2

    # Step 5: Multiply surface mask with original labels
    surface_labels = surface_mask * labelled_image
    del surface_mask

    # Step 6: Extract histograms
    histograms = Bulk_particle_histograms(surface_labels, image)

    # Step 7: Fill missing labels with zero histograms
    all_labels = np.unique(labelled_image)
    all_labels = all_labels[all_labels != 0]
    missing_indices = pd.Index(all_labels).difference(histograms.index)

    if len(missing_indices) > 0:
        missing_rows = pd.DataFrame(0, index=missing_indices, columns=histograms.columns)
        histograms = pd.concat([histograms, missing_rows])

    histograms = histograms.sort_index()
    del all_labels, missing_indices

    return histograms


def convert_and_save_as_h5ad(histograms_df,Path_to_save_histograms):
    
    """
    Converts a histogram DataFrame into an AnnData object and saves it as an .h5ad file.

    Parameters:
    - histograms_df (pd.DataFrame): DataFrame with particles as rows and intensity bins as columns.
    - Path_to_save_histograms (str): Full path where the .h5ad file will be saved.

    Returns:
    - histograms_df (pd.DataFrame): The modified histogram DataFrame with string index/columns.
    """
    # Ensure the index (label IDs) is of string type (required by AnnData)
    histograms_df.index = histograms_df.index.astype(str)
    
    # Ensure the column names (intensity bins) are also strings
    histograms_df.columns = histograms_df.columns.astype(str)
    
    # Prefix all column names with "bin_" to clearly indicate histogram bins
    histograms_df.columns = ['bin_' + str(col) for col in histograms_df.columns]
    
    # Create empty `obs` DataFrame (per-particle metadata) using the index of histograms
    obs = pd.DataFrame(index=histograms_df.index)
    
    # Create empty `var` DataFrame (per-feature metadata) using the columns (bins)
    var = pd.DataFrame(index=histograms_df.columns)

    print("obs DataFrame index dtype:", obs.index.dtype)
    print("obs DataFrame index values:", obs.index.values)
    
    # Create AnnData object using histogram matrix as `X`, and empty metadata frames
    adata = anndata.AnnData(X=histograms_df, obs=obs, var=var)
    
    # Save the AnnData object to the specified path as an .h5ad file
    adata.write(Path_to_save_histograms)

    return histograms_df



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



def run_batch_processing(
    labelled_image,
    CT_image,
    Properties,
    Background_mean,
    labels_per_chunk,
    Size_threshold,
    voxel_size,
    angle_spacing,
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
        angle_spacing : int
            Step in degrees for Feret diameter rotation.
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
                chunk_labels, chunk_nonbinary, Properties,
                angle_spacing, voxel_size, step_size
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



from skimage.measure import regionprops
import napari


def view_particle(label_image, label_id, image_list=None, image_names=None, pad=2):
    """
    View bounding box of a single label in Napari.

    Parameters:
    - label_image: np.ndarray, labeled 3D image
    - label_id: int, label to visualize
    - image_list: list of np.ndarrays, associated images (same shape as label_image)
    - image_names: list of str, names for associated images
    - pad: int, padding around bounding box
    """
    if image_list is None:
        image_list = []
    if image_names is None:
        image_names = [f"Image_{i+1}" for i in range(len(image_list))]

    assert all(img.shape == label_image.shape for img in image_list), "All images must match label image shape."

    # Create binary mask for the label
    binary_mask = (label_image == label_id).astype(np.uint8)

    props = regionprops(binary_mask)
    if not props:
        print(f"Label {label_id} not found.")
        return

    # Get bounding box: (min_z, min_y, min_x, max_z, max_y, max_x)
    bbox = props[0].bbox
    min_z, min_y, min_x, max_z, max_y, max_x = bbox

    # Apply padding
    min_z = max(0, min_z - pad)
    min_y = max(0, min_y - pad)
    min_x = max(0, min_x - pad)
    max_z = min(label_image.shape[0], max_z + pad)
    max_y = min(label_image.shape[1], max_y + pad)
    max_x = min(label_image.shape[2], max_x + pad)

    # Crop label and associated images
    cropped_label = label_image[min_z:max_z, min_y:max_y, min_x:max_x]

    viewer = napari.Viewer()
    viewer.add_labels(cropped_label, name=f'Label_{label_id}')
    
    # Add any additional cropped images
    for img, name in zip(image_list, image_names):
        cropped_img = img[min_z:max_z, min_y:max_y, min_x:max_x]
        viewer.add_image(cropped_img, name=name)

    napari.run()
    
    
def upload_histograms_h5ad(file_path):
    
    """
    Load a .h5ad file and convert it into a clean DataFrame with label-based indexing.
    
    Parameters:
        file_path (str): Path to the .h5ad file.
    
    Returns:
        df (DataFrame): A cleaned pandas DataFrame with histogram data and proper label index.
    """

    try:
        # Load the .h5ad file using anndata with read-only memory mode
        adata = anndata.read_h5ad(file_path, backed='r')
        
        # Show progress while converting the anndata to a DataFrame
        with tqdm(total=100, desc="Converting to DataFrame") as pbar_convert:
            df = adata.to_df()
            pbar_convert.update(100)

        # Convert numeric-like column names to integers
        df.columns = [
            int(''.join(filter(str.isdigit, col))) if isinstance(col, str) and any(char.isdigit() for char in col)
            else col for col in df.columns
        ]

        label_keys = {'labels', 'label', 'particle labels', 'particle label'}

        # Step 1: Check for label column (case-insensitive)
        label_col = None
        for col in df.columns:
            if isinstance(col, str) and col.strip().lower() in label_keys:
                label_col = col
                break

        if label_col:
            df.set_index(label_col, inplace=True)
            df.index.name = 'Label'

        # Step 2: Index name matches known label keys
        elif isinstance(df.index.name, str) and df.index.name.strip().lower() in label_keys:
            df.index.name = 'Label'

        # Step 3: Index has a non-label name → rename it
        elif df.index.name:
            df.index.name = 'Label'

        # Step 4: Unnamed column exists → use it as index
        elif None in df.columns:
            df.rename(columns={None: 'Label'}, inplace=True)
            df.set_index('Label', inplace=True)
            df.index.name = 'Label'

        # Step 5: No label column or index name at all → assign generic name 'Labels'
        else:
            df.index.name = 'Label'

        df.index = df.index.astype(int)

        print("h5ad file uploaded and converted to DataFrame successfully.")
        return df

    except Exception as e:
        print(f"An error occurred while uploading and converting the h5ad file: {e}")
        

def bin_histograms(histograms, binning=1):
    """
    Bin histogram data row-wise using the specified binning factor.

    Parameters:
        Bulk_histograms (pd.DataFrame): Input histogram DataFrame (rows = samples, cols = bins).
        binning (int): Number of original bins to combine into one.
        n_jobs (int): Number of parallel jobs to use.

    Returns:
        pd.DataFrame: Binned histogram DataFrame.
    """
    array = np.array(histograms.columns).astype(int)

    def process_row(row):
        num = row.to_numpy()
        num = np.pad(num, (0, 1), 'constant')
        num = num.ravel()
        
        # Pad with one zero to ensure coverage of edge bins
        rang = int(round(len(num) / binning))
        bins = np.linspace(0, max(array) + 1, rang)
        
        # Create digitized mapping from original bin edges to new bins
        full_range = np.linspace(0, max(array), len(array) + 1)
        digitized = np.digitize(full_range, bins)
        
        # Sum values for each bin group
        bin_sum = [num[digitized == i].sum() for i in range(1, len(bins))]
        return bin_sum
    
    # Process all rows in parallel using joblib
    results = Parallel(n_jobs=-1)(
        delayed(process_row)(row)
        for _, row in tqdm(histograms.iterrows(), total=histograms.shape[0], desc="Binning Histograms")
    )

    # Create new bin column labels
    rang = int(round(len(histograms.columns) / binning))
    x = np.linspace(0, len(histograms.columns) - 1, rang - 1).astype(int)
    
    # Construct final binned DataFrame
    binned_df = np.array(results).reshape(len(results), -1)
    binned_df = pd.DataFrame(binned_df, columns=x)
    
    # Keep original row indexing
    binned_df.index = histograms.index
    
    # Remove any negative values (if caused by numerical errors)
    binned_df[binned_df < 0] = 0
    binned_df.index = binned_df.index.astype(int)

    return binned_df

from scipy.signal import savgol_filter

def smooth_histograms(binned_histograms, window=11, polyorder=3):
    
    """
    Apply Savitzky-Golay smoothing to each histogram row in a DataFrame.

    Parameters:
        binned_histograms (pd.DataFrame): Histogram data (rows = particles, columns = binned intensities).
        window (int): Maximum window length for smoothing filter.
        polyorder (int): Polynomial order for smoothing filter.

    Returns:
        pd.DataFrame: Smoothed histogram DataFrame.
    """
    
    def smooth_row(row):
        values = row.to_numpy()
        bin_sum = values.copy()
        row1 = bin_sum[bin_sum > 0]

        # Too few values — return as-is
        if len(row1) <= 4:
            yhat = row1
        else:
            # Adjust window size to data length
            win = len(row1)
            win = win - 1 if win < 11 and win % 2 == 0 else min(win, 11)
            yhat = savgol_filter(row1, window_length=win, polyorder=polyorder)
        
        # Replace original values with smoothed ones
        result = bin_sum.copy()
        result[bin_sum > 0] = yhat
        result[result < 0] = 0
        return result
    
    # Apply smoothing row-wise in parallel
    results = Parallel(n_jobs=-1)(
        delayed(smooth_row)(row)
        for _, row in tqdm(binned_histograms.iterrows(), total=binned_histograms.shape[0], desc="Smoothing histograms")
    )

    smoothed_df = pd.DataFrame(results, columns=binned_histograms.columns, index=binned_histograms.index)
    smoothed_df.index = smoothed_df.index.astype(int)
    return smoothed_df

import matplotlib.pyplot as plt

def plot_all_rows(df, title='Line Plot of All Particles',
                  xlabel='Grey value', ylabel='Number of voxels', figsize=(10, 6)):
    """
    Plots each row of a DataFrame as a line.

    Parameters:
        df (pd.DataFrame): DataFrame where each row represents a line to plot.
        title (str): Title of the plot.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)

    x_vals = df.columns.astype(float) if df.columns.dtype.kind in "iufc" else range(len(df.columns))

    for idx in df.index:
        plt.plot(x_vals, df.loc[idx], label=f'Row {idx}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    


def find_peaks_and_arrange(
    file,
    distance,
    height_ratio,
    Binning,
    Voxel_size,
    Background_peak,
    Height_threshold,
    *thresholds
):
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
        on changing with the histrogram based on the peak having maximum height in the particle.
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
    
    Returns:
    Peaks_phases : pd.DataFrame
        DataFrame containing the highest valid peak position and height for each phase per particle,
        along with the threshold values used for classification.
    """
    from scipy.signal import find_peaks
    # Define up to 11 phase thresholds. Fill missing ones with 65536 (max greyscale).
    phase_keys = ['Phase_1', 'Phase_2', 'Phase_3', 'Phase_4', 'Phase_5', 'Phase_6', 'Phase_7', 'Phase_8', 'Phase_9', 'Phase_10', 'Phase_11']
    if len(thresholds) < 1:
        raise ValueError("At least Phase_1 threshold must be provided.")

    thresholds_dict = {phase_keys[i]: thresholds[i] if i < len(thresholds) else 65536 for i in range(len(phase_keys))}
    Phase_1_threshold = thresholds_dict['Phase_1']
    Phase_2_threshold = thresholds_dict['Phase_2']
    Phase_3_threshold = thresholds_dict['Phase_3']
    Phase_4_threshold = thresholds_dict['Phase_4']
    Phase_5_threshold = thresholds_dict['Phase_5']
    Phase_6_threshold = thresholds_dict['Phase_6']
    Phase_7_threshold = thresholds_dict['Phase_7']
    Phase_8_threshold = thresholds_dict['Phase_8']
    Phase_9_threshold = thresholds_dict['Phase_9']
    Phase_10_threshold = thresholds_dict['Phase_10']
    Phase_11_threshold = thresholds_dict['Phase_11']

    # Process Peaks
    Peaks_Position = []
    Peaks_Height = []
    
    # Loop over all particles (rows in histogram DataFrame)
    for index, row in tqdm(file.iterrows(), total=file.shape[0], desc="Processing Peaks"):
        file_row_1 = np.array(row).ravel().astype(float)
        file_row_1 = np.pad(file_row_1, (0, 1), constant_values=0)

        Grey_scale = np.arange(0, 65536, dtype=float)
        Grey_scale = np.pad(Grey_scale, (0, 1), constant_values=0).astype(int)

        file_row_1[np.isnan(file_row_1)] = 0
        file_row_1[file_row_1 < 0] = 0
        
        # Set dynamic peak height threshold and prominence
        height_peak_finder = max(file_row_1) / height_ratio
        prominence = max(file_row_1) / 50

        peaks = find_peaks(file_row_1, height=height_peak_finder, threshold=None,
                           distance=distance, prominence=prominence)

        height = peaks[1]['peak_heights']
        peak_pos = Grey_scale[peaks[0]] * Binning

        Peaks_Position.append([peak_pos])
        Peaks_Height.append([height])
        
    # Convert peak data into DataFrames
    Peaks_Positions = pd.DataFrame(Peaks_Position)
    Peaks_Height_df = pd.DataFrame(Peaks_Height)
    
    # Expand list of peak positions/heights into separate columns
    Peaks_Positions = pd.concat([Peaks_Positions[0].str[i] for i in range(22)], axis=1)
    Peaks_Height_df = pd.concat([Peaks_Height_df[0].str[i] for i in range(22)], axis=1)

    Peaks_Positions.columns = [f'Peak_position_{i+1}' for i in range(22)]
    Peaks_Height_df.columns = [f'Peak_height_{i+1}' for i in range(22)]
    
    # Combine into a single DataFrame and clean
    Peaks = pd.concat([Peaks_Positions.fillna(0), Peaks_Height_df.fillna(0)], axis=1)
    Peaks.index = file.index
    Peaks = Peaks.astype(float)
    Peaks.replace([np.inf, -np.inf], 0, inplace=True)

    Peaks1 = Peaks.copy()
    cols = [f"Peak_position_{i}" for i in range(1, 23)]
    peaks_height_cols = [f"Peak_height_{i}" for i in range(1, 23)]

    mask = (Peaks1[cols] <= Background_peak).all(axis=1)
    Peaks1.loc[mask, cols[0]] = Background_peak + 10
    Peaks1.loc[mask, peaks_height_cols[0]] = 1
    
    # Helper function to extract the best peak within a given phase threshold range
    def process_phase(Peaks, phase_start, phase_end, phase_label):
        if Peaks.empty:
            return pd.DataFrame(0, index=Peaks.index,
                                columns=[f'Peak_position_{phase_label}', f'Peak_height_{phase_label}'])

        # Filter peaks within the range
        Peaks_filtered = Peaks.where((Peaks >= phase_start) & (Peaks < phase_end), np.nan)
        Peaks_height = Peaks1[peaks_height_cols]
        Peaks_filtered = Peaks_filtered.loc[Peaks_filtered.any(axis=1)].fillna(0)
        Peaks_filtered = Peaks_filtered.merge(Peaks_height, left_index=True, right_index=True)

        # Zero-out irrelevant peak heights and ensure background filtering
        for i in range(1, 23):
            Peaks_filtered[f"Peak_position_{i}"] = Peaks_filtered[f"Peak_position_{i}"].clip(lower=0)
            Peaks_filtered[f"Peak_height_{i}"] = Peaks_filtered[f"Peak_height_{i}"].where(
                (Peaks_filtered[f"Peak_position_{i}"] >= phase_start) &
                (Peaks_filtered[f"Peak_position_{i}"] < phase_end), 0
            ).where(Peaks_filtered[f"Peak_position_{i}"] >= Background_peak, 0)

        # Select the peak with the maximum height
        if Peaks_filtered[peaks_height_cols].notna().any().any():
            max_peak_idx = Peaks_filtered[peaks_height_cols].idxmax(axis=1)
            peaks_data = pd.DataFrame(0, index=Peaks_filtered.index,
                                      columns=[f'Peak_position_{phase_label}', f'Peak_height_{phase_label}'])
            for i, col_name in enumerate(peaks_height_cols):
                mask = max_peak_idx == col_name
                peaks_data[f'Peak_position_{phase_label}'] = np.where(mask,
                                                                       Peaks_filtered[f'Peak_position_{i+1}'],
                                                                       peaks_data[f'Peak_position_{phase_label}'])
                peaks_data[f'Peak_height_{phase_label}'] = np.where(mask,
                                                                     Peaks_filtered[col_name],
                                                                     peaks_data[f'Peak_height_{phase_label}'])
        else:
            peaks_data = pd.DataFrame(0, index=Peaks.index,
                                      columns=[f'Peak_position_{phase_label}', f'Peak_height_{phase_label}'])
        return peaks_data

    # Process all defined phases using their threshold ranges
    phase_thresholds = [
        (Background_peak, Phase_1_threshold),
        (Phase_1_threshold, Phase_2_threshold),
        (Phase_2_threshold, Phase_3_threshold),
        (Phase_3_threshold, Phase_4_threshold),
        (Phase_4_threshold, Phase_5_threshold),
        (Phase_5_threshold, Phase_6_threshold),
        (Phase_6_threshold, Phase_7_threshold),
        (Phase_7_threshold, Phase_8_threshold),
        (Phase_8_threshold, Phase_9_threshold),
        (Phase_9_threshold, Phase_10_threshold),
        (Phase_10_threshold, Phase_11_threshold),
        (Phase_11_threshold, 65536),
    ]

    all_peaks_data = [
        process_phase(Peaks1[cols], start, end, i + 1)
        for i, (start, end) in enumerate(phase_thresholds)
        if not (start == np.inf or end == np.inf)
    ]

     # Combine peaks from all phases
    Peaks_phases = pd.concat([df for df in all_peaks_data if not df.empty], axis=1)
    Peaks_phases = Peaks_phases.loc[:, ~Peaks_phases.columns.duplicated()]
    Peaks_phases = Peaks_phases.fillna(0).astype(float)

    # Apply replacement conditions
    peak_pos_cols = [col for col in Peaks_phases.columns if col.startswith('Peak_position_')]
    peak_height_cols = [col for col in Peaks_phases.columns if col.startswith('Peak_height_')]
    
    # Replace peaks below height thresholds with background_mean
    for pos_col, height_col in zip(peak_pos_cols, peak_height_cols):
        Peaks_phases[pos_col] = np.where(Peaks_phases[height_col] <= float(Height_threshold),
                                         Background_peak,
                                         Peaks_phases[pos_col])
        Peaks_phases[height_col] = np.where(Peaks_phases[pos_col] < Background_peak,
                                            0,
                                            Peaks_phases[height_col])
        
    
    # Add a column for the peak of highest grey value
    Peaks_phases['Max_peak'] = Peaks_phases[peak_pos_cols].max(axis=1)
    Peaks_phases.index = Peaks_phases.index.astype(int)
    Peaks_phases = Peaks_phases.sort_index()
    
    Peaks_phases.index.name = 'Label'

    # Append threshold values as columns 
    threshold_values = {
        'Background_peak': Background_peak,
        'Phase_1_threshold': Phase_1_threshold,
        'Phase_2_threshold': Phase_2_threshold,
        'Phase_3_threshold': Phase_3_threshold,
        'Phase_4_threshold': Phase_4_threshold,
        'Phase_5_threshold': Phase_5_threshold,
        'Phase_6_threshold': Phase_6_threshold,
        'Phase_7_threshold': Phase_7_threshold,
        'Phase_8_threshold': Phase_8_threshold,
        'Phase_9_threshold': Phase_9_threshold,
        'Phase_10_threshold': Phase_10_threshold,
        'Phase_11_threshold': Phase_11_threshold

    }
    
    for key, value in threshold_values.items():
        Peaks_phases[key] = value
        Peaks[key] = value
        
    # Remove infinite values (safety)
    Peaks_phases = Peaks_phases.loc[:, ~Peaks_phases.isin([np.inf, -np.inf]).any()]
    Peaks = Peaks.loc[:, ~Peaks.isin([np.inf, -np.inf]).any()]

    Peaks['Peak_position_1'] = np.where(Peaks['Peak_position_1'] == Background_peak + 1,
    Background_peak + 2, Peaks['Peak_position_1'])
    
    return Peaks_phases


def map_peaks_on_bulk_histograms(histograms, Peaks):
    """
    Often we calculate peaks on processed binned histograms for better accuracy but in
    processed histograms peaks shifts a bit. So we have to map the peaks back on bulk_histograms to 
    adjust the shifts.
    
    Computes peaks per particle using intensity thresholds.
    Then, for each particle, if original Peak_position_n == Background_peak,
    set position to background and height to 0 in the result.

    Automatically detects which columns in histograms are intensity bins.

    Args:
        histograms (pd.DataFrame): Bulk_histogram
        Peaks (pd.DataFrame): rows = particles, with original peak metadata

    Returns:
        pd.DataFrame: Computed peak positions and heights for each particle
    """
    _, Background_peak_global, thresholds = extract_peaks_and_thresholds(Peaks)
    peak_results = []

    # Step 0: Detect bin columns (e.g., '0', '1', ..., '65535' or bin centers)
    bin_cols = [col for col in histograms.columns if str(col).isdigit()]
    bin_cols_sorted = sorted(bin_cols, key=lambda x: int(x))  # ensure sorted numerically

    for idx, row in histograms.iterrows():
        row_result = {}

        # restrict to bin values only
        spectrum = row[bin_cols_sorted]
        spectrum.index = spectrum.index.astype(int)  # ensure integer comparison

        for i in range(len(thresholds) - 1):
            start = thresholds[i]
            end = thresholds[i + 1]

            subrange = spectrum[(spectrum.index >= start) & (spectrum.index < end)]

            if not subrange.empty:
                max_pos = int(subrange.idxmax())
                max_height = float(subrange[max_pos])
            else:
                max_pos = Background_peak_global
                max_height = 0

            row_result[f'Peak_position_{i+1}'] = max_pos
            row_result[f'Peak_height_{i+1}'] = max_height

        peak_results.append(row_result)

    peaks_df = pd.DataFrame(peak_results, index=Peaks.index)

    # Overwrite any computed peak if original peak == background
    for i in range(1, 12):
        peak_col = f'Peak_position_{i}'
        height_col = f'Peak_height_{i}'
        if peak_col in Peaks.columns:
            mask = Peaks[peak_col] == Peaks['Background_peak']
            peaks_df.loc[mask, peak_col] = Peaks.loc[mask, 'Background_peak']
            peaks_df.loc[mask, height_col] = 0.0
            peaks_df = peaks_df.fillna(0)
    
    for col in peaks_df.columns:
        if col in Peaks.columns:
            for idx in peaks_df.index:
                if idx in Peaks.index:
                    val_peaks = Peaks.at[idx, col]
                    val_peaks_df = peaks_df.at[idx, col]
                    if val_peaks != val_peaks_df:
                        Peaks.at[idx, col] = val_peaks_df

    Peaks['Peak_position_1'] = np.where(Peaks['Peak_position_1'] == Background_peak_global + 1,
    Background_peak_global + 2, Peaks['Peak_position_1'])

    return Peaks



def bias_dense_minerals(Peak_data, Properties_Bulk, phases_to_correct=None):
    
    """
    Apply bias correction for detecting dense minerals in Peak data by replacing 
    missed peaks with maximum intensity values from region properties. 
    
    Motivation:
    - Some peaks (especially from small inclusions) may be missed by automatic peak detection.
    - Since high-density (high-attenuation) minerals like gold or REEs are more critical,
      this function biases the correction towards denser phases.
    - For example, missing a small quartz grain is less important than missing a small gold inclusion.
    
    Parameters:
        Peak_data (pd.DataFrame): Peak positions and phase thresholds per label (index = 'Label').
        Properties_Bulk (pd.DataFrame): Region properties including max intensity per label.
        phases_to_correct (list of int, optional): Phase numbers to apply correction to (e.g., [2, 4]).
    
    Returns:
        pd.DataFrame: Corrected Peak_data with adjusted peak positions for selected phases.
    """
    
    Peak_data = standardize_index(Peak_data)
    Properties_Bulk = standardize_index(Properties_Bulk)

    # Automatically detect Background_peak from Peaks 
    background_cols = [col for col in Peak_data.columns if re.search(r'background', col, re.IGNORECASE)]
    if not background_cols:
        raise ValueError("No 'background' column found in Peaks.")
    Background_peak = Peak_data[background_cols[0]].max()

    # Extract threshold columns 
    threshold_cols = [col for col in Peak_data.columns if re.search(r'threshold', col, re.IGNORECASE)]
    if not threshold_cols:
        raise ValueError("No threshold columns found in Peaks. Expected columns containing 'threshold' in the name.")
    threshold_cols_sorted = sorted(threshold_cols, key=lambda x: int(re.search(r'\d+', x).group()))
    threshold_values = [Peak_data[col].max() for col in threshold_cols_sorted]
    threshold_values.append(float('inf'))
    thresholds = list(zip(threshold_values[:-1], threshold_values[1:]))

    # Convert user-friendly input like [2, 4] into actual column names
    if phases_to_correct is not None:
        phases_to_correct = [f'Peak_position_{i}' for i in phases_to_correct]

    # Apply bias correction 
    for i, (low, high) in enumerate(thresholds, start=2):  # Start from Peak_position_2
        peak_col = f'Peak_position_{i}'

        if peak_col not in Peak_data.columns:
            continue

        if phases_to_correct is not None and peak_col not in phases_to_correct:
            continue
        
        # Create mask for labels that:
        # - fall within the current threshold range based on their max_intensity
        # - but had their peak missed (still labeled as Background_peak)
        mask = (
            (Properties_Bulk['max_intensity'] > low) &
            (Properties_Bulk['max_intensity'] <= high) &
            (Peak_data[peak_col] == Background_peak)
        )
        
        # Replace missed peak with max intensity from the particle
        if mask.any():
            Peak_data.loc[mask, peak_col] = Properties_Bulk.loc[mask, 'max_intensity']
            Peak_data.loc[mask, 'Max_peak'] = Properties_Bulk.loc[mask, 'max_intensity']

    return Peak_data



def plot_histograms_with_peaks(df, Peaks, selected_particles=None,
                               xlabel='Grey value', ylabel='Number of voxels',
                               figsize=(10, 6), dpi=100, font_size=12, Grid=False):
    """
    Plots rows from a DataFrame and overlays peak positions from Peaks DataFrame.
    Does not plot line segments where y-values are zero.
    """
    df.index = df.index.astype(int)
    Peaks.index = Peaks.index.astype(int)
    df = df.sort_index()
    Peaks = Peaks.sort_index()

    if selected_particles is None:
        selected_particles = df.index.tolist()

    plt.figure(figsize=figsize, dpi=dpi)
    x_vals = df.columns.astype(float) if df.columns.dtype.kind in "iufc" else np.arange(len(df.columns))

    for pid in selected_particles:
        if pid not in df.index:
            continue
        y_vals = df.loc[pid].values.astype(float)

        # Mask zero values
        nonzero_mask = y_vals != 0
        x_nonzero = np.array(x_vals)[nonzero_mask]
        y_nonzero = y_vals[nonzero_mask]

        if len(y_nonzero) == 0:
            continue  # Skip plotting if all y-values are zero

        plt.plot(x_nonzero, y_nonzero, label=f'{pid}')

        peak_cols = [col for col in Peaks.columns if col.startswith("Peak_position_")]
        height_cols = [col for col in Peaks.columns if col.startswith("Peak_height_")]

        if pid in Peaks.index:
            peak_positions = Peaks.loc[pid, peak_cols].values
            peak_heights = Peaks.loc[pid, height_cols].values
            plt.scatter(peak_positions, peak_heights, color='red', zorder=5)

    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    plt.grid(Grid)
    plt.legend(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()
    plt.show()
    

def extract_peaks_and_thresholds(Peaks):
    """
    Extracts:
    - Sorted peak columns into a new DataFrame called `array`
    - Background peak position (assumes constant for all rows)
    - Phase thresholds (as a list), padded and prepended with background

    Returns:
        array (DataFrame), Background_peak_pos (int), phase_thresholds (list of float)
    """
    import re
    Peaks.columns = Peaks.columns.str.strip()

    # Step 1: Extract and sort peak columns
    peak_cols = []
    for col in Peaks.columns:
        match = re.match(r'(?i)^peak[\s_]?((position|phase)[\s_]?)?(\d+)$', col)
        if match:
            peak_num = int(match.group(3))
            peak_cols.append((peak_num, col))
    peak_cols_sorted = [col for _, col in sorted(peak_cols)]
    array = Peaks[peak_cols_sorted].copy().fillna(0)

    # Step 2: Extract background_mean value
    background_col = next((col for col in Peaks.columns if 'background' in col.lower()), None)
    if background_col is None:
        raise ValueError("No column with 'background' in name found in Peaks DataFrame.")
    Background_peak_pos = int(Peaks[background_col].iloc[0])

    # Step 3: Extract and sort phase thresholds
    threshold_cols = []
    for col in Peaks.columns:
        match = re.match(r'(?i)^phase[_\s]*(\d+)[_\s]*threshold$', col)
        if match:
            phase_num = int(match.group(1))
            threshold_cols.append((phase_num, col))

    threshold_cols_sorted = [col for _, col in sorted(threshold_cols)]
    phase_thresholds = [int(Peaks[col].iloc[0]) for col in threshold_cols_sorted]

    # Prepend background
    phase_thresholds.insert(0, Background_peak_pos)
    
    if phase_thresholds[-1] < 65535:
        phase_thresholds.append(65535)

    # Pad up to Phase 12 (or 11 thresholds + background)
    while len(phase_thresholds) <= 12:
        phase_thresholds.append(float('inf'))
    


    return array, Background_peak_pos, phase_thresholds



def Liberated_Particles(Peaks, Bulk_histograms, Surface_mesh_histogram):
    
    """
    Process particles containing exactly one peak above background and compute their core, outer, 
    and surface quantification. Organizes the result into phase-sorted DataFrame.
    
    Parameters:
        Peaks (pd.DataFrame): Peak positions and thresholds, indexed by particle label.
        Bulk_histograms (pd.DataFrame): Histogram of voxel intensities for each particle.
        Surface_mesh_histogram (pd.DataFrame): Histogram of voxel intensities of surface.
    
    Returns:
        pd.DataFrame: Quantification results sorted by phase thresholds, indexed by Label.
    """
        
    # Standardize Peak DataFrame to ensure index is 'Label'
    Peaks = standardize_index(Peaks)
    
    # Extract peak values, background position, and threshold intervals
    array, Background_peak_pos, phase_thresholds = extract_peaks_and_thresholds(Peaks)


    # Step 5: Initialize containers
    Quantification_1_phases_append = []
    Index_1_phase = []
    Peaks_1_phase = []
    Quantification_Outer_phase_1_append = []
    Surface_quantification_append = []

    # # Loop through each particle to process single-peak cases
    for i, (index, row) in enumerate(Bulk_histograms.iterrows()):
        Peaks_row = array.iloc[i].values

        # Find which peaks are above background
        above_bg_indices = np.where(Peaks_row > Background_peak_pos)[0]
        if len(above_bg_indices) == 1:
            valid_peak_idx = above_bg_indices[0]
            Partical_peak = int(Peaks_row[valid_peak_idx])
            Peaks_1_phase.append([Partical_peak])
            Index_1_phase.append([index])

            # Core quantification
            Sum_phase = row.iloc[Partical_peak:].sum()
            Quantification_1_phases_append.append([Sum_phase])

            # Outer phase quantification
            voxels = row.iloc[Background_peak_pos:Partical_peak]
            weights = np.linspace(0, 1, Partical_peak+1 - Background_peak_pos)
            weights = weights[:len(voxels)]
            Quantification_Outer = (voxels * weights).sum()
            Quantification_Outer_phase_1_append.append([Quantification_Outer])

            # Surface quantification
            Surface_quant = Surface_mesh_histogram.iloc[i, Background_peak_pos:].sum()
            Surface_quantification_append.append([Surface_quant])

    # Build DataFrames from collected results
    df_outer = pd.DataFrame(Quantification_Outer_phase_1_append, columns=['Quantification_Outer_phase_1'])
    df_core = pd.DataFrame(Quantification_1_phases_append, columns=['Quantification_phase_1'])
    df_surface = pd.DataFrame(Surface_quantification_append, columns=['Surface_quantification'])
    df_core['total_quantification_phase_1'] = df_core['Quantification_phase_1'] + df_outer['Quantification_Outer_phase_1']
    df_index = pd.DataFrame(Index_1_phase, columns=['Label'])
    df_peaks = pd.DataFrame(Peaks_1_phase, columns=['Peak_1'])

    Quantification_1_phase_sorted = pd.DataFrame(index=df_index['Label'])
    
    # Assign quantification values to corresponding phase based on peak position thresholds
    for i in range(1, len(phase_thresholds)):  
        lower = phase_thresholds[i - 1]
        upper = phase_thresholds[i] if i < len(phase_thresholds) else float('inf')
        mask = (df_peaks['Peak_1'] > lower) & (df_peaks['Peak_1'] <= upper)

        Quantification_1_phase_sorted[f'Peak_{i}'] = np.where(mask, df_peaks['Peak_1'], 0)
        Quantification_1_phase_sorted[f'Phase_{i}_quantification'] = np.where(mask, df_core['total_quantification_phase_1'], 0)
        Quantification_1_phase_sorted[f'Phase_{i}_surface_quantification'] = np.where(mask, df_surface['Surface_quantification'], 0)
        
    # Sort columns by type and number (e.g., Peak_1, Phase_1_quantification...)
    Quantification_1_phase_sorted = Quantification_1_phase_sorted[
        sorted(
            Quantification_1_phase_sorted.columns,
            key=lambda name: (re.sub(r'\d+', '', name), int(re.search(r'\d+', name).group()) if re.search(r'\d+', name) else -1)
        )
    ]

    Quantification_1_phase_sorted.index = df_index['Label']
    cols = ['Peak_1', 'Peak_2', 'Peak_3', 'Peak_4', 'Peak_5', 'Peak_6', 'Peak_7', 'Peak_8', 'Peak_9', 'Peak_10', 'Peak_11', 'Peak_12']
    Quantification_1_phase_sorted[cols] = Quantification_1_phase_sorted[cols].replace(0, Background_peak_pos)
    return Quantification_1_phase_sorted
    



def Binary_Particles(Peaks, Bulk_histograms, Inner_volume_histograms, Outer_volume_histograms, Surface_mesh_histogram, Gradient, Gradient_threshold):
    
    """
    Quantify particles that contain exactly two peaks (i.e., two phases).
    Calculates the inner, outer, and surface contributions of each phase and returns them as a threshold-sorted DataFrame.

    Parameters:
        Peaks (DataFrame): Peak positions and thresholds per particle (must include background and phase threshold info).
        Bulk_histograms (DataFrame): Full voxel intensity histograms per particle.
        Inner_volume_histograms (DataFrame): Intensity histograms limited to inner voxels.
        Outer_volume_histograms (DataFrame): Histograms for outer voxels (potential PVE correction).
        Surface_mesh_histogram (DataFrame): Intensity histograms from surface voxels.
        Gradient (DataFrame): Gradient values used to scale surface estimation.
        Gradient_threshold (float): Lower limit to control scaling for surface estimation.

    Returns:
        DataFrame: Quantification results for binary-phase particles sorted by threshold-defined phase regions.
    """
    
    #Step 1: Clean column names and extract necessary data
    Peaks = standardize_index(Peaks)
    Gradient = standardize_index(Gradient)
    
    array, Background_peak_pos, phase_thresholds = extract_peaks_and_thresholds(Peaks)
    
    # Step 2: Initialize containers to collect particle-specific results
    Quantification_all_2_phases_1 = []
    Quantification_all_2_phases_2 = []
    Quantification_out_of_peaks_phase_1 = []
    Quantification_out_of_peaks_phase_2 = []
    Surface_volume_phase_1_append = []
    Surface_volume_phase_2_append = []
    Quantification_Outer_phase_1 = []
    Quantification_Outer_phase_2 = []
    Peaks_1_phase = []
    Peaks_2_phase = []
    Index_2_phase = []

    # step 3: Loop through particles to calculate quantifications
    for i, (index, row) in enumerate(Inner_volume_histograms.iterrows()):
        Peaks_row = array.iloc[i].values
        above_bg = Peaks_row[Peaks_row > Background_peak_pos]

        if len(above_bg) == 2:
            # Extract two valid peak positions
            Partical_peak_1 = int(above_bg.flat[0])
            Partical_peak_2 = int(above_bg.flat[1])
            
            # Gradient-based surface scaling
            Gradient_ratio = Gradient['Gradient_3'].iloc[i]
            if Gradient_ratio < Gradient_threshold:
                Gradient_ratio = Gradient_threshold

            # Inner core quantification from histogram
            Sum_phase_1 = Inner_volume_histograms.iloc[i, Background_peak_pos+1:Partical_peak_1+1].sum()
            Sum_phase_2 = Inner_volume_histograms.iloc[i, Partical_peak_2:].sum()
            

            Quantification_all_2_phases_2.append([Sum_phase_2])
            Peaks_1_phase.append([Partical_peak_1])
            Peaks_2_phase.append([Partical_peak_2])

            No_of_voxels = Inner_volume_histograms.iloc[i, Partical_peak_1+1:Partical_peak_2].values
            multiples = np.linspace(0, 1, max(1, Partical_peak_2+1 - Partical_peak_1))
            multiples_towards_1 = multiples[1:len(No_of_voxels)+1]
            multiples_towards_2 = multiples_towards_1[::-1]

            out_of_peak_volume_1 = np.sum(No_of_voxels * multiples_towards_2)
            out_of_peak_volume_2 = np.sum(No_of_voxels * multiples_towards_1)

            Quantification_all_2_phases_1.append([Sum_phase_1])
            Quantification_out_of_peaks_phase_1.append([out_of_peak_volume_1])
            Quantification_out_of_peaks_phase_2.append([out_of_peak_volume_2])

            Outer_volume_full_phase_2 = Outer_volume_histograms.iloc[i, Partical_peak_2:].sum()

            voxels_bg_1 = Outer_volume_histograms.iloc[i, Background_peak_pos+1:Partical_peak_1]
            weights_bg_1 = np.linspace(0, 1, max(1, Partical_peak_1 - Background_peak_pos+1))[1:len(voxels_bg_1)+1]
            
            Quantification_Outer_phase_1_array = np.sum(voxels_bg_1 * weights_bg_1)

            voxels_bg_2 = Outer_volume_histograms.iloc[i, Background_peak_pos+1:Partical_peak_2]
            weights_bg_2 = np.linspace(0, 1, max(1, Partical_peak_2 - Background_peak_pos+1))[1:len(voxels_bg_2)+1]
            Quantification_Outer_phase_2_array = np.sum(voxels_bg_2 * weights_bg_2) - np.sum((voxels_bg_2 * weights_bg_2)[:Partical_peak_1 - Background_peak_pos])

            PVE_adjusted_volume = Outer_volume_full_phase_2 + Quantification_Outer_phase_1_array + Quantification_Outer_phase_2_array

            # Determine phase limit from thresholds
            for j in range(1, len(phase_thresholds)):
                if phase_thresholds[j - 1] <= Partical_peak_1 < phase_thresholds[j]:
                    Phase_limit = phase_thresholds[j]
                    break
            else:
                Phase_limit = 65535

            surface_row = Surface_mesh_histogram.iloc[i]
            Surface_ratio = surface_row[Background_peak_pos+1:int(Gradient_ratio * Phase_limit)].sum() / surface_row[Background_peak_pos+1:].sum()

            Phase_1_surface_volume = surface_row[Background_peak_pos+1:].sum() * Surface_ratio
            Phase_2_surface_volume = surface_row[Background_peak_pos+1:].sum() - Phase_1_surface_volume

            Surface_volume_phase_1_append.append([Phase_1_surface_volume])
            Surface_volume_phase_2_append.append([Phase_2_surface_volume])

            Quantification_Outer_phase_1.append([PVE_adjusted_volume * Surface_ratio])
            Quantification_Outer_phase_2.append([PVE_adjusted_volume * (1 - Surface_ratio)])
            Index_2_phase.append([index])

    # Step 4: Build DataFrames
    Index_2_phase = pd.DataFrame(Index_2_phase, columns=["Label"])
    Peaks_1_phase = pd.DataFrame(Peaks_1_phase, columns=['Peak_1'], index=Index_2_phase['Label'])
    Peaks_2_phase = pd.DataFrame(Peaks_2_phase, columns=['Peak_2'], index=Index_2_phase['Label'])

    Quant_1 = pd.DataFrame(Quantification_all_2_phases_1, columns=['Phase_1_quantification_outer'], index=Index_2_phase['Label'])
    Quant_2 = pd.DataFrame(Quantification_all_2_phases_2, columns=['Phase_2_quantification_outer'], index=Index_2_phase['Label'])

    Out_1 = pd.DataFrame(Quantification_out_of_peaks_phase_1, columns=["Quantification_out_of_peaks_1_outer"], index=Index_2_phase['Label'])
    Out_2 = pd.DataFrame(Quantification_out_of_peaks_phase_2, columns=["Quantification_out_of_peaks_2_outer"], index=Index_2_phase['Label'])

    Outer_1 = pd.DataFrame(Quantification_Outer_phase_1, columns=["Quantification_Outer_phase_1"], index=Index_2_phase['Label'])
    Outer_2 = pd.DataFrame(Quantification_Outer_phase_2, columns=["Quantification_Outer_phase_2"], index=Index_2_phase['Label'])

    Surf_1 = pd.DataFrame(Surface_volume_phase_1_append, columns=['Surface_volume_phase_1'], index=Index_2_phase['Label'])
    Surf_2 = pd.DataFrame(Surface_volume_phase_2_append, columns=['Surface_volume_phase_2'], index=Index_2_phase['Label'])

    Quantification_2_phases_inner = pd.concat([Peaks_1_phase, Peaks_2_phase, Quant_1, Quant_2, Out_1, Out_2], axis=1)
    Quantification_2_phases_inner['Phase_1_inner_quantification'] = Quant_1['Phase_1_quantification_outer'] + Out_1['Quantification_out_of_peaks_1_outer']
    Quantification_2_phases_inner['Phase_2_inner_quantification'] = Quant_2['Phase_2_quantification_outer'] + Out_2['Quantification_out_of_peaks_2_outer']
    Quantification_2_phases_inner = Quantification_2_phases_inner[['Peak_1', 'Peak_2', 'Phase_1_inner_quantification', 'Phase_2_inner_quantification']]

    Quantification_2_phases = pd.concat([Quantification_2_phases_inner, Outer_1, Outer_2], axis=1)
    Quantification_2_phases['total_quantification_phase_1'] = Quantification_2_phases['Phase_1_inner_quantification'] + Quantification_2_phases['Quantification_Outer_phase_1']
    Quantification_2_phases['total_quantification_phase_2'] = Quantification_2_phases['Phase_2_inner_quantification'] + Quantification_2_phases['Quantification_Outer_phase_2']
    Quantification_2_phases['Phase_1_surface_quantification'] = Surf_1['Surface_volume_phase_1']
    Quantification_2_phases['Phase_2_surface_quantification'] = Surf_2['Surface_volume_phase_2']

    # Step 5: Assign by thresholds
    Quantification_2_phase_sorted = pd.DataFrame(index=Quantification_2_phases.index)
    temp_sorted = pd.DataFrame(index=Quantification_2_phases.index)

    for i in range(1, len(phase_thresholds)):
        lower = phase_thresholds[i - 1]
        upper = phase_thresholds[i]

        mask1 = (Peaks_1_phase['Peak_1'] > lower) & (Peaks_1_phase['Peak_1'] <= upper)
        Quantification_2_phase_sorted[f'Peak_{i}'] = np.where(mask1, Peaks_1_phase['Peak_1'], 0)
        Quantification_2_phase_sorted[f'Phase_{i}_quantification'] = np.where(mask1, Quantification_2_phases['total_quantification_phase_1'], 0)
        Quantification_2_phase_sorted[f'Phase_{i}_surface_quantification'] = np.where(mask1, Quantification_2_phases['Phase_1_surface_quantification'], 0)
        Quantification_2_phase_sorted[f'Phase_{i}_outer_quantification'] = np.where(mask1, Quantification_2_phases['Quantification_Outer_phase_1'], 0)

        mask2 = (Peaks_2_phase['Peak_2'] > lower) & (Peaks_2_phase['Peak_2'] <= upper)
        temp_sorted[f'Peak_{i}'] = np.where(mask2, Peaks_2_phase['Peak_2'], 0)
        temp_sorted[f'Phase_{i}_quantification'] = np.where(mask2, Quantification_2_phases['total_quantification_phase_2'], 0)
        temp_sorted[f'Phase_{i}_surface_quantification'] = np.where(mask2, Quantification_2_phases['Phase_2_surface_quantification'], 0)
        temp_sorted[f'Phase_{i}_outer_quantification'] = np.where(mask2, Quantification_2_phases['Quantification_Outer_phase_2'], 0)

    Quantification_2_phase_sorted = Quantification_2_phase_sorted.mask(Quantification_2_phase_sorted == 0, temp_sorted)
    cols = ['Peak_1', 'Peak_2', 'Peak_3', 'Peak_4', 'Peak_5', 'Peak_6', 'Peak_7', 'Peak_8', 'Peak_9', 'Peak_10', 'Peak_11', 'Peak_12']
    Quantification_2_phase_sorted[cols] = Quantification_2_phase_sorted[cols].replace(0, Background_peak_pos)

    return Quantification_2_phase_sorted



def Ternary_Particles(Peaks, Bulk_histograms, Inner_volume_histograms, Outer_volume_histograms, Surface_mesh_histogram, Gradient, Gradient_threshold):
    
    """
    Quantifies particles that contain exactly 3 peaks (i.e., three distinct mineral phases).
    Computes the inner volume, outer volume (with PVE correction), and surface contributions
    for each phase and returns a threshold-sorted DataFrame.

    Parameters:
        Peaks (DataFrame): Peak data for each particle (must include background and phase thresholds).
        Bulk_histograms (DataFrame): Histogram of intensities across all voxels per particle.
        Inner_volume_histograms (DataFrame): Histogram within eroded particle cores.
        Outer_volume_histograms (DataFrame): Histogram of outer (border) voxels.
        Surface_mesh_histogram (DataFrame): Histogram from mesh-based surface voxel map.
        Gradient (DataFrame): Gradient-derived surface sharpness per particle.
        Gradient_threshold (float): Minimum ratio for scaling surface contributions.

    Returns:
        DataFrame: Quantified contributions from three phases, sorted by peak threshold bins.
    """
    
    # Standardize indices for consistency
    Peaks = standardize_index(Peaks)
    Gradient = standardize_index(Gradient)
    
    # Extract peak data and thresholds
    array, Background_peak_pos, phase_thresholds = extract_peaks_and_thresholds(Peaks)
    
    # Initialize lists for results
    Quantification_all_3_phases_1 = []
    Quantification_all_3_phases_2 = []
    Quantification_all_3_phases_3 = []

    Peaks_1_phase = []
    Peaks_2_phase = []
    Peaks_3_phase = []

    Index_3_phase = []
    
    Surface_volume_phase_1_append = []
    Surface_volume_phase_2_append = []
    Surface_volume_phase_3_append = []
    
    Quantification_Outer_phase_1_append = []
    Quantification_Outer_phase_2_append = []
    Quantification_Outer_phase_3_append = []
    

    i=0     
    # Loop over all particles
    for index,row in Bulk_histograms.iterrows():
        Peaks = (array.iloc[[i]].values)
        if (np.count_nonzero(Peaks > Background_peak_pos) == 3) and i >-1:
            Partical_peak = Peaks[Peaks >Background_peak_pos]
        
            Partical_peak_1 = Partical_peak.flat[0]
            Partical_peak_1 = int(float(Partical_peak_1))
            Partical_peak_2 = Partical_peak.flat[1]
            Partical_peak_2 = int(float(Partical_peak_2))
            Partical_peak_3 = Partical_peak.flat[2]
            Partical_peak_3 = int(float(Partical_peak_3))
        
            #Taking the sum of ith row from phase 1 minimum threshold greyscale value till Phase1_max_limit
            Sum_phase_1 = Inner_volume_histograms.iloc[i,Background_peak_pos+1:int((Partical_peak_1+Partical_peak_2)/2)].sum()
        

            #Taking the sum of ith row from phase 1 minimum threshold greyscale value till Phase1_max_limit
            Sum_phase_2 = Inner_volume_histograms.iloc[i,int((Partical_peak_1+Partical_peak_2)/2):int((Partical_peak_2+Partical_peak_3)/2)].sum()
            
        
            #Taking the sum of ith row from phase 1 minimum threshold greyscale value till Phase1_max_limit
            Sum_phase_3 = Inner_volume_histograms.iloc[i,int((Partical_peak_2+Partical_peak_3)/2):].sum()
        
        
            Index_3_phase.append([index])
            
            Peaks_1_phase.append([Partical_peak_1])
            Peaks_2_phase.append([Partical_peak_2])
            Peaks_3_phase.append([Partical_peak_3])
            
            Gradient_ratio = Gradient['Gradient_3'].iloc[i]
            if Gradient_ratio < Gradient_threshold:
                Gradient_ratio = Gradient_threshold
            
            def get_phase_limit(peak, phase_thresholds):
                for i in range(1, len(phase_thresholds)):
                    if peak < phase_thresholds[i]:
                        return phase_thresholds[i]
                return phase_thresholds[-1]
            
            Phase_limit_1 = get_phase_limit(Partical_peak_1, phase_thresholds)
            Phase_limit_2 = get_phase_limit(Partical_peak_2, phase_thresholds)

                
            
            Phase_1_surface_volume = Surface_mesh_histogram.iloc[i,Background_peak_pos+1:int(Phase_limit_1*Gradient_ratio)].sum()
            Phase_2_surface_volume = Surface_mesh_histogram.iloc[i,int(Phase_limit_1*Gradient_ratio):int(Phase_limit_2*Gradient_ratio)].sum()
            Phase_3_surface_volume = Surface_mesh_histogram.iloc[i,int(Phase_limit_2*Gradient_ratio):].sum()
            
            Surface_ratio_1 = Phase_1_surface_volume/(Phase_1_surface_volume+Phase_2_surface_volume+Phase_3_surface_volume)
            Surface_ratio_2 = Phase_2_surface_volume/(Phase_1_surface_volume+Phase_2_surface_volume+Phase_3_surface_volume)
            Surface_ratio_3 = Phase_3_surface_volume/(Phase_1_surface_volume+Phase_2_surface_volume+Phase_3_surface_volume)
            
            Surface_volume_phase_1_append.append([Phase_1_surface_volume])
            Surface_volume_phase_2_append.append([Phase_2_surface_volume])
            Surface_volume_phase_3_append.append([Phase_3_surface_volume]) 
            
            Outer_volume_full_phase_3 = Outer_volume_histograms.iloc[i,Partical_peak_3:].sum()
            
            def calculate_phase_quantification_array(particle_peak_pos):
                no_of_voxels_towards_background = Outer_volume_histograms.iloc[i,Background_peak_pos+1:particle_peak_pos]
                # Calculate the multiples array
                multiples_towards_background = np.linspace(0, 1, (particle_peak_pos) - Background_peak_pos+1)
                multiples_towards_background = multiples_towards_background[1:len(no_of_voxels_towards_background)+1]
                # Calculate and return the quantification result
                quantification_outer_phase = no_of_voxels_towards_background * multiples_towards_background
                return quantification_outer_phase
            Quantification_Outer_phase_1_array = calculate_phase_quantification_array(Partical_peak_1)
            Quantification_Outer_phase_2_array = calculate_phase_quantification_array(Partical_peak_2)
            Quantification_Outer_phase_2_array = Quantification_Outer_phase_2_array[Partical_peak_1-Background_peak_pos:]
            Quantification_Outer_phase_3_array = calculate_phase_quantification_array(Partical_peak_3)
            Quantification_Outer_phase_3_array = Quantification_Outer_phase_3_array[Partical_peak_2-Background_peak_pos: Partical_peak_3 - Background_peak_pos]
            
            PVE_adjusted_volume = Outer_volume_full_phase_3 + Quantification_Outer_phase_1_array.sum()+Quantification_Outer_phase_2_array.sum()+Quantification_Outer_phase_3_array.sum()
            
            # Scale outer volume using surface ratio
            Quantification_Outer_phase_1_volume = PVE_adjusted_volume*Surface_ratio_1
            Quantification_Outer_phase_2_volume = PVE_adjusted_volume*Surface_ratio_2
            Quantification_Outer_phase_3_volume = PVE_adjusted_volume*Surface_ratio_3
            
            Quantification_Outer_phase_1_append.append([Quantification_Outer_phase_1_volume])
            Quantification_Outer_phase_2_append.append([Quantification_Outer_phase_2_volume])
            Quantification_Outer_phase_3_append.append([Quantification_Outer_phase_3_volume])
            
            Sum_phase_1 = Sum_phase_1.sum() + Quantification_Outer_phase_1_volume
            Sum_phase_2 = Sum_phase_2.sum() + Quantification_Outer_phase_2_volume
            Sum_phase_3 = Sum_phase_3.sum() + Quantification_Outer_phase_3_volume
            Quantification_all_3_phases_1.append([Sum_phase_1])
            Quantification_all_3_phases_2.append([Sum_phase_2])
            Quantification_all_3_phases_3.append([Sum_phase_3])

        i = i+1
    
    #Creating Quantification_all of quantification of voxels which have 100% phase 1
    Quantification_all_3_phases_1 = pd.DataFrame(Quantification_all_3_phases_1,columns = ['total_quantification_phase_1'])
    Quantification_all_3_phases_2 = pd.DataFrame(Quantification_all_3_phases_2,columns = ['total_quantification_phase_2'])
    Quantification_all_3_phases_3 = pd.DataFrame(Quantification_all_3_phases_3,columns = ['total_quantification_phase_3'])
    
    
    Quantification_Outer_phase_1 = pd.DataFrame(Quantification_Outer_phase_1_append,columns = ['Outer_quantification_phase_1'])
    Quantification_Outer_phase_2 = pd.DataFrame(Quantification_Outer_phase_2_append,columns = ['Outer_quantification_phase_2'])
    Quantification_Outer_phase_3 = pd.DataFrame(Quantification_Outer_phase_3_append,columns = ['Outer_quantification_phase_3'])
    
    

    Index_3_phase = pd.DataFrame(Index_3_phase,columns = ['Label'])

    Peaks_1_phase = pd.DataFrame(Peaks_1_phase,columns = ['Peak_1'])
    Peaks_2_phase = pd.DataFrame(Peaks_2_phase,columns = ['Peak_2'])
    Peaks_3_phase = pd.DataFrame(Peaks_3_phase,columns = ['Peak_3'])
    
    Surface_volume_phase_1 = pd.DataFrame(Surface_volume_phase_1_append,columns = ['Surface_volume_phase_1'])
    Surface_volume_phase_1.index = Index_3_phase['Label']
    Surface_volume_phase_2 = pd.DataFrame(Surface_volume_phase_2_append,columns = ['Surface_volume_phase_2'])
    Surface_volume_phase_2.index = Index_3_phase['Label']
    Surface_volume_phase_3 = pd.DataFrame(Surface_volume_phase_3_append,columns = ['Surface_volume_phase_3'])
    Surface_volume_phase_3.index = Index_3_phase['Label']

    Quantification_3_phases = pd.concat([Index_3_phase,Quantification_all_3_phases_1,Quantification_all_3_phases_2,Quantification_all_3_phases_3,
                                         Peaks_1_phase,Peaks_2_phase,Peaks_3_phase,Quantification_Outer_phase_1,
                                         Quantification_Outer_phase_2,Quantification_Outer_phase_3],axis = 1)
    cols = ['Peak_1', 'Peak_2', 'Peak_3', 'Peak_4', 'Peak_5', 'Peak_6', 'Peak_7', 'Peak_8', 'Peak_9', 'Peak_10', 'Peak_11', 'Peak_12']

    thresholds = phase_thresholds
    Quantification_3_phase_sorted = pd.DataFrame(columns= cols + [f'Phase_{i}_quantification' for i in range(1, 6)])
    Quantification_3_phase_sorted_1 = Quantification_3_phase_sorted.copy()
    Quantification_3_phase_sorted_2 = Quantification_3_phase_sorted.copy()
    for i in range(1,  len(phase_thresholds)):
        mask = (Peaks_1_phase['Peak_1'] > thresholds[i - 1]) & (Peaks_1_phase['Peak_1'] <= thresholds[i])
        Quantification_3_phase_sorted[f'Peak_{i}'] = np.where(mask, Peaks_1_phase['Peak_1'], 0)
        Quantification_3_phase_sorted[f'Phase_{i}_quantification'] = np.where(mask, Quantification_3_phases['total_quantification_phase_1'], 0)
        Quantification_3_phase_sorted[f'Phase_{i}_surface_quantification'] = np.where(mask, Surface_volume_phase_1['Surface_volume_phase_1'], 0)
        Quantification_3_phase_sorted[f'Phase_{i}_outer_quantification'] = np.where(mask, Quantification_Outer_phase_1['Outer_quantification_phase_1'], 0)
    
    for i in range(1,  len(phase_thresholds)):
        mask = (Peaks_2_phase['Peak_2'] > thresholds[i-1]) & (Peaks_2_phase['Peak_2'] <= thresholds[i])
        Quantification_3_phase_sorted_1[f'Peak_{i}'] = np.where(mask, Peaks_2_phase['Peak_2'], 0)
        Quantification_3_phase_sorted_1[f'Phase_{i}_quantification'] = np.where(mask, Quantification_3_phases['total_quantification_phase_2'], 0)
        Quantification_3_phase_sorted_1[f'Phase_{i}_surface_quantification'] = np.where(mask, Surface_volume_phase_2['Surface_volume_phase_2'], 0)
        Quantification_3_phase_sorted_1[f'Phase_{i}_outer_quantification'] = np.where(mask, Quantification_Outer_phase_2['Outer_quantification_phase_2'], 0)
    
    for i in range(1,  len(phase_thresholds)):
        mask = (Peaks_3_phase['Peak_3'] > thresholds[i-1]) & (Peaks_3_phase['Peak_3'] <= thresholds[i])
        Quantification_3_phase_sorted_2[f'Peak_{i}'] = np.where(mask, Peaks_3_phase['Peak_3'], 0)
        Quantification_3_phase_sorted_2[f'Phase_{i}_quantification'] = np.where(mask, Quantification_3_phases['total_quantification_phase_3'], 0)
        Quantification_3_phase_sorted_2[f'Phase_{i}_surface_quantification'] = np.where(mask, Surface_volume_phase_3['Surface_volume_phase_3'], 0)
        Quantification_3_phase_sorted_2[f'Phase_{i}_outer_quantification'] = np.where(mask, Quantification_Outer_phase_3['Outer_quantification_phase_3'], 0)
    
    Quantification_3_phase_sorted = Quantification_3_phase_sorted.mask(Quantification_3_phase_sorted == 0, Quantification_3_phase_sorted_1)
    Quantification_3_phase_sorted = Quantification_3_phase_sorted.mask(Quantification_3_phase_sorted == 0, Quantification_3_phase_sorted_2)
    Quantification_3_phase_sorted.index = Quantification_3_phases['Label']
    Quantification_3_phase_sorted[cols] = Quantification_3_phase_sorted[cols].replace(0, Background_peak_pos)
    
    return Quantification_3_phase_sorted





def Quaternary_Particles(Peaks, Bulk_histograms, Inner_volume_histograms, Outer_volume_histograms, Surface_mesh_histogram, Gradient, Gradient_threshold):
    
    """
    Quantifies particles that contain exactly 4 peaks (i.e., four distinct mineral phases).
    Computes the inner volume, outer volume (with PVE correction), and surface contributions
    for each phase and returns a threshold-sorted DataFrame.

    Parameters:
        Peaks (DataFrame): Peak data for each particle (must include background and phase thresholds).
        Bulk_histograms (DataFrame): Histogram of intensities across all voxels per particle.
        Inner_volume_histograms (DataFrame): Histogram within eroded particle cores.
        Outer_volume_histograms (DataFrame): Histogram of outer (border) voxels.
        Surface_mesh_histogram (DataFrame): Histogram from mesh-based surface voxel map.
        Gradient (DataFrame): Gradient-derived surface sharpness per particle.
        Gradient_threshold (float): Minimum ratio for scaling surface contributions.

    Returns:
        DataFrame: Quantified contributions from three phases, sorted by peak threshold bins.
    """
    Peaks = standardize_index(Peaks)
    Gradient = standardize_index(Gradient)
    
    array, Background_peak_pos, phase_thresholds = extract_peaks_and_thresholds(Peaks)
    
    Quantification_all_4_phases_1 = []
    Quantification_all_4_phases_2 = []
    Quantification_all_4_phases_3 = []
    Quantification_all_4_phases_4 = []

    Peaks_1_phase = []
    Peaks_2_phase = []
    Peaks_3_phase = []
    Peaks_4_phase = []

    Index_4_phase = []
    
    Surface_volume_phase_1_append = []
    Surface_volume_phase_2_append = []
    Surface_volume_phase_3_append = []
    Surface_volume_phase_4_append = []
    
    
    Quantification_Outer_phase_1_append = []
    Quantification_Outer_phase_2_append = []
    Quantification_Outer_phase_3_append = []
    Quantification_Outer_phase_4_append = []
    
    i=0     
    for index,row in Bulk_histograms.iterrows():
        Peaks = (array.iloc[[i]].values)
        if (np.count_nonzero(Peaks > Background_peak_pos) == 4) and i >-1:
            Partical_peak = Peaks[Peaks >Background_peak_pos]
        
            Partical_peak_1 = Partical_peak.flat[0]
            Partical_peak_1 = int(float(Partical_peak_1))
            Partical_peak_2 = Partical_peak.flat[1]
            Partical_peak_2 = int(float(Partical_peak_2))
            Partical_peak_3 = Partical_peak.flat[2]
            Partical_peak_3 = int(float(Partical_peak_3))
            Partical_peak_4 = Partical_peak.flat[3]
            Partical_peak_4 = int(float(Partical_peak_4))
        
            Sum_phase_1 = Inner_volume_histograms.iloc[i,Background_peak_pos+1:int((Partical_peak_1+Partical_peak_2)/2)].sum()
        
            Sum_phase_2 = Inner_volume_histograms.iloc[i,int((Partical_peak_1+Partical_peak_2)/2):int((Partical_peak_2+Partical_peak_3)/2)].sum()
            
            Sum_phase_3 = Inner_volume_histograms.iloc[i,int((Partical_peak_2+Partical_peak_3)/2):int((Partical_peak_3+Partical_peak_4)/2)].sum()
                   
            Sum_phase_4 = Inner_volume_histograms.iloc[i,int((Partical_peak_3+Partical_peak_4)/2):].sum()
        
            Index_4_phase.append([index])
        
            Peaks_1_phase.append([Partical_peak_1])
            Peaks_2_phase.append([Partical_peak_2])
            Peaks_3_phase.append([Partical_peak_3])
            Peaks_4_phase.append([Partical_peak_4])
            
                
            
            
            Gradient_ratio = Gradient['Gradient_3'].iloc[i]
            if Gradient_ratio < Gradient_threshold:
                Gradient_ratio = Gradient_threshold
            
            def get_phase_limit(peak, phase_thresholds):
                for i in range(1, len(phase_thresholds)):
                    if peak < phase_thresholds[i]:
                        return phase_thresholds[i]
                return phase_thresholds[-1]
            
            Phase_limit_1 = get_phase_limit(Partical_peak_1, phase_thresholds)
            Phase_limit_2 = get_phase_limit(Partical_peak_2, phase_thresholds)
            Phase_limit_3 = get_phase_limit(Partical_peak_3, phase_thresholds)
                

            
            Phase_1_surface_volume = Surface_mesh_histogram.iloc[i,Background_peak_pos+1:int(Phase_limit_1*Gradient_ratio)].sum()
            Phase_2_surface_volume = Surface_mesh_histogram.iloc[i,int(Phase_limit_1*Gradient_ratio):int(Phase_limit_2*Gradient_ratio)].sum()
            Phase_3_surface_volume = Surface_mesh_histogram.iloc[i,int(Phase_limit_2*Gradient_ratio):int(Phase_limit_3*Gradient_ratio)].sum()
            Phase_4_surface_volume = Surface_mesh_histogram.iloc[i,int(Phase_limit_3*Gradient_ratio):].sum()
            
            Surface_ratio_1 = Phase_1_surface_volume/(Phase_1_surface_volume+Phase_2_surface_volume+Phase_3_surface_volume+Phase_4_surface_volume)
            Surface_ratio_2 = Phase_2_surface_volume/(Phase_1_surface_volume+Phase_2_surface_volume+Phase_3_surface_volume+Phase_4_surface_volume)
            Surface_ratio_3 = Phase_3_surface_volume/(Phase_1_surface_volume+Phase_2_surface_volume+Phase_3_surface_volume+Phase_4_surface_volume)
            Surface_ratio_4 = Phase_4_surface_volume/(Phase_1_surface_volume+Phase_2_surface_volume+Phase_3_surface_volume+Phase_4_surface_volume)
            
            Surface_volume_phase_1_append.append([Phase_1_surface_volume])
            Surface_volume_phase_2_append.append([Phase_2_surface_volume])
            Surface_volume_phase_3_append.append([Phase_3_surface_volume])    
            Surface_volume_phase_4_append.append([Phase_4_surface_volume]) 
            
            
            Outer_volume_full_phase_4 = Outer_volume_histograms.iloc[i,Partical_peak_4:].sum()

            def calculate_phase_quantification_array(particle_peak_pos):
                no_of_voxels_towards_background = Outer_volume_histograms.iloc[i,Background_peak_pos+1:particle_peak_pos]
                # Calculate the multiples array
                multiples_towards_background = np.linspace(0, 1, (particle_peak_pos) - Background_peak_pos+1)
                multiples_towards_background = multiples_towards_background[1:len(no_of_voxels_towards_background)+1]
                # Calculate and return the quantification result
                quantification_outer_phase = no_of_voxels_towards_background * multiples_towards_background
                return quantification_outer_phase

            Quantification_Outer_phase_1_array = calculate_phase_quantification_array(Partical_peak_1)
            Quantification_Outer_phase_2_array = calculate_phase_quantification_array(Partical_peak_2)
            Quantification_Outer_phase_2_array = Quantification_Outer_phase_2_array[Partical_peak_1-Background_peak_pos:]
            Quantification_Outer_phase_3_array = calculate_phase_quantification_array(Partical_peak_3)
            Quantification_Outer_phase_3_array = Quantification_Outer_phase_3_array[Partical_peak_2-Background_peak_pos:]
            Quantification_Outer_phase_4_array = calculate_phase_quantification_array(Partical_peak_4)
            Quantification_Outer_phase_4_array = Quantification_Outer_phase_4_array[Partical_peak_3-Background_peak_pos: Partical_peak_4 - Background_peak_pos]
            
            PVE_adjusted_volume = (Outer_volume_full_phase_4 + Quantification_Outer_phase_1_array.sum()+Quantification_Outer_phase_2_array.sum()
                                   +Quantification_Outer_phase_3_array.sum()+Quantification_Outer_phase_4_array.sum())
            
            print("s",Outer_volume_histograms.iloc[i,Background_peak_pos:].sum()-PVE_adjusted_volume)
            
            Quantification_Outer_phase_1_volume = PVE_adjusted_volume*Surface_ratio_1
            Quantification_Outer_phase_2_volume = PVE_adjusted_volume*Surface_ratio_2
            Quantification_Outer_phase_3_volume = PVE_adjusted_volume*Surface_ratio_3
            Quantification_Outer_phase_4_volume = PVE_adjusted_volume*Surface_ratio_4
            
            
            Quantification_Outer_phase_1_append.append([Quantification_Outer_phase_1_volume])
            Quantification_Outer_phase_2_append.append([Quantification_Outer_phase_2_volume])
            Quantification_Outer_phase_3_append.append([Quantification_Outer_phase_3_volume])
            Quantification_Outer_phase_4_append.append([Quantification_Outer_phase_4_volume])
            
            
            Sum_phase_1 = Sum_phase_1.sum() + Quantification_Outer_phase_1_volume
            Sum_phase_2 = Sum_phase_2.sum() + Quantification_Outer_phase_2_volume
            Sum_phase_3 = Sum_phase_3.sum() + Quantification_Outer_phase_3_volume
            Sum_phase_4 = Sum_phase_4.sum() + Quantification_Outer_phase_4_volume
            
            Quantification_all_4_phases_1.append([Sum_phase_1])
            Quantification_all_4_phases_2.append([Sum_phase_2])
            Quantification_all_4_phases_3.append([Sum_phase_3])
            Quantification_all_4_phases_4.append([Sum_phase_4])

        i = i+1

    Quantification_all_4_phases_1 = pd.DataFrame(Quantification_all_4_phases_1,columns = ['total_quantification_phase_1'])
    Quantification_all_4_phases_2 = pd.DataFrame(Quantification_all_4_phases_2,columns = ['total_quantification_phase_2'])
    Quantification_all_4_phases_3 = pd.DataFrame(Quantification_all_4_phases_3,columns = ['total_quantification_phase_3'])
    Quantification_all_4_phases_4 = pd.DataFrame(Quantification_all_4_phases_4,columns = ['total_quantification_phase_4'])
    
    
    Quantification_Outer_phase_1 = pd.DataFrame(Quantification_Outer_phase_1_append,columns = ['Outer_quantification_phase_1'])
    Quantification_Outer_phase_2 = pd.DataFrame(Quantification_Outer_phase_2_append,columns = ['Outer_quantification_phase_2'])
    Quantification_Outer_phase_3 = pd.DataFrame(Quantification_Outer_phase_3_append,columns = ['Outer_quantification_phase_3'])
    Quantification_Outer_phase_4 = pd.DataFrame(Quantification_Outer_phase_4_append,columns = ['Outer_quantification_phase_4'])
    

    Index_4_phase = pd.DataFrame(Index_4_phase,columns = ['Label'])

    Peaks_1_phase = pd.DataFrame(Peaks_1_phase,columns = ['Peak_1'])
    Peaks_2_phase = pd.DataFrame(Peaks_2_phase,columns = ['Peak_2'])
    Peaks_3_phase = pd.DataFrame(Peaks_3_phase,columns = ['Peak_3'])
    Peaks_4_phase = pd.DataFrame(Peaks_4_phase,columns = ['Peak_4'])
    
    Surface_volume_phase_1 = pd.DataFrame(Surface_volume_phase_1_append,columns = ['Surface_volume_phase_1'])
    Surface_volume_phase_1.index = Index_4_phase['Label']
    Surface_volume_phase_2 = pd.DataFrame(Surface_volume_phase_2_append,columns = ['Surface_volume_phase_2'])
    Surface_volume_phase_2.index = Index_4_phase['Label']
    Surface_volume_phase_3 = pd.DataFrame(Surface_volume_phase_3_append,columns = ['Surface_volume_phase_3'])
    Surface_volume_phase_3.index = Index_4_phase['Label']
    Surface_volume_phase_4 = pd.DataFrame(Surface_volume_phase_4_append,columns = ['Surface_volume_phase_4'])
    Surface_volume_phase_4.index = Index_4_phase['Label']

    Quantification_4_phases = pd.concat([Index_4_phase,Quantification_all_4_phases_1,Quantification_all_4_phases_2,Quantification_all_4_phases_3,
                                         Quantification_all_4_phases_4, Peaks_1_phase,Peaks_2_phase,Peaks_3_phase,Peaks_4_phase,
                                         Quantification_Outer_phase_1,Quantification_Outer_phase_2,
                                         Quantification_Outer_phase_3,Quantification_Outer_phase_4],axis = 1)
    cols = ['Peak_1', 'Peak_2', 'Peak_3', 'Peak_4', 'Peak_5', 'Peak_6', 'Peak_7', 'Peak_8', 'Peak_9', 'Peak_10', 'Peak_11', 'Peak_12']

    thresholds = phase_thresholds
    
    Quantification_4_phase_sorted = pd.DataFrame(columns= cols + [f'Phase_{i}_quantification' for i in range(1, 6)])
    Quantification_4_phase_sorted_1 = Quantification_4_phase_sorted.copy()
    Quantification_4_phase_sorted_2 = Quantification_4_phase_sorted.copy()
    Quantification_4_phase_sorted_3 = Quantification_4_phase_sorted.copy()
    
    for i in range(1, len(thresholds)):
        mask = (Peaks_1_phase['Peak_1'] > thresholds[i - 1]) & (Peaks_1_phase['Peak_1'] <= thresholds[i])
        Quantification_4_phase_sorted[f'Peak_{i}'] = np.where(mask, Peaks_1_phase['Peak_1'], 0)
        Quantification_4_phase_sorted[f'Phase_{i}_quantification'] = np.where(mask, Quantification_4_phases['total_quantification_phase_1'], 0)
        Quantification_4_phase_sorted[f'Phase_{i}_surface_quantification'] = np.where(mask, Surface_volume_phase_1['Surface_volume_phase_1'], 0)
        Quantification_4_phase_sorted[f'Phase_{i}_outer_quantification'] = np.where(mask, Quantification_Outer_phase_1['Outer_quantification_phase_1'], 0)
    
    for i in range(1, len(thresholds)):
        mask = (Peaks_2_phase['Peak_2'] > thresholds[i-1]) & (Peaks_2_phase['Peak_2'] <= thresholds[i])
        Quantification_4_phase_sorted_1[f'Peak_{i}'] = np.where(mask, Peaks_2_phase['Peak_2'], 0)
        Quantification_4_phase_sorted_1[f'Phase_{i}_quantification'] = np.where(mask, Quantification_4_phases['total_quantification_phase_2'], 0)
        Quantification_4_phase_sorted_1[f'Phase_{i}_surface_quantification'] = np.where(mask, Surface_volume_phase_2['Surface_volume_phase_2'], 0)
        Quantification_4_phase_sorted_1[f'Phase_{i}_outer_quantification'] = np.where(mask, Quantification_Outer_phase_2['Outer_quantification_phase_2'], 0)
    
    for i in range(1, len(thresholds)):
        mask = (Peaks_3_phase['Peak_3'] > thresholds[i-1]) & (Peaks_3_phase['Peak_3'] <= thresholds[i])
        Quantification_4_phase_sorted_2[f'Peak_{i}'] = np.where(mask, Peaks_3_phase['Peak_3'], 0)
        Quantification_4_phase_sorted_2[f'Phase_{i}_quantification'] = np.where(mask, Quantification_4_phases['total_quantification_phase_3'], 0)
        Quantification_4_phase_sorted_2[f'Phase_{i}_surface_quantification'] = np.where(mask, Surface_volume_phase_3['Surface_volume_phase_3'], 0)
        Quantification_4_phase_sorted_2[f'Phase_{i}_outer_quantification'] = np.where(mask, Quantification_Outer_phase_3['Outer_quantification_phase_3'], 0)
        
    for i in range(1, len(thresholds)):
        mask = (Peaks_4_phase['Peak_4'] > thresholds[i-1]) & (Peaks_4_phase['Peak_4'] <= thresholds[i])
        Quantification_4_phase_sorted_3[f'Peak_{i}'] = np.where(mask, Peaks_4_phase['Peak_4'], 0)
        Quantification_4_phase_sorted_3[f'Phase_{i}_quantification'] = np.where(mask, Quantification_4_phases['total_quantification_phase_4'], 0)
        Quantification_4_phase_sorted_3[f'Phase_{i}_surface_quantification'] = np.where(mask, Surface_volume_phase_4['Surface_volume_phase_4'], 0)
        Quantification_4_phase_sorted_3[f'Phase_{i}_outer_quantification'] = np.where(mask, Quantification_Outer_phase_4['Outer_quantification_phase_4'], 0)
    
    Quantification_4_phase_sorted = Quantification_4_phase_sorted.mask(Quantification_4_phase_sorted == 0, Quantification_4_phase_sorted_1)
    Quantification_4_phase_sorted = Quantification_4_phase_sorted.mask(Quantification_4_phase_sorted == 0, Quantification_4_phase_sorted_2)
    Quantification_4_phase_sorted = Quantification_4_phase_sorted.mask(Quantification_4_phase_sorted == 0, Quantification_4_phase_sorted_3)
    Quantification_4_phase_sorted.index = Quantification_4_phases['Label']

    Quantification_4_phase_sorted[cols] = Quantification_4_phase_sorted[cols].replace(0, Background_peak_pos)
    
    return Quantification_4_phase_sorted



def Quinary_Particles(Peaks, Bulk_histograms, Inner_volume_histograms, Outer_volume_histograms, Surface_mesh_histogram, Gradient, Gradient_threshold):
    
    """
    Quantifies particles that contain exactly 5 peaks (i.e., five distinct mineral phases).
    Computes the inner volume, outer volume (with PVE correction), and surface contributions
    for each phase and returns a threshold-sorted DataFrame.

    Parameters:
        Peaks (DataFrame): Peak data for each particle (must include background and phase thresholds).
        Bulk_histograms (DataFrame): Histogram of intensities across all voxels per particle.
        Inner_volume_histograms (DataFrame): Histogram within eroded particle cores.
        Outer_volume_histograms (DataFrame): Histogram of outer (border) voxels.
        Surface_mesh_histogram (DataFrame): Histogram from mesh-based surface voxel map.
        Gradient (DataFrame): Gradient-derived surface sharpness per particle.
        Gradient_threshold (float): Minimum ratio for scaling surface contributions.

    Returns:
        DataFrame: Quantified contributions from three phases, sorted by peak threshold bins.
    """
    Peaks = standardize_index(Peaks)
    Gradient = standardize_index(Gradient)
    
    array, Background_peak_pos, phase_thresholds = extract_peaks_and_thresholds(Peaks)
    
    Quantification_all_5_phases_1 = []
    Quantification_all_5_phases_2 = []
    Quantification_all_5_phases_3 = []
    Quantification_all_5_phases_4 = []
    Quantification_all_5_phases_5 = []

    Peaks_1_phase = []
    Peaks_2_phase = []
    Peaks_3_phase = []
    Peaks_4_phase = []
    Peaks_5_phase = []

    Index_5_phase = []
    
    Surface_volume_phase_1_append = []
    Surface_volume_phase_2_append = []
    Surface_volume_phase_3_append = []
    Surface_volume_phase_4_append = []
    Surface_volume_phase_5_append = []
    
    Quantification_Outer_phase_1_append = []
    Quantification_Outer_phase_2_append = []
    Quantification_Outer_phase_3_append = []
    Quantification_Outer_phase_4_append = []
    Quantification_Outer_phase_5_append = []

    i=0     
    for index,row in Bulk_histograms.iterrows():
        Peaks = (array.iloc[[i]].values)
        if (np.count_nonzero(Peaks > Background_peak_pos) == 5) and i >-1:
            Partical_peak = Peaks[Peaks >Background_peak_pos]
        
            Partical_peak_1 = Partical_peak.flat[0]
            Partical_peak_1 = int(float(Partical_peak_1))
            Partical_peak_2 = Partical_peak.flat[1]
            Partical_peak_2 = int(float(Partical_peak_2))
            Partical_peak_3 = Partical_peak.flat[2]
            Partical_peak_3 = int(float(Partical_peak_3))
            Partical_peak_4 = Partical_peak.flat[3]
            Partical_peak_4 = int(float(Partical_peak_4))
            Partical_peak_5 = Partical_peak.flat[4]
            Partical_peak_5 = int(float(Partical_peak_5))
        
            #Taking the sum of ith row from phase 1 minimum threshold greyscale value till Phase1_max_limit
            Sum_phase_1 = Inner_volume_histograms.iloc[i,Background_peak_pos+1:int((Partical_peak_1+Partical_peak_2)/2)].sum()
        
            Sum_phase_2 = Inner_volume_histograms.iloc[i,int((Partical_peak_1+Partical_peak_2)/2):int((Partical_peak_2+Partical_peak_3)/2)].sum()

            Sum_phase_3 = Inner_volume_histograms.iloc[i,int((Partical_peak_2+Partical_peak_3)/2):int((Partical_peak_3+Partical_peak_4)/2)].sum()

            Sum_phase_4 = Inner_volume_histograms.iloc[i,int((Partical_peak_3+Partical_peak_4)/2):int((Partical_peak_4+Partical_peak_5)/2)].sum()
            
            Sum_phase_5 = Inner_volume_histograms.iloc[i,:int((Partical_peak_4+Partical_peak_5)/2):].sum()
        
            Index_5_phase.append([index])
        
            Peaks_1_phase.append([Partical_peak_1])
            Peaks_2_phase.append([Partical_peak_2])
            Peaks_3_phase.append([Partical_peak_3])
            Peaks_4_phase.append([Partical_peak_4])
            Peaks_5_phase.append([Partical_peak_5])
            
            
            Gradient_ratio = Gradient['Gradient_3'].iloc[i]
            if Gradient_ratio < Gradient_threshold:
                Gradient_ratio = Gradient_threshold
            
            def get_phase_limit(peak, phase_thresholds):
                for i in range(1, len(phase_thresholds)):
                    if peak < phase_thresholds[i]:
                        return phase_thresholds[i]
                return phase_thresholds[-1]
            
            Phase_limit_1 = get_phase_limit(Partical_peak_1, phase_thresholds)
            Phase_limit_2 = get_phase_limit(Partical_peak_2, phase_thresholds)
            Phase_limit_3 = get_phase_limit(Partical_peak_3, phase_thresholds)
            Phase_limit_4 = get_phase_limit(Partical_peak_4, phase_thresholds)
                
            
            Phase_1_surface_volume = Surface_mesh_histogram.iloc[i,Background_peak_pos+1:int(Phase_limit_1*Gradient_ratio)].sum()
            Phase_2_surface_volume = Surface_mesh_histogram.iloc[i,int(Phase_limit_1*Gradient_ratio):int(Phase_limit_2*Gradient_ratio)].sum()
            Phase_3_surface_volume = Surface_mesh_histogram.iloc[i,int(Phase_limit_2*Gradient_ratio):int(Phase_limit_3*Gradient_ratio)].sum()
            Phase_4_surface_volume = Surface_mesh_histogram.iloc[i,int(Phase_limit_3*Gradient_ratio):int(Phase_limit_4*Gradient_ratio)].sum()
            Phase_5_surface_volume = Surface_mesh_histogram.iloc[i,int(Phase_limit_4*Gradient_ratio):].sum()
            
            Surface_ratio_1 = Phase_1_surface_volume/(Phase_1_surface_volume+Phase_2_surface_volume+Phase_3_surface_volume+Phase_4_surface_volume+Phase_5_surface_volume)
            Surface_ratio_2 = Phase_2_surface_volume/(Phase_1_surface_volume+Phase_2_surface_volume+Phase_3_surface_volume+Phase_4_surface_volume+Phase_5_surface_volume)
            Surface_ratio_3 = Phase_3_surface_volume/(Phase_1_surface_volume+Phase_2_surface_volume+Phase_3_surface_volume+Phase_4_surface_volume+Phase_5_surface_volume)
            Surface_ratio_4 = Phase_4_surface_volume/(Phase_1_surface_volume+Phase_2_surface_volume+Phase_3_surface_volume+Phase_4_surface_volume+Phase_5_surface_volume)
            Surface_ratio_5 = Phase_5_surface_volume/(Phase_1_surface_volume+Phase_2_surface_volume+Phase_3_surface_volume+Phase_4_surface_volume+Phase_5_surface_volume)
            
            Surface_volume_phase_1_append.append([Phase_1_surface_volume])
            Surface_volume_phase_2_append.append([Phase_2_surface_volume])
            Surface_volume_phase_3_append.append([Phase_3_surface_volume])    
            Surface_volume_phase_4_append.append([Phase_4_surface_volume]) 
            Surface_volume_phase_5_append.append([Phase_5_surface_volume]) 
            
            
            Outer_volume_full_phase_5 = Outer_volume_histograms.iloc[i,Partical_peak_4:].sum()
            

            Outer_volume_full_phase_4 = Outer_volume_histograms.iloc[i,Partical_peak_4:].sum()

            def calculate_phase_quantification_array(particle_peak_pos):
                no_of_voxels_towards_background = Outer_volume_histograms.iloc[i,Background_peak_pos+1:particle_peak_pos]
                # Calculate the multiples array
                multiples_towards_background = np.linspace(0, 1, (particle_peak_pos) - Background_peak_pos+1)
                multiples_towards_background = multiples_towards_background[1:len(no_of_voxels_towards_background)+1]
                # Calculate and return the quantification result
                quantification_outer_phase = no_of_voxels_towards_background * multiples_towards_background
                return quantification_outer_phase

            Quantification_Outer_phase_1_array = calculate_phase_quantification_array(Partical_peak_1)
            Quantification_Outer_phase_2_array = calculate_phase_quantification_array(Partical_peak_2)
            Quantification_Outer_phase_2_array = Quantification_Outer_phase_2_array[Partical_peak_1-Background_peak_pos:]
            Quantification_Outer_phase_3_array = calculate_phase_quantification_array(Partical_peak_3)
            Quantification_Outer_phase_3_array = Quantification_Outer_phase_3_array[Partical_peak_2-Background_peak_pos:]
            Quantification_Outer_phase_4_array = calculate_phase_quantification_array(Partical_peak_4)
            Quantification_Outer_phase_4_array = Quantification_Outer_phase_4_array[Partical_peak_3-Background_peak_pos: Partical_peak_4 - Background_peak_pos]
            
            PVE_adjusted_volume = (Outer_volume_full_phase_4 + Quantification_Outer_phase_1_array.sum()+Quantification_Outer_phase_2_array.sum()
                                   +Quantification_Outer_phase_3_array.sum()+Quantification_Outer_phase_4_array.sum())
            
            Quantification_Outer_phase_1_array = calculate_phase_quantification_array(Partical_peak_1)
            Quantification_Outer_phase_2_array = calculate_phase_quantification_array(Partical_peak_2)
            Quantification_Outer_phase_2_array = Quantification_Outer_phase_2_array[Partical_peak_1-Background_peak_pos:]
            Quantification_Outer_phase_3_array = calculate_phase_quantification_array(Partical_peak_3)
            Quantification_Outer_phase_3_array = Quantification_Outer_phase_3_array[Partical_peak_2-Background_peak_pos:]
            Quantification_Outer_phase_4_array = calculate_phase_quantification_array(Partical_peak_4)
            Quantification_Outer_phase_4_array = Quantification_Outer_phase_4_array[Partical_peak_3-Background_peak_pos:]
            Quantification_Outer_phase_5_array = calculate_phase_quantification_array(Partical_peak_5)
            Quantification_Outer_phase_5_array = Quantification_Outer_phase_5_array[Partical_peak_4-Background_peak_pos:: Partical_peak_5 - Background_peak_pos]
            
            PVE_adjusted_volume = (Outer_volume_full_phase_5 + Quantification_Outer_phase_1_array.sum()+Quantification_Outer_phase_2_array.sum()
                                   +Quantification_Outer_phase_3_array.sum()+Quantification_Outer_phase_4_array.sum()+
                                   Quantification_Outer_phase_5_array.sum())
            
            Quantification_Outer_phase_1_volume = PVE_adjusted_volume*Surface_ratio_1
            Quantification_Outer_phase_2_volume = PVE_adjusted_volume*Surface_ratio_2
            Quantification_Outer_phase_3_volume = PVE_adjusted_volume*Surface_ratio_3
            Quantification_Outer_phase_4_volume = PVE_adjusted_volume*Surface_ratio_4
            Quantification_Outer_phase_5_volume = PVE_adjusted_volume*Surface_ratio_5
            
            Quantification_Outer_phase_1_append.append([Quantification_Outer_phase_1_volume])
            Quantification_Outer_phase_2_append.append([Quantification_Outer_phase_2_volume])
            Quantification_Outer_phase_3_append.append([Quantification_Outer_phase_3_volume])
            Quantification_Outer_phase_4_append.append([Quantification_Outer_phase_4_volume])
            Quantification_Outer_phase_5_append.append([Quantification_Outer_phase_5_volume])
            
            Sum_phase_1 = Sum_phase_1.sum() + Quantification_Outer_phase_1_volume
            Sum_phase_2 = Sum_phase_2.sum() + Quantification_Outer_phase_2_volume
            Sum_phase_3 = Sum_phase_3.sum() + Quantification_Outer_phase_3_volume
            Sum_phase_4 = Sum_phase_4.sum() + Quantification_Outer_phase_4_volume
            Sum_phase_5 = Sum_phase_5.sum() + Quantification_Outer_phase_5_volume
            
            Quantification_all_5_phases_1.append([Sum_phase_1])
            Quantification_all_5_phases_2.append([Sum_phase_2])
            Quantification_all_5_phases_3.append([Sum_phase_3])
            Quantification_all_5_phases_4.append([Sum_phase_4])
            Quantification_all_5_phases_5.append([Sum_phase_5])

        i = i+1
    
    #Creating Quantification_all of quantification of voxels which have 100% phase 1
    Quantification_all_5_phases_1 = pd.DataFrame(Quantification_all_5_phases_1,columns = ['total_quantification_phase_1'])
    Quantification_all_5_phases_2 = pd.DataFrame(Quantification_all_5_phases_2,columns = ['total_quantification_phase_2'])
    Quantification_all_5_phases_3 = pd.DataFrame(Quantification_all_5_phases_3,columns = ['total_quantification_phase_3'])
    Quantification_all_5_phases_4 = pd.DataFrame(Quantification_all_5_phases_4,columns = ['total_quantification_phase_4'])
    Quantification_all_5_phases_5 = pd.DataFrame(Quantification_all_5_phases_5,columns = ['total_quantification_phase_5'])
    
    
    Quantification_Outer_phase_1 = pd.DataFrame(Quantification_Outer_phase_1_append,columns = ['Outer_quantification_phase_1'])
    Quantification_Outer_phase_2 = pd.DataFrame(Quantification_Outer_phase_2_append,columns = ['Outer_quantification_phase_2'])
    Quantification_Outer_phase_3 = pd.DataFrame(Quantification_Outer_phase_3_append,columns = ['Outer_quantification_phase_3'])
    Quantification_Outer_phase_4 = pd.DataFrame(Quantification_Outer_phase_4_append,columns = ['Outer_quantification_phase_4'])
    Quantification_Outer_phase_5 = pd.DataFrame(Quantification_Outer_phase_5_append,columns = ['Outer_quantification_phase_5'])

    Index_5_phase = pd.DataFrame(Index_5_phase,columns = ['Label'])

    Peaks_1_phase = pd.DataFrame(Peaks_1_phase,columns = ['Peak_1'])
    Peaks_2_phase = pd.DataFrame(Peaks_2_phase,columns = ['Peak_2'])
    Peaks_3_phase = pd.DataFrame(Peaks_3_phase,columns = ['Peak_3'])
    Peaks_4_phase = pd.DataFrame(Peaks_4_phase,columns = ['Peak_4'])
    Peaks_5_phase = pd.DataFrame(Peaks_5_phase,columns = ['Peak_5'])
    
    Surface_volume_phase_1 = pd.DataFrame(Surface_volume_phase_1_append,columns = ['Surface_volume_phase_1'])
    Surface_volume_phase_1.index = Index_5_phase['Label']
    Surface_volume_phase_2 = pd.DataFrame(Surface_volume_phase_2_append,columns = ['Surface_volume_phase_2'])
    Surface_volume_phase_2.index = Index_5_phase['Label']
    Surface_volume_phase_3 = pd.DataFrame(Surface_volume_phase_3_append,columns = ['Surface_volume_phase_3'])
    Surface_volume_phase_3.index = Index_5_phase['Label']
    Surface_volume_phase_4 = pd.DataFrame(Surface_volume_phase_4_append,columns = ['Surface_volume_phase_4'])
    Surface_volume_phase_4.index = Index_5_phase['Label']
    Surface_volume_phase_5 = pd.DataFrame(Surface_volume_phase_5_append,columns = ['Surface_volume_phase_5'])
    Surface_volume_phase_5.index = Index_5_phase['Label']

    Quantification_5_phases = pd.concat([Index_5_phase,Quantification_all_5_phases_1,Quantification_all_5_phases_2,Quantification_all_5_phases_3,
                                         Quantification_all_5_phases_4,Quantification_all_5_phases_5, Peaks_1_phase,Peaks_2_phase,Peaks_3_phase,
                                         Peaks_4_phase,Peaks_5_phase,Quantification_Outer_phase_1,Quantification_Outer_phase_2,
                                         Quantification_Outer_phase_3,Quantification_Outer_phase_4,Quantification_Outer_phase_5],axis = 1)

    cols = ['Peak_1', 'Peak_2', 'Peak_3', 'Peak_4', 'Peak_5', 'Peak_6', 'Peak_7', 'Peak_8', 'Peak_9', 'Peak_10', 'Peak_11', 'Peak_12']

    thresholds = phase_thresholds

    Quantification_5_phase_sorted = pd.DataFrame(columns= cols + [f'Phase_{i}_quantification' for i in range(1, 6)])
    Quantification_5_phase_sorted_1 = Quantification_5_phase_sorted.copy()
    Quantification_5_phase_sorted_2 = Quantification_5_phase_sorted.copy()
    Quantification_5_phase_sorted_3 = Quantification_5_phase_sorted.copy()
    Quantification_5_phase_sorted_4 = Quantification_5_phase_sorted.copy()

    for i in range(1,len(thresholds)):
        mask = (Peaks_1_phase['Peak_1'] > thresholds[i - 1]) & (Peaks_1_phase['Peak_1'] <= thresholds[i])
        Quantification_5_phase_sorted[f'Peak_{i}'] = np.where(mask, Peaks_1_phase['Peak_1'], 0)
        Quantification_5_phase_sorted[f'Phase_{i}_quantification'] = np.where(mask, Quantification_5_phases['total_quantification_phase_1'], 0)
        Quantification_5_phase_sorted[f'Phase_{i}_surface_quantification'] = np.where(mask, Surface_volume_phase_1['Surface_volume_phase_1'], 0)
        Quantification_5_phase_sorted[f'Phase_{i}_outer_quantification'] = np.where(mask, Quantification_Outer_phase_1['Outer_quantification_phase_1'], 0)
    
    for i in range(1, len(thresholds)):
        mask = (Peaks_2_phase['Peak_2'] > thresholds[i-1]) & (Peaks_2_phase['Peak_2'] <= thresholds[i])
        Quantification_5_phase_sorted_1[f'Peak_{i}'] = np.where(mask, Peaks_2_phase['Peak_2'], 0)
        Quantification_5_phase_sorted_1[f'Phase_{i}_quantification'] = np.where(mask, Quantification_5_phases['total_quantification_phase_2'], 0)
        Quantification_5_phase_sorted_1[f'Phase_{i}_surface_quantification'] = np.where(mask, Surface_volume_phase_2['Surface_volume_phase_2'], 0)
        Quantification_5_phase_sorted_1[f'Phase_{i}_outer_quantification'] = np.where(mask, Quantification_Outer_phase_2['Outer_quantification_phase_2'], 0)
    
    for i in range(1, len(thresholds)):
        mask = (Peaks_3_phase['Peak_3'] > thresholds[i-1]) & (Peaks_3_phase['Peak_3'] <= thresholds[i])
        Quantification_5_phase_sorted_2[f'Peak_{i}'] = np.where(mask, Peaks_3_phase['Peak_3'], 0)
        Quantification_5_phase_sorted_2[f'Phase_{i}_quantification'] = np.where(mask, Quantification_5_phases['total_quantification_phase_3'], 0)
        Quantification_5_phase_sorted_2[f'Phase_{i}_surface_quantification'] = np.where(mask, Surface_volume_phase_3['Surface_volume_phase_3'], 0)
        Quantification_5_phase_sorted_2[f'Phase_{i}_outer_quantification'] = np.where(mask, Quantification_Outer_phase_3['Outer_quantification_phase_3'], 0)
        
    for i in range(1, len(thresholds)):
        mask = (Peaks_4_phase['Peak_4'] > thresholds[i-1]) & (Peaks_4_phase['Peak_4'] <= thresholds[i])
        Quantification_5_phase_sorted_3[f'Peak_{i}'] = np.where(mask, Peaks_4_phase['Peak_4'], 0)
        Quantification_5_phase_sorted_3[f'Phase_{i}_quantification'] = np.where(mask, Quantification_5_phases['total_quantification_phase_4'], 0)
        Quantification_5_phase_sorted_3[f'Phase_{i}_surface_quantification'] = np.where(mask, Surface_volume_phase_4['Surface_volume_phase_4'], 0)
        Quantification_5_phase_sorted_3[f'Phase_{i}_outer_quantification'] = np.where(mask, Quantification_Outer_phase_4['Outer_quantification_phase_4'], 0)
        
    for i in range(1, len(thresholds)):
        mask = (Peaks_5_phase['Peak_5'] > thresholds[i-1]) & (Peaks_5_phase['Peak_5'] <= thresholds[i])
        Quantification_5_phase_sorted_4[f'Peak_{i}'] = np.where(mask, Peaks_5_phase['Peak_5'], 0)
        Quantification_5_phase_sorted_4[f'Phase_{i}_quantification'] = np.where(mask, Quantification_5_phases['total_quantification_phase_5'], 0)
        Quantification_5_phase_sorted_4[f'Phase_{i}_surface_quantification'] = np.where(mask, Surface_volume_phase_5['Surface_volume_phase_5'], 0)
        Quantification_5_phase_sorted_4[f'Phase_{i}_outer_quantification'] = np.where(mask, Quantification_Outer_phase_5['Outer_quantification_phase_5'], 0)
    
    Quantification_5_phase_sorted = Quantification_5_phase_sorted.mask(Quantification_5_phase_sorted == 0, Quantification_5_phase_sorted_1)
    Quantification_5_phase_sorted = Quantification_5_phase_sorted.mask(Quantification_5_phase_sorted == 0, Quantification_5_phase_sorted_2)
    Quantification_5_phase_sorted = Quantification_5_phase_sorted.mask(Quantification_5_phase_sorted == 0, Quantification_5_phase_sorted_3)
    Quantification_5_phase_sorted = Quantification_5_phase_sorted.mask(Quantification_5_phase_sorted == 0, Quantification_5_phase_sorted_4)
    Quantification_5_phase_sorted.index = Quantification_5_phases['Label']
    
    Quantification_5_phase_sorted[cols] = Quantification_5_phase_sorted[cols].replace(0, Background_peak_pos)
    
    return Quantification_5_phase_sorted

def Concatenate(*dfs):
    import re
    """
    Reorders the columns of each DataFrame based on numeric values in column names,
    then concatenates them row-wise.

    Parameters:
        *dfs: Variable number of pandas DataFrames

    Returns:
        A single concatenated DataFrame with columns sorted by number in the name.
    """
    
    def extract_number(col):
        match = re.search(r'\d+', col)
        return int(match.group()) if match else float('inf')

    # Reorder each DataFrame's columns
    reordered_dfs = []
    for df in dfs:
        sorted_cols = sorted(df.columns, key=extract_number)
        reordered_dfs.append(df[sorted_cols])

    # Concatenate all reordered DataFrames
    combined = pd.concat(reordered_dfs, axis=0, ignore_index=False)
    combined = combined.sort_index()
    
    combined = combined.fillna(0)

    return combined

def calculate_bootstrapping_error_bulk(dataset, fractions):
    
    """
    This function performs bootstrapping to estimate uncertainty in phase quantification.
    
    It resamples the input dataset 'fractions' times with replacement, splits it into
    'fractions' number of sub-datasets, and calculates the mineral composition for each.
    
    The function then computes the 2.5th and 97.5th percentiles to estimate the 95% confidence interval
    for each phase, along with the actual percentage from the original dataset.
    
    Parameters:
        dataset (pd.DataFrame): DataFrame containing columns 'Phase_1_quantification' to 'Phase_9_quantification'.
        fractions (int): Number of resamples/subsets to create.
    
    Returns:
        pd.DataFrame: A DataFrame containing the min, max, and actual percentages for each phase.
    """
    import numpy as np
    # Define the phase column names
    phase_cols = [f'Phase_{i}_quantification' for i in range(1, 12)]

    # Resample the dataset
    resampled_dataset = dataset.sample(frac=fractions, replace=True, random_state=42).copy()

    # Calculate total sum column for normalization
    resampled_dataset['Sum'] = resampled_dataset[phase_cols].sum(axis=1)

    # Determine chunk size
    chunk_size = len(resampled_dataset) / fractions

    # Initialize dictionary to collect mean values
    phase_means = {col: [] for col in phase_cols}

    for i in range(fractions):
        chunk = resampled_dataset.iloc[int(i * chunk_size):int((i + 1) * chunk_size)]
        for col in phase_cols:
            mean_val = chunk[col].sum() * 100 / chunk['Sum'].sum()
            phase_means[col].append(mean_val)

    # Calculate percentiles and actual values
    bootstrapping_error = {
        'Class': [f'Class {i}' for i in range(1, 12)],
        'Max': [round(np.percentile(phase_means[col], 97.5), 5) for col in phase_cols],
        'Min': [round(np.percentile(phase_means[col], 2.5), 5) for col in phase_cols],
        'Actual_Percentage': [round(dataset[col].sum() * 100 / dataset[phase_cols].sum().sum(), 5) for col in phase_cols]
    }

    return pd.DataFrame(bootstrapping_error)


def calculate_bootstrapping_error_surface(dataset, fraction):
    
    """
    Estimate the uncertainty in surface phase quantification using bootstrapping.
    
    This function performs bootstrapping by:
    - Creating a large resampled version of the dataset by sampling with replacement.
    - Splitting the resampled data into 'fraction' number of chunks.
    - Recomputing phase-wise surface quantification percentages for each chunk.
    - Calculating the 2.5th and 97.5th percentiles to provide a 95% confidence interval.
    - Returning these bounds along with the actual percentage based on the original dataset.
    
    Parameters:
        dataset (pd.DataFrame): The input DataFrame containing surface quantification columns
                                named as 'Phase_1_surface_quantification', ..., 'Phase_9_surface_quantification'.
        fraction (int): Number of bootstrapped samples and sub-chunks. Default is 1000.
    
    Returns:
        pd.DataFrame: A summary DataFrame with:
            - Class: phase class name (e.g., Class 1, Class 2, ...).
            - Max: 97.5th percentile of bootstrapped percentage estimates.
            - Min: 2.5th percentile of bootstrapped percentage estimates.
            - Actual_Percentage: Actual percentage from the input dataset.
    """
    import numpy as np
    phase_cols = [f'Phase_{i}_surface_quantification' for i in range(1, 12)]

    resampled_dataset = dataset.sample(frac=fraction, replace=True, random_state=42).copy()
    resampled_dataset['Sum'] = resampled_dataset[phase_cols].sum(axis=1)
    chunk_size = len(resampled_dataset) / fraction
    phase_means = {col: [] for col in phase_cols}

    for i in range(fraction):
        chunk = resampled_dataset.iloc[int(i * chunk_size):int((i + 1) * chunk_size)]
        for col in phase_cols:
            mean_val = chunk[col].sum() * 100 / chunk['Sum'].sum()
            phase_means[col].append(mean_val)

    bootstrapping_error = {
        'Class': [f'Class {i}' for i in range(1, 12)],
        'Max': [round(np.percentile(phase_means[col], 97.5), 5) for col in phase_cols],
        'Min': [round(np.percentile(phase_means[col], 2.5), 5) for col in phase_cols],
        'Actual_Percentage': [round(dataset[col].sum() * 100 / dataset[phase_cols].sum().sum(), 5) for col in phase_cols]
    }

    return pd.DataFrame(bootstrapping_error)



def compute_particle_quantification_percentages(df):
    """
    Adds 'Bulk_sum' and Phase_n_quantification (%) 
    """
    phase_cols = [f'Phase_{i}_quantification' for i in range(1, 13)]

    df['Bulk_sum'] = df[phase_cols].sum(axis=1)

    for col in phase_cols:
        percent_col = col.replace('_quantification', '_quantification (%)')
        df[percent_col] = (df[col]*100 / df['Bulk_sum']).fillna(0)
        
    df.drop(columns='Bulk_sum', inplace=True)

    return df


def compute_particle_surface_percentages(df):
    """
    Adds 'Surface_sum' and Phase_n_surface_quantification (%) columns
    """
    surface_cols = [f'Phase_{i}_surface_quantification' for i in range(1, 13)]

    df['Surface_sum'] = df[surface_cols].sum(axis=1)

    for col in surface_cols:
        percent_col = col.replace('_surface_quantification', '_surface_quantification (%)')
        df[percent_col] = (df[col] *100/ df['Surface_sum']).fillna(0)
    
    df.drop(columns='Surface_sum', inplace=True)

    return df

def compute_particle_outer_percentages(df):
    """
    Adds Phase_n_outer_quantification (%) columns 
    """
    outer_cols = [f'Phase_{i}_outer_quantification' for i in range(1, 13)]

    df['Outer_sum'] = df[outer_cols].sum(axis=1)

    for col in outer_cols:
        percent_col = col.replace('_outer_quantification', '_outer_quantification (%)')
        df[percent_col] = (df[col]*100 / df['Outer_sum']).fillna(0)
    
    df.drop(columns='Outer_sum', inplace=True)

    return df


