# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:09:33 2025

@author: gupta46
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.signal import savgol_filter, find_peaks
from scipy import ndimage 
import re

from .utils import extract_peaks_and_thresholds, standardize_index


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