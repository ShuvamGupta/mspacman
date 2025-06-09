# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:45:28 2025

@author: gupta46
"""

import anndata
import pandas as pd
from tqdm import tqdm

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
        