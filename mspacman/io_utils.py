# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 13:53:55 2025

@author: gupta46
"""

import numpy as np
import os
import glob
import re
from skimage import io
from joblib import Parallel, delayed
from tqdm import tqdm
import tifffile
import anndata
import pandas as pd
import gc


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





    
