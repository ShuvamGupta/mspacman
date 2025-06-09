# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:44:21 2025

@author: gupta46
"""
import numpy as np
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
