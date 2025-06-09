# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 15:21:34 2025

@author: gupta46
"""

import numpy as np
import pandas as pd
import re
from .utils import standardize_index, extract_peaks_and_thresholds


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

            voxels_bg_1 = Outer_volume_histograms.iloc[i, Background_peak_pos+1:Partical_peak_1+1]
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
                no_of_voxels_towards_background = Outer_volume_histograms.iloc[i,Background_peak_pos+1:particle_peak_pos+1]
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
            Quantification_Outer_phase_3_array = Quantification_Outer_phase_3_array[Partical_peak_2-Background_peak_pos: Partical_peak_3 - Background_peak_pos-1]
            
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
                no_of_voxels_towards_background = Outer_volume_histograms.iloc[i,Background_peak_pos+1:particle_peak_pos+1]
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
            Quantification_Outer_phase_4_array = Quantification_Outer_phase_4_array[Partical_peak_3-Background_peak_pos: Partical_peak_4 - Background_peak_pos-1]
            
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
            

            def calculate_phase_quantification_array(particle_peak_pos):
                no_of_voxels_towards_background = Outer_volume_histograms.iloc[i,Background_peak_pos+1:particle_peak_pos+1]
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
            Quantification_Outer_phase_4_array = Quantification_Outer_phase_4_array[Partical_peak_3-Background_peak_pos:]
            Quantification_Outer_phase_5_array = calculate_phase_quantification_array(Partical_peak_5)
            Quantification_Outer_phase_5_array = Quantification_Outer_phase_5_array[Partical_peak_4-Background_peak_pos: Partical_peak_5 - Background_peak_pos-1]
            
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