# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:41:33 2025

@author: gupta46
"""

import matplotlib.pyplot as plt
import numpy as np

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
    