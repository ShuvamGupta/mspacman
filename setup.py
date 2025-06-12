# -*- coding: utf-8 -*-
"""
Created on Wed May 14 20:37:45 2025

@author: gupta46
"""

from setuptools import setup, find_packages

setup(
    name='mspacman',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy<2.0.0',
        'scipy',
        'pandas',
        'tifffile',
        'joblib',
        'anndata',
        'tqdm',
        'scikit-image',
        'pykuwahara',
        'matplotlib',
        'napari'
    ],
    author='Shuvam Gupta',
    description='MSPaCMAn: 3D particle and phase analysis',
)
