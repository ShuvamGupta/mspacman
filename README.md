# mapsacman: Mounted Single Particle Compositional and Mineralogical Analyses

**mapsacman** is a Python toolkit for 3D micro-CT particle analysis, enabling advanced mineralogical and compositional quantification while accounting for partial volume blur (PVB) effects in CT data.
It supports:

- Particle property extraction (volume, surface area, Feret diameters, intensity stats)

- Greyscale histogram computation and smoothing

- Peak detection with phase mapping and bias correction for dense minerals

- Multi-phase quantification (liberated to quinary phases)

- Bootstrapping uncertainty estimation for both bulk and surface compositions

- Complete unit/regression testing against reference datasets


## Installation

Install directly using pip and git:

pip install git+https://github.com/ShuvamGupta/mspacman.git

## Example Usage & Unit Testing
We provide a combined example + unit testing script:
mspacman.core_example_usage_with_unit_test.py

This script:

- Demonstrates the complete MSPaCMAn processing pipeline using the publicly available Kemi +850 µm micro-CT dataset.

- Runs regression tests by comparing freshly generated outputs to pre-computed “golden” reference datasets located in:
  Kemi_+850_unit_test_and_example_usage/

Running the Example + Unit Test
-  Download the sample dataset from Figshare: https://figshare.com/articles/dataset/Kemi_850_m_CT_data_and_Particle_Labels/29836160

-  Download the reference datasets from: https://github.com/ShuvamGupta/mspacman/tree/main/Kemi_%2B850_unit_test_and_example_usage

-  Run the example script: python mspacman.core_example_usage_with_unit_test.py

The script automatically tests:

-  Data upload integrity (shape, index name, etc.)

-  Batch-processed outputs vs. reference data

-  Individual processing step outputs (properties, histograms, gradients, smoothed histograms)

-  Saving and reloading .h5ad and .csv formats without data loss

-  Quantification results (bulk, outer, surface)

-  Pass criteria: All DataFrames must match exactly with reference datasets (ignoring dtype differences).


All DataFrames must match exactly with reference datasets (ignoring dtype differences).

## Dependencies
Built on top of:

-  NumPy, SciPy, Pandas, Matplotlib, tifffile – Scientific computing and plotting

-  scikit-image – 3D image processing

-  Napari – Interactive visualization

-  AnnData – .h5ad data handling

-  pykuwahara – Edge-aware filtering

-  joblib, tqdm – Parallel processing and progress bars

## Acknowledge
This project utilizes several open-source Python libraries, including NumPy, SciPy, Pandas, Matplotlib, 
Napari, tifffile, scikit-image, joblib, anndata, tqdm, pykuwahara, and standard Python libraries 
(os, glob, re, and gc). We gratefully acknowledge the developers and communities of these tools 
for enabling efficient scientific computing, visualization, image processing, and data analysis.

## Citation

If you use this repository or the method/code provided, please cite the following paper:
Gupta, S., Moutinho, V., Godinho, J. R., Guy, B. M., & Gutzmer, J. (2025). 3D mineral quantification of particulate 
materials with rare earth mineral inclusions: Achieving sub-voxel resolution by considering the partial volume and blurring effect. 
Tomography of Materials and Structures, 7, 100050. https://doi.org/10.1016/j.tmater.2025.100050

You may also cite this GitLab repository as a secondary reference:
Gupta, S. (2025). *MSPaCMAn* [Python Package]. GitLab. https://gitlab.com/ShuvamGupta1/mspacman
Developed while affiliated with Helmholtz Institute Freiberg for Resource Technology, Germany.

