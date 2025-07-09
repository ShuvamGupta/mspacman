# MSPaCMAn: Mounted Single Particle Compositional and Mineralogical Analyses

**MSPaCMAn** is a Python toolkit for analyzing 3D micro-CT images of particles.  
It supports property extraction, greyscale histogram analysis, peak detection, and multi-phase quantification — 
considering partial volume blur for advanced CT analyses.


## Installation

Install directly using pip:

pip install git+https://github.com/ShuvamGupta/mspacman.git

## Example usage
Refer mspacman.core_example_usage.py file

## Acknowledge
This project utilizes several open-source Python libraries including NumPy, SciPy, Pandas, Matplotlib, 
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
