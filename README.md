# RTGen

This repository contains the pipeline to run radiative transfer simulations of galaxies using the [RADMC-3D](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/) code by Cornelis Dullemond. 

Please refer to the RADMC-3D documentation for an extensive and detailed discussion of all the possible options and computations that the code can perform.

If you use this pipeline, or part of it, please cite:
* the [RADMC-3D](https://ui.adsabs.harvard.edu/abs/2012ascl.soft02015D/abstract) code
* the paper presenting and validating the pipeline: [Sinigaglia et al. (2025), A&A, in press](https://ui.adsabs.harvard.edu/abs/2024arXiv241208609S/abstract) 

## Installation and usage

As recommended by the documentation, it is strongly advised to clear the main directory from any input/configuration file, coming e.g. from previos runs. This can be done by either initializing a new fresh direcotry, or by executing
```
python3 cleanoutdirall.py
```

The pipeline consists in a series of scripts which performs all the modelling required by the RADMC-3D code and coordinate execution of the different phases (dust continuum transfer, H2/HI splitting, line transfer). 

The code requires the following packages: NumPy (tested version 1.26.0), SciPy (tested version 1.12.0) and [Numba](https://numba.pydata.org) (tested version 0.58.1). 

Also, the opacity computation requires adding the 'compute_opacity' directory to your PYTHONPATH by either executing the following command before running (must be done every time one runs in a new shell):
```
export PYTHONPATH="${PYTHONPATH}:/your_path_to_compute_opacity_directory"
```

Or, one can add the previous line to the ~/.bashrc file and then execute (can be done only once): 
```
source ~/.bashrc
```


**Flowchart**

1) from the 'sph_interpolation' directory: submit the SPH interpolation job executing
```
sbatch submit_interpolation.sh
```


2) from the 'stars_model' directory: submit the stellar modelling job executing 
```
sbatch submit_stellar_modelling.sh
```

3) from the main directory: submit the full pipeline executing
```
sbatch submit_master_pipeline.sh
```

To fit the H2-to-CO conversion factor/function, from the 'line_transfer' directory execute
```
sbatch submit_fit_cocoversion.sh
```


All the relevant input parameters can be changed in 'input/input_params.py'.

The code and the input parameters come with no warrranty. Please, always check the physical meaning of you results, 



