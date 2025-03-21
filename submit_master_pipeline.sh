#!/usr/bin/env bash
#SBATCH --cpus-per-task=32
#SBATCH --mem=20GB
#SBATCH --time=61:00:00
#SBATCH --output=job.out

# First, copy the input params and constants files everywhere
cp input/input_params.py dust_transfer/
cp input/input_params.py line_transfer/
cp input/input_params.py H2_HI_splitting/

cp input/constants.py dust_transfer/
cp input/constants.py line_transfer/
cp input/constants.py H2_HI_splitting/

# --------------------------------------  
# --------------------------------------
# WORKFLOW:
# 1) dust RT
# 2) dust RT images
# 3) H2/HI splitting
# 4) line transfer image  
# 5) H-alpha photoionization
# 6) plotting

# --------------------------------------
# --------------------------------------  
# 1) dust RT

cd dust_transfer; python3 problem_setup_dust_continuum.py; cd ..
radmc3d mctherm setthreads 32

# --------------------------------------
# 2) dust RT image
# Choose just one of the following
#radmc3d image lambda 150 incl 90 secondorder sizepc 20000 npixx 256 npixy 256 setthreads 32
radmc3d image loadlambda incl 90 secondorder sizepc 20000 npixx 256 npixy 256 setthreads 32   
mv image.out image_cont.out

# --------------------------------------                                       
# 3) H2/HI splitting                                                                       
cp mcmono_wavelength_micron_1000A.inp mcmono_wavelength_micron.inp              
radmc3d mcmono setthreads 32                                             
mv mean_intensity.out mean_intensity_1000A.out

cd H2_HI_splitting; python3 split_HI_H2.py; cd ..

# TESTED SO FAR

# --------------------------------------                                                                  
# 4) line transfer image  
cd line_transfer; python3 problem_setup_line_transfer.py; cd ..
radmc3d image incl 90 iline 1 widthkms 250 linenlam 20 secondorder sizepc 20000 npixx 256 npixy 256 setthreads 16
mv image.out image_line.out

rm dust_transfer/input_params.py
rm line_transfer/input_params.py
rm H2_HI_splitting/input_params.py

rm dust_transfer/constants.py
rm line_transfer/constants.py
rm H2_HI_splitting/constants.py
