#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=20GB
#SBATCH --time=6:00:00
#SBATCH --output=job.out

cp mcmono_wavelength_micron_1000A.inp mcmono_wavelength_micron.inp 

radmc3d mcmono setthreads 16

mv mean_intensity.out mean_intensity_1000A.out
