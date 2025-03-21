#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=5GB
#SBATCH --time=6:00:00
#SBATCH --output=job.out

python3 problem_setup_dust_continuum.py
radmc3d mctherm setthreads 16
