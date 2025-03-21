#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=5GB
#SBATCH --time=01:00:00
#SBATCH --output=job_sph_interpolation.out

export NUMBA_NUM_THREADS=8

cp ../input/input_params.py .
cp ../input/constants.py .

python3 rewrite_sim.py
conda run -n numba-env python3 sph_interpolation.py

rm input_params.py
rm constants.py
