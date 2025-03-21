#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --output=job_sph_interpolation.out

export NUMBA_NUM_THREADS=16

cp ../input/input_params.py .
cp ../input/constants.py .

python3 use_filter.py

rm input_params.py
rm constants.py
