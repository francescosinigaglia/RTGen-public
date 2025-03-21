#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=5GB
#SBATCH --time=02:00:00
#SBATCH --output=job.out

#python3 run_pipeline.py
export NUMBA_NUM_THREADS=16
cp ../input/input_params.py .
cp ../input/constants.py .
conda run -n numba-env python3 prepare_stellar_grid_s99.py
rm input_params.py
