#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=20GB
#SBATCH --time=6:00:00
#SBATCH --output=job.out

python3 split_HI_H2.py
