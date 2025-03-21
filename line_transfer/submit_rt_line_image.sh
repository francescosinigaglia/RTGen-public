#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=20GB
#SBATCH --time=6:00:00
#SBATCH --output=job.out

python3 problem_setup_line_transfer.py

radmc3d image incl 90 iline 1 widthkms 250 linenlam 20 secondorder sizepc 20000 npixx 256 npixy 256 setthreads 16      
#radmc3d image lambda 2600.757 incl 90 secondorder sizepc 20000 npixx 256 npixy 256 setthreads 16
#radmc3d image incl 90 allwl sizepc 20000 npixx 256 npixy 256 setthreads 16 
#widthkms 10 linenlam 100
