import os

#os.system("python3 cleanoutdirall.py")
os.system('cp ../input/input_params.py .')
#os.system("python3 prepare_sample_stars.py")
os.system('conda run -n numba-env python3 prepare_stellar_grid_s99.py')
os.system('rm input_params.py')
