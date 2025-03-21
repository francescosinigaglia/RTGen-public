import os

os.system('cp ../input/input_params.py .')
os.system('cp ../input/constants.py .')

os.system('python3 use_filter.py')

os.system('rm input_params.py')
os.system('rm constants.py')
