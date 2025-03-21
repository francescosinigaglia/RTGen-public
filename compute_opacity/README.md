Bohren and Huffman Mie scattering code for computing mass absorption 
coefficients. The code allows the user to compute opacity tables for 
arbitrary grain size distributions.

The code was ported to Python3 from the Python2 version developed by Laszlo Szucs (https://gitlab.mpcdf.mpg.de/szucs/bh-mie-scat), which is based on the Bohren and Huffman Mie scattering code in [RADMC-3D](http://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/) Python modules, developed by C. P. Dullemond.

Usage
-----

**As stand alone program**

Run example code to compute multiple opacity tables for grain size distributions 
with increasing a_max.

```
python run_bhmie_scat.py
```

**Using functions in other Python code**

The bhmie-scat folder need to be added to the $PYTHONPATH environment variable, 
or the python interpreter must run from the folder containing the code.
In this case, the main function can be called as:

```
import run_bhmie_scat
run_bhmie_scat.compute_opac_dist(amin_mic=0.1,amax_mic=100,pwl=-3.0,na=50)
```
The above instructions will compute the opacity of 
[magnesium-iron silicates](http://www.astro.uni-jena.de/Laboratory/OCDB/amsilicates.html#B) 
(default, lnk/pyrmg70.lnk) grains between 0.1 and 100 micron size and 
n(a) \~ a^(-3) distribution.

Requirements
------------

* python 3.x
* numpy
* scipy
