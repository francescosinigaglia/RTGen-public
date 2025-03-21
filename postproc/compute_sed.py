import numpy as np

# """"""""""""""""""""""""""""""""""""""""""""""
# //////////////////////////////////////////////
# RADMC-3D yields images in units emisivity:
# [j_nu] = erg / s / cm^2 / Hz / ster
#
#
# To pass to luminosity:
#
# 1) compute intensity: I_nu = j_nu * (4 x pi) 
#
# 2) integrate over the image plane:
# L_nu = I_nu * pix^2

# //////////////////////////////////////////////
# """""""""""""""""""""""""""""""""""""""""""""" 

nlam = 30

# Read image file                                                                                                                          
dummy = np.genfromtxt('../image.out', usecols=0)
pix = dummy[3]
ngrid = int(dummy[1])

# Read wavelengths                                                                                                                                   
raw = np.genfromtxt('../image.out', skip_header=4)

lam = raw[:nlam]
sed = np.zeros(nlam)


img = raw[nlam:]
img = np.reshape(img, (ngrid,ngrid,nlam), order='F')

# Integrate over image plane
sed = np.sum(img, axis=(0,1)) * (pix**2)

# Pass from emissivity to intensity: integrate over solid angle
sed *= (4*np.pi)

# Now write to file
f = open('sed.txt', 'w')

f.write('# This file contains the modelled SED.\n')
f.write('# lambda [um]        luminosity [erg/s] \n')
for ii in range(nlam):
    f.write(str(lam[ii]) + '        ' + str(sed[ii]) + '\n')

f.close()


sed = sed[np.where(np.logical_and(lam>80, lam<100))]
lam = lam[np.where(np.logical_and(lam>80, lam<100))]


cc = 3e5*1e9
nu = cc / lam
print(lam)
print(nu)
Lsun = 3.826e33
#lir = np.trapz(sed*nu,nu)/Lsun
lir = sed*nu/Lsun  
print(lir)
