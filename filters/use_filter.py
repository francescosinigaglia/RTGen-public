# ***********************************************************************
# This script applies the optical filters to the produced continuum image
# ***********************************************************************

import numpy as np
import input_params as pars
#import constants as const
import astropy.constants as const
from scipy.interpolate import interp1d

cc = const.c.to_value() # in km/s                                                                                                                            
Lsun = 3.8525e33          # Solar luminosity        [erg/s]                                                      

nlam = pars.nbins_uvopt_cam + pars.nbins_nir_cam + pars.nbins_fir_cam

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

# Get ird of the cm'^2 dependence
img = img * (pix**2)

# Pass from emissivity to intensity: integrate over solid angle
img *= (4*np.pi)

# Now pass from rest-frame to observed wavelenghts
lam = lam * (1 + pars.z_optical)

# Now read the filter
instrument = pars.instrument 
filtername = pars.filtername
filt = np.genfromtxt(instrument + '/' + filtername)

lamfilt = filt[:,0]
throughput = filt[:,1]

# Express the wavelenght of throughputs in units micron, for consistency with the image
lamfilt/= 1e4

img_interpolator = interp1d(lam, img, kind='cubic', axis=2, fill_value='extrapolate')

imgnew = np.sum(img_interpolator(lamfilt) * throughput, axis=2)

print(imgnew.shape)
