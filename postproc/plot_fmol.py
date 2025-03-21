import matplotlib.pyplot as plt
import numpy as np

ngrid = 256

raw = np.fromfile('../input/fmol.DAT', dtype=np.float64)

gasmass = np.fromfile('../input/mass_gas.DAT', dtype=np.float64)

# First, plot histogram
fmol = raw[np.where(raw>=0)]
plt.hist(fmol, bins=30, alpha=0.7, density=True)
plt.xlabel(r'$f_{\rm{mol}}$')
plt.ylabel(r'Probability density')
plt.yscale('log')
plt.grid()
plt.savefig('fmol_hist.pdf', bbox_inches='tight')

plt.clf()

raw = np.reshape(raw,(ngrid,ngrid,ngrid), order='F')
gasmass = np.reshape(gasmass,(ngrid,ngrid,ngrid), order='F')

# The slice must be of mass-weighted fmol, not fmol
#slce = np.sum(raw[:,ngrid//2-2:ngrid//2+3,:] * gasmass[:,ngrid//2-2:ngrid//2+3,:], axis=1) / np.sum(gasmass[:,ngrid//2-2:ngrid//2+3,:], axis=1)
slce = np.mean(raw[:,ngrid//2-2:ngrid//2+3,:] , axis=1)
#slce = np.sum(raw * gasmass, axis=1) / np.sum(gasmass, axis=1)
#slce = raw[:,ngrid//2,:] 
#slce[np.where(slce<0)]=np.nan
slce = np.log10(slce)

plt.imshow(np.flipud(slce.T),
           interpolation='bicubic',
           vmin=-10.,
           extent=[-10,10,-10,10],
           cmap='Blues')
plt.colorbar(label=r'$f_{\rm{mol}}$')
plt.xlabel('x [kpc]')
plt.ylabel('y [kpc]')
plt.savefig('fmol_slice.pdf', bbox_inches='tight')
