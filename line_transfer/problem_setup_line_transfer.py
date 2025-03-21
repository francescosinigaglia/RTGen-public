#
# Import NumPy for array handling
#
import numpy as np
import os
import input_params as pars
import constants as const
#
# Import plotting libraries (start Python with ipython --matplotlib)
#
#from mpl_toolkits.mplot3d import axes3d
#from matplotlib import pyplot as plt
#
# Some natural constants
#
au  = const.au                 # Astronomical Unit       [cm]
pc  = const.pc                 # Parsec                  [cm]
ms  = const.ms                 # Solar mass              [g]
ts  = const.ms                 # Solar temperature       [K]
ls  = const.ls                 # Solar luminosity        [erg/s]
rs  = const.rs                 # Solar radius            [cm]
kpc = 1e3*pc
kb = const.kb                  # Boltzmann constant      [erg/K]
cc = const.cc                  # Speed of light          [cm/s]
hh = const.hh                  # Planck constant        
amu = const.amu                # atomic mass unit        [g]
CO_molmass = const.CO_molmass  # molar mass of CO        [g/mol]
avnum = const.avnum            # Avogradro number

#
# Monte Carlo parameters
#
nphot    = pars.nphot
nphot_scat = pars.nphot_scat
nphot_spec = pars.nphot_spec
nphot_mono = pars.nphot_mono

scattering_mode_max = pars.scattering_mode_max

h2_to_co_convfact = pars.h2_to_co_convfact
lines_mode = pars.lines_mode

#
# Grid parameters
#

nx       = pars.nx
ny       = pars.ny
nz       = pars.nz
sizex    = pars.sizex * kpc
sizey    = pars.sizey * kpc
sizez    = pars.sizez * kpc
lcx = sizex / nx
lcy = sizey / ny
lcz = sizez / nz
#

max_gas_temp_line = pars.max_gas_temp_line


ortho_to_para_H2_perc = pars.ortho_to_para_H2_perc
tgas_eq_tdust = pars.tgas_eq_tdust

# ***********************************************
# ***********************************************
# ***********************************************


#
# Read gas density
# To compute the H2 density, multiply by the molecular fraction
# Then, convert to CO (molecular) density assuming a conersion factor
# Finally, conert to number density by dividing by the molecular mass

# Read gas density and molecular fraction
rhog = np.fromfile(pars.input_dir+pars.mass_gas_filename, dtype=np.float64) * ms / (lcx*lcy*lcz)
fmol = np.fromfile(pars.input_dir+pars.fmol_filename, dtype=np.float64)

vx = np.fromfile(pars.input_dir+pars.vx_gas_filename, dtype=np.float64) * 1e5 # pass from km/s to cm/s
vy = np.fromfile(pars.input_dir+pars.vy_gas_filename, dtype=np.float64) * 1e5 # pass from km/s to cm/s
vz = np.fromfile(pars.input_dir+pars.vz_gas_filename, dtype=np.float64) * 1e5 # pass from km/s to cm/s

temp =  np.fromfile(pars.input_dir+pars.temperature_gas_filename, dtype=np.float64)
temp[temp>max_gas_temp_line] = max_gas_temp_line

# Reshape velocity vectors
vx = np.reshape(vx, (nx,ny,nz), order='F')
vy = np.reshape(vy, (nx,ny,nz), order='F')
vz = np.reshape(vz, (nx,ny,nz), order='F')

# Compute H2 density
rhog = fmol * rhog

# Pass from H2 to molecular density
rhog = h2_to_co_convfact * rhog

# Compute molecular number density
rhog = rhog / (CO_molmass/avnum)

# Write the molecule number density file. 
#

with open('../numberdens_co.inp','w+') as f:
    f.write('1\n')                       # Format number
    f.write('%d\n'%(nx*ny*nz))           # Nr of cells
    data = rhog.ravel(order='F')          # Create a 1-D view, fortran-style indexing
    data.tofile(f, sep='\n', format="%13.6e")
    f.write('\n')

# Write the number density of collisional partners: ortho-H2 and para-H2
# At tmeperatures T>200-300 K, Ortho/Para = 3 (Sternberg & Neufeld 1998) 

with open('../numberdens_p-h2.inp','w+') as f:
    f.write('1\n')                       # Format number                                     
    f.write('%d\n'%(nx*ny*nz))           # Nr of cells                
    data = (1.-ortho_to_para_H2_perc)*rhog.ravel(order='F')          # Create a 1-D view, fortran-style indexing                                 
    data.tofile(f, sep='\n', format="%13.6e")
    f.write('\n')

with open('../numberdens_o-h2.inp','w+') as f:
    f.write('1\n')                       # Format number        
    f.write('%d\n'%(nx*ny*nz))           # Nr of cells   
    data = (ortho_to_para_H2_perc)*rhog.ravel(order='F')          # Create a 1-D view, fortran-style indexing        
    data.tofile(f, sep='\n', format="%13.6e")
    f.write('\n')
    
#
# Write the gas velocity field
#
with open('../gas_velocity.inp','w+') as f:
    f.write('1\n')                       # Format number
    f.write('%d\n'%(nx*ny*nz))           # Nr of cells
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                f.write('%13.6e %13.6e %13.6e\n'%(vx[ix,iy,iz],vz[ix,iy,iz],vy[ix,iy,iz]))

"""
#
# Write the microturbulence file
#
with open('microturbulence.inp','w+') as f:
    f.write('1\n')                       # Format number
    f.write('%d\n'%(nx*ny*nz))           # Nr of cells
    data = vturb.ravel(order='F')          # Create a 1-D view, fortran-style indexing
    data.tofile(f, sep='\n', format="%13.6e")
    f.write('\n')
"""

#
# Write the gas temperature
#
with open('../gas_temperature.inp','w+') as f:
    f.write('1\n')                       # Format number
    f.write('%d\n'%(nx*ny*nz))           # Nr of cells
    data = temp.ravel(order='F')          # Create a 1-D view, fortran-style indexing
    data.tofile(f, sep='\n', format="%13.6e")
    f.write('\n')

#
# Write the lines.inp control file
#
with open('../lines.inp','w') as f:
    f.write('2\n')
    f.write('1\n')
    f.write('co    leiden    0    0    2\n')
    f.write('p-h2\n')
    f.write('o-h2\n')

# Write the radmc3d.inp control file
#

# First make, a copy of the dust continuum input param file
os.system('cp ../radmc3d.inp ../radmc3d_dust_continuum.inp')

with open('../radmc3d.inp','w+') as f:
    f.write('nphot = %d\n'%(nphot))
    f.write('nphot_scat = %d\n'%(nphot_scat))
    f.write('nphot_spec = %d\n'%(nphot_spec))
    f.write('nphot_mono = %d\n'%(nphot_mono))
    f.write('scattering_mode_max = %d\n'%(scattering_mode_max))   # Put this to 1 for isotropic scattering
    #f.write('iranfreqmode = 1\n')
    f.write('lines_mode = %d\n'%(lines_mode))
    f.write('tgas_eq_tdust = %d\n' %tgas_eq_tdust)
    f.write('countwrite = 1000000\n')
    f.write('catch_doppler_resolution = 0.3\n')

print('LINE TRANSFER SETUP DONE!')
