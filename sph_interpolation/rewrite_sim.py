import numpy as np
import pynbody as pn
import pynbody.plot.sph as sph
from astropy.cosmology import FlatLambdaCDM
import input_params as pars
import constants as const

# INPUT
input_galaxy_dir = pars.input_galaxy_dir
filename = pars.input_galaxy_dir + pars.input_galaxy_filename

# ****************************************         
# ****************************************                                                             
# **************************************** 

# Define some important variables
zstart = pars.zstart

cosmo = FlatLambdaCDM(H0=pars.H0, Om0=pars.Om0, Tcmb0=pars.Tcmb)

tstart = cosmo.lookback_time(zstart).to_value()
age_universe = cosmo.age(0).to_value()
tstart = age_universe - tstart

zs = const.zs

# Read simulation snapshot

gal = pn.load(filename)

# Compute time and redshift

time = gal.properties['time'].ratio('Gyr')

time_real = tstart + time

zarr_dummy = np.linspace(5., 1., num=10000)
tarr_dummy = age_universe - cosmo.lookback_time(zarr_dummy).to_value()

z_real = np.interp(time_real, tarr_dummy, zarr_dummy)

# Write time and redshift to file

ff = open('time_and_z.txt', 'w')

ff.write('# This file contains information about time and redshift of te snapshot. It assumes the simulation starts at z=4 \n')
ff.write('# sim time [Gyr]      real time [Gyr]      z\n')
ff.write(str(tstart) + '      ' + str(time_real) + '     ' + str(z_real) + '\n')
ff.close()

# Read gas properties

met = gal.gas['metals'] / zs
met[np.where(met<0)] = 0

pos = gal.gas['pos']#.in_units('cm')                                                                                                                                 
temp = gal.gas['temp']
vel = gal.gas['vel']
dens = gal.gas['rho']#.in_units('g cm**-3')
#mass = 2.3262e+05*gal.gas['mass']#.in_units('g')
mass = gal.gas['mass'].in_units('Msol')

# Read star properties
posstar = gal.star['pos']            
massstar = gal.star['mass'].in_units('Msol')
metstar = gal.star['metals'] / zs
#print('Diagnostics stellar metallicity: ', np.amin(metstar), np.amax(metstar), np.mean(metstar))
metstar[np.where(metstar<0.)] = 0.

print('Total M* (1e9 M_sum): ', np.sum(massstar)/1e9)
print('Total Mgas (1e9 M_sum): ', np.sum(mass)/1e9)
print('')

tformstar = gal.star['tform']
tformstar[tformstar>=0.] = pars.tsnap - tformstar[tformstar>=0.]
tformstar[tformstar<0.] = pars.tsnap # Negative formation time means the stars were there since the ICs

xx = pos[:,0]
yy = pos[:,1]
zz = pos[:,2]

vx = vel[:,0]
vy = vel[:,1]
vz = vel[:,2]

np.save('posx_gas.npy', xx)
np.save('posy_gas.npy', yy)
np.save('posz_gas.npy', zz)
np.save('vx_gas.npy', vx)
np.save('vy_gas.npy', vy)
np.save('vz_gas.npy', vz)
np.save('mass_gas.npy', mass)
np.save('temperature_gas.npy', temp)
np.save('metallicity.npy', met)

xx = posstar[:,0]
yy = posstar[:,1]
zz = posstar[:,2]

np.save('posx_star.npy', xx)
np.save('posy_star.npy', yy)
np.save('posz_star.npy', zz)
np.save('mass_star.npy', massstar)
np.save('metallicity_star.npy', metstar)
np.save('tform_star.npy', tformstar)
