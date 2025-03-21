#
# Import NumPy for array handling
#
import numpy as np
import os
import input_params as pars
import constants as const
import run_bhmie_scat
import scipy

#
# Import plotting libraries (start Python with ipython --matplotlib)
#
# from mpl_toolkits.mplot3d import axes3d
# from matplotlib import pyplot as plt
#
# Some natural constants
#
au  = const.au     # Astronomical Unit       [cm]
pc  = const.pc     # Parsec                  [cm]
ms  = const.ms     # Solar mass              [g]
ts  = const.ts      # Solar temperature       [K]
ls  = const.ls     # Solar luminosity        [erg/s]
rs  = const.rs     # Solar radius            [cm]
kpc = 1e3 * pc
kb = const.kb      # Boltzmann constant      [erg/K]
cc = const.cc      # Speed of light          [cm/s]
hh = const.hh      # Planck constant

#
# Monte Carlo parameters
#
nphot    = pars.nphot
nphot_scat = pars.nphot_scat
nphot_spec = pars.nphot_spec
nphot_mono = pars.nphot_mono

scattering_mode_max = pars.scattering_mode_max

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

# Make the coordinates of the grid                                                               
#                                                                                                                                                                  
xi       = np.linspace(-sizex/2,sizex/2,nx+1)
yi       = np.linspace(-sizey/2,sizey/2,ny+1)
zi       = np.linspace(-sizez/2,sizez/2,nz+1)


# Physical input parameters
dust_to_gas_ratio = pars.dust_to_gas_ratio
dust_to_gas_model = pars.dust_to_gas_model
dust_to_gas_slope = pars.dust_to_gas_slope
dust_to_gas_zeropoint = pars.dust_to_gas_zeropoint
dust_to_gas_scatter = pars.dust_to_gas_scatter
fsil = pars.fsil
fcarb = pars.fcarb
grain_size_model = pars.grain_size_model
grain_size_distr_exp = pars.grain_size_distr_exp
numbinsdust = pars.numbinsdust
dustsizemin = pars.dustsizemin
dustsizemax = pars.dustsizemax
sil_dust_density = const.sil_dust_density
carb_dust_density = const.carb_dust_density

# Miscellaneous
numsmall = pars.numsmall

# ***********************************************
# ***********************************************
# ***********************************************

# Make copies of some key files
os.system('cp ../input/dustkappa_silicate.INP ../dustkappa_silicate.inp')
os.system('cp ../input/dustkappa_carbon.INP ../dustkappa_carbon.inp')
os.system('cp ../input/molecule_co.INP ../molecule_co.inp' )

#
# Make the dust density model
#
# read gas mass (in units M_sun) and metallicity (in units Z_sun)
rhog = np.fromfile(pars.input_dir+pars.mass_gas_filename, dtype=np.float64) * ms / (lcx*lcy*lcz)
metg = np.fromfile(pars.input_dir+pars.metallicity_filename, dtype=np.float64)
# Make sure there is no zero metallcity
metg[np.where(metg<=0)] = numsmall

if dust_to_gas_model == 'fixed_ratio':
    rhod  = dust_to_gas_ratio * rhog
    
elif dust_to_gas_model == 'metallicity_dependent':
    scat = np.random.normal(0,dust_to_gas_scatter, size=len(rhog))
    rhod = rhog * 10**(dust_to_gas_slope * np.log10(metg) + dust_to_gas_zeropoint + scat)
    
else:
    print('Error: this model is not yet implemented. Falling back to a fixed dust-to-gas ratio')
    rhod  = dust_to_gas_ratio * rhog

# Make the stellar mass density model
mass_star = np.fromfile(pars.input_dir+pars.mass_star_filename, dtype=np.float64)
rhos  = mass_star * ms / (lcx*lcy*lcz)

#
# Write the wavelength_micron.inp file
#

# Read the stellar spectrum                                                
rawstarspec = np.genfromtxt(pars.input_dir + pars.stellar_spectrum_filename)
lam = rawstarspec[:,0] # lambda is in micron
#nlam = len(lam)
star_spectrum = rawstarspec[:,1] # spectrum is in units erg/s/A

# The original spectrum is in units erg/s/A. I have first to convert to erg/s, then to erg/s/Hz.
# So, I multiply the spectrum by lambda in A and divide it by frequency 
cc_tmp = cc * 1e-2 # cc in m/s 
nu = cc_tmp / (lam * 1e-6) # lam here is in meters
star_spectrum = star_spectrum * (lam * 1e4) / nu

# Fill in the second part of the spectrum, the final grid point must be ~10^4 um
lam_fill = np.logspace(np.log10(lam[-1]),np.log10(1.0e4),101,endpoint=True)

lam_fill = lam_fill[1:]

# The interpolation is done is log space, because in the FIR part of the spectrum it's well approximated by a linear relation
interp = scipy.interpolate.interp1d(np.log10(lam), np.log10(star_spectrum), kind='linear', fill_value='extrapolate')

star_spectrum_fill = 10**interp(np.log10(lam_fill))

# Now concatenate
lam = np.concatenate((lam, lam_fill))
star_spectrum = np.concatenate((star_spectrum, star_spectrum_fill))

nlam = len(lam)

#
# Write the wavelength file
#
with open('../wavelength_micron.inp','w+') as f:
    f.write('%d\n'%(nlam))
    for value in lam:
        f.write('%13.6e\n'%(value))

    f.close()


# Write the wavelenght files used to compute the radiation field to split HI and H2
with open('../mcmono_wavelength_micron_1000A.inp','w+') as f:
    f.write('1\n')
    f.write('0.1\n') # 1000A = 0.1um
    f.close()

# Write the wavelenght files used to compute the radiation field to compute Halpha photoionization   

lam1bis     = 0.01e0
lam2bis     = 0.0912e0
n12bis      = 10
lambis    = np.logspace(np.log10(lam1bis),np.log10(lam2bis),n12bis,endpoint=True)
nlambis = lambis.size

with open('../mcmono_wavelength_micron_photoion.inp','w+') as f:
    f.write('%d\n'%(nlambis))
    for value in lambis:
        f.write('%13.6e\n'%(value))
    f.close()

# Write the wavelenght files used to compute the dust continuum images      
lam1ter     = 0.01e0
lam2ter     = 7.0e0
lam3ter     = 25.e0
lam4ter     = 1.0e4
n12ter      = pars.nbins_uvopt_cam
n23ter      = pars.nbins_nir_cam
n34ter      = pars.nbins_fir_cam
lam12ter    = np.logspace(np.log10(lam1ter),np.log10(lam2ter),n12ter,endpoint=False)
lam23ter    = np.logspace(np.log10(lam2ter),np.log10(lam3ter),n23ter,endpoint=False)
lam34ter    = np.logspace(np.log10(lam3ter),np.log10(lam4ter),n34ter,endpoint=True)
lamter      = np.concatenate([lam12ter,lam23ter,lam34ter])
nlamter     = lamter.size

with open('../camera_wavelength_micron.inp','w+') as f:
    f.write('%d\n'%(nlamter))
    for value in lamter:
        f.write('%13.6e\n'%(value))
    f.close()
    
#
#
# Write the grid file
#
with open('../amr_grid.inp','w+') as f:
    f.write('1\n')                       # iformat
    f.write('0\n')                       # AMR grid style  (0=regular grid, no AMR)
    f.write('0\n')                       # Coordinate system
    f.write('0\n')                       # gridinfo
    f.write('1 1 1\n')                   # Include x,y,z coordinate
    f.write('%d %d %d\n'%(nx,ny,nz))     # Size of grid
    for value in xi:
        f.write('%13.6e\n'%(value))      # X coordinates (cell walls)
    for value in yi:
        f.write('%13.6e\n'%(value))      # Y coordinates (cell walls)
    for value in zi:
        f.write('%13.6e\n'%(value))      # Z coordinates (cell walls)
    f.close()  


# Convert the stellar spectrum in proper units
star_spectrum = star_spectrum / (4*np.pi * np.sum(mass_star*ms))

with open('../stellarsrc_templates.inp','w+') as f:
    f.write('2\n')                       # Format number                    
    f.write('1\n')                       # Nr of templates
    f.write('%d\n'%(nlam))               # Nr of wavelenghts of the template(s)
    for value in lam:
        f.write('%13.6e\n'%(value))      # Wavelenghts of templates
    for value2 in star_spectrum:
        f.write('%13.6e\n'%(value2))  
    #f.write('-%3.6e\n'%(2*ts))
    #f.write('%13.6e\n'%(2*rs))
    #f.write('%13.6e\n'%(2*ms))
    f.write('\n')
    f.close()
#

# Write the stellarsrc density file                                                                                       
#                                                                                                                                                          
with open('../stellarsrc_density.inp','w+') as f:
    f.write('1\n')                       # Format number                             
    f.write('%d\n'%(nx*ny*nz))           # Nr of cells                                                              
    f.write('1\n')            # Nr of templates                                                                                
    data = rhos.ravel(order='F')         # Create a 1-D view, fortran-style indexing                                                       
    data.tofile(f, sep='\n', format="%13.6e")
    f.write('\n')
    f.close()
#

#
# Dust opacity control file
#
# Two possible cases:
# 1) fixed size model: 2 species (silicates and carbonaceous), simpler and quicker
# 2) grain size distribution treated as 
# 3) explicit grain size distribution. 2 x numbins dust species, slower and more involved, but more realistic

# if 3), initialize first a list of strings containing the extensions of the opacity files. Then, compute opacities
if grain_size_model == 'power_law':
    silopaclist = []
    carbopaclist = []
    opacsize = np.logspace(np.log10(dustsizemin), np.log10(dustsizemax), num=numbinsdust+1)
    opacsizegrid = 0.5*(opacsize[1:]+opacsize[:-1]) 
    for ii in range(numbinsdust):
        silsizestring = 'silicate_' + str(opacsize[ii])
        carbsizestring = 'carbon_' + str(opacsize[ii])
        silopaclist.append(silsizestring)
        carbopaclist.append(carbsizestring)
        # Silicates
        run_bhmie_scat.compute_opac_dist(amin_mic=opacsize[ii]-1e-6,amax_mic=opacsize[ii]+1e-6,pwl=grain_size_distr_exp,na=1, optconst="../compute_opacity/lnk/pyrmg70.lnk", matdens=sil_dust_density, outfolder='../', outname='dustkappa_%s.inp' %silsizestring)
        # Carbonaceous
        run_bhmie_scat.compute_opac_dist(amin_mic=opacsize[ii]-1e-6,amax_mic=opacsize[ii]+1e-6,pwl=grain_size_distr_exp,na=1, optconst="../compute_opacity/lnk/mix_compact_bruggeman.lnk", matdens=carb_dust_density, outfolder='../', outname='dustkappa_%s.inp' %carbsizestring)

    os.system('mv dustkappa_*.inp ../')
        

# if 2), compute first a new opacities file
if grain_size_model == 'mixture':
    # Silicates
    run_bhmie_scat.compute_opac_dist(amin_mic=dustsizemin,amax_mic=dustsizemax,pwl=grain_size_distr_exp,na=250, optconst="../compute_opacity/lnk/pyrmg70.lnk", matdens=sil_dust_density, outfolder='.', outname='dustkappa_silicate_mixture.inp')
    os.system('mv dustkappa_silicate_mixture.inp ../')
    
    # Carbonaceous
    run_bhmie_scat.compute_opac_dist(amin_mic=dustsizemin,amax_mic=dustsizemax,pwl=grain_size_distr_exp,na=250, optconst="../compute_opacity/lnk/mix_compact_bruggeman.lnk", matdens=carb_dust_density, outfolder='.', outname='dustkappa_carbon_mixture.inp')
    os.system('mv dustkappa_carbon_mixture.inp ../')

with open('../dustopac.inp','w+') as f:
    if grain_size_model == 'fixed':
        f.write('2               Format number of this file\n')
        f.write('2               Nr of dust species\n')
        f.write('============================================================================\n')
        f.write('1               Way in which this dust species is read\n')
        f.write('0               0=Thermal grain\n')
        f.write('silicate        Extension of name of dustkappa_***.inp file\n')
        f.write('----------------------------------------------------------------------------\n')
        f.write('1               Way in which this dust species is read\n')
        f.write('0               0=Thermal grain\n')
        f.write('carbon          Extension of name of dustkappa_***.inp file\n')
        f.write('----------------------------------------------------------------------------\n')

    elif grain_size_model == 'mixture':
        f.write('2               Format number of this file\n')
        f.write('2               Nr of dust species\n')
        f.write('============================================================================\n')
        f.write('1               Way in which this dust species is read\n')
        f.write('0               0=Thermal grain\n')
        f.write('silicate_mixture        Extension of name of dustkappa_***.inp file\n')
        f.write('----------------------------------------------------------------------------\n')
        f.write('1               Way in which this dust species is read\n')
        f.write('0               0=Thermal grain\n')
        f.write('carbon_mixture          Extension of name of dustkappa_***.inp file\n')
        f.write('----------------------------------------------------------------------------\n')

    elif grain_size_model == 'power_law':
        f.write('2               Format number of this file\n')
        f.write('%d               Nr of dust species\n' %int(2*numbinsdust))
        f.write('============================================================================\n')
        for kk in range(len(silopaclist)):
            f.write('1               Way in which this dust species is read\n')
            f.write('0               0=Thermal grain\n')
            f.write('%s        Extension of name of dustkappa_***.inp file\n' %silopaclist[kk])
            f.write('----------------------------------------------------------------------------\n')
        for kk in range(len(carbopaclist)):
            f.write('1               Way in which this dust species is read\n')
            f.write('0               0=Thermal grain\n')
            f.write('%s          Extension of name of dustkappa_***.inp file\n' %carbopaclist[kk])
            f.write('----------------------------------------------------------------------------\n')

    else:
        print('Error: this case is not yet implemented. Falling back to fixed size distribution.')
        f.write('2               Format number of this file\n')
        f.write('2               Nr of dust species\n')
        f.write('============================================================================\n')
        f.write('1               Way in which this dust species is read\n')
        f.write('0               0=Thermal grain\n')
        f.write('silicate        Extension of name of dustkappa_***.inp file\n')
        f.write('----------------------------------------------------------------------------\n')
        f.write('1               Way in which this dust species is read\n')
        f.write('0               0=Thermal grain\n')
        f.write('carbon          Extension of name of dustkappa_***.inp file\n')
        f.write('----------------------------------------------------------------------------\n')

    f.close()
        
#                                                                                                                            
# Write the dust density file                                                                                                         
#                                                                                                                                     
with open('../dust_density.inp','w+') as f:
    if grain_size_model == 'fixed' or grain_size_model == 'mixture':
        f.write('1\n')                       # Format number
        f.write('%d\n'%(nx*ny*nz))           # Nr of cells
        f.write('2\n')                       # Nr of dust species
        data = (rhod*fsil).ravel(order='F')         # Create a 1-D view, fortran-style indexing
        data.tofile(f, sep='\n', format="%13.6e")
        f.write('\n')
        data = (rhod*fcarb).ravel(order='F')         # Create a 1-D view, fortran-style indexing  
        data.tofile(f, sep='\n', format="%13.6e")
        f.write('\n')
        f.close()

    elif grain_size_model == 'power_law':

        # First, set the probabilities
        func = opacsizegrid**grain_size_distr_exp
        prob = func/np.sum(func)
        
        f.write('1\n')                       # Format number                                                                          
        f.write('%d\n'%(nx*ny*nz))           # Nr of cells                                                                            
        f.write('%d\n' %int(2*numbinsdust))   # Nr of dust species
        for jj in range(len(silopaclist)):
            data = (rhod*fsil*prob[jj]).ravel(order='F')         # Create a 1-D view, fortran-style indexing
            data.tofile(f, sep='\n', format="%13.6e")
            f.write('\n')
        for jj in range(len(carbopaclist)):
            data = (rhod*fcarb*prob[jj]).ravel(order='F')         # Create a 1-D view, fortran-style indexing
            data.tofile(f, sep='\n', format="%13.6e")
            f.write('\n')

        f.close()

    else:

        print('Error: this case is not yet implemented. Falling back to fixed size distribution.')
        f.write('1\n')                       # Format number
        f.write('%d\n'%(nx*ny*nz))           # Nr of cells
        f.write('2\n')                       # Nr of dust species
        data = (rhod*fsil).ravel(order='F')         # Create a 1-D view, fortran-style indexing
        data.tofile(f, sep='\n', format="%13.6e")
        f.write('\n')
        data = (rhod*fcarb).ravel(order='F')         # Create a 1-D view, fortran-style indexing
        data.tofile(f, sep='\n', format="%13.6e")
        f.write('\n')
        f.close()

#
# Write the radmc3d.inp control file
#
with open('../radmc3d.inp','w+') as f:
    f.write('nphot = %d\n'%(nphot))
    f.write('nphot_scat = %d\n'%(nphot_scat))
    f.write('nphot_spec = %d\n'%(nphot_spec))
    f.write('nphot_mono = %d\n'%(nphot_mono))
    f.write('scattering_mode_max = %d\n'%(scattering_mode_max))   # Put this to 1 for isotropic scattering
    f.write('countwrite = 1000000\n')
    f.close()

print('DUST CONTINUUM TRANSFER SETUP DONE!')
print('')
