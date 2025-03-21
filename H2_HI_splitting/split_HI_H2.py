import numpy as np
import input_params as pars
import constants as const

print('---------------------------------')
print('Computing the H2/HI splitting ...')

ngridx = pars.nx
ngridy = pars.ny
ngridz = pars.nz

lxbox = pars.sizex
lybox = pars.sizey
lzbox = pars.sizez

lxcell = lxbox / ngridx
lycell = lybox / ngridy
lzcell = lzbox / ngridz

dust_to_gas_model = pars.dust_to_gas_model                                                                          
dust_to_gas_ratio = pars.dust_to_gas_ratio
dust_to_gas_slope = pars.dust_to_gas_slope                                                                                   
dust_to_gas_zeropoint = pars.dust_to_gas_zeropoint                                                                                   
dust_to_gas_scatter = pars.dust_to_gas_scatter

splitting_method = pars.splitting_method

# Read the gas field, in units M*/kpc^3
rhog = np.fromfile(pars.input_dir+pars.mass_gas_filename, dtype=np.float64) / (lxcell * lycell * lzcell)
metg = np.fromfile(pars.input_dir+pars.metallicity_filename, dtype=np.float64)
# Make sure there is no zero metallcity                                                                                                                         
metg[np.where(metg<=0)] = pars.numsmall

# Dust-to-gas ratio: it's assumed equal to Z in Diemer et al. (2018)

if dust_to_gas_model == 'fixed_ratio':
    D_mw  = dust_to_gas_ratio

elif dust_to_gas_model == 'metallicity_dependent':
    scat = np.random.normal(0,dust_to_gas_scatter, size=len(rhog))
    D_mw = 10**(dust_to_gas_slope * np.log10(metg) + dust_to_gas_zeropoint + scat)

else:
    print('Error: this model is not yet implemented. Falling back to a fixed dust-to-gas ratio')
    D_mw  = dust_to_gas_ratio
    
# Read the radiation field
rad = np.genfromtxt('../mean_intensity_1000A.out', skip_header=4) # This is in units erg / s / cm^2 / Hz / ster

#rad = np.reshape(rad, (ngridx, ngridy, ngridz), order='F') 

# See the end of Appendix A in Diemer et al. (2018): this is in units photons / s / cm^2 / Hz
# Let's multiply by 4*pi in order to match J_nu definition and convert photons into erg
# Computation:
# lam = 1000A = 1000 x 1e-10 m
# nu = cc / lam [Hz] = 2.99792458e-5 Hz
# 2.99792458e-5 Hz = 1.986445e-31 erg
rad_draine = 3.43e-8 * (4*np.pi) * 1.986445e-31


# UV field normalized at 1000A
U_mw = rad / rad_draine

# **************************************************
# **************************************************
# GNEDIN & KRAVTSOV (2011)
# Define the formulas

if splitting_method == 'GK11':

    ss = 0.04 / ( D_mw + 1.5e-3 * np.log( 1 + (3 * U_mw)**1.7) )

    alpha = 5. * (0.5 * U_mw) / ( 1 + (0.5 * U_mw)**2 )

    gg = (1 + alpha*ss + ss**2) / (1 + ss)

    # Now compute the critical density, in units M* / kpc^3
    sigmac = 2e7 * ( np.log(1 + gg*D_mw**(3./7.) * (U_mw / 15.)**(4./7.) )**(4./7.) ) / ( D_mw * np.sqrt(1 + U_mw * D_mw**2) )

    # Finally, compute the molecular fraction
    fmol = rhog.copy()
    fmol[np.where(rhog!=0.)] = 1 / (1 + sigmac[np.where(rhog!=0.)]/rhog[np.where(rhog!=0.)])**2
    fmol[np.where(rhog==0.)] = -1.
    fmol = fmol.flatten()

# GNEDIN & DRAINE (2014)
elif splitting_method == 'GD14':

    ss = lxcell / 100.

    Dstar = 0.17 * (2. + ss**5) / (1. + ss**5)

    g = np.sqrt(D_mw**2 + Dstar**2)

    sigmac = 5e7 * np.sqrt(0.001 + 0.1*U_mw) / (gg * (1. + 1.69* np.sqrt(0.001 + 0.1*U_mw)))

    alpha = 0.5 + 1./(1. + np.sqrt(U_mw * D_mw**2 / 600.))

    # Finally, compute the molecular fraction                     
    fmol = rhog.copy()
    fmol[np.where(rhog!=0.)] = 1 / (1 + (sigmac[np.where(rhog!=0.)]/rhog[np.where(rhog!=0.)])**(-alpha) )**2
    fmol[np.where(rhog==0.)] = -1.
    fmol = fmol.flatten()

else:
    print('No other methods immplemented. Exit.')
    exit()
    
fmol.astype('float64').tofile(pars.input_dir + pars.fmol_filename)

print('... done!')
print('---------------------------------')
print('')
