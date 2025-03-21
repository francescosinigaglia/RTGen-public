import numpy as np
import os
from numba import njit, prange
from numba.typed import List
import time
import numba as nb
import input_params as pars
import constants as const

# INPUT          
srcdir = pars.input_galaxy_dir
basedir = pars.input_dir

filename = srcdir + pars.input_galaxy_filename

nthreads = pars.nthreads

mass_star_filename = '../sph_interpolation/mass_star.npy'#pars.input_dir + pars.mass_star_filename
metallicity_star_filename = '../sph_interpolation/metallicity_star.npy' #pars.input_dir + pars.metallicity_star_filename
tform_star_filename = '../sph_interpolation/tform_star.npy' #pars.input_dir + pars.tform_star_filename

ntempgrid = 301
nlamgrid = 1221

os.system('export NUMBA_NUM_THREADS=%d' %nthreads)

nx = pars.nx
ny = pars.ny
nz = pars.nz

zs = const.zs

mstargrid = np.array([1e3, 5e3, 1e4, 2.5e4, 5e4, 7.5e4, 1e5, 2.5e5, 5e5, 7.5e5, 1e6, 2.5e6, 5e6, 7.5e6, 1e7, 2.5e7, 5e7, 7.5e7, 1e8, 5e8, 1e9])
zgrid = np.array([0.0004,0.004, 0.008, 0.020, 0.050]) / zs

# *************************************************
# *************************************************
# *************************************************
@njit(parallel=False,fastmath=True, cache=True)
def BilinearInterpolation(x_in, y_in, x_grid, y_grid, f_grid):

    idx = np.searchsorted(x_grid, x_in)
    idy = np.searchsorted(y_grid, y_in)
    
    if idx==0 or idx>=(len(x_grid)-1): # x outside left bondary
        if idy==0 or idy>=(len(y_grid)-1): # y outside left or right boundary

            fout = f_grid[idx,idy]

        else: # y inside boundary
            y2 = y_grid[idy]
            y1 = y_grid[idy-1]

            f2 = f_grid[idx,idy]
            f1 = f_grid[idx,idy-1]
            
            fout = (f2  * (y2 - y_in) + f1  * (y_in - y1)) / (y2 - y1)            
            

    else: # if idx!=0, x inside boundary   

        if idy==0 or idy>=(len(y_grid)-1): # y outside boundary

            x2 = x_grid[idx]
            x1 = x_grid[idx-1]

            f2 = f_grid[idx,idy]
            f1 = f_grid[idx-1,idy]

            fout = (f2  * (x2 - x_in) + f1  * (x_in - x1)) / (x2 - x1)

        else: # both x and y inside boundaries

            y2 = y_grid[idy]
            y1 = y_grid[idy-1]

            x2 = x_grid[idx]
            x1 = x_grid[idx-1]

            f22 = f_grid[idx,idy]
            f12 = f_grid[idx-1,idy]
            f21 = f_grid[idx,idy-1]
            f11 = f_grid[idx-1,idy-1]

            fout = (f22 *(x2-x_in)*(y2-y_in) + f12 *(x_in-x1)*(y2-y_in) + f21 *(x2-x_in)*(y_in-y1) + f11 *(x_in-x1)*(y_in-y1)) / ((x2 - x1)*(y2 - y1))

    return fout

# *************************************************
@njit(parallel=False,fastmath=True, cache=True)
def TrilinearInterpolation(x_in, y_in, z_in, x_grid, y_grid, z_grid, f_grid):

    idx = np.searchsorted(x_grid, x_in)
    idy = np.searchsorted(y_grid, y_in)
    idz = np.searchsorted(z_grid, z_in)

    #print(idx, idy,idz)
    
    if idx==0 or idx>=(len(x_grid)-1): # x outside left boundary or right boundary
        if idy==0 or idy>=(len(y_grid)-1): # y outside left or right boundary
            if idz<=0 or idz>=(len(z_grid)-1): # z outside left or right boundary

                fout = f_grid[idx,idy,idz]

            else: # z inside boundary
                z2 = z_grid[idz]
                z1 = z_grid[idz-1]
                
                f2 = f_grid[idx,idy,idz]
                f1 = f_grid[idx,idy,idz-1]
                
                fout = (f2  * (z2 - z_in) + f1  * (z_in - z1)) / (z2 - z1)  

        elif idy>0 or idy<(len(y_grid)-1): # y inside boundary
            if idz<=0 or idz>=(len(z_grid)-1): # z outside left or right boundary
                
                y2 = y_grid[idy]
                y1 = y_grid[idy-1]
                
                f2 = f_grid[idx,idy,idz]
                f1 = f_grid[idx,idy-1,idz]
                
                fout = (f2  * (y2 - y_in) + f1  * (y_in - y1)) / (y2 - y1)            
            
            else: # z inside boundary

                y2 = y_grid[idy]
                y1 = y_grid[idy-1]

                z2 = z_grid[idz]
                z1 = z_grid[idz-1]

                f022 = f_grid[idx,idy,idz]
                f012 = f_grid[idx,idy-1,idz]
                f021 = f_grid[idx,idy,idz-1]
                f011 = f_grid[idx,idy-1,idz-1]

                fout = (f022 *(y2-y_in)*(z2-z_in) + f012 *(y_in-y1)*(z2-z_in) + f021 *(y2-y_in)*(z_in-z1) + f011 *(y_in-y1)*(z_in-z1)) / ((y2 - y1)*(z2 - z1))


    else: # if idx!=0, x inside boundary   

        if idy==0 or idy>=(len(y_grid)-1): # y outside boundary
            if idz<=0 or idz>=(len(z_grid)-1): # z outside left or right boundary:

                x2 = x_grid[idx]
                x1 = x_grid[idx-1]

                f2 = f_grid[idx,idy,idz]
                f1 = f_grid[idx-1,idy,idz]

                fout = (f2  * (x2 - x_in) + f1  * (x_in - x1)) / (x2 - x1)

            elif idz>0 or idz<(len(z_grid)-1): # z inside boundary
                
                z2 = z_grid[idz]
                z1 = z_grid[idz-1]

                x2 = x_grid[idx]
                x1 = x_grid[idx-1]

                f101 = f_grid[idx-1,idy,idz-1]
                f102 = f_grid[idx-1,idy,idz]
                f201 = f_grid[idx,idy,idz-1]
                f202 = f_grid[idx,idy,idz]

                fout = (f101 *(x_in-x1)*(z_in-z1) + f102 *(x_in-x1)*(z2-z_in) + f201 *(x2-x_in)*(z_in-z1) + f202 *(x2-x_in)*(z2-z_in)) / ((x2 - x1)*(z2 - z1))

        elif idy>0 or idy<(len(y_grid)-1): # y inside boundary
            if idz<=0 or idz>=(len(z_grid)-1): # z outside left or right boundary:

                y2 = y_grid[idy]
                y1 = y_grid[idy-1]

                x2 = x_grid[idx]
                x1 = x_grid[idx-1]

                f110 = f_grid[idx-1,idy-1,idz]
                f120 = f_grid[idx-1,idy,idz]
                f210 = f_grid[idx,idy-1,idz]
                f220 = f_grid[idx,idy,idz]

                fout = (f110 *(x_in-x1)*(y_in-y1) + f120 *(x_in-x1)*(y2-y_in) + f210 *(x2-x_in)*(y_in-y1) + f220 *(x2-x_in)*(y2-y_in)) / ((x2 - x1)*(y2 - y1))


            else: # all variables inside boundaries
                
                x2 = x_grid[idx]
                x1 = x_grid[idx-1]

                y2 = y_grid[idy]
                y1 = y_grid[idy-1]

                z2 = z_grid[idz]
                z1 = z_grid[idz-1]

                f111 = f_grid[idx-1,idy-1,idz-1]
                f112 = f_grid[idx-1,idy-1,idz]
                f121 = f_grid[idx-1,idy,idz-1]
                f211 = f_grid[idx,idy-1,idz-1]
                f122 = f_grid[idx-1,idy,idz]
                f212 = f_grid[idx,idy-1,idz]
                f221 = f_grid[idx,idy,idz-1]
                f222 = f_grid[idx,idy,idz]

                fout = (f111*(x_in-x1)*(y_in-y1)*(z_in-z1) + f112*(x_in-x1)*(y_in-y1)*(z2-z_in) + f121*(x_in-x1)*(y2-y_in)*(z_in-z1) 
                        + f211*(x2-x_in)*(y_in-y1)*(z_in-z1) + f122*(x_in-x1)*(y2-y_in)*(z2-z_in) + f212*(x2-x_in)*(y_in-y1)*(z2-z_in)
                        + f221*(x2-x_in)*(y2-y_in)*(z_in-z1) + f222*(x2-x_in)*(y2-y_in)*(z2-z_in))/ (((x2 - x1)*(y2 - y1)*(z2-z1)))


    return fout

# *************************************************
# This function loops over the wavelenght grid and at each wavelenght:
# 1) performs a bilinear interpolation of the spectrum, cell by cell
# 2) performs the mass-weighted average of all cells
@njit(parallel=True,fastmath=True, cache=True)
def GetSpectrum(lam, spec_arr, zstar, mstar, tformstar, zgrid, mstargrid, tformgrid, ngridtot):

    """
    @ lam: wavelenght array, N-dim array
    @ spec_arr: array of spectra (Z,M,N)-im array, Z=len(zgrid), M=len(tformgrid), N=len(lam)
    @ mstar: M* array, flattened array of dimension ngrid**3
    @ zstar: metallicity array, flattened array of dimension ngrid**3
    @ tformstar: formation time array, flattened array of dimension ngrid**3
    @ zgrid: metallicity grid array
    @ mstargrid: M* array grid array
    @ tformgrid: formation time grid array
    """

    specout = np.zeros(len(lam))
    
    for ii in range(len(lam)): # Loop over spectrum

        #if ii>0:
        #    break

        dummymesh = np.zeros((ngridtot))

        specgrid = spec_arr[:,:,:,ii]

        for jj in prange(ngridtot): # Loop over meshes

            if mstar[jj]>0.:
                #print(mstar[jj], zstar[jj], tformstar[jj])
                spec_interp = TrilinearInterpolation(zstar[jj], mstar[jj], tformstar[jj], zgrid, mstargrid,tformgrid, specgrid)
                dummymesh[jj] = spec_interp

        spec_final = np.sum(dummymesh)

        specout[ii] = spec_final

    return specout
        
# *************************************************
# *************************************************
# *************************************************

# READ INPUT FILES
print('Reading input files...')

mass_star = np.load(mass_star_filename) #np.fromfile(mass_star_filename, dtype=np.float64)
metallicity_star = np.load(metallicity_star_filename) #np.fromfile(metallicity_star_filename, dtype=np.float64)
tform_star = np.load(tform_star_filename) #np.fromfile(tform_star_filename, dtype=np.float64)

#print(np.amin(mass_star), np.amax(mass_star), np.mean(mass_star), np.sum(mass_star))

ngridtot = len(mass_star) #int(nx * ny * nz)

# *************************************************
# Now read the Starburst 99 files
# Mstar = 1e3 M_sun
z00004_1e3_raw = np.genfromtxt('Padova_tracks/mstar_1e3/Padova_z0.0004_1e3.spectrum')
z0004_1e3_raw = np.genfromtxt('Padova_tracks/mstar_1e3/Padova_z0.004_1e3.spectrum')
z0008_1e3_raw = np.genfromtxt('Padova_tracks/mstar_1e3/Padova_z0.008_1e3.spectrum')
z0020_1e3_raw = np.genfromtxt('Padova_tracks/mstar_1e3/Padova_z0.020_1e3.spectrum')
z0050_1e3_raw = np.genfromtxt('Padova_tracks/mstar_1e3/Padova_z0.050_1e3.spectrum')

# Mstar = 5e3 M_sun
z00004_5e3_raw = np.genfromtxt('Padova_tracks/mstar_5e3/Padova_z0.0004_5e3.spectrum')
z0004_5e3_raw = np.genfromtxt('Padova_tracks/mstar_5e3/Padova_z0.004_5e3.spectrum')
z0008_5e3_raw = np.genfromtxt('Padova_tracks/mstar_5e3/Padova_z0.008_5e3.spectrum')
z0020_5e3_raw = np.genfromtxt('Padova_tracks/mstar_5e3/Padova_z0.020_5e3.spectrum')
z0050_5e3_raw = np.genfromtxt('Padova_tracks/mstar_5e3/Padova_z0.050_5e3.spectrum')

# Mstar = 1e4 M_sun
z00004_1e4_raw = np.genfromtxt('Padova_tracks/mstar_1e4/Padova_z0.0004_1e4.spectrum')
z0004_1e4_raw = np.genfromtxt('Padova_tracks/mstar_1e4/Padova_z0.004_1e4.spectrum')
z0008_1e4_raw = np.genfromtxt('Padova_tracks/mstar_1e4/Padova_z0.008_1e4.spectrum')
z0020_1e4_raw = np.genfromtxt('Padova_tracks/mstar_1e4/Padova_z0.020_1e4.spectrum')
z0050_1e4_raw = np.genfromtxt('Padova_tracks/mstar_1e4/Padova_z0.050_1e4.spectrum')

# Mstar = 2.5e4 M_sun
z00004_25e4_raw = np.genfromtxt('Padova_tracks/mstar_2.5e4/Padova_z0.0004_2.5e4.spectrum')
z0004_25e4_raw = np.genfromtxt('Padova_tracks/mstar_2.5e4/Padova_z0.004_2.5e4.spectrum')
z0008_25e4_raw = np.genfromtxt('Padova_tracks/mstar_2.5e4/Padova_z0.008_2.5e4.spectrum')
z0020_25e4_raw = np.genfromtxt('Padova_tracks/mstar_2.5e4/Padova_z0.020_2.5e4.spectrum')
z0050_25e4_raw = np.genfromtxt('Padova_tracks/mstar_2.5e4/Padova_z0.050_2.5e4.spectrum')

# Mstar = 5e4 M_sun
z00004_5e4_raw = np.genfromtxt('Padova_tracks/mstar_5e4/Padova_z0.0004_5e4.spectrum')
z0004_5e4_raw = np.genfromtxt('Padova_tracks/mstar_5e4/Padova_z0.004_5e4.spectrum')
z0008_5e4_raw = np.genfromtxt('Padova_tracks/mstar_5e4/Padova_z0.008_5e4.spectrum')
z0020_5e4_raw = np.genfromtxt('Padova_tracks/mstar_5e4/Padova_z0.020_5e4.spectrum')
z0050_5e4_raw = np.genfromtxt('Padova_tracks/mstar_5e4/Padova_z0.050_5e4.spectrum')

# Mstar = 7.5e4 M_sun
z00004_75e4_raw = np.genfromtxt('Padova_tracks/mstar_7.5e4/Padova_z0.0004_7.5e4.spectrum')
z0004_75e4_raw = np.genfromtxt('Padova_tracks/mstar_7.5e4/Padova_z0.004_7.5e4.spectrum')
z0008_75e4_raw = np.genfromtxt('Padova_tracks/mstar_7.5e4/Padova_z0.008_7.5e4.spectrum')
z0020_75e4_raw = np.genfromtxt('Padova_tracks/mstar_7.5e4/Padova_z0.020_7.5e4.spectrum')
z0050_75e4_raw = np.genfromtxt('Padova_tracks/mstar_7.5e4/Padova_z0.050_7.5e4.spectrum')

# Mstar = 1e5 M_sun
z00004_1e5_raw = np.genfromtxt('Padova_tracks/mstar_1e5/Padova_z0.0004_1e5.spectrum')
z0004_1e5_raw = np.genfromtxt('Padova_tracks/mstar_1e5/Padova_z0.004_1e5.spectrum')
z0008_1e5_raw = np.genfromtxt('Padova_tracks/mstar_1e5/Padova_z0.008_1e5.spectrum')
z0020_1e5_raw = np.genfromtxt('Padova_tracks/mstar_1e5/Padova_z0.020_1e5.spectrum')
z0050_1e5_raw = np.genfromtxt('Padova_tracks/mstar_1e5/Padova_z0.050_1e5.spectrum')

# Mstar = 2.5e5 M_sun
z00004_25e5_raw = np.genfromtxt('Padova_tracks/mstar_2.5e5/Padova_z0.0004_2.5e5.spectrum')
z0004_25e5_raw = np.genfromtxt('Padova_tracks/mstar_2.5e5/Padova_z0.004_2.5e5.spectrum')
z0008_25e5_raw = np.genfromtxt('Padova_tracks/mstar_2.5e5/Padova_z0.008_2.5e5.spectrum')
z0020_25e5_raw = np.genfromtxt('Padova_tracks/mstar_2.5e5/Padova_z0.020_2.5e5.spectrum')
z0050_25e5_raw = np.genfromtxt('Padova_tracks/mstar_2.5e5/Padova_z0.050_2.5e5.spectrum')

# Mstar = 5e5 M_sun
z00004_5e5_raw = np.genfromtxt('Padova_tracks/mstar_5e5/Padova_z0.0004_5e5.spectrum')
z0004_5e5_raw = np.genfromtxt('Padova_tracks/mstar_5e5/Padova_z0.004_5e5.spectrum')
z0008_5e5_raw = np.genfromtxt('Padova_tracks/mstar_5e5/Padova_z0.008_5e5.spectrum')
z0020_5e5_raw = np.genfromtxt('Padova_tracks/mstar_5e5/Padova_z0.020_5e5.spectrum')
z0050_5e5_raw = np.genfromtxt('Padova_tracks/mstar_5e5/Padova_z0.050_5e5.spectrum')

# Mstar = 7.5e5 M_sun
z00004_75e5_raw = np.genfromtxt('Padova_tracks/mstar_7.5e5/Padova_z0.0004_7.5e5.spectrum')
z0004_75e5_raw = np.genfromtxt('Padova_tracks/mstar_7.5e5/Padova_z0.004_7.5e5.spectrum')
z0008_75e5_raw = np.genfromtxt('Padova_tracks/mstar_7.5e5/Padova_z0.008_7.5e5.spectrum')
z0020_75e5_raw = np.genfromtxt('Padova_tracks/mstar_7.5e5/Padova_z0.020_7.5e5.spectrum')
z0050_75e5_raw = np.genfromtxt('Padova_tracks/mstar_7.5e5/Padova_z0.050_7.5e5.spectrum')

# Mstar = 1e6 M_sun
z00004_1e6_raw = np.genfromtxt('Padova_tracks/mstar_1e6/Padova_z0.0004_1e6.spectrum')
z0004_1e6_raw = np.genfromtxt('Padova_tracks/mstar_1e6/Padova_z0.004_1e6.spectrum')
z0008_1e6_raw = np.genfromtxt('Padova_tracks/mstar_1e6/Padova_z0.008_1e6.spectrum')
z0020_1e6_raw = np.genfromtxt('Padova_tracks/mstar_1e6/Padova_z0.020_1e6.spectrum')
z0050_1e6_raw = np.genfromtxt('Padova_tracks/mstar_1e6/Padova_z0.050_1e6.spectrum')

# Mstar = 2.5e6 M_sun
z00004_25e6_raw = np.genfromtxt('Padova_tracks/mstar_2.5e6/Padova_z0.0004_2.5e6.spectrum')
z0004_25e6_raw = np.genfromtxt('Padova_tracks/mstar_2.5e6/Padova_z0.004_2.5e6.spectrum')
z0008_25e6_raw = np.genfromtxt('Padova_tracks/mstar_2.5e6/Padova_z0.008_2.5e6.spectrum')
z0020_25e6_raw = np.genfromtxt('Padova_tracks/mstar_2.5e6/Padova_z0.020_2.5e6.spectrum')
z0050_25e6_raw = np.genfromtxt('Padova_tracks/mstar_2.5e6/Padova_z0.050_2.5e6.spectrum')

# Mstar = 5e6 M_sun
z00004_5e6_raw = np.genfromtxt('Padova_tracks/mstar_5e6/Padova_z0.0004_5e6.spectrum')
z0004_5e6_raw = np.genfromtxt('Padova_tracks/mstar_5e6/Padova_z0.004_5e6.spectrum')
z0008_5e6_raw = np.genfromtxt('Padova_tracks/mstar_5e6/Padova_z0.008_5e6.spectrum')
z0020_5e6_raw = np.genfromtxt('Padova_tracks/mstar_5e6/Padova_z0.020_5e6.spectrum')
z0050_5e6_raw = np.genfromtxt('Padova_tracks/mstar_5e6/Padova_z0.050_5e6.spectrum')

# Mstar = 7.5e6 M_sun
z00004_75e6_raw = np.genfromtxt('Padova_tracks/mstar_7.5e6/Padova_z0.0004_7.5e6.spectrum')
z0004_75e6_raw = np.genfromtxt('Padova_tracks/mstar_7.5e6/Padova_z0.004_7.5e6.spectrum')
z0008_75e6_raw = np.genfromtxt('Padova_tracks/mstar_7.5e6/Padova_z0.008_7.5e6.spectrum')
z0020_75e6_raw = np.genfromtxt('Padova_tracks/mstar_7.5e6/Padova_z0.020_7.5e6.spectrum')
z0050_75e6_raw = np.genfromtxt('Padova_tracks/mstar_7.5e6/Padova_z0.050_7.5e6.spectrum')

# Mstar = 1e7 M_sun
z00004_1e7_raw = np.genfromtxt('Padova_tracks/mstar_1e7/Padova_z0.0004_1e7.spectrum')
z0004_1e7_raw = np.genfromtxt('Padova_tracks/mstar_1e7/Padova_z0.004_1e7.spectrum')
z0008_1e7_raw = np.genfromtxt('Padova_tracks/mstar_1e7/Padova_z0.008_1e7.spectrum')
z0020_1e7_raw = np.genfromtxt('Padova_tracks/mstar_1e7/Padova_z0.020_1e7.spectrum')
z0050_1e7_raw = np.genfromtxt('Padova_tracks/mstar_1e7/Padova_z0.050_1e7.spectrum')

# Mstar = 2.5e7 M_sun
z00004_25e7_raw = np.genfromtxt('Padova_tracks/mstar_2.5e7/Padova_z0.0004_2.5e7.spectrum')
z0004_25e7_raw = np.genfromtxt('Padova_tracks/mstar_2.5e7/Padova_z0.004_2.5e7.spectrum')
z0008_25e7_raw = np.genfromtxt('Padova_tracks/mstar_2.5e7/Padova_z0.008_2.5e7.spectrum')
z0020_25e7_raw = np.genfromtxt('Padova_tracks/mstar_2.5e7/Padova_z0.020_2.5e7.spectrum')
z0050_25e7_raw = np.genfromtxt('Padova_tracks/mstar_2.5e7/Padova_z0.050_2.5e7.spectrum')

# Mstar = 5e7 M_sun
z00004_5e7_raw = np.genfromtxt('Padova_tracks/mstar_5e7/Padova_z0.0004_5e7.spectrum')
z0004_5e7_raw = np.genfromtxt('Padova_tracks/mstar_5e7/Padova_z0.004_5e7.spectrum')
z0008_5e7_raw = np.genfromtxt('Padova_tracks/mstar_5e7/Padova_z0.008_5e7.spectrum')
z0020_5e7_raw = np.genfromtxt('Padova_tracks/mstar_5e7/Padova_z0.020_5e7.spectrum')
z0050_5e7_raw = np.genfromtxt('Padova_tracks/mstar_5e7/Padova_z0.050_5e7.spectrum')

# Mstar = 7.5e7 M_sun
z00004_75e7_raw = np.genfromtxt('Padova_tracks/mstar_7.5e7/Padova_z0.0004_7.5e7.spectrum')
z0004_75e7_raw = np.genfromtxt('Padova_tracks/mstar_7.5e7/Padova_z0.004_7.5e7.spectrum')
z0008_75e7_raw = np.genfromtxt('Padova_tracks/mstar_7.5e7/Padova_z0.008_7.5e7.spectrum')
z0020_75e7_raw = np.genfromtxt('Padova_tracks/mstar_7.5e7/Padova_z0.020_7.5e7.spectrum')
z0050_75e7_raw = np.genfromtxt('Padova_tracks/mstar_7.5e7/Padova_z0.050_7.5e7.spectrum')

# Mstar = 1e8 M_sun
z00004_1e8_raw = np.genfromtxt('Padova_tracks/mstar_1e8/Padova_z0.0004_1e8.spectrum')
z0004_1e8_raw = np.genfromtxt('Padova_tracks/mstar_1e8/Padova_z0.004_1e8.spectrum')
z0008_1e8_raw = np.genfromtxt('Padova_tracks/mstar_1e8/Padova_z0.008_1e8.spectrum')
z0020_1e8_raw = np.genfromtxt('Padova_tracks/mstar_1e8/Padova_z0.020_1e8.spectrum')
z0050_1e8_raw = np.genfromtxt('Padova_tracks/mstar_1e8/Padova_z0.050_1e8.spectrum')

# Mstar = 5e8 M_sun
z00004_5e8_raw = np.genfromtxt('Padova_tracks/mstar_5e8/Padova_z0.0004_5e8.spectrum')
z0004_5e8_raw = np.genfromtxt('Padova_tracks/mstar_5e8/Padova_z0.004_5e8.spectrum')
z0008_5e8_raw = np.genfromtxt('Padova_tracks/mstar_5e8/Padova_z0.008_5e8.spectrum')
z0020_5e8_raw = np.genfromtxt('Padova_tracks/mstar_5e8/Padova_z0.020_5e8.spectrum')
z0050_5e8_raw = np.genfromtxt('Padova_tracks/mstar_5e8/Padova_z0.050_5e8.spectrum')

# Mstar = 1e9 M_sun
z00004_1e9_raw = np.genfromtxt('Padova_tracks/mstar_1e9/Padova_z0.0004_1e9.spectrum')
z0004_1e9_raw = np.genfromtxt('Padova_tracks/mstar_1e9/Padova_z0.004_1e9.spectrum')
z0008_1e9_raw = np.genfromtxt('Padova_tracks/mstar_1e9/Padova_z0.008_1e9.spectrum')
z0020_1e9_raw = np.genfromtxt('Padova_tracks/mstar_1e9/Padova_z0.020_1e9.spectrum')
z0050_1e9_raw = np.genfromtxt('Padova_tracks/mstar_1e9/Padova_z0.050_1e9.spectrum')

print('... done!')
print('')

# *************************************************

print('Prepare interpolation grids ...')
# Lambda template
lam_template = z00004_1e3_raw[:nlamgrid,1] / 1e4 # Divide by 1e4 to pass from Angstroms (output of SB99) to um (input of RADMC-3D)

# Formation time
tformgrid = []

tform_arr = z00004_1e3_raw[:,0]
for kk in range(ntempgrid):
    tformgrid.append(tform_arr[kk*nlamgrid])

tformgrid = np.array(tformgrid) / 1e9 # Divide by 1e9 to pass from yr to Gyr
    
# Spectra
# Mstar = 1e3 M_sun
z00004_1e3 = 10**z00004_1e3_raw[:,3]
z0004_1e3 = 10**z0004_1e3_raw[:,3]
z0008_1e3 = 10**z0008_1e3_raw[:,3]
z0020_1e3 = 10**z0020_1e3_raw[:,3]
z0050_1e3 = 10**z0050_1e3_raw[:,3]

# Mstar = 5e3 M_sun
z00004_5e3 = 10**z00004_5e3_raw[:,3]
z0004_5e3 = 10**z0004_5e3_raw[:,3]
z0008_5e3 = 10**z0008_5e3_raw[:,3]
z0020_5e3 = 10**z0020_5e3_raw[:,3]
z0050_5e3 = 10**z0050_5e3_raw[:,3]

# Mstar = 1e4 M_sun
z00004_1e4 = 10**z00004_1e4_raw[:,3]
z0004_1e4 = 10**z0004_1e4_raw[:,3]
z0008_1e4 = 10**z0008_1e4_raw[:,3]
z0020_1e4 = 10**z0020_1e4_raw[:,3]
z0050_1e4 = 10**z0050_1e4_raw[:,3]

# Mstar = 2.5e4 M_sun
z00004_25e4 = 10**z00004_25e4_raw[:,3]
z0004_25e4 = 10**z0004_25e4_raw[:,3]
z0008_25e4 = 10**z0008_25e4_raw[:,3]
z0020_25e4 = 10**z0020_25e4_raw[:,3]
z0050_25e4 = 10**z0050_25e4_raw[:,3]

# Mstar = 5e4 M_sun
z00004_5e4 = 10**z00004_5e4_raw[:,3]
z0004_5e4 = 10**z0004_5e4_raw[:,3]
z0008_5e4 = 10**z0008_5e4_raw[:,3]
z0020_5e4 = 10**z0020_5e4_raw[:,3]
z0050_5e4 = 10**z0050_5e4_raw[:,3]

# Mstar = 7.5e4 M_sun
z00004_75e4 = 10**z00004_75e4_raw[:,3]
z0004_75e4 = 10**z0004_75e4_raw[:,3]
z0008_75e4 = 10**z0008_75e4_raw[:,3]
z0020_75e4 = 10**z0020_75e4_raw[:,3]
z0050_75e4 = 10**z0050_75e4_raw[:,3]

# Mstar = 1e5 M_sun
z00004_1e5 = 10**z00004_1e5_raw[:,3]
z0004_1e5 = 10**z0004_1e5_raw[:,3]
z0008_1e5 = 10**z0008_1e5_raw[:,3]
z0020_1e5 = 10**z0020_1e5_raw[:,3]
z0050_1e5 = 10**z0050_1e5_raw[:,3]

# Mstar = 2.5e5 M_sun
z00004_25e5 = 10**z00004_25e5_raw[:,3]
z0004_25e5 = 10**z0004_25e5_raw[:,3]
z0008_25e5 = 10**z0008_25e5_raw[:,3]
z0020_25e5 = 10**z0020_25e5_raw[:,3]
z0050_25e5 = 10**z0050_25e5_raw[:,3]

# Mstar = 5e5 M_sun
z00004_5e5 = 10**z00004_5e5_raw[:,3]
z0004_5e5 = 10**z0004_5e5_raw[:,3]
z0008_5e5 = 10**z0008_5e5_raw[:,3]
z0020_5e5 = 10**z0020_5e5_raw[:,3]
z0050_5e5 = 10**z0050_5e5_raw[:,3]

# Mstar = 7.5e5 M_sun
z00004_75e5 = 10**z00004_75e5_raw[:,3]
z0004_75e5 = 10**z0004_75e5_raw[:,3]
z0008_75e5 = 10**z0008_75e5_raw[:,3]
z0020_75e5 = 10**z0020_75e5_raw[:,3]
z0050_75e5 = 10**z0050_75e5_raw[:,3]

# Mstar = 1e6 M_sun
z00004_1e6 = 10**z00004_1e6_raw[:,3]
z0004_1e6 = 10**z0004_1e6_raw[:,3]
z0008_1e6 = 10**z0008_1e6_raw[:,3]
z0020_1e6 = 10**z0020_1e6_raw[:,3]
z0050_1e6 = 10**z0050_1e6_raw[:,3]

# Mstar = 2.5e6 M_sun
z00004_25e6 = 10**z00004_25e6_raw[:,3]
z0004_25e6 = 10**z0004_25e6_raw[:,3]
z0008_25e6 = 10**z0008_25e6_raw[:,3]
z0020_25e6 = 10**z0020_25e6_raw[:,3]
z0050_25e6 = 10**z0050_25e6_raw[:,3]

# Mstar = 5e6 M_sun
z00004_5e6 = 10**z00004_5e6_raw[:,3]
z0004_5e6 = 10**z0004_5e6_raw[:,3]
z0008_5e6 = 10**z0008_5e6_raw[:,3]
z0020_5e6 = 10**z0020_5e6_raw[:,3]
z0050_5e6 = 10**z0050_5e6_raw[:,3]

# Mstar = 7.5e6 M_sun
z00004_75e6 = 10**z00004_75e6_raw[:,3]
z0004_75e6 = 10**z0004_75e6_raw[:,3]
z0008_75e6 = 10**z0008_75e6_raw[:,3]
z0020_75e6 = 10**z0020_75e6_raw[:,3]
z0050_75e6 = 10**z0050_75e6_raw[:,3]

# Mstar = 1e7 M_sun
z00004_1e7 = 10**z00004_1e7_raw[:,3]
z0004_1e7 = 10**z0004_1e7_raw[:,3]
z0008_1e7 = 10**z0008_1e7_raw[:,3]
z0020_1e7 = 10**z0020_1e7_raw[:,3]
z0050_1e7 = 10**z0050_1e7_raw[:,3]

# Mstar = 2.5e7 M_sun
z00004_25e7 = 10**z00004_25e7_raw[:,3]
z0004_25e7 = 10**z0004_25e7_raw[:,3]
z0008_25e7 = 10**z0008_25e7_raw[:,3]
z0020_25e7 = 10**z0020_25e7_raw[:,3]
z0050_25e7 = 10**z0050_25e7_raw[:,3]

# Mstar = 5e7 M_sun
z00004_5e7 = 10**z00004_5e7_raw[:,3]
z0004_5e7 = 10**z0004_5e7_raw[:,3]
z0008_5e7 = 10**z0008_5e7_raw[:,3]
z0020_5e7 = 10**z0020_5e7_raw[:,3]
z0050_5e7 = 10**z0050_5e7_raw[:,3]

# Mstar = 7.5e7 M_sun
z00004_75e7 = 10**z00004_75e7_raw[:,3]
z0004_75e7 = 10**z0004_75e7_raw[:,3]
z0008_75e7 = 10**z0008_75e7_raw[:,3]
z0020_75e7 = 10**z0020_75e7_raw[:,3]
z0050_75e7 = 10**z0050_75e7_raw[:,3]

# Mstar = 1e8 M_sun
z00004_1e8 = 10**z00004_1e8_raw[:,3]
z0004_1e8 = 10**z0004_1e8_raw[:,3]
z0008_1e8 = 10**z0008_1e8_raw[:,3]
z0020_1e8 = 10**z0020_1e8_raw[:,3]
z0050_1e8 = 10**z0050_1e8_raw[:,3]

# Mstar = 5e8 M_sun
z00004_5e8 = 10**z00004_5e8_raw[:,3]
z0004_5e8 = 10**z0004_5e8_raw[:,3]
z0008_5e8 = 10**z0008_5e8_raw[:,3]
z0020_5e8 = 10**z0020_5e8_raw[:,3]
z0050_5e8 = 10**z0050_5e8_raw[:,3]

# Mstar = 1e9 M_sun
z00004_1e9 = 10**z00004_1e9_raw[:,3]
z0004_1e9 = 10**z0004_1e9_raw[:,3]
z0008_1e9 = 10**z0008_1e9_raw[:,3]
z0020_1e9 = 10**z0020_1e9_raw[:,3]
z0050_1e9 = 10**z0050_1e9_raw[:,3]

# Now RESHAPE them in such a way to have the (spec,tform) array
# Mstar = 1e3 M_sun
z00004_1e3 = np.reshape(z00004_1e3, (ntempgrid,nlamgrid))
z0004_1e3 = np.reshape(z0004_1e3, (ntempgrid,nlamgrid))
z0008_1e3 = np.reshape(z0008_1e3, (ntempgrid,nlamgrid))
z0020_1e3 = np.reshape(z0020_1e3, (ntempgrid,nlamgrid))
z0050_1e3 = np.reshape(z0050_1e3, (ntempgrid,nlamgrid))

# Mstar = 5e3 M_sun
z00004_5e3 = np.reshape(z00004_5e3, (ntempgrid,nlamgrid))
z0004_5e3 = np.reshape(z0004_5e3, (ntempgrid,nlamgrid))
z0008_5e3 = np.reshape(z0008_5e3, (ntempgrid,nlamgrid))
z0020_5e3 = np.reshape(z0020_5e3, (ntempgrid,nlamgrid))
z0050_5e3 = np.reshape(z0050_5e3, (ntempgrid,nlamgrid))

# Mstar = 1e4 M_sun
z00004_1e4 = np.reshape(z00004_1e4, (ntempgrid,nlamgrid))
z0004_1e4 = np.reshape(z0004_1e4, (ntempgrid,nlamgrid))
z0008_1e4 = np.reshape(z0008_1e4, (ntempgrid,nlamgrid))
z0020_1e4 = np.reshape(z0020_1e4, (ntempgrid,nlamgrid))
z0050_1e4 = np.reshape(z0050_1e4, (ntempgrid,nlamgrid))

# Mstar = 2.5e4 M_sun
z00004_25e4 = np.reshape(z00004_25e4, (ntempgrid,nlamgrid))
z0004_25e4 = np.reshape(z0004_25e4, (ntempgrid,nlamgrid))
z0008_25e4 = np.reshape(z0008_25e4, (ntempgrid,nlamgrid))
z0020_25e4 = np.reshape(z0020_25e4, (ntempgrid,nlamgrid))
z0050_25e4 = np.reshape(z0050_25e4, (ntempgrid,nlamgrid))

# Mstar = 5e4 M_sun
z00004_5e4 = np.reshape(z00004_5e4, (ntempgrid,nlamgrid))
z0004_5e4 = np.reshape(z0004_5e4, (ntempgrid,nlamgrid))
z0008_5e4 = np.reshape(z0008_5e4, (ntempgrid,nlamgrid))
z0020_5e4 = np.reshape(z0020_5e4, (ntempgrid,nlamgrid))
z0050_5e4 = np.reshape(z0050_5e4, (ntempgrid,nlamgrid))

# Mstar = 7.5e4 M_sun
z00004_75e4 = np.reshape(z00004_75e4, (ntempgrid,nlamgrid))
z0004_75e4 = np.reshape(z0004_75e4, (ntempgrid,nlamgrid))
z0008_75e4 = np.reshape(z0008_75e4, (ntempgrid,nlamgrid))
z0020_75e4 = np.reshape(z0020_75e4, (ntempgrid,nlamgrid))
z0050_75e4 = np.reshape(z0050_75e4, (ntempgrid,nlamgrid))

# Mstar = 1e5 M_sun
z00004_1e5 = np.reshape(z00004_1e5, (ntempgrid,nlamgrid))
z0004_1e5 = np.reshape(z0004_1e5, (ntempgrid,nlamgrid))
z0008_1e5 = np.reshape(z0008_1e5, (ntempgrid,nlamgrid))
z0020_1e5 = np.reshape(z0020_1e5, (ntempgrid,nlamgrid))
z0050_1e5 = np.reshape(z0050_1e5, (ntempgrid,nlamgrid))

# Mstar = 2.5e5 M_sun
z00004_25e5 = np.reshape(z00004_25e5, (ntempgrid,nlamgrid))
z0004_25e5 = np.reshape(z0004_25e5, (ntempgrid,nlamgrid))
z0008_25e5 = np.reshape(z0008_25e5, (ntempgrid,nlamgrid))
z0020_25e5 = np.reshape(z0020_25e5, (ntempgrid,nlamgrid))
z0050_25e5 = np.reshape(z0050_25e5, (ntempgrid,nlamgrid))

# Mstar = 5e5 M_sun
z00004_5e5 = np.reshape(z00004_5e5, (ntempgrid,nlamgrid))
z0004_5e5 = np.reshape(z0004_5e5, (ntempgrid,nlamgrid))
z0008_5e5 = np.reshape(z0008_5e5, (ntempgrid,nlamgrid))
z0020_5e5 = np.reshape(z0020_5e5, (ntempgrid,nlamgrid))
z0050_5e5 = np.reshape(z0050_5e5, (ntempgrid,nlamgrid))

# Mstar = 7.5e5 M_sun
z00004_75e5 = np.reshape(z00004_75e5, (ntempgrid,nlamgrid))
z0004_75e5 = np.reshape(z0004_75e5, (ntempgrid,nlamgrid))
z0008_75e5 = np.reshape(z0008_75e5, (ntempgrid,nlamgrid))
z0020_75e5 = np.reshape(z0020_75e5, (ntempgrid,nlamgrid))
z0050_75e5 = np.reshape(z0050_75e5, (ntempgrid,nlamgrid))

# Mstar = 1e6 M_sun
z00004_1e6 = np.reshape(z00004_1e6, (ntempgrid,nlamgrid))
z0004_1e6 = np.reshape(z0004_1e6, (ntempgrid,nlamgrid))
z0008_1e6 = np.reshape(z0008_1e6, (ntempgrid,nlamgrid))
z0020_1e6 = np.reshape(z0020_1e6, (ntempgrid,nlamgrid))
z0050_1e6 = np.reshape(z0050_1e6, (ntempgrid,nlamgrid))

# Mstar = 2.5e6 M_sun
z00004_25e6 = np.reshape(z00004_25e6, (ntempgrid,nlamgrid))
z0004_25e6 = np.reshape(z0004_25e6, (ntempgrid,nlamgrid))
z0008_25e6 = np.reshape(z0008_25e6, (ntempgrid,nlamgrid))
z0020_25e6 = np.reshape(z0020_25e6, (ntempgrid,nlamgrid))
z0050_25e6 = np.reshape(z0050_25e6, (ntempgrid,nlamgrid))

# Mstar = 5e6 M_sun
z00004_5e6 = np.reshape(z00004_5e6, (ntempgrid,nlamgrid))
z0004_5e6 = np.reshape(z0004_5e6, (ntempgrid,nlamgrid))
z0008_5e6 = np.reshape(z0008_5e6, (ntempgrid,nlamgrid))
z0020_5e6 = np.reshape(z0020_5e6, (ntempgrid,nlamgrid))
z0050_5e6 = np.reshape(z0050_5e6, (ntempgrid,nlamgrid))

# Mstar = 7.5e6 M_sun
z00004_75e6 = np.reshape(z00004_75e6, (ntempgrid,nlamgrid))
z0004_75e6 = np.reshape(z0004_75e6, (ntempgrid,nlamgrid))
z0008_75e6 = np.reshape(z0008_75e6, (ntempgrid,nlamgrid))
z0020_75e6 = np.reshape(z0020_75e6, (ntempgrid,nlamgrid))
z0050_75e6 = np.reshape(z0050_75e6, (ntempgrid,nlamgrid))

# Mstar = 1e7 M_sun
z00004_1e7 = np.reshape(z00004_1e7, (ntempgrid,nlamgrid))
z0004_1e7 = np.reshape(z0004_1e7, (ntempgrid,nlamgrid))
z0008_1e7 = np.reshape(z0008_1e7, (ntempgrid,nlamgrid))
z0020_1e7 = np.reshape(z0020_1e7, (ntempgrid,nlamgrid))
z0050_1e7 = np.reshape(z0050_1e7, (ntempgrid,nlamgrid))

# Mstar = 2.5e7 M_sun
z00004_25e7 = np.reshape(z00004_25e7, (ntempgrid,nlamgrid))
z0004_25e7 = np.reshape(z0004_25e7, (ntempgrid,nlamgrid))
z0008_25e7 = np.reshape(z0008_25e7, (ntempgrid,nlamgrid))
z0020_25e7 = np.reshape(z0020_25e7, (ntempgrid,nlamgrid))
z0050_25e7 = np.reshape(z0050_25e7, (ntempgrid,nlamgrid))

# Mstar = 5e7 M_sun
z00004_5e7 = np.reshape(z00004_5e7, (ntempgrid,nlamgrid))
z0004_5e7 = np.reshape(z0004_5e7, (ntempgrid,nlamgrid))
z0008_5e7 = np.reshape(z0008_5e7, (ntempgrid,nlamgrid))
z0020_5e7 = np.reshape(z0020_5e7, (ntempgrid,nlamgrid))
z0050_5e7 = np.reshape(z0050_5e7, (ntempgrid,nlamgrid))

# Mstar = 7.5e7 M_sun
z00004_75e7 = np.reshape(z00004_75e7, (ntempgrid,nlamgrid))
z0004_75e7 = np.reshape(z0004_75e7, (ntempgrid,nlamgrid))
z0008_75e7 = np.reshape(z0008_75e7, (ntempgrid,nlamgrid))
z0020_75e7 = np.reshape(z0020_75e7, (ntempgrid,nlamgrid))
z0050_75e7 = np.reshape(z0050_75e7, (ntempgrid,nlamgrid))

# Mstar = 1e8 M_sun
z00004_1e8 = np.reshape(z00004_1e8, (ntempgrid,nlamgrid))
z0004_1e8 = np.reshape(z0004_1e8, (ntempgrid,nlamgrid))
z0008_1e8 = np.reshape(z0008_1e8, (ntempgrid,nlamgrid))
z0020_1e8 = np.reshape(z0020_1e8, (ntempgrid,nlamgrid))
z0050_1e8 = np.reshape(z0050_1e8, (ntempgrid,nlamgrid))

# Mstar = 5e8 M_sun
z00004_5e8 = np.reshape(z00004_5e8, (ntempgrid,nlamgrid))
z0004_5e8 = np.reshape(z0004_5e8, (ntempgrid,nlamgrid))
z0008_5e8 = np.reshape(z0008_5e8, (ntempgrid,nlamgrid))
z0020_5e8 = np.reshape(z0020_5e8, (ntempgrid,nlamgrid))
z0050_5e8 = np.reshape(z0050_5e8, (ntempgrid,nlamgrid))

# Mstar = 1e9 M_sun
z00004_1e9 = np.reshape(z00004_1e9, (ntempgrid,nlamgrid))
z0004_1e9 = np.reshape(z0004_1e9, (ntempgrid,nlamgrid))
z0008_1e9 = np.reshape(z0008_1e9, (ntempgrid,nlamgrid))
z0020_1e9 = np.reshape(z0020_1e9, (ntempgrid,nlamgrid))
z0050_1e9 = np.reshape(z0050_1e9, (ntempgrid,nlamgrid))

# Now transpose (to get (tform,spec)) and then stack. The final array is (metallicity,tform,spec)
"""
specarr = np.stack((z00004_1e3, z0004_1e3, z0008_1e3, z0020_1e3, z0050_1e3),
                   (z00004_5e3, z0004_5e3, z0008_5e3, z0020_5e3, z0050_5e3),
                   (z00004_1e4, z0004_1e4, z0008_1e4, z0020_1e4, z0050_1e4),
                   (z00004_25e4, z0004_25e4, z0008_25e4, z0020_25e4, z0050_25e4),
                   (z00004_5e4, z0004_5e4, z0008_5e4, z0020_5e4, z0050_5e4),
                   (z00004_75e4, z0004_75e4, z0008_75e4, z0020_75e4, z0050_75e4),
                   (z00004_1e5, z0004_1e5, z0008_1e5, z0020_1e5, z0050_1e5),
                   (z00004_25e5, z0004_25e5, z0008_25e5, z0020_25e5, z0050_25e5),
                   (z00004_5e5, z0004_5e5, z0008_5e5, z0020_5e5, z0050_5e5),
                   (z00004_75e5, z0004_75e5, z0008_75e5, z0020_75e5, z0050_75e5),
                   (z00004_1e6, z0004_1e6, z0008_1e6, z0020_1e6, z0050_1e6),
                   (z00004_25e6, z0004_25e6, z0008_25e6, z0020_25e6, z0050_25e6),
                   (z00004_5e6, z0004_5e6, z0008_5e6, z0020_5e6, z0050_5e6),
                   (z00004_75e6, z0004_75e6, z0008_75e6, z0020_75e6, z0050_75e6),
                   (z00004_1e7, z0004_1e7, z0008_1e7, z0020_1e7, z0050_1e7),
                   (z00004_25e7, z0004_25e7, z0008_25e7, z0020_25e7, z0050_25e7),
                   (z00004_5e7, z0004_5e7, z0008_5e7, z0020_5e7, z0050_5e7),
                   (z00004_75e7, z0004_75e7, z0008_75e7, z0020_75e7, z0050_75e7),
                   (z00004_1e8, z0004_1e8, z0008_1e8, z0020_1e8, z0050_1e8),
                   (z00004_5e8, z0004_5e8, z0008_5e8, z0020_5e8, z0050_5e8),
                   (z00004_1e9, z0004_1e9, z0008_1e9, z0020_1e9, z0050_1e9),
                   )
"""

specarr_z00004 = np.stack((z00004_1e3,z00004_5e3,z00004_1e4,z00004_25e4,z00004_5e4,z00004_75e4,z00004_1e5,z00004_25e5,z00004_5e5,z00004_75e5,z00004_1e6,z00004_25e6,z00004_5e6,z00004_75e6,z00004_1e7,z00004_25e7,z00004_5e7,z00004_75e7,z00004_1e8,z00004_5e8,z00004_1e9))
specarr_z0004 = np.stack((z0004_1e3,z0004_5e3,z0004_1e4,z0004_25e4,z0004_5e4,z0004_75e4,z0004_1e5,z0004_25e5,z0004_5e5,z0004_75e5,z0004_1e6,z0004_25e6,z0004_5e6,z0004_75e6,z0004_1e7,z0004_25e7,z0004_5e7,z0004_75e7,z0004_1e8,z0004_5e8,z0004_1e9))
specarr_z0008 = np.stack((z0008_1e3,z0008_5e3,z0008_1e4,z0008_25e4,z0008_5e4,z0008_75e4,z0008_1e5,z0008_25e5,z0008_5e5,z0008_75e5,z0008_1e6,z0008_25e6,z0008_5e6,z0008_75e6,z0008_1e7,z0008_25e7,z0008_5e7,z0008_75e7,z0008_1e8,z0008_5e8,z0008_1e9))
specarr_z0020 = np.stack((z0020_1e3,z0020_5e3,z0020_1e4,z0020_25e4,z0020_5e4,z0020_75e4,z0020_1e5,z0020_25e5,z0020_5e5,z0020_75e5,z0020_1e6,z0020_25e6,z0020_5e6,z0020_75e6,z0020_1e7,z0020_25e7,z0020_5e7,z0020_75e7,z0020_1e8,z0020_5e8,z0020_1e9))
specarr_z0050 = np.stack((z0050_1e3,z0050_5e3,z0050_1e4,z0050_25e4,z0050_5e4,z0050_75e4,z0050_1e5,z0050_25e5,z0050_5e5,z0050_75e5,z0050_1e6,z0050_25e6,z0050_5e6,z0050_75e6,z0050_1e7,z0050_25e7,z0050_5e7,z0050_75e7,z0050_1e8,z0050_5e8,z0050_1e9))

specarr = np.stack((specarr_z00004,specarr_z0004,specarr_z0008,specarr_z0020,specarr_z0050))

#specarr[specarr<100.] = 100.

print(specarr.shape)

print('... done!')
print('')

"""
plt.plot(lam_template, specarr[0,49,:], label='z=0.0001, age=100 Myr')
plt.plot(lam_template, specarr[2,49,:], label='z=0.0008, age=100 Myr')
plt.plot(lam_template, specarr[4,49,:], label='z=0.0040, age=100 Myr')
plt.xscale('log')
#plt.yscale('log')
plt.xlim([0.025, 1.])
plt.ylim([25, 40])
plt.xlabel(r'$\lambda$ [$\mu$m]', fontsize=14)
plt.ylabel(r'$\log_{10}(L)$ [erg s$^{-1}$]', fontsize=14)
plt.legend()
plt.savefig('../../rt-sims-plot/age100_plot.png', bbox_inches='tight')
plt.show()
"""

print('Computing final spectrum ...')
spec_output = GetSpectrum(lam_template, specarr, metallicity_star, mass_star, tform_star, zgrid, mstargrid, tformgrid, ngridtot)
print('... done!')
print('')

# Write to file
ff = open(pars.input_dir + pars.stellar_spectrum_filename, 'w')

for ii in range(len(lam_template)):

    ff.write(str(lam_template[ii]) + '      ' + str(spec_output[ii]) + '\n')

ff.close()
