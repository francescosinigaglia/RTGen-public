import numpy as np
import os
from numba import njit, prange 
from numba.typed import List
import time
import numba as nb
import input_params as pars

# INPUT
srcdir = pars.input_galaxy_dir
basedir = pars.input_dir

filename = srcdir + pars.input_galaxy_filename

nthreads = pars.nthreads

ngridx = pars.nx
ngridy = pars.ny
ngridz = pars.nz

imftype = pars.imftype

nmassbins = pars.numbins

minmass = pars.minmassstar
maxmass = pars.maxmassstar

xlbox = pars.sizex
ylbox = pars.sizey
zlbox = pars.sizez

thx = xlbox/2.
thy = ylbox/2.
thz = zlbox/2.

os.system('export NUMBA_NUM_THREADS=%d' %nthreads)

# ****************************************                          
# ****************************************                                                                                
# **************************************** 

def set_imf():

    # Set the M* grid bin edges
    mdummy = np.linspace(minmass, maxmass, num=nmassbins+1)
    # Pass to log scale
    logm = np.log10(mdummy)
    # Set the M* grid central points
    logmc = 0.5*(logm[1:] + logm[:-1])
    # Initialize empty vector for the IMF
    logimf = np.zeros(len(logmc))
    imf = np.zeros(len(logmc))

    # IMF parameters
    csi1 = 100. # Arbitrary zero point: the others will be computed self-consistently
    alpha4 = 2.35
    alpha3 = 2.7
    alpha2 = 1.3
    alpha1 = 0.3
    mtr1 = 0.08
    mtr2 = 0.5
    mtr3 = 1.0
    
    if imftype == 'Kroupa':

        logmtr1 = np.log10(mtr1)
        logmtr2 = np.log10(mtr2)
        logmtr3 = np.log10(mtr3)
        logcsi1 = np.log10(csi1)

        # First interval
        logimf[np.where(logmc<=logmtr1)] = logcsi1 - alpha1 * logmc[np.where(logmc<=logmtr1)]

        # Second interval: first compute zero-point, then the IMF
        logcsi2 = (alpha2 - alpha1) * logmtr1 + logcsi1
        logimf[np.where(np.logical_and(logmc>logmtr1, logmc<=logmtr2))] = logcsi2 - alpha2 * logmc[np.where(np.logical_and(logmc>logmtr1, logmc<=logmtr2))]

        # This interval                                                            
        logcsi3 = (alpha3 - alpha2) * logmtr2 + logcsi2
        logimf[np.where(np.logical_and(logmc>logmtr2, logmc<=logmtr3))] = logcsi3 - alpha3 * logmc[np.where(np.logical_and(logmc>logmtr2, logmc<=logmtr3))]

        # This interval                                                                                                                                                  
        logcsi4 = (alpha4 - alpha3) * logmtr3 + logcsi3
        logimf[np.where(logmc>logmtr3)] = logcsi4 - alpha4 * logmc[np.where(logmc>logmtr3)]

    # Get the IMF in linear scale
    imf = 10**logimf

    # Integrate the IMF under the curve to get probabilities

    area = np.sum(imf)

    imf /= area


    return logm, logmc, imf

# ****************************************  
@njit(cache=True, fastmath=True)
def ngp(xpos, ypos, zpos, ngridx, ngridy, ngridz, xlbox, ylbox, zlbox, imf, logmcent, logmedg, mstararr, nmassbin):

    # This function does NGP interpolation for stars from one single star cluster
    # It is able to do different grids depending on the desired number of specified stellar mass bins 
    
    # First, create the bin edges
    #massgrid = np.logspace(np.log10(minmass), np.log10(maxmass), num=nmassbin+1)
    masscent = 10**logmcent #0.5*(massgrid[1:] + massgrid[:-1])
    massedg = 10**logmedg

    ngridycopy = ngridy
    ngridy = ngridz
    ngridz = ngridycopy

    ylboxcopy = ylbox
    ylbox = zlbox
    zlbox = ylboxcopy

    grid = np.zeros((ngridx,ngridy,ngridz,nmassbin))

    xlcell = xlbox / ngridx
    ylcell = ylbox / ngridy
    zlcell = zlbox / ngridz
    
    for ii in range(len(xpos)):

        imfnew = (np.around((imf * mstararr[ii]))).astype('int')

        xind = int(xpos[ii]/xlcell)
        yind = int(ypos[ii]/ylcell)
        zind = int(zpos[ii]/zlcell)
        
        for jj in range(nmassbin):

            #print(imfnew)

            grid[xind,zind,yind,jj] += imfnew[jj]

    return grid

# ****************************************
# ****************************************
# ****************************************

ti = time.time()

# Read star cluster masses

raw = np.genfromtxt(basedir + 'star_clusters.TXT')
xx = raw[:,0] 
yy = raw[:,1] 
zz = raw[:,2] 
massstar = raw[:,3]

yy = yy[np.where(np.logical_and(xx>-thx, xx<thx))]
zz = zz[np.where(np.logical_and(xx>-thx, xx<thx))]
xx = xx[np.where(np.logical_and(xx>-thx, xx<thx))]

xx = xx[np.where(np.logical_and(yy>-thy, yy<thy))]
zz = zz[np.where(np.logical_and(yy>-thy, yy<thy))]
yy = yy[np.where(np.logical_and(yy>-thy, yy<thy))]

xx = xx[np.where(np.logical_and(zz>-thz, zz<thz))]
yy = yy[np.where(np.logical_and(zz>-thz, zz<thz))]
zz = zz[np.where(np.logical_and(zz>-thz, zz<thz))]

xx = xx + xlbox / 2.
yy = yy + xlbox / 2.
zz = zz + xlbox / 2.

print('Tot star cluster: ', len(massstar))
print('')

# Set IMF
print('Setting IMF ...')
logm_edg, logm_cent, imf = set_imf()
print('... done!')
print('')

# Now do the NGP for the chosen number of clusters. The NGP is done is stellar mass bins
print('Doing NGP for the chosen number of clusters...')
grid = ngp(xx, yy, zz, ngridx, ngridy, ngridz, xlbox, ylbox, zlbox, imf, logm_cent, logm_edg, massstar, nmassbins)
print('... done!')
print('')

# Now interpolate stellar masses to get radii and temperature
# Tabulated star data                                                                                                                                                    
masslist = [0.07, 0.08, 0.15, 0.60, 0.69, 0.78, 0.93, 1., 1.10, 1.3, 1.7, 2.1, 3.2, 6.5, 18, 35, 100] # In units M_sun   
radlist  = [0.09, 0.11, 0.18, 0.51, 0.74, 0.85, 0.93, 1., 1.05, 1.2, 1.3, 1.7, 2.5, 3.8, 7.4, 9.8, 12] # In units R_sun                                                 
templist = [2200., 2650., 3120., 3800., 4410., 5240., 5610., 5780., 5920., 6540., 7240., 8620., 10800., 16400., 30000, 38000, 50000]

# Interpolate of stellar parameters as a function of mass                                                                                            

masslin = np.asarray(10**logm_cent)
radlin = np.interp(masslin, np.asarray(masslist),np.asarray(radlist))
templin = np.interp(masslin, np.asarray(masslist),np.asarray(templist))

ff = open(basedir + pars.stellar_parameters_filename, 'w')
ff.write('# This file contains the stellar parameters of the modelled Main Sequence stars \n')

for ii in range(len(masslin)):

    ff.write(str(masslin[ii]) + '      ' + str(radlin[ii]) + '      ' + str(templin[ii]) + '\n')

ff.close()

# Write the stellar grid to file
grid = grid * masslin / (4*np.pi)

grid = grid.flatten(order='F')
print('... done!')
print('')

tf = time.time()

print('Elapsed %s s ...' %str(tf-ti))
                                                                                                                                         
grid.tofile(basedir + pars.mass_star_filename)

