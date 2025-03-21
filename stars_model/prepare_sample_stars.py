import numpy as np
import pynbody as pn
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt

# INPUT

basedir = "../"
srcdir = "../../../../run27_FB_NEW/"

filename = srcdir + 'galaxy_cc6_v180_g05_FB.01200'

nthreads = 4

imftype = 'Kroupa'

os.system('export NUMBA_NUM_THREADS=4')

# ****************************************                                                                                                              
# ****************************************                                                                                                              
# **************************************** 

def set_imf():

    # Set the M* grid bin edges
    mdummy = np.linspace(0.05, 120, num=10001)
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

        print(logmtr1, logmtr2, logmtr3)

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

def sample_stars(massstar, xx, yy, zz, radinterp, tempinterp, logm_edg, imf):
    
    totmassarr_dummy = np.zeros(len(massstar))

    dummy_bins = np.arange(len(imf))

    f = open('stars.txt', 'w')
    f.write('# x [kpc]      y [kpc]      z [kpc]      M* [M_sun]      R [R_sun]      T [T_sun] \n')

    for ii in range(len(massstar)):

        while totmassarr_dummy[ii]<massstar[ii]:

            logm_new_ind = np.random.choice(dummy_bins, size=1, replace=True, p=imf)
            logmstar_new = np.random.uniform(logm_edg[logm_new_ind], logm_edg[logm_new_ind+1])[0]
            #print(mstar_new[0])
            mstar_new = 10**logmstar_new
            temp_new = tempinterp(mstar_new)
            rad_new = radinterp(mstar_new)

            omega = 1. # galacocentric angular velocity of a star cluster  
            clusterrad = ((GG * massstar[ii]) / (2 * omega**2))**(1./3.)  # Eq. (16) from Choksi & Kruijsen (2021) (https://arxiv.org/abs/1912.05560)
            xnew = xx[ii] + np.random.normal(0., clusterrad)
            ynew = yy[ii] + np.random.normal(0., clusterrad)
            znew = zz[ii] + np.random.normal(0., clusterrad)
            f.write(str(xnew) + '      ' + str(ynew) + '      ' + str(znew) + '      ' + str(mstar_new) + '      ' + str(rad_new) + '      ' + str(temp_new) + '\n')
            totmassarr_dummy[ii] += mstar_new

        mstar_new = massstar[ii] - totmassarr_dummy[ii]
        temp_new = tempinterp(mstar_new)
        rad_new = radinterp(mstar_new)
        f.write(str(xnew) + '      ' + str(ynew) + '      ' + str(znew) + '      ' + str(mstar_new) + '      ' + str(rad_new) + '      ' + str(temp_new) + '\n')                                   

# ****************************************
# ****************************************
# ****************************************

# Tabulated star data
masslist = [0.07, 0.08, 0.15, 0.60, 0.69, 0.78, 0.93, 1., 1.10, 1.3, 1.7, 2.1, 3.2, 6.5, 18, 35, 100] # In units M_sun
radlist  = [0.09, 0.11, 0.18, 0.51, 0.74, 0.85, 0.93, 1., 1.05, 1.2, 1.3, 1.7, 2.5, 3.8, 7.4, 9.8, 12] # In units R_sun
templist = [2200., 2650., 3120., 3800., 4410., 5240., 5610., 5780., 5920., 6540., 7240., 8620., 10800., 16400., 30000, 38000, 50000]

# Interpolate stellar parameters as a function of mass
radinterp = interp1d(masslist, radlist, kind='cubic', fill_value="extrapolate")
tempinterp = interp1d(masslist, templist, kind='cubic', fill_value="extrapolate")

# Load galaxy simulation with pynbody - here we just need positions and masses
gal = pn.load(filename)

posstar = gal.star['pos']
massstar = 2.33e+05*gal.star['mass']

xx = posstar[:,0]
yy = posstar[:,1]
zz = posstar[:,2]

f = open(basedir + 'star_clusters.TXT', 'w+')
f.write('# x [kpc]      y [kpc]      z [kpc]      M* [M_sun] \n')

for ii in range(len(xx)):
    f.write(str(xx[ii]) + '      ' + str(yy[ii]) + '      ' + str(zz[ii]) + '      ' + str(massstar[ii]) +  '\n')

f.close()