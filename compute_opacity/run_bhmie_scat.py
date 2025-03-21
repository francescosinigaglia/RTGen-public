from makedustopac import *
import numpy as np
import os

def compute_opac_dist(amin_mic=0.01, amax_mic=100, pwl=-3.5, \
                      na=50, optconst="lnk/pyrmg70.lnk", matdens=3.01, \
                      lam_min=0.005, lam_max=1e4, nlam=1000, extrapol=True, \
                      outfolder='.', outname=None, verbose=False):
    '''
    This function computes opacity file for a given grain size distribution.
    The size distribution is given by power law n(a) ~ a**(pwl), between
    amin_min and amax_min grain sizes (in micron).

    Parameters:
      amin_mic      minimum grain size in micron
      amax_mic      maximum grain size in micron
      pwl           power exponent of the grain size distribution
      na            number of grain sizes in the [amin_mic,amin_max] range
      optcont       filename of optical constant (n,k) data
      matdens       material density in gram / cm^3
      outname       opacity table is written with this filename
      verbose       write out status information
      extrapol      extrapolate optical constants beyond its wavelength grid, if necessary
      outfolder     folder name where results are written
      lam_min       shortest wavelength of results
      lam_max       longest wavelength of results
      nlam          number of wavelength points
    '''

    print("\nComputing grain opacity with grain size distribution:\n")
    print("amin = {} micron\namax = {} micron".format(amin_mic, amax_mic))
    print("n(a) \prop a**{}".format(pwl))
    print("optical constants: {}".format(optconst))
    print("grain bulk density = {} g cm**-3".format(matdens))

    amincm   = amin_mic * 1e-4
    amaxcm   = amax_mic * 1e-4

    agraincm = np.exp(np.linspace(np.log(amincm),np.log(amaxcm),na))
    dum      = agraincm**(pwl+4)
    wgt      = dum / sum(dum) # normalized weighting factors

    #
    # Set up a wavelength grid upon which we want to compute the opacities
    #
    lamcm    = 10.0**np.linspace(np.log10(lam_min),np.log10(lam_max),nlam)*1e-4

    #
    # Now make the opacity with the bhmie code
    #
    opac = compute_opac_mie(optconst,matdens,agraincm,lamcm,theta=None,wgt=wgt,
                            extrapolate=extrapol,chopforward=False,
                            verbose=verbose)

    #
    # Now write it out to a RADMC-3D opacity file:
    #
    print("Writing the opacity to kappa file")
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)
    if not outname:
        optconst = ( optconst.split('/')[-1] ).split('.lnk')[0]
        outname = outfolder + "/dustkappa_" + optconst + \
                  "_amin{}_amax{}.inp".format(amin_mic,amax_mic)
    write_radmc3d_kappa_file(opac,None,filename=outname)
    #
    print("Done\n")

if __name__ == "__main__":
    '''
    Example to create multiple opacity tables at once for different maximum 
    grain sizes.
    '''

    amin     = 0.01
    amax     = np.arange(100,350, 50)
    pwl      = -3.0
    optconst = "lnk/mix_compact_bruggeman.lnk"
    matdens  = 1.36

    for am in amax:
        compute_opac_dist(amin_mic=amin, amax_mic=am, \
                          pwl=pwl, optconst=optconst, \
                          verbose=False,outfolder='opac',\
                          matdens=matdens,na=100, \
                          lam_max=1e5,nlam=500)
