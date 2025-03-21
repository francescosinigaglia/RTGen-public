import numpy as np
from numba import njit, prange
from scipy.spatial import KDTree
from scipy.integrate import tplquad
import time
import input_params as pars

# INPUT
basedir = ""
outdir = pars.input_dir 

# Filenames
xfile = basedir + 'posx_gas.npy'
yfile = basedir + 'posy_gas.npy'
zfile = basedir + 'posz_gas.npy'
mass_file = basedir + 'mass_gas.npy'

vx_file = 'vx_gas.npy'
vy_file = 'vy_gas.npy'
vz_file = 'vz_gas.npy'
temp_file = 'temperature_gas.npy'
metallicity_file = 'metallicity.npy'

xfile_star = basedir + 'posx_star.npy'
yfile_star = basedir + 'posy_star.npy'
zfile_star = basedir + 'posz_star.npy'
mass_file_star = basedir + 'mass_star.npy'
metallicity_file_star = basedir + 'metallicity_star.npy'
tform_file_star = basedir + 'tform_star.npy'

# OUTPUT
outname_mass = outdir + pars.mass_gas_filename
outname_mass_star = outdir + pars.mass_star_filename
outname_vx = outdir + pars.vx_gas_filename
outname_vy = outdir + pars.vy_gas_filename
outname_vz = outdir + pars.vz_gas_filename
outname_temperature = outdir + pars.temperature_gas_filename
outname_metallicity = outdir + pars.metallicity_filename
outname_metallicity_star = outdir + pars.metallicity_star_filename
outname_tform_star = outdir + pars.tform_star_filename

# Grid size
nx = pars.nx
ny = pars.ny
nz = pars.nz

# Box size
xlbox = pars.sizex # in kpc
ylbox = pars.sizey # in kpc
zlbox = pars.sizez # in kpc

# If True, it sets a threshold in the coordinates in the three directions
setThreshold = True

# If threshold = True, the natural cell for threshold is lbox/2.
thx = xlbox/2. # in kpc
thy = ylbox/2. # in kpc
thz = zlbox/2. # in kpc

nLastNeigh = pars.nLastNeigh # default = 32 inside KDTree, following Gasoline documentation

kernel = pars.sph_kernel
gas_kernel = pars.gas_kernel
star_kernel = pars.star_kernel

nthreads = pars.nthreads

parallel = pars.parallel

# ****************************************
# ****************************************
# ****************************************

# This function cannot be Numba-zed, as it contains a KDTree algorithm, which is not supported by Numba at the moment
def getSmoothingLenght(xx,yy,zz, nlast=32):

    # Prepare positions array
    dummyarr = np.zeros((len(xx), 3))

    dummyarr[:,0] = xx.copy()
    dummyarr[:,1] = yy.copy()
    dummyarr[:,2] = zz.copy()

    # Initialize array of smoothing lenghts
    hharr = np.zeros((len(xx)))
    
    # Initialize KDTree
    kdtree = KDTree(dummyarr)

    # Now loop over particles. No parallelization here, as the query inside the loop is parallelized
    for ii in range(len(xx)):

        dd, ind = kdtree.query([xx[ii], yy[ii], zz[ii]], k=[nlast])

        #printd)

        hharr[ii] = dd[0]/2.

    return hharr

# ****************************************   
# This function computes the Wendland C2 kernel in 3D      
@njit(cache=True, fastmath=True)
def WendlandC2Kernel3D(dx, dy, dz, hh):

    rr = np.sqrt(dx**2 + dy**2 + dz**2) 

    qq = rr/hh
    ad = 21./(16. * np.pi * hh**3)

    #print(qq)

    if qq<=2.:
        kern = ad * (1. - qq/2.)**4 * (2.*qq + 1.)
    else:
        kern = 0.
        
    return kern

# ****************************************   
# This function integrates the SPH kernel in a given cell
@njit(cache=True, fastmath=True)
def integrateKernel(xx,yy,zz, xlcell, ylcell, zlcell, xind, yind, zind, hh, mult):

    #print(xx,yy,zz, xlcell, ylcell, zlcell, xind, yind, zind)
    
    xinf = xind * xlcell - xx
    xsup = (xind + 1) * xlcell - xx

    yinf = yind * ylcell - yy
    ysup = (yind + 1) * ylcell - yy

    zinf = zind * zlcell - zz
    zsup = (zind + 1) * zlcell - zz

    # Compute triple integral

    itg = 0.

    nc = 51
    
    xxint = np.linspace(xinf,xsup, num=nc)
    yyint = np.linspace(yinf,ysup, num=nc)
    zzint = np.linspace(zinf,zsup, num=nc)

    #xmid = 0.5*(xxint[1:]+xxint[:-1])
    #ymid = 0.5*(yyint[1:]+yyint[:-1])
    #zmid = 0.5*(zzint[1:]+zzint[:-1])

    hx = (xxint[1]-xxint[0])
    hy = (yyint[1]-yyint[0])
    hz = (zzint[1]-zzint[0])

    for ii in range(len(xxint)):
        for jj in range(len(yyint)):
            for kk in range(len(zzint)):

                if ii==0 or ii==(len(xxint)-1):
                    if jj==0 or jj==(len(yyint)-1):
                        if kk==0 or kk==(len(zzint)-1):
                            const = 1.
                        else:
                            const = 2.
                    else:
                        const = 4.

                elif jj==0 or jj==(len(yyint)-1):
                    if kk==0 or kk==(len(zzint)-1):
                        const = 2.
                    else: const = 4.

                elif kk==0 or kk==(len(zzint)-1):
                    const = 4.

                else:
                    const = 8.
                
                height = WendlandC2Kernel3D(xxint[ii], yyint[jj], zzint[kk], hh)
                #kernint = (xxint[ii+1]-xxint[ii]) * (yyint[jj+1]-yyint[jj]) * (zzint[kk+1]-zzint[kk]) * height
                kernint = hx * hy * hz / 8. * const * height

                itg += kernint
    
    return itg


# ****************************************     
@njit(parallel=parallel, cache=True, fastmath=True)
def SPHInterpolator(posx, posy, posz, mass, vx, vy, vz, temperature, metallicity, tform, xlbox, ylbox, zlbox, ncx, ncy, ncz, hharr, parttype):

    xlcell = xlbox / ncx
    ylcell = ylbox / ncy
    zlcell = zlbox / ncz
    
    delta = np.zeros((ncx, ncy, ncz))
    vxint = np.zeros((ncx, ncy, ncz))
    vyint = np.zeros((ncx, ncy, ncz))
    vzint = np.zeros((ncx, ncy, ncz))
    meint = np.zeros((ncx, ncy, ncz))
    teint = np.zeros((ncx, ncy, ncz))
    tformint = np.zeros((ncx, ncy, ncz))

    for ii in prange(len(posx)):
        xx = posx[ii]
        yy = posy[ii]
        zz = posz[ii]

        hh = hharr[ii]

        # Verify that this is not close to the boundary, otherwise shorten h   
        #if (xx-2*hh)<0. or (xx+2*hh)>xlbox or (yy-2*hh)<0. or (yy+2*hh)>ylbox or (zz-2*hh)<0. or (zz+2*hh)>zlbox:
        #    hh = np.amin(np.array([xx/2., yy/2., zz/2., (xlbox-xx)/2., (ylbox-yy)/2., (zlbox-zz)/2.]))
            #print('new q: ', 2*hh)

        # Now evaluate how many cells should we loop over
        qtmp = 2*hh
        mult = qtmp / xlcell

        deltacell = int(np.ceil(mult))

        indxc = int(xx/xlcell)
        indyc = int(yy/ylcell)
        indzc = int(zz/zlcell)

        # Establish let and right limits over which to loop
        indxr = indxc + deltacell + 1
        indxl = indxc - deltacell

        indyr = indyc + deltacell + 1
        indyl = indyc - deltacell

        indzr = indzc + deltacell + 1
        indzl = indzc - deltacell

        if indxl < 0:
            indxl = 0
        if indxr > nx:
            indxr = nx

        if indyl < 0:
            indyl = 0
        if indyr > ny:
            indyr = ny

        if indzl < 0:
            indzl = 0
        if indzr > nz:
            indzr = nz

        wtot = 0.

        for ll in range(indxl, indxr):
            for mm in range(indyl, indyr):
                for nn in range(indzl, indzr):
            
                    ww = integrateKernel(xx,yy,zz, xlcell, ylcell, zlcell, ll, mm, nn, hh, deltacell)
                    
                    delta[ll,mm,nn] += mass[ii]*ww
                    meint[ll,mm,nn] += metallicity[ii]*mass[ii]*ww
                    
                    if parttype=='gas':
                        vxint[ll,mm,nn] += vx[ii]*mass[ii]*ww
                        vyint[ll,mm,nn] += vy[ii]*mass[ii]*ww
                        vzint[ll,mm,nn] += vz[ii]*mass[ii]*ww
                        teint[ll,mm,nn] += temperature[ii]*mass[ii]*ww

                    if parttype == 'star':
                        tformint[ll,mm,nn] += tform[ii]*mass[ii]*ww

                    wtot += ww
                    
    return delta, vxint, vyint, vzint, teint, meint, tformint

# ****************************************
# **********************************************
@njit(parallel=False, cache=True, fastmath=True)
def get_cic(posx, posy, posz, weight, lbox, ngrid):

    lcell = lbox/ngrid

    delta = np.zeros((ngrid,ngrid,ngrid))

    for ii in prange(len(posx)):
        xx = posx[ii]
        yy = posy[ii]
        zz = posz[ii]
        indxc = int(xx/lcell)
        indyc = int(yy/lcell)
        indzc = int(zz/lcell)

        wxc = xx/lcell - indxc
        wyc = yy/lcell - indyc
        wzc = zz/lcell - indzc

        if wxc <=0.5:
            indxl = indxc - 1
            if indxl<0:
                indxl += ngrid
            wxc += 0.5
            wxl = 1 - wxc
        elif wxc >0.5:
            indxl = indxc + 1
            if indxl>=ngrid:
                indxl -= ngrid
            wxl = wxc - 0.5
            wxc = 1 - wxl

        if wyc <=0.5:
            indyl = indyc - 1
            if indyl<0:
                indyl += ngrid
            wyc += 0.5
            wyl = 1 - wyc
        elif wyc >0.5:
            indyl = indyc + 1
            if indyl>=ngrid:
                indyl -= ngrid
            wyl = wyc - 0.5
            wyc = 1 - wyl

        if wzc <=0.5:
            indzl = indzc - 1
            if indzl<0:
                indzl += ngrid
            wzc += 0.5
            wzl = 1 - wzc
        elif wzc >0.5:
            indzl = indzc + 1
            if indzl>=0:
                indzl -= ngrid
            wzl = wzc - 0.5
            wzc = 1 - wzl

        ww = weight[ii]

        delta[indxc,indyc,indzc] += ww*wxc*wyc*wzc
        delta[indxl,indyc,indzc] += ww*wxl*wyc*wzc
        delta[indxc,indyl,indzc] += ww*wxc*wyl*wzc
        delta[indxc,indyc,indzl] += ww*wxc*wyc*wzl
        delta[indxl,indyl,indzc] += ww*wxl*wyl*wzc
        delta[indxc,indyl,indzl] += ww*wxc*wyl*wzl
        delta[indxl,indyc,indzl] += ww*wxl*wyc*wzl
        delta[indxl,indyl,indzl] += ww*wxl*wyl*wzl

    return delta

# ****************************************                                                       
# ****************************************                                                    
# **************************************** 

# Start time
tin = time.time()

print('SPH INTERPOLATION')
print('')

# Check size of the cells. At the moment only for cubic cells 
lxcell = xlbox / nx
lycell = ylbox / ny
lzcell = zlbox / nz

if lxcell!=lycell or lycell!=lzcell or lxcell!=lzcell:
    print('Error: non-cubic cells found. The code works only with cubic cells. Exiting.')
    exit()

# Set SPH kernel
if kernel == 'WendlandC2Kernel3D':
    kern = WendlandC2Kernel3D

else: 
    print('Kernel not found. Exiting.')


print('Reading input ...')

# Now read files
xx = np.load(xfile)
yy = np.load(yfile)
zz = np.load(zfile)
mass = np.load(mass_file)
vx= np.load(vx_file)
vy = np.load(vy_file)
vz = np.load(vz_file)
temperature = np.load(temp_file)
metallicity = np.load(metallicity_file)

xx_star = np.load(xfile_star)
yy_star = np.load(yfile_star)
zz_star = np.load(zfile_star)
mass_star = np.load(mass_file_star)
metallicity_star = np.load(metallicity_file_star)
tform_star = np.load(tform_file_star)

if setThreshold == True:

    # GAS
    vx = vx[np.where(np.logical_and(xx>-thx, xx<thx))]
    vy = vy[np.where(np.logical_and(xx>-thx, xx<thx))]
    vz = vz[np.where(np.logical_and(xx>-thx, xx<thx))]
    mass = mass[np.where(np.logical_and(xx>-thx, xx<thx))]
    temperature = temperature[np.where(np.logical_and(xx>-thx, xx<thx))]
    metallicity = metallicity[np.where(np.logical_and(xx>-thx, xx<thx))]
    #metallicity_star = metallicity_star[np.where(np.logical_and(xx>-thx, xx<thx))]
    yy = yy[np.where(np.logical_and(xx>-thx, xx<thx))]
    zz = zz[np.where(np.logical_and(xx>-thx, xx<thx))]
    xx = xx[np.where(np.logical_and(xx>-thx, xx<thx))]

    vx = vx[np.where(np.logical_and(yy>-thy, yy<thy))]
    vy = vy[np.where(np.logical_and(yy>-thy, yy<thy))]
    vz = vz[np.where(np.logical_and(yy>-thy, yy<thy))]
    mass = mass[np.where(np.logical_and(yy>-thy, yy<thy))]
    temperature = temperature[np.where(np.logical_and(yy>-thy, yy<thy))]
    metallicity = metallicity[np.where(np.logical_and(yy>-thy, yy<thy))]
    #metallicity_star = metallicity_star[np.where(np.logical_and(yy>-thy, yy<thy))]
    xx = xx[np.where(np.logical_and(yy>-thy, yy<thy))]
    zz = zz[np.where(np.logical_and(yy>-thy, yy<thy))]
    yy = yy[np.where(np.logical_and(yy>-thy, yy<thy))]

    vx = vx[np.where(np.logical_and(zz>-thz, zz<thz))]
    vy = vy[np.where(np.logical_and(zz>-thz, zz<thz))]
    vz = vz[np.where(np.logical_and(zz>-thz, zz<thz))]
    mass = mass[np.where(np.logical_and(zz>-thz, zz<thz))]
    temperature = temperature[np.where(np.logical_and(zz>-thz, zz<thz))]
    metallicity = metallicity[np.where(np.logical_and(zz>-thz, zz<thz))]
    #metallicity_star = metallicity_star[np.where(np.logical_and(zz>-thz, zz<thz))]
    xx = xx[np.where(np.logical_and(zz>-thz, zz<thz))]
    yy = yy[np.where(np.logical_and(zz>-thz, zz<thz))]
    zz = zz[np.where(np.logical_and(zz>-thz, zz<thz))]

    # STARS
    mass_star = mass_star[np.where(np.logical_and(xx_star>-thx, xx_star<thx))]
    metallicity_star = metallicity_star[np.where(np.logical_and(xx_star>-thx, xx_star<thx))]
    tform_star = tform_star[np.where(np.logical_and(xx_star>-thx, xx_star<thx))]
    yy_star = yy_star[np.where(np.logical_and(xx_star>-thx, xx_star<thx))]
    zz_star = zz_star[np.where(np.logical_and(xx_star>-thx, xx_star<thx))]
    xx_star = xx_star[np.where(np.logical_and(xx_star>-thx, xx_star<thx))]

    mass_star = mass_star[np.where(np.logical_and(yy_star>-thy, yy_star<thy))]
    metallicity_star = metallicity_star[np.where(np.logical_and(yy_star>-thy, yy_star<thy))]
    tform_star = tform_star[np.where(np.logical_and(yy_star>-thy, yy_star<thy))]
    xx_star = xx_star[np.where(np.logical_and(yy_star>-thy, yy_star<thy))]
    zz_star = zz_star[np.where(np.logical_and(yy_star>-thy, yy_star<thy))]
    yy_star = yy_star[np.where(np.logical_and(yy_star>-thy, yy_star<thy))]

    mass_star = mass_star[np.where(np.logical_and(zz_star>-thz, zz_star<thz))]
    metallicity_star = metallicity_star[np.where(np.logical_and(zz_star>-thz, zz_star<thz))]
    tform_star = tform_star[np.where(np.logical_and(zz_star>-thz, zz_star<thz))]
    xx_star = xx_star[np.where(np.logical_and(zz_star>-thz, zz_star<thz))]
    yy_star = yy_star[np.where(np.logical_and(zz_star>-thz, zz_star<thz))]
    zz_star = zz_star[np.where(np.logical_and(zz_star>-thz, zz_star<thz))]

# Coordinates are zero-centered. First of all, move them by lbox/2 
xx += xlbox/2
yy += ylbox/2
zz += zlbox/2

xx_star += xlbox/2
yy_star += ylbox/2
zz_star += zlbox/2

print('... done!')

print('')

# Sommthing lengths
print('Computing smoothing lenghts ...')
hharr = getSmoothingLenght(xx,yy,zz, nlast=nLastNeigh)
#hharr_star = pars.softlength * np.ones(len(xx_star))  #getSmoothingLenght(xx_star,yy_star,zz_star, nlast=nLastNeigh)
print('... done!')

print('')

# Stellar particles interpolation
if star_kernel == 'CIC':
    print('Doing CIC to interpolate collisionless star particles ...')
    weights_star = get_cic(xx_star, yy_star, zz_star, np.ones(len(xx_star)), xlbox, nx)
    massnew_star = get_cic(xx_star, yy_star, zz_star, mass_star, xlbox, nx)
    metnew_star = get_cic(xx_star, yy_star, zz_star, metallicity_star, xlbox, nx)
    tformnew_star = get_cic(xx_star, yy_star, zz_star, tform_star, xlbox, nx)

elif star_kernel == 'SPH':
    print('Doing SPH interpolation of star particles ...')
    massnew_star, vxnew_empty, vynew_empty, vznew_empty, tempnew_empty, metnew_star, tformnew_star = SPHInterpolator(xx_star, yy_star, zz_star, mass_star, vx, vy, vz, temperature, metallicity_star, tform_star, xlbox, ylbox, zlbox, nx, ny, nz, hharr, "star")

else:
    print('No other options implemented so far. Revise zour choice. Falling back to CIC.')
    weights_star = get_cic(xx_star, yy_star, zz_star, np.ones(len(xx_star)), xlbox, nx)
    massnew_star = get_cic(xx_star, yy_star, zz_star, mass_star, xlbox, nx)
    metnew_star = get_cic(xx_star, yy_star, zz_star, metallicity_star, xlbox, nx)
    tformnew_star = get_cic(xx_star, yy_star, zz_star, tform_star, xlbox, nx)

# Gas particles interpolation
print('Doing SPH interpolation of gas particles ...')
massnew, vxnew, vynew, vznew, tempnew, metnew, tformnew_empty = SPHInterpolator(xx, yy, zz, mass, vx, vy, vz, temperature, metallicity, tform_star, xlbox, ylbox, zlbox, nx, ny, nz, hharr, "gas")

print('... done!')

print('')

# Normaliye the grids to the gas/stellar mass
print('Normalizing by gas mass...')
vxnew[np.where(massnew!=0)] = vxnew[np.where(massnew!=0)] / massnew[np.where(massnew!=0)]
vynew[np.where(massnew!=0)] = vynew[np.where(massnew!=0)] / massnew[np.where(massnew!=0)]
vznew[np.where(massnew!=0)] = vznew[np.where(massnew!=0)] / massnew[np.where(massnew!=0)]
tempnew[np.where(massnew!=0)] = tempnew[np.where(massnew!=0)] / massnew[np.where(massnew!=0)]
metnew[np.where(massnew!=0)] = metnew[np.where(massnew!=0)] / massnew[np.where(massnew!=0)]

if star_kernel == 'SPH':
    metnew_star[np.where(massnew_star!=0)] = metnew_star[np.where(massnew_star!=0)] / massnew_star[np.where(massnew_star!=0)]
    tformnew_star[np.where(massnew_star!=0)] = tformnew_star[np.where(massnew_star!=0)] / massnew_star[np.where(massnew_star!=0)]

elif star_kernel == 'CIC':
    metnew_star[np.where(weights_star!=0)] = metnew_star[np.where(weights_star!=0)] / weights_star[np.where(weights_star!=0)]
    tformnew_star[np.where(weights_star!=0)] = tformnew_star[np.where(weights_star!=0)] / weights_star[np.where(weights_star!=0)]

else:
    print('No other options implemented so far. Revise zour choice. Falling back to CIC.')
    metnew_star[np.where(weights_star!=0)] = metnew_star[np.where(weights_star!=0)] / weights_star[np.where(weights_star!=0)]
    tformnew_star[np.where(weights_star!=0)] = tformnew_star[np.where(weights_star!=0)] / weights_star[np.where(weights_star!=0)]
    

print('... done!')

print('')


print('Writing output ...')
massnew = np.swapaxes(massnew, 1, 2)
massnew = massnew.flatten(order='F')
massnew.astype('float64').tofile(outname_mass)

massnew_star = np.swapaxes(massnew_star, 1, 2)
massnew_star = massnew_star.flatten(order='F')
massnew_star.astype('float64').tofile(outname_mass_star)

vxnew = np.swapaxes(vxnew, 1, 2)
vxnew = vxnew.flatten(order='F')
vxnew.astype('float64').tofile(outname_vx)

vynew = np.swapaxes(vynew, 1, 2)
vynew = vynew.flatten(order='F')
vynew.astype('float64').tofile(outname_vy)

vznew = np.swapaxes(vznew, 1, 2)
vznew = vznew.flatten(order='F')
vznew.astype('float64').tofile(outname_vz)

tempnew = np.swapaxes(tempnew, 1, 2)
tempnew = tempnew.flatten(order='F')
tempnew.astype('float64').tofile(outname_temperature)

metnew = np.swapaxes(metnew, 1, 2)
metnew = metnew.flatten(order='F')
metnew.astype('float64').tofile(outname_metallicity)
print('... done!')

metnew_star = np.swapaxes(metnew_star, 1, 2)
metnew_star = metnew_star.flatten(order='F')
metnew_star.astype('float64').tofile(outname_metallicity_star)
print('... done!')

tformnew_star = np.swapaxes(tformnew_star, 1, 2)
tformnew_star = tformnew_star.flatten(order='F')
tformnew_star.astype('float64').tofile(outname_tform_star)
print('... done!')

print('')

print('THE END!')

tf = time.time()

dt = tf-tin
uu = 's'

if dt>60:
    dt /= 60.
    uu = 'min'

    if dt>60:

        dt /= 60.
        uu = 'hours'

print('Elapsed ' + str(dt) + ' ' +  uu + ' ...')

