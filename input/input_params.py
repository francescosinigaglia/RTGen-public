# INPUT PARAMETERS
# **********************************************
# **********************************************
# I/O
# Input directory
input_dir = '../input/'
output_dir = '../output/'

input_galaxy_dir = '/data/fsinig/simulations/run13/'
input_galaxy_filename = 'galaxy_cc6_v150_gf0.3_lres_MC.01200'

tsnap = 1.2 # In Gyr

# Input file names: these should be standard, do not change         
mass_gas_filename = 'mass_gas.DAT'
vx_gas_filename = 'vx_gas.DAT'
vy_gas_filename = 'vy_gas.DAT'
vz_gas_filename = 'vz_gas.DAT'
temperature_gas_filename = 'temperature_gas.DAT'
metallicity_filename = 'metallicity.DAT'
mass_star_filename = 'mass_star.DAT'
fmol_filename = 'fmol.DAT'
metallicity_star_filename = 'metallicity_star.DAT'
tform_star_filename = 'tform_star.DAT'

# **********************************************                         
# **********************************************
# Grid parameters
nx       = 256
ny       = 256
nz       = 256

sizex    = 30. # in kpc
sizey    = 30. # in kpc
sizez    = 30. # in kpc


# **********************************************     
# **********************************************
# RADMC-3D INTERNAL PARAMETERS
# Monte Carlo parameters
nphot    = int(1e7)
nphot_scat = int(1e7)
nphot_spec = int(1e7)
nphot_mono = int(1e7)

scattering_mode_max = 2

# Inclination used to render images
# incl = 0 -->edge-on, incl = 90 --> face-on
incl = 90.

# **********************************************
# **********************************************
# STARS)
# Stellar IMF density field binning
stellar_spectrum_filename = 'stellar_spectrum.TXT'

# **********************************************
# **********************************************
# DUST
# Physical input parameters
dust_to_gas_model = 'metallicity_dependent' # 'fixed_ratio' or 'metallicity_dependent'
dust_to_gas_ratio = 0.001
dust_to_gas_slope = 2.445       # fiducial:  2.445 (Li, Narayanan, Davé 2019)
dust_to_gas_zeropoint = -2.029  # fiducial: -2.029 (Li, Narayanan, Davé 2019)
dust_to_gas_scatter = 0.3  # fiducial: 0.3 (Li, Narayanan, Davé 2019)  

# By default, the dust is a mixture of silicate and carbon grains
# Set here the relative fraction. Fiducial: fsil=0.55, fcarb=0.45, based on [C/O]~1.85
fsil = 0.55
fcarb = 1-fsil

grain_size_model = 'mixture' # 'fixed', 'mixture', 'power_law'. If 'mixture' or 'power_law', uses a distribution n(a)~a^(-3.5)

# If dust grain size follows a power law, then set th number of bins. If 'fixed', these are not used  
grain_size_distr_exp = -3.5
numbinsdust = 20
dustsizemin = 0.0005 # [um]; this number should be fixed
dustsizemax = 0.2500 # [um]; this number should be fixed

# **********************************************
# **********************************************
# LINE TRANSFER
h2_to_co_model = 'fixed_ratio' # 'fixed_ratio' or 'metallicity_dependent'
h2_to_co_convfact = 7.5e-8
h2_to_co_slope = 1e6 # CHANGE THIS
h2_to_co_zeropoint = 1e6 # CHANGE THIS

tgas_eq_tdust = 1 # 0: read gas temperature, 1: Tgas=Tdust

ortho_to_para_H2_perc = 0.75 # Ortho/Para H2 = 3, from Sternberg & Neufeld (1998)

lines_mode = 1 # LTE:1, LVG NLTE:3, OptThin NLTE:4

max_gas_temp_line = 99999. # Maximum temperature for line transfer, needed because the maximum T in the partition function is 1e5

# **********************************************
# **********************************************
# WAVELENGHT GRIDDING

# Camera
nbins_uvopt_cam = 90
nbins_nir_cam = 30
nbins_fir_cam = 30

# **********************************************
# **********************************************
# H2/HI splitting 
splitting_method = 'GK11' # 'GK11' or 'GD14'

# **********************************************
# **********************************************
# Telescope filters
instrument = 'HST' # Optins: 'Euclid', 'HST', 'JWST'
filtername = 'HST_WFC3_UVIS1.F438W.dat' # have a look at the filters directory for the name
# F 814 W (red HST)
# F 606 W (green HST)
# F 438 W (blue HST)

# **********************************************                                                        
# **********************************************   
# ALMA SIMULATIONS
# See https://casadocs.readthedocs.io/en/stable/api/tt/casatasks.simulation.simalma.html
# For more information and documentation about the "simalma" CASA task

project = "sim"                                         # Name of the project
dryrun = False                                          # False to really run the simulation
skymodel = "dummy.fits"                                 # Name of the original FITS image
inbright = 'compute'#"1Jy/pixel"                        # Maximum flux in pixel expressed in Jy/pixel --> OVERRIDDEN
indirection = "J2000 19h00m00 -40d00m00"                # Pointing coordinates, can also be taken from the FITS header  
incell = 'compute'#"0.1arcsec"                          # Image pixel size
incenter = "89GHz"                                      # Frequency of center channel
inwidth = "10MHz"                                       # Channel width
complist = ""                                           # Componentlist to observe --> usually not used
compwidth = ""                                          # Width of the component to observe --> usually not used
ptgfile = '$project.ptg.txt'                            # List of pointing positions, do not change 
setpointings = True                                     # Whether to compute or not the pointings, if True computes a list of pointings
integration = '10s'                                     # Integration/sampling time
direction = ""                                          # String with direction, or "" to center on model
mapsize = 'compute'#['10arcsec', '10arcsec']            # Size of the map in pixels, e.g. ['10arcsec', '10arcsec'] or "" to match model
antennalist = ['alma.cycle2.1.cfg']#, 'aca.cycle1.cfg']   # Antenna position files of ALMA
hourangle = 'transit'                                   # Hour angle of observation center 
totaltime = ['16h']#, '3h']                             # Total time of observation; vector corresponding to antennalist
tpnant = 0                                              # Number of total power antennas to use (0-4)
tptime = '0s'                                           # Total observation time for total power, only relevatn if tptime!=0
pwv = 0.5                                               # Precipitable Water Vapor in mm. 0 for noise-free simulation
image = True                                            # True to umage the simulations
imsize = [64,64]                                        # Output image size in pixels (x,y) e.g. [128,128], or 0 to match model 
imdirection = ""                                        # Set output image direction, (otherwise center on the model) 
cell = ""                                               # Cell size width units or “” to equal model
niter = 0                                               # Maximum number of iterations (0 for dirty image)
threshold = '0.1mJy'                                    # flux level (+units) to stop cleaning
graphics = 'none'                                       # display graphics at each stage to [screen|file|both|none], do not change
verbose = True                                          # Print extra information to the logger and terminal
overwrite =  True                                       # Overwrite existing run with the same model name?

# **********************************************        
# **********************************************           
# OPACITY COMPUTATION TABLES TEMPLATE       
lam_min_kappa = 0.005 # [um]    
lam_max_kappa = 1e4   # [um]            
nlam_kappa = 1000

# **********************************************
# **********************************************
# PARALLELIZATION
nthreads = 16
parallel = True

# **********************************************    
# ********************************************** 
# SPH INTERPOLATION

gas_kernel = 'SPH' # Only 'SPH' working so far
star_kernel = 'SPH' # 'SPH' or 'CIC'

sph_kernel = 'WendlandC2Kernel3D' # At the moment, only 'WendlandC2Kernel3D' is implemented
nLastNeigh = 32 # Number of nearest neighbours for SPH kernel interpolation
softlength = 0.1 # in kpc

# **********************************************     
# **********************************************                                 
# COSMOLOGY
H0 = 73.
Om0 = 0.27
Tcmb = 2.725

zstart = 4. # Starting redshift of the simulation
zreal = 2.17347
z_optical = 2.
z_alma = 2.

# **********************************************                                                                             
# **********************************************               
# Miscellaneous
numsmall = 1e-8
