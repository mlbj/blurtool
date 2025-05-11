import blurtool
import torch
import matplotlib.pyplot as plt
from skimage import data

# Load camera man image
image = data.camera() 
image = torch.from_numpy(image).float()/255.0  

# Create Blur operators
# 1. Stationary blur operator
gauss_kernel = blurtool.Kernel((25,25), 'gaussian', {'sx':0.25, 'sy':0.05, 'angle':45})
blur = blurtool.StationaryBlur(gauss_kernel)

# 2. Nonstationary blur operator
# Create a Nonstationary lattice of kernels based on a rotated Gaussian model.
# Note that this lattice has the same shape as the test image, that is, for each 
# point in the original image, we have a corresponding stationary kernel in the lattice.
lat = blurtool.RotGaussian(
    shape=image.shape,
    kernel_shape=(25,25),
    lattice_pars={'ax':0.01, 'ay':0.01, 'bx':0.5, 'by':0.1, 'gamma_x':1, 'gamma_y':1})

# Since the original lattice is too big and redundant, we can sample it in only (8,8) 
# stationary kernels in a zero order regular fashion.
sampled_lat = lat.sample(sampled_shape = (8,8))

# Compute a PCA decomposition of the sampled lattice and instantiate a Blur object with it 
pca_lat = blurtool.DecomposeLattice(sampled_lat, 
                                    image.shape,
                                    mode = 'pca',
                                    decompose_pars = {'r':15})
ns_blur = blurtool.NonstationaryBlur(pca_lat)

# Blur
blurred = blur(image)
ns_blurred = ns_blur(image)

# Save results
plt.imsave('original.png', image, cmap='gray')
plt.imsave('stationary_blur.png', blurred, cmap='gray')
plt.imsave('nonstationary_blur.png', ns_blurred, cmap='gray')
