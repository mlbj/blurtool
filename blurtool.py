import math
import torch
import torch.nn as nn
import torch.fft


from timeit import default_timer as timer

import scipy
import numpy as np
pi = torch.tensor(math.pi)  # convert to tensor

'''
Blur class. This class Holds both stationary and nonstationary blur subclasses.
'''
class Blur(nn.Module):
	def __init__(self):
		
		
		
		# misc parameters 
		self.forward_timer=-1
		
	
	# x.shape: (batch_size,fig_shape[0],fig_shape[1]) 
	# y.shape: (batch_size,fig_shape[0],fig_shape[1])
	def forward(self,x):
        	# start timer
		start = timer()
    		
		# make sure input is float64
		x=x.double()

		# store original input shape
		og_x_shape=x.shape	

		# add a batch dimension to the input, if necessary
		if x.dim()==2:
			x=x.unsqueeze(0)

        	# call internal forward method 
		y=self._forward(x)
        
		# update timer
		end = timer()    
		self.forward_timer=end-start
		
		# return
		return y.reshape(og_x_shape)
		
	# specific internal forward method
	def _forward(self,x):
		return None
		

'''
Stationary class. 
'''
class StationaryBlur(Blur):
	def __init__(self,kernel):
		super().__init__()

		# this holds the module's parameters, which is necessary for proper initialization of the nn.Module object 
		self._modules = {}

		# kernel must be a Kernel object or a psf data stored in a torch tensor
		if torch.is_tensor(kernel) is True:
			kernel=Kernel(kernel.shape,psf_data=kernel)

		self.kernel=kernel

	def _forward(self,x):
		pad_width=[x.shape[1]-self.kernel.shape[0],x.shape[2]-self.kernel.shape[1]]
		shifted_padded_kernel=cshift(pad(self.kernel(),pad_width))

		fft_x = torch.fft.fftn(x)
		fft_shifted_padded_kernel = torch.fft.fftn(shifted_padded_kernel, s=x.shape[-2:])
		fft_product = fft_x*fft_shifted_padded_kernel
		x_conv_kernel = torch.fft.ifftn(fft_product).real
		return x_conv_kernel

	def forward_check(self,x):
		from numpy.fft import fft2,ifft2
		ret=ifft2(fft2(x.clone().detach().numpy())*fft2(cshift(self.kernel()).clone().detach().numpy(),s=x.shape)).real
		return torch.tensor(ret)




'''
NonstationaryBlur class. 
'''
class NonstationaryBlur(Blur):
	def __init__(self,lattice):
		super().__init__()
		
		# this holds the module's parameters, which is necessary for proper initialization of the nn.Module object 
		self._modules = {}


'''
Kernel class
'''

class Kernel:
	def __init__(self,shape,mode='gaussian',psf_pars=None,psf_data=None,eps=1e-5):
		self.shape=shape
		self.mode=mode
		self.psf_pars=psf_pars
		self.eps=eps
		self.psf_data=psf_data
		
		# passing mode='from_data' is not necessary if psf_data is set
		if self.psf_data is not None:
			self.mode='from_data'
		
		

	def __call__(self):
		if self.mode=='gaussian':
			kernel=self._gaussian()
		elif self.mode=='from_data':
			kernel=self.psf_data	
		else:
			print('mode named \'',self.mode,'\' is unkown.')
			return -1
	
		return kernel
	
	
	# 2d rotated gaussian model 
	def _gaussian(self):
		# unpack parameters
		sx=torch.tensor(self.psf_pars.get('sx',0.1))
		sy=torch.tensor(self.psf_pars.get('sy',0.1))
		angle=self.psf_pars.get('angle',45.0)
		mx=torch.tensor(self.psf_pars.get('mx',0.0))
		my=torch.tensor(self.psf_pars.get('my',0.0))


		# initialize gaussian kernel with zeros
		gaussian_kernel=torch.zeros(self.shape).double()
		
		# create grid 
		x = torch.linspace(-1, 1, steps=self.shape[0])
		y = torch.linspace(-1, 1, steps=self.shape[1])
		xx, yy = torch.meshgrid(x, y)
		
		# fill kernel values
		gaussian_kernel = 1. / (2. * pi * sx * sy) * torch.exp(-((xx - mx)**2. / (2. * sx**2.) + (yy - my)**2. / (2. * sy**2.)))
    
    		# rotate using scipy
		rotated_kernel=scipy.ndimage.rotate(gaussian_kernel.detach().clone().numpy(),angle,mode='nearest',reshape=False).transpose()
		rotated_kernel=torch.tensor(rotated_kernel)
    		
		return rotated_kernel.double()
    		
		
		# window size (w)
		#self._find_window_size()


'''
Lattice class. Objects from this class are data structures the store nonstationary blur kernels
'''
class Lattice:
	def __init__(self,shape,lattice_pars=None):

		# main object is a dictionary of kernels
		self.kernels={}

		# other pars
		self.shape=shape
		self.lattice_pars=lattice_pars

	def kernel(self,pos):
		return kernels[tuple(pos)]

	# this is the function the properly draws the grid when needed
	def grid(self,grid_shape):
		return 0


	# samples target_lattice using self.shape indexes 
	def sample(self,target_lattice):
		return 0

	# writes the eigenkernels of self.kernels into self.kernels
	def pca(self,r=None):
		return 0

	def pcp_rpca(self,r=None,mu=None,lamb=None,max_iter=100,tol=1e-5,debug=False):
		return 0 

'''
RotatedGaussian class implements the rotated gaussian model of nonstationary blur. 

This class inherits the structure of the Lattice class, and by overloading it's kernel method, we
don't have to store all Kernel objects in the memory. As opposed to the pca/rpca based Lattices, 
in principe, a model like the rotated gaussian model needs to cast a different kernel in order to
blur each position of a target image. 
'''
class RotGaussian(Lattice):
	def __init__(self,img_shape,lattice_pars):
		super().__init__(img_shape,lattice_pars=lattice_pars)

		# find center point
		self.cent_i=int(np.round(self.shape[0]/2))
		self.cent_j=int(np.round(self.shape[1]/2))
		
		# half diagonal length normalization 
		self.half_diag_length=np.sqrt(self.cent_i**2 + self.cent_j**2)
		
		# radius
		# self.r=np.sqrt(self.cent_i**2 + self.cent_j**2)/self.half_diag_length
		
	def kernel(self,pos):
		# calculate parameters and return a Kernel() object call directly.

		# unpack parameters
		ax=self.lattice_pars.get('ax',0.1)
		ay=self.lattice_pars.get('ay',0.1)
		bx=self.lattice_pars.get('bx',0.1)
		by=self.lattice_pars.get('by',0.1)
		gamma_x=self.lattice_pars.get('gamma_x',0.1)
		gamma_y=self.lattice_pars.get('gamma_y',0.1)

		# transform pos into co,ca,theta and r 
		i,j=pos
		co=i-self.cent_i 
		ca=j-self.cent_j 
		theta=(np.arctan2(co,ca)/pi)*180.0
		r=np.sqrt(co**2 + ca**2)/self.half_diag_length

		# standard deviation in each axis
		sx=ax+bx*(r**gamma_x)
		sy=ay+by*(r**gamma_y)

		# cast a Kernel object and return	
		psf_pars={'sx':sx,'sy':sy,'angle':theta}

		return Kernel(self.shape,mode='gaussian',psf_pars=psf_pars)()


''' 
Deblur class. 
'''
class Deblur(nn.Module):
	def __init__(self):
		super().__init__()
		
		
		# misc parameters 
		self.forward_timer=-1
		
		return 0
		
	# y.shape: (batch_size,fig_shape[0],fig_shape[1])
	# x.shape: (batch_size,fig_shape[0],fig_shape[1]) 
	def forward(self,x):
        	# start timer
		start = timer()
    		
    		
		# add a batch dimension to the input, if necessary
		if x.dim()==2:
			x=x.unsqueeze(0)

        	# call internal forward method 
		y=self._forward(x)
        
		# update timer
		end = timer()    
		self.forward_timer=end-start
		
		# return
		return y
		
	# specific internal forward method
	def _forward(self,x):
		return None


''' 
misc functions
'''

# rectangular pad of specified width. output at dimension i is centered if width[i] is even.
def pad(x,width):

	# attemps to suppress batch dimension
	x=x.squeeze()

	# if batch dimension was suppressed, include it again
	if x.dim()==2:
		x=x.unsqueeze(0)
		no_batch_dimension_flag=True

	pn1,pn2=width
	batch_size,n1,n2=x.shape

	px=torch.zeros(batch_size,n1+pn1,n2+pn2)
	px[:,:n1,:n2]=x.clone().detach()

	new_center=((pn1+1)//2,(pn2+1)//2)

	if no_batch_dimension_flag is True:
		return cshift(px,new_center).reshape(n1+pn1,n2+pn2)
	else:
		return cshift(px,new_center).reshape(x.shape)


'''
shifts input f to an arbitrary position t. 
in my tests, when t=None, it is completely identical to torch.fft.fftshift(f,axis=None)
'''
def cshift(f, t=None):
	# store original input shape 
	og_f_shape=f.shape 

	# attemps to suppress batch dimension
	f=f.squeeze()

	# if batch dimension was suppressed, include it again
	if f.dim()==2:
		f=f.unsqueeze(0)

	batch_size,n1,n2 = f.shape
	h = torch.zeros([batch_size,2*n1, 2*n2])
	g = torch.zeros([batch_size,n1,n2])

	if t is None:
		# t = f.shape[1]//2+1, f.shape[2]//2+1
		t = f.shape[1]//2, f.shape[2]//2 # works better when the lengths of f are even

	rr, cc = t

	r = n1 - rr % n1
	c = n2 - cc % n2

	h[:,0:n1, 0:n2] = f.clone().detach()
	h[:,n1:2*n1, 0:n2] = f.clone().detach()
	h[:,0:n1, n2:2*n2] = f.clone().detach()
	h[:,n1:2*n1, n2:2*n2] = f.clone().detach()

	g = h[:,r:r+n1, c:c+n2]

	return g.reshape(og_f_shape)

# crops h around pos using a square window of half side size w
def crop_around(h, w, pos=None):
    k1, k2 = h.shape
    if pos is None:
        i = (k1)//2
        j = (k2)//2
    else:
        i, j = pos
    
    #h_pad = torch.nn.functional.pad(h, (w, w, w, w), mode='circular')
    h_pad=np.pad(h.clone().detach().numpy(),w,mode='wrap')
    h_pad=torch.tensor(h_pad)
    
    ipad, jpad = i+w, j+w
    cropped_h = h_pad[ipad-w:ipad+w+1, jpad-w:jpad+w+1]
    
    return cropped_h
