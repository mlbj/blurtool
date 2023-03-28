import math
import torch
import torch.nn as nn
import torch.fft
import torchvision
import PIL
import cv2
import copy

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')


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
		
	
	# to-do: transform time measuring and batch handling into function decorators
	# x.shape: (batch_size,fig_shape[0],fig_shape[1]) 
	# y.shape: (batch_size,fig_shape[0],fig_shape[1])
	def forward(self,x,n_pool=1,debug=False):
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
		y=self._forward(x,n_pool=n_pool,debug=debug)
        
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

	def _forward(self,x,n_pool=1,debug=False):
		pad_width=[x.shape[1]-self.kernel.shape[0],x.shape[2]-self.kernel.shape[1]]
		shifted_padded_kernel=cshift(pad(self.kernel(),pad_width))

		fft_x = torch.fft.fftn(x)
		fft_shifted_padded_kernel = torch.fft.fftn(shifted_padded_kernel, s=x.shape[-2:])
		fft_product = fft_x*fft_shifted_padded_kernel
		x_conv_kernel = torch.fft.ifftn(fft_product).real
		return x_conv_kernel

	# def forward_check(self,x):
	# 	from numpy.fft import fft2,ifft2
	# 	ret=ifft2(fft2(x.clone().detach().numpy())*fft2(cshift(self.kernel()).clone().detach().numpy(),s=x.shape)).real
	# 	return torch.tensor(ret)


'''
NonstationaryBlur class. 
'''
class NonstationaryBlur(Blur):
	def __init__(self,lattice):
		super().__init__()
	
		# this holds the module's parameters, which is necessary for proper initialization of the nn.Module object 
		self._modules = {}

		self.lattice=lattice
		self.shape=self.lattice.shape

		# other attributes
		self._temp_x=None

		# set forward method
		if self.lattice.decomposed is True:
			self._forward=self._forward_decomposed
		else:
			self._forward=self._forward_nondecomposed

	# nonstationary blur can for forwarded in a different ways depending whether self.lattice is decomposed or not. 
	# since kernels are computed using eigenkernels normally when the lattice is decomposed, it would be fine to just 
	# use the _forward_nondecomposed always. however, for decomposed lattices, _forward_decomposed is way faster.
	def _forward_nondecomposed(self,x,n_pool=1,debug=False):
		y=torch.zeros(x.shape)

		# linear forward
		if n_pool==1:
			for i in range(x.shape[0]):
				for j in range(x.shape[1]):
					for k in range(x.shape[2]):
						pos=[j,k]

						# define operands
						kernel=self.lattice.kernel(pos)

						w=kernel.shape[0]//2
						# x_window=crop_around_check(x[i,:,:].squeeze().clone().detach().numpy(),w,pos=pos)
						# x_window=torch.tensor(x_window)
						x_window=crop_around(x[i,:,:].squeeze(),kernel.shape,pos=pos)
						
						# define a stationary blur and operate
						stat_blur=StationaryBlur(kernel)

						y[i,j,k]=stat_blur.forward(x_window)[kernel.shape[0]//2,kernel.shape[1]//2]

					if debug is True:
						print('finished row ',j)
		
		# parallel forward
		else:
			# store x as an attribute temporarily
			self._temp_x=x#.clone()

			# yjs=[]

			# # compute each row in parallel using torch.multiprocessing.pool
			# with mp.get_context("spawn").Pool(n_pool) as pool:
			# # with mp.Pool(n_pool) as pool: 
			# 	for j in range(x.shape[1]):
			# 		args=[(i,j,k) for i in range(x.shape[0]) for k in range(x.shape[2])]
			# 		yj=torch.tensor(pool.map(self._forward_multi,args)).reshape(x.shape[0],x.shape[2])
			# 		yjs.append(yj)

			# 		if debug is True:
			# 			print('finished row ',j)

			# 	y=torch.stack(yjs,dim=1)

			#compute everything in parallel at once using torch.multiprocessing.pool
			with mp.get_context("spawn").Pool(n_pool) as pool:
				args=[(i,j,k) for i in range(x.shape[0]) for j in range(x.shape[1]) for k in range(x.shape[2])]
				y=torch.tensor(pool.map(self._multi_forwardfun,args)).reshape(x.shape)

		return y


	# function to be mapped into pool
	def _multi_forwardfun(self,args):
		# unpack parameters
		i,j,k=args
		pos=[j,k]

		# define operands
		kernel=self.lattice.kernel(pos)
		x_window=crop_around(self._temp_x[i,:,:].squeeze(),kernel.shape,pos=pos)

		# define a stationary blur and operate
		stat_blur=StationaryBlur(kernel)

		yijk=stat_blur.forward(x_window)[kernel.shape[0]//2,kernel.shape[1]//2]

		return yijk


	# decomposed forward method
	def _forward_decomposed(self,x,n_pool=1,debug=False):
		y=torch.zeros(x.shape)
		pad_shape=(x.shape[1]-self.lattice.kernel_shape[0],x.shape[2]-self.lattice.kernel_shape[1])

		for c in range(self.lattice.rank):
			# create a blur object using the current padded eigenkernel
			eigenkernel_data=self.lattice.eigenkernels[c]()
			padded_eigenkernel_data=pad(eigenkernel_data,pad_shape)
			padded_eigenkernel=Kernel(self.shape,psf_data=padded_eigenkernel_data,mode='from_data')
			stat_blur=StationaryBlur(padded_eigenkernel)

			# representation coefficients
			coef=self.lattice.eigencoefs[c]

			# define a stationary blur and operate
			y=y+stat_blur.forward(coef*x)

		return y

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
			raise TypeError('Unrecognized mode in Kernel')
	
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
		x = torch.linspace(-1,1, steps=self.shape[0])
		y = torch.linspace(-1,1, steps=self.shape[1])
		xx,yy = torch.meshgrid(x, y,indexing='ij')

		# fill kernel values
		gaussian_kernel = 1. / (2. * pi * sx * sy) * torch.exp(-((xx - mx)**2. / (2. * sx**2.) + (yy - my)**2. / (2. * sy**2.)))

		# rotate using scipy
		# TODO: implement with torch
		rotated_kernel=scipy.ndimage.rotate(gaussian_kernel.detach().clone().numpy(),angle,mode='nearest',reshape=False).transpose()
		rotated_kernel=torch.tensor(rotated_kernel).double()

		return rotated_kernel 


'''
Lattice class. Objects from this class are data structures the store nonstationary blur kernels
'''
class Lattice:
	def __init__(self,shape,kernel_shape=None,lattice_pars=None):

		# main object is a dictionary of kernels
		self.kernels={}

		# other pars
		self.shape=shape
		self.lattice_pars=lattice_pars
		self.kernel_shape=None
		self.decomposed=False

		# find out kernel_shape, if needed/possible
		if kernel_shape is None:
			if self.kernel((0,0)) is not None:
				# currently all kernels must be of the same shape, so use the first one.
				self.kernel_shape=self.kernel((0,0))().shape
			else:
				self.kernel_shape=None
		else:
			self.kernel_shape=kernel_shape

	# this method returns the kernel object in a given position of the lattice
	def kernel(self,pos):
		return self.kernels.get(tuple(pos),None)


	# this method returns a sampled lattice of shape sampled_shape whose kernels are 
	# regularly sampled from the current lattice.
	# sample_mode may be 'deep_copy' for copying the original Kernel objects, not only their reference. 
	# or 'copy_data', which also creates new Kernel objects, but using 'from_data' Kernel psf_mode.
	# TODO: unit test of sample regularity 
	def sample(self,sampled_shape,sample_mode='copy_from_data'):
		# cast a grid of the same img_shape as the current lattice, but using the new sampled lattice shape
		sample_grid,table=self.grid(sampled_shape,return_table=True)
		table_i,table_j=table

		# define sampled kernels
		sampled_kernels={}

		# map indexes of sampled lattice to the ones in the current lattice and fill the dictionary
		for i in range(sampled_shape[0]):
			for j in range(sampled_shape[1]):
				sampled_pos=(table_i[i].item(),table_j[j].item())
				
				# sample the current kernel
				if sample_mode=='deep_copy':
					sampled_kernels[(i,j)]=copy.deepcopy(self.kernel(sampled_pos))
				if sample_mode=='copy_from_data':
					psf_data=self.kernel(sampled_pos)()
					sampled_kernels[(i,j)]=Kernel(self.kernel_shape,psf_data=psf_data,mode='from_data')


		# define output sampled lattice
		sampled_lattice=Lattice(sampled_shape)
		sampled_lattice.kernels=sampled_kernels	
		sampled_lattice.kernel_shape=self.kernel_shape

		return sampled_lattice

	# this is the function the properly draws a test grid
	# if return_table is True, it returns the table needed for lattice sampling.
	def grid(self,sampled_shape,return_table=False,blur=False):

		# define output variables
		table_i=torch.zeros(sampled_shape[0]).int()
		table_j=torch.zeros(sampled_shape[1]).int()
		test_grid=torch.zeros(self.shape)

		mod1=int(np.round(self.shape[0]/(sampled_shape[0]+1)))
		mod2=int(np.round(self.shape[1]/(sampled_shape[1]+1)))
		column_sum=0
		for i in range(self.shape[0]):
			line_sum=0
			for j in range(self.shape[1]):
				if column_sum<sampled_shape[0]:
					if i%mod1==0 and j%mod2==0 and i!=0 and j!=0 and line_sum<sampled_shape[1]:
						# self.points.append((i,j))
						table_i[column_sum]=i
						table_j[line_sum]=j
						test_grid[i,j]=1
						line_sum+=1
				else:
					break
			if line_sum==sampled_shape[1]:
				column_sum+=1

			# self.table_i=self.table_i.astype(int)
			# self.table_j=self.table_j.astype(int)

		# return blurred test grid if needed
		# TODO: fix number of pools
		if blur is True:
			test_blur=NonstationaryBlur(self)
			test_grid=test_blur.forward(test_grid,n_pool=4)

		if return_table is True:
			return test_grid,(table_i,table_j)
		else:
			return test_grid


	# stack kernels into a matrix of the form [kernel_shape[0]*kernel_shape[1],self.shape[0]*self.shape[1]]
	# WARNING: i don't recommend runnning this if self.shape is too big, since it stores the kernel.__call__() data for all kernels. 
	#		  the correct way to do it, is to sample the big lattice into a smaller one, and only then stacking the sampled kernels 
	# 		  to be decomposed.	 
	def stack_kernels(self):

		# allocate memory
		stacked_kernels=torch.zeros((self.kernel_shape[0]*self.kernel_shape[1],self.shape[0]*self.shape[1]))

		# fill one kernel at the time
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				pos=(i,j)
				raveled_pos=ravel_multi_index((i,j),(self.shape[0],self.shape[1]))
				stacked_kernels[:,raveled_pos]=self.kernel(pos)().reshape(self.kernel_shape[0]*self.kernel_shape[1])

		return stacked_kernels

	# unstack kernels from a matrix of shape [kernel_shape[0]*kernel_shape[1],self.shape[0]*self.shape[1]] to a 
	# dictionary. stacked_kernels is the same number of kernels as the number of points of the current lattice object.
	# Also, each kernel is expected to have the same shape as in self.kernel_shape.
	# TODO: write a unit test using this function 
	def unstack_kernels(self,stacked_kernels):
		# check dimensions
		if stacked_kernels.shape[0]!=self.kernel_shape[0]*self.kernel_shape[1]:
			raise ValueError('Kernel shape mismatch in unstack_kernels()')
		if stacked_kernels.shape[1]!=self.shape[0]*self.shape[1]:
			raise ValueError('Lattice shape mismatch in unstack_kernels()')

		kernels={}
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				pos=(i,j)
				ravaled_pos=ravel_multi_index((i,j),(self.shape[0],self.shape[1]))
				psf_data=stacked_kernels[:,ravaled_pos].reshape(self.kernel_shape[0],self.kernel_shape[1])
				kernel=Kernel(self.kernel_shape,mode='from_data',psf_data=psf_data)
				kernels[pos]=kernel

		self.kernels=kernels


'''
RotatedGaussian class implements the rotated gaussian model of nonstationary blur. 

This class inherits the structure of the Lattice class, and by overloading it's kernel method, we
don't have to store all Kernel objects in the memory. As opposed to the pca/rpca based Lattices, 
in principe, a model like the rotated gaussian model needs to cast a different kernel in order to
blur each position of a target image. 
'''
class RotGaussian(Lattice):
	def __init__(self,shape,kernel_shape,lattice_pars):
		super().__init__(shape,kernel_shape=kernel_shape,lattice_pars=lattice_pars)

		# kernel shape
		self.kernel_shape=kernel_shape

		# find center point
		self.cent_i=int(np.round(self.shape[0]/2))
		self.cent_j=int(np.round(self.shape[1]/2))
		
		# half diagonal length normalization 
		self.half_diag_length=np.sqrt(self.cent_i**2 + self.cent_j**2)
		
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

		return Kernel(self.kernel_shape,mode='gaussian',psf_pars=psf_pars)


'''
DecomposeLattice subclass. 
eigenkernels are sorted and stored in a list as an attribute.
TODO: instead of having modes set inside DecomposeLattice, in a future version, it would be appropriate to 
      have a nn.Module object which decomposes the lattice as a parameter.
'''
class DecomposeLattice(Lattice):
	def __init__(self,input_lattice,img_shape,mode='pca',decompose_pars=None):
		super().__init__(img_shape,kernel_shape=input_lattice.kernel_shape)

		# eigenkernel related attributes
		self.eigenkernels=[]
		self.eigencoefs=[]
		self.noninterpolated_eigencoefs=[]
		self.rank=0

		# other attributes
		self.decompose_pars=decompose_pars
		self.input_lattice_shape=input_lattice.shape

		# no kernel information has been passed so for to super(), so we need to copy it manually
		self.kernel_shape=input_lattice.kernel_shape

		if mode=='pca':
			self._pca(input_lattice)
		elif mode=='pcp_rpca':
			self._pcp_rpca(input_lattice)
		else:
			raise TypeError('Unrecognized mode in DecomposeLattice')


	# unstack a matrix of eigenkernels (data) into a list of callable kernel objects with from_data mode.
	# TODO: reordering parameter if needed
	def unstack_eigenkernels(self,stacked_eigenkernels):
		# check dimensions
		if stacked_eigenkernels.shape[0]!=self.kernel_shape[0]*self.kernel_shape[1]:
			raise ValueError('Kernel shape mismatch in unstack_eigenkernels()')

		eigenkernels=[]
		for i in range(stacked_eigenkernels.shape[1]):
			psf_data=stacked_eigenkernels[:,i].reshape(self.kernel_shape[0],self.kernel_shape[1])
			eigenkernel=Kernel(self.kernel_shape,mode='from_data',psf_data=psf_data)
			eigenkernels.append(eigenkernel)

		self.eigenkernels=eigenkernels


	# overload the kernel method to interpolated eigenkernel representation 
	def kernel(self,pos):
		pos=tuple(pos)
		psf_data=torch.zeros(self.kernel_shape)#+self.mean_kernel
		for c in range(self.rank):
			psf_data=psf_data+self.eigencoefs[c][pos].item()*self.eigenkernels[c]()

		return Kernel(self.kernel_shape,psf_data=psf_data,mode='from_data')

	def noninterpolated_kernel(self,pos):
		pos=tuple(pos)
		psf_data=torch.zeros(self.kernel_shape)#+self.mean_kernel
		for c in range(self.rank):
			psf_data=psf_data+self.noninterpolated_eigencoefs[c][pos].item()*self.eigenkernels[c]()

		return Kernel(self.kernel_shape,psf_data=psf_data,mode='from_data')


	# return a lattice whose kernels are the eigenkernels of self.kernels 
	def _pca(self,input_lattice):
		# check if it's already decomposed

		# unpack decompose_pars
		rank=self.decompose_pars.get('r',input_lattice.shape[0]*input_lattice.shape[1])

		# mean bias is stored in the last element, so add one to rank
		self.rank=rank+1

		# stack input kernels and normalize 
		#stacked_kernels=normalize(input_lattice.stack_kernels())
		stacked_kernels=input_lattice.stack_kernels()

		# store and subtract the mean kernel of each (stacked) kernel
		# TODO: check if that's correct again
		stacked_mean_kernel=torch.mean(stacked_kernels,dim=1)
		for i in range(input_lattice.shape[0]):
			for j in range(input_lattice.shape[1]):
				unraveled_pos=(i,j)
				index=ravel_multi_index(unraveled_pos,input_lattice.shape)
				stacked_kernels[:,index]=stacked_kernels[:,index]-stacked_mean_kernel

		# save mean_kernel 
		self.mean_kernel=stacked_mean_kernel.reshape(self.kernel_shape)  

        # allocate coefs.
        # TODO: check if it is really 'self.rank+1'.
		stacked_coefs=torch.zeros((self.kernel_shape[0]*self.kernel_shape[1],self.rank+1))
        
		# run SVD
		# TODO: use torch.
		U,D,Vt=scipy.linalg.svd(stacked_kernels.clone().detach().numpy())     
		stacked_eigenkernels=torch.from_numpy(U[:,:self.rank])
		stacked_noninterpolated_coefs=torch.from_numpy((np.diag(D[:self.rank]).dot(Vt[:self.rank,:])).transpose())

		# insert mean kernel information at the begining 
		stacked_ones_coefs=torch.ones(self.input_lattice_shape[0]*self.input_lattice_shape[1],1)
		stacked_noninterpolated_coefs=torch.cat((stacked_ones_coefs,stacked_noninterpolated_coefs),dim=1)

		stacked_mean_kernel=stacked_mean_kernel.reshape(self.kernel_shape[0]*self.kernel_shape[1],1)
		stacked_eigenkernels=torch.cat((stacked_mean_kernel,stacked_eigenkernels),dim=1)

		# set self.eigenkernels
		self.unstack_eigenkernels(stacked_eigenkernels)

		# interpolate coefs
		coefs=[]
		noninterpolated_coefs=[]
		for j in range(stacked_noninterpolated_coefs.shape[1]):
			noninterpolated_coef=stacked_noninterpolated_coefs[:,j].reshape(self.input_lattice_shape[0],self.input_lattice_shape[1])

 			# interpolate coef into self.shape 
 			# other possible choices for interpolation_mode: [cv2.INTER_CUBIC,cv2.INTER_LINEAR,cv2.INTER_NEAREST,cv2.INTER_AREA]
			interpolation_mode=cv2.INTER_LANCZOS4
			coef=cv2.resize(noninterpolated_coef.clone().detach().numpy(),self.shape,interpolation_mode)
			coef=torch.from_numpy(coef).double()

			# fill 
			coefs.append(coef.double())
			noninterpolated_coefs.append(noninterpolated_coef)

			# interpolation debug
			# print('------------------------------------------')
			# print('j=',j)
			# print('ni_min=',torch.min(noninterpolated_coef))
			# print('ni_max=',torch.max(noninterpolated_coef))
			# print('ni_norm=',torch.norm(noninterpolated_coef))	
			# print('ni_mean=',torch.mean(noninterpolated_coef))	
			# print('ni_std=',torch.std(noninterpolated_coef))	
			# print('------------------------------------------')
			# print('i_min=',torch.min(coef))
			# print('i_max=',torch.max(coef))
			# print('i_norm=',torch.norm(coef))	
			# print('i_mean=',torch.mean(coef))
			# print('i_std=',torch.std(coef))		
			# print('------------------------------------------\n')
			

		# set coefs attribute
		self.eigencoefs=coefs
		self.noninterpolated_eigencoefs=noninterpolated_coefs

		# set singular values
		self.eigenvals=torch.tensor(D[:self.rank],dtype=torch.float64)

		# set decomposed flag to True
		self.decomposed=True

	def _pcp_rpca(self,input_lattice):
		# reset decomposed tag
		self.decomposed=False

		# stack kernels
		stacked_kernels=input_lattice.stack_kernels()


		# normalize 
		# mi,ma=torch.min(stacked_kernels),torch.max(stacked_kernels)
		# stacked_kernels=(stacked_kernels-mi)/(ma-mi)

		# unpack decompose_pars
		# mu=self.decompose_pars.get('mu',0.25/torch.abs(torch.mean(stacked_kernels)))
		mu=self.decompose_pars.get('mu',0.25/torch.abs(torch.mean(stacked_kernels)))

		lamb=self.decompose_pars.get('lamb',1/torch.sqrt(torch.max(torch.tensor(stacked_kernels.shape))))
		tol=self.decompose_pars.get('tol',1e-7)
		max_iter=self.decompose_pars.get('max_iter',500)
		debug=self.decompose_pars.get('debug',True)

		# # change to torch
		# if torch.is_tensor(mu):
		# 	mu=mu.clone().detach().numpy()
		# if torch.is_tensor(lamb):
		# 	lamb=lamb.clone().detach().numpy()

		# stacked_kernels=stacked_kernels.clone().detach().numpy()
		# B=np.zeros(stacked_kernels.shape)
		# A=np.zeros(stacked_kernels.shape)
		# D=np.zeros(stacked_kernels.shape)
		B=torch.zeros(stacked_kernels.shape)
		A=torch.zeros(stacked_kernels.shape)
		M=torch.zeros(stacked_kernels.shape)

		rank=0
		err=tol+1
		ite=0
		while ite<max_iter: # and  err>tol:        
			# B - subproblem  
			# U, sigmas, V = np.linalg.svd(stacked_kernels - A + M/mu, full_matrices=False);
			# rank = (sigmas > 1/mu).sum()
			# Sigma = np.diag(sigmas[0:rank] - 1/mu)
			# B = np.dot(np.dot(U[:,0:rank], Sigma), V[0:rank,:])
			U, S, Vt = torch.linalg.svd(stacked_kernels-A+(1/mu)*M,full_matrices=False)
			B=torch.matmul(U, torch.matmul(torch.diag(torch.nn.Softshrink(1/mu)(S)), Vt))

			# A - subproblem 
			# A = torch.nn.Softshrink(lamb/mu)(torch.from_numpy(stacked_kernels - B + M/mu)).clone().detach().numpy() 
			A=torch.nn.Softshrink(lamb/mu)(stacked_kernels-B+(1/mu)*M)
            
			# Update Lambda (dual variable)
			# M = M + mu*(stacked_kernels - A - B)
			M=M+mu*(stacked_kernels-B-A)
            
			# err
			err=torch.norm(stacked_kernels-B-A,p='fro')/torch.norm(stacked_kernels,p='fro')
            
			if debug is True:
				print('ite=',ite,'. err=',err)
            
			# update ite
			ite=ite+1
        
		# reports
		print('mu=',mu,'. lamb=',lamb,'. rank=',rank)
  
		# pca fit
		# stacked_kernels=B.copy()

		# create new copy lattice
		bg_input_lattice=copy.deepcopy(input_lattice)

		# to torch
		# stacked_kernels=torch.from_numpy(B.copy())
		stacked_kernels=B.clone()

		# denormalize
		# stacked_kernels=stacked_kernels*(ma-mi)+mi	

		# copy bg component to the new lattice
		bg_input_lattice.unstack_kernels(stacked_kernels)
		
		# run pca 
		self.rank=rank
		pca_decompose_pars={'r':self.decompose_pars.get('r',self.rank-1)}
		self.decompose_pars=pca_decompose_pars
		self._pca(bg_input_lattice)


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

# rectangular pad of specified width. output at dimension i is centered if pad_shape[i] is even.
# TODO: directly used torch.pad instead of this
def pad(x,pad_shape):

	# attemps to suppress batch dimension
	x=x.squeeze()

	# if batch dimension was suppressed, include it again
	if x.dim()==2:
		x=x.unsqueeze(0)
		no_batch_dimension_flag=True

	pn1,pn2=pad_shape
	batch_size,n1,n2=x.shape

	px=torch.zeros(batch_size,n1+pn1,n2+pn2)
	px[:,:n1,:n2]=x.clone().detach()

	# find new center
	new_center=( (pn1+1)//2,(pn2+1)//2 )

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


# crops h around pos using a rectangular window of arbitrary shape
def crop_around(h,crop_shape,pos=None):
	k1,k2=h.shape
	if pos is None:
		i,j=k1//2,k2//2
	else:
		i,j=pos

	# unpack crop_shape
	c1,c2=crop_shape[0],crop_shape[1]
	w1,w2=crop_shape[0]//2,crop_shape[1]//2
	h1,h2=h.shape[0],h.shape[1]

	# fix shapes
	add_begin_1,add_begin_2,add_end_1,add_end_2=0,0,0,0
	if h1%2==0:
		add_begin_1=c1%2
	else:
		add_end_1=c1%2
	if h2%2==0:
		add_begin_2=c2%2
	else:
		add_end_2=c2%2

	# circularly pad the original image
	pad_amount=(w1+1,w1+1,w2+1,w2+1)
	h_pad = torch.nn.functional.pad(h.unsqueeze(0).unsqueeze(0),pad_amount, mode='circular')
	h_pad=h_pad.squeeze()

	# slice the padded image
	ipad=i+w1+1
	jpad=j+w2+1
	cropped_h=h_pad[ipad-w1-add_begin_1:ipad+w1+add_end_1,jpad-w2-add_begin_2:jpad+w2+add_end_2]

	return cropped_h


# [min,max] normalization 
# normalize x in [a,b].
def normalize(x,a=0.0,b=1.0):
	mi=torch.min(x)
	ma=torch.max(x)
	if ma>mi:
		return a+((x-mi)*(b-a))/(ma-mi)

	if mi>=a and mi<=b:
		rx=x*0.0 + mi
	else:
		rx=x*0.0 + (mi+ma)/2

	return rx

# salt and pepper noise
def spnoise(img, amount=0.5, s_vs_p=0.5):
	out = img.clone()
	amount = torch.tensor(amount, dtype=torch.float64)
	s_vs_p = torch.tensor(s_vs_p, dtype=torch.float64)
	# salt
	num_salt = torch.ceil(amount * torch.numel(img) * s_vs_p).int()
	coords = [torch.randint(0, i - 1, (num_salt,)) for i in img.shape]
	out[tuple(coords)] = torch.max(img) #1
	# pepper
	num_pepper = torch.ceil(amount * torch.numel(img) * (1. - s_vs_p)).int()
	coords = [torch.randint(0, i - 1, (num_pepper,)) for i in img.shape]
	out[tuple(coords)] = torch.min(img) #0
	return out


# unravel_index
def unravel_multi_index(index, shape):
#  https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/2
	# out = []
	# for dim in reversed(shape):
	# 	out.append(index % dim)
	# 	index = index // dim
	# return tuple(reversed(out))
	return np.unravel_multi_index(index,shape)

# ravel_index
def ravel_multi_index(index, shape):
	return np.ravel_multi_index(index,shape)


# peak signal to noise ratio
def psnr(original, compressed):
    # normalize in [0,1] both original and compressed signals
    original=(original-torch.min(original))/(torch.max(original)-torch.min(original))
    compressed=(compressed-torch.min(compressed))/(torch.max(compressed)-torch.min(compressed))
    
    # calculate mse
    mse = torch.mean((original - compressed) ** 2)
    
    # mse is zero means no noise is present in the signal 
    if(mse == 0):           
        return torch.tensor(float('inf'))
    
    # calculate psnr
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr