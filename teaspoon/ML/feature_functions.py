# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:15:11 2018

@author: khasawn3
"""
from teaspoon.Misc import printPrettyTime
import teaspoon.TDA.Persistence as pP

import time
import numpy as np
import pandas as pd 


from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, RidgeClassifierCV, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.preprocessing import scale, PolynomialFeatures
from scipy.special import comb
import itertools
##


import numpy as np
from mpl_toolkits import mplot3d
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# -------------------------------------------- #
# -------------------------------------------- #
# ----------Featurization--------------------- #
# -------------------------------------------- #
# -------------------------------------------- #


## Applies the tent function to a diagram.
# @param Dgm
# 	A persistence diagram, given as a $K \times 2$ numpy array
# @param params
# 	An tents.ParameterBucket object.  Really, we need d, delta, and epsilon from that.
# @param type
#	This code accepts diagrams either 
#	* in (birth, death) coordinates, in which case `type = 'BirthDeath'`, or 
#	* in (birth, lifetime) = (birth, death-birth) coordinates, in which case `type = 'BirthLifetime'`
# @return \f$\sum_{x,y \in \text{Dgm}}g_{i,j}(x,y)\f$ where
# \f[g_{i,j}(x,y) = 
# \bigg| 1- \max\left\{ \left|\frac{x}{\delta} - i\right|, \left|\frac{y-x}{\delta} - j\right|\right\} \bigg|_+\f]
# where
# \f$| * |_+\f$ is positive part; equivalently, min of \f$*\f$ and 0.
# @note This code does not take care of the maxPower polynomial stuff.  The build_G() function does it after all the rows have been calculated.
def tent(Dgm, params, type = 'BirthDeath'):
	d = params.d
	delta = params.delta 
	epsilon = params.epsilon
	# print(Dgm[:3])
	# Move to birth,lifetime plane
	if type == 'BirthDeath':
		T = np.array(((1,-1),(0,1)))
		A = np.dot( Dgm, T)
	elif type == 'BirthLifetime':
		A = Dgm
	else:
		print('Your choices for type are "BirthDeath" or "BirthLifetime".')
		print('Exiting...')
		return

	I,J = np.meshgrid(range(d+1), range(1,d+1))

	Iflat = delta*I.reshape(np.prod(I.shape))
	Jflat = delta*J.reshape(np.prod(I.shape)) + epsilon

	Irepeated = Iflat.repeat(Dgm.shape[0])
	Jrepeated = Jflat.repeat(Dgm.shape[0])

	DgmRepeated = np.tile(A,(len(Iflat),1))

	BigIJ = np.array((Irepeated,Jrepeated)).T

	B = DgmRepeated - BigIJ
	B = np.abs(B)
	B = np.max(B, axis = 1)
	B = delta-B
	B = np.where(B >=0, B, 0)
	B = B.reshape((Iflat.shape[0],Dgm.shape[0]))
	out = np.sum(B,axis = 1)

	out = out.reshape((d,d+1)).T.flatten()
	out = out/delta

	# TO BE REMOVED.... THIS HAS BEEN MOVED TO build_G()
	# if params.maxPower >1:


	# 	BigOuts = [out]
	# 	# Make 2 using np.triu_indices
	# 	indices = np.array(np.triu_indices(len(out)))
	# 	C = out[indices.T]
	# 	C = np.prod(C,1)
	# 	BigOuts.append(C)
	# 	# Make 3 or above using itertools
	# 	# NOTE: This is incredibly slow and should be improved.
	# 	for i in range(3,params.maxPower + 1):
	# 		C = [a for a in itertools.combinations_with_replacement(out,i)]
	# 		C = np.array(C)
	# 		C = np.prod(C,1)
	# 		BigOuts.append(C)
	# 	# turn all of them into one long vector
	# 	out = np.concatenate(BigOuts)


	return out 



# coverts subscripts to indices
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind


# convert indices to subscripts
def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return rows, cols


# find the quadrature/interpolation weights for common orthognal functions
# define the function blocks
# Chebyshev-Gauss points of the first kind
def quad_cheb1(npts=10):
    # get the Chebyshev nodes of the first kind
    x = np.cos(np.pi * np.arange(0, npts+1) / npts)
    
    # get the corresponding quadrature weights
    w = np.pi / (1 + np.repeat(npts, npts + 1)) 
    
    return x, w


# Computes the Legendre-Gauss-Lobatto nodes, and their quadrate weights.
# The LGL nodes are the zeros of (1-x**2)*P'_N(x), wher P_N is the nth Legendre
# polynomial
def quad_legendre(npts=10):
    # Truncation + 1
    nptsPlusOne = npts + 1
    eps = np.spacing(1)  # find epsilon, the machine precision
    
    # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
    x = np.cos(np.pi* np.arange(0, npts+1) / npts)
    
    # Get the Legendre Vandermonde Matrix
    P = np.zeros((nptsPlusOne, nptsPlusOne))
    
    # Compute P_(N) using the recursion relation
    # Compute its first and second derivatives and 
    # update x using the Newton-Raphson method.
    
    xold = 2
    
    while np.max(np.abs(x-xold)) > eps:           
        xold = x
            
        P[:, 0] = 1
        P[:, 1] = x
        
        for k in range(1, npts):
            P[:, k+1] = ((2 * k + 1) * x * P[:, k] - k * P[:, k-1] ) / (k+1)
            
        x = xold - ( x * P[:, npts] - P[:, npts-1] ) \
        / ( nptsPlusOne * P[:, npts] )

    # get the corresponding quadrature weights
    w = 2 / (npts * nptsPlusOne * P[:, npts] ** 2)
    
    return x, w
    

# map the inputs to the function blocks
# you can invoke the desired function block using:
# quad_pts_and_weights['name_of_function'](npts)
quad_pts_and_weights = {'cheb1' : quad_cheb1,
           'legendre' : quad_legendre
           }


# find the barycentric interplation weights
def bary_weights(x):
    # Calculate Barycentric weights for the base points x.
    #
    # Note: this algorithm may be numerically unstable for high degree
    # polynomials (e.g. 15+). If using linear or Chebyshev spacing of base
    # points then explicit formulae should then be used. See Berrut and
    # Trefethen, SIAM Review, 2004.

    # find the length of x
    n = x.size

    # Rescale for numerical stability
    eps = np.spacing(1)  # find epsilon, the machine precision
    x = x * (1 / np.prod(x[x > -1 + eps] + 1)) ** (1/n)

    # find the weights
    w = np.zeros((1, n))
    for i in np.arange(n):
        w[0, i] = np.prod(x[i] - np.delete(x, i))

    return 1 / w


def bary_diff_matrix(xnew, xbase, w=None):
    # Calculate both the derivative and plain Lagrange interpolation matrix
    # using Barycentric formulae from Berrut and Trefethen, SIAM Review, 2004.
    # xnew     Interpolation points

    # xbase    Base points for interpolation
    # w        Weights calculated for base points (optional)

    # if w is not set, set it
    if w is None:
        w = bary_weights(xbase)

    # get the length of the base points
    n = xbase.size

    # get the length of the requested points
    nn = xnew.size

    # replicate the weights vector into a matrix
    wex = np.tile(w, (nn, 1))

    # Barycentric Lagrange interpolation (from Berrut and Trefethen, SIAM Review, 2004)
    xdiff = np.tile(xnew[np.newaxis].T, (1, n)) - np.tile(xbase, (nn, 1))

    M = wex / xdiff

    divisor = np.tile(np.sum(M, axis=1)[np.newaxis].T, (1, n))
    divisor[np.isinf(divisor)] = float("inf")

    M[np.isinf(M)] = float("inf")
    M = M / divisor

    M[np.isnan(M)] = 0
    M[xdiff == 0] = 1

    # Construct the derivative (Section 9.3 of Berrut and Trefethen)
    xdiff2 = xdiff ** 2

    frac1 = wex / xdiff
    frac1[np.isinf(frac1)] = float("inf")
    frac2 = wex / xdiff2
    DM = (M * np.tile(np.sum(frac2, axis=1)[np.newaxis].T, (1, n)) - frac2) / \
         np.tile(np.sum(frac1, axis=1)[np.newaxis].T, (1, n))
    row, col = np.where(xdiff == 0)

    if np.all(row == 0):  # or, row.size == 0:
        DM[row, ] = (wex[row, ] / np.tile(w[col].T, (1, n))) / xdiff[row, ]
        idx = sub2ind(DM.shape, row, col)
        DM[idx] = 0
        DM[idx] = -np.sum(DM[row, ], axis=1)

    return DM, M

# this is the function that will get the interpolation weights
# test interpolation in 2D
nx = 100  # number of interpolation points in x
ny = 50  # number of interpolation points in y

## create the and y vectors
#x = np.linspace(0, 1, nx)
#y = np.linspace(0, 1, ny)
#
## define the mesh
#[xmesh, ymesh] = np.meshgrid(x, y)
#
## these lines show a sample 3D plot
#fig2 = plt.figure(num=2)
#ax = plt.axes(projection='3d')
#
#ax.scatter(xmesh, ymesh, np.sin(xmesh+ymesh))
## ax.legend()
#
#plt.show()

# These lines show the algorithm for finding a score for each mesh point in the
# persistence diagram. The example used is for 2D interpolation of a function

# 0) create test data
# create the data:
# define a function of two variables which translates and scales Gaussiand.
# This is equivalent to Matlab's peaks function. 
z =  lambda x, y : 3 * (1-x) ** 2 * np.exp(-(x ** 2) - (y+1) ** 2) \
   - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(-x ** 2-y ** 2) \
   - 1/3 * np.exp(-(x + 1) ** 2 - y ** 2) 
   
# define the query points. These ar the points where we want to interpolate.
# get the 1D query points along x
num_query_pts = 100
xq = np.linspace(start=-1, stop=1, num=num_query_pts)

# get the 1D query points along y
yq = np.linspace(start=-1, stop=1, num=num_query_pts)

# get the query mesh
#xquery, yquery = np.meshgrid(xq, yq, sparse=False, indexing='ij')
#xquery_flat = np.concatenate(xquery)
#yquery_flat = np.concatenate(yquery)

# evaluate the function at the (xq, yq) mesh tuples
zq = z(xq, yq)

# define a base mesh on legendre-gauss-lobatto points
# specify the order of the interpolating polynomials (num_pts+1 nodes)
#num_pts = 100

# 1) Get the base nodes:
# get the 1D base nodes in x and y
xmesh, w = quad_pts_and_weights['legendre'](nx)
ymesh, w = quad_pts_and_weights['legendre'](ny)
xmesh = np.sort(xmesh)
ymesh = np.sort(ymesh)

# define a mesh on the base points
x_base, y_base = np.meshgrid(xmesh, ymesh, sparse=False, indexing='ij')

# flatten the vectors for later plotting
xbase_flat =  np.concatenate(x_base)
ybase_flat = np.concatenate(y_base)

# get the values of the function at the base mesh points
z_base = z(np.concatenate(x_base), np.concatenate(y_base))

# get the x and y interpolation matrices
# get the 1D interpolation matrix for x
x_meshdiff_mat, x_interp_mat = bary_diff_matrix(xnew=xq, xbase=xmesh)
x_interp_mat = x_interp_mat.T  # transpose the x-interplation matrix

# get the 1D interpolation matrix for y
y_meshdiff_mat, y_interp_mat = bary_diff_matrix(xnew=yq, xbase=ymesh)

# replicate each column in the x-interpolation matrix n times
Gamma = np.repeat(x_interp_mat, ny+1, axis=1)
# unravel, then replicate each row in the y-interpolation matrix m times
y_interp_mat.shape = (1, y_interp_mat.size)
Phi = np.repeat(y_interp_mat, nx+1, axis=0)

# element-wise multiply Gamma and Phi
Psi = Gamma * Phi

# split column-wise, then concatenate row-wise
Psi = np.concatenate(np.split(Psi, num_query_pts, axis=1), axis=0)

# now reshape Psi so that each row corresponds to the weights of one query pt
Psi = np.reshape(Psi, (num_query_pts, -1))

# get the weights for each interpolation function/base-point
interp_weights = np.sum((Psi), axis=0)