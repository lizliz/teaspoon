## @package teaspoon.ML.tents
# Machine learning featurization method
# 
# If you make use of this code, please cite the following paper:
# J.A. Perea, E. Munch, and F. Khasawneh.  "Approximating Continuous Functions On Persistence Diagrams." Preprint, 2017.
#
# 
#


# from teaspoon.Misc import printPrettyTime

import time
import numpy as np
import pandas as pd 


from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, RidgeClassifierCV, LassoCV
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score





## 	A class for storing all of the chosen parameters for a data frame and computation of ML.
#
class ParameterBucket(object):
	def __init__(self, description = ''):
		"""!@brief Creates a new ParameterBucket object.

	    This object is being used to keep track of all the parameters needed
	    for the tents ML featurization.

	    Parameters that are included in the ParameterBucket initially:

	    @param description A description, has no effect on code. This can be set on initialization.
	    @param d 
	    @param delta
	    @param epsilon
	    	The bounding box for the persistence diagram in the (birth, lifetime) coordinates is [0,d * delta] x [epsilon, d* delta + epsilon].  In the usual coordinates, this creates a parallelogram.
	    @param maxPower 
	    	The maximum degree used for the monomial combinations of the tent functions.  Testing suggests we usually want this to be 1.  Increasing causes large increase in number of features.
	    @param clfFunction 
	    	The choice of tool used for classification or regression, passed as the function.  This code has been tested using `sklearn` functions `RidgeClassiferCV` for classification and `RidgeCV` for regression. 

	    """
		self.description = description
		self.d = None
		self.delta = None
		self.epsilon = None

		self.maxPower = None
		self.clfFunction = None

		# @todo The following settings don't appear to be ever used.  Commented in case that's not correct, but should be removed eventually.
		# self.minBirth = None
		# self.maxBirth = None
		# self.minPers = None
		# self.maxPers = None
		# self.remove0cols = False

	def __str__(self):
		"""!
		@brief Nicely prints all currently set values in the ParameterBucket.
		"""
		attrs = vars(self)
		output = ''
		output += '---'
		output += 'Variables in parameter bucket\n'
		output += '---'
		for key in attrs.keys():
			output += str(key) + ' : '
			output += str(attrs[key])+ '\n'
		output += '---\n'
		return output



# -------------------------------------------- #
# -------------------------------------------- #
# ----------Featurization--------------------- #
# -------------------------------------------- #
# -------------------------------------------- #


## @brief Evaluates the (i,j)th tent function on the point (x,y) in (birth, death) plane.
#
# Tent (i,j) evaluates to 1 at \f$x=i,y = j + i + \epsilon\f$ and diameter of support is controlled by \f$\delta\f$.
#
#
#
# 
# We are shifting the entire grid up by a predetermined value epsilon.
# This is to keep the tent functions from being non-zero 
# in every nbhd of the diagonal.
#
# @param x,y
# 	Point in persistence diagram given as (birth, death). 
# 	This code assumes all points above diagonal,
# 	so y>x
# @param i,j
# 	Integers giving coordinate of the tent function.
#
# @return \f$g_{i,j}(x,y)\f$ where
# \f[g_{i,j}(x,y) = 
# \bigg| 1- \max\left\{ \left|\frac{x}{\delta} - i\right|, \left|\frac{y-x}{\delta} - j\right|\right\} \bigg|_+\f]
# where
# \f$| * |_+\f$ is positive part; equivalently, min of \f$*\f$ and 0.
#
def tent(x, y, i,j, delta = 1, epsilon = 0):


	#Send point (x,y) to the skewed plane
	def f(x,y):
		T = np.array(((1,0),(-1,1)))
		vec =  np.array(((x,),(y-epsilon,)))
		ans = 1/delta * np.dot(T,vec)
		return ans

	# Compute tent function there
	def tentSkew(a,b):
		val_a = abs(a-i)[0]
		val_b = abs(b-j)[0]
		ans = 1 - max(val_a,val_b)
		ans = max(ans,0)
		return ans

	A = f(x,y) #Point mapped to skewed (square) plane
	out = tentSkew(A[0],A[1])
	return out






## Generates row of features for the given diagram
#
# 
# 	@param Dgms
# 		A list or pd.Series. Length = M.
# 		Each entry is an N x 2 numpy array with off diagonal 
# 		points from the persistence diagram
# 	@param d 
# 		An integer, the number of elements for griding up the x and y
# 		axis of the diagram.  Will result in dx(d+1) tent functions 
# 	@param delta
# 		Width of mesh elements for x and y axis of the diagram.
#
# 	@param epsilon
# 		The amount to shift the grid up.
#	@param returnSquare
#		Stops the code from doing the final flattening.  
#
# 	The resulting parallelogram in the diagram has vertices
# 	(0,0 + epsilon)
# 	(0,d*delta  + epsilon)
# 	(d*delta, d*delta  + epsilon)
# 	(d*delta,2*d*delta  + epsilon)
#
# @return 
# 	A numpy ndarray with 
#	- shape:
# 		`( d*(d+1),  )` if returnSquare = False
# 	- entries:
# 		\f$ \displaystyle{\sum_{x \in Dgm}} g_{i,j}(x) \f$
# 		for tent functions \f$g_{i,j}\f$
# @todo Check for shape output if returnSquare = True.
def build_Gm(Dgm,d = 10,delta = 1, epsilon = 0, returnSquare = False):

	# N is number of points in the diagram
	N = np.shape(Dgm)[0]

	# Add birth times of zero
	if len(np.shape(Dgm)) <2:
		Z = np.zeros(N)
		Dgm = np.concatenate([[Z],[Dgm]]).T


	H = np.zeros((d+1, d, N))

	for i in range(d+1):
		for j in range(1,d+1):
			for  dgmRow in range(N):
				entry = tent(Dgm[dgmRow,0],Dgm[dgmRow,1], i,j,delta, epsilon)
				H[i,j-1,dgmRow ] = entry

	Gm = np.sum(H,axis = 2)
	# print('Shape of square Gm is', np.shape(Gm))
	if returnSquare:
		return(Gm)
	# print(Gm)
	Gm = np.ndarray.flatten(Gm)
	return Gm 


## Builds matrix of features, where each row corresponds to an input persistence diagram.
# 
#
#
#
# 	@param Dgms
# 		A list or pd.Series with length M.
# 		Each entry is an N x 2 numpy array with off diagonal 
# 		points from the persistence diagram.
# 	@param d 
# 		An integer, the number of elements for griding up the x and y
# 		axis of the diagram.  Will result in \f$d\times(d+1)\f$ tent functions.
# 	@param delta
# 		Width of mesh elements for x and y axis of the diagram.
#
# 	@param epsilon
# 		The amount to shift the grid up.  
# 	@param maxPower
# 		We want to output a matrix with monomials in the
# 		given base features up to power maxPower.
#
# 	Specifically, the support in the (birth, death) plane is a parallelogram with  vertices
# 	(0,0 + epsilon)
# 	(0,d*delta  + epsilon)
# 	(d*delta, d*delta  + epsilon)
# 	(d*delta,2*d*delta  + epsilon)
#
#
# @return G a numpy ndarray
# 	shape = d*(d+1)*choose( d*(d+1), maxPower)
#
# @todo Add figure for parallelogram.
def build_G(Dgms, d = 10, delta=1, epsilon = 0,maxPower = 1):
	M = len(Dgms)

	G = np.zeros((M,d**2+d))
	for m, Dgm in enumerate(Dgms):
		G[m,:] = build_Gm(Dgm,d,delta,epsilon)

	if maxPower > 1:
		numCols = np.shape(G)[1] #num of cols in G

		numColsInBigG = int(sum([comb(numCols, k, repetition = True) for k in range(1, maxPower +1)]))
		# print("M:", type(M), M)
		# print("numColsInBigG:", type(numColsInBigG), numColsInBigG)
		BigG = np.zeros((M,numColsInBigG))
		BigG[:,:numCols] = G

        
		count = numCols
		for L in range(2, maxPower + 1):
		    for subset in itertools.combinations_with_replacement(range(numCols),L):
				# #print('subset is', subset)
				# #print('cols are\n', testG[:,subset])
				# #print('product is\n', np.prod(testG[:,subset],axis = 1))
		        BigG[:, count] = np.prod(G[:,subset],axis = 1)
		        count = count + 1

		G = BigG
	# print('Max power is ', maxPower)
	# print('Shape of G is ', np.shape(G))


	return G
