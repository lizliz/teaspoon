## @package teaspoon.ML.tents
# Machine learning featurization method
# 
# If you make use of this code, please cite the following paper:
# J.A. Perea, E. Munch, and F. Khasawneh.  "Approximating Continuous Functions On Persistence Diagrams." Preprint, 2017.
#
# 
#


from teaspoon.Misc import printPrettyTime

import time
import numpy as np
import pandas as pd 


from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, RidgeClassifierCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score





## 	A class for storing all of the chosen parameters for a data frame and computation of ML.
#
class ParameterBucket(object):
	def __init__(self, description = '',
					d=10,
					delta = 1,
					epsilon = 0,
					maxPower = 1,
					clfClass = RidgeClassifierCV):
		"""!@brief Creates a new ParameterBucket object.

	    This object is being used to keep track of all the parameters needed
	    for the tents ML featurization.

	    Parameters that are included in the ParameterBucket initially:

	    @param description A description, has no effect on code. This can be set on initialization.
	    @param d, delta, epsilon
	    	The bounding box for the persistence diagram in the (birth, lifetime) coordinates is [0,d * delta] x [epsilon, d* delta + epsilon].  In the usual coordinates, this creates a parallelogram.
	    @param maxPower 
	    	The maximum degree used for the monomial combinations of the tent functions.  Testing suggests we usually want this to be 1.  Increasing causes large increase in number of features. 
	    @param clfClass
	    	The choice of tool used for classification or regression, passed as the function.  This code has been tested using `sklearn` functions `RidgeClassiferCV` for classification and `RidgeCV` for regression. 

	    """
		self.description = description
		self.d = d
		self.delta = delta
		self.epsilon = epsilon

		self.maxPower = maxPower
		self.clfClass = clfClass

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
#	@param featureFunc
#		If we want to use a different feature function than the tent
#		function, it can be set here.  This function needs to accept
#		inputs in the same format as the `tent` function.
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
def build_Gm(Dgm,
			d = 10,
			delta = 1,
			epsilon = 0, 
			featureFunc = tent,
			returnSquare = False):

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
				entry = featureFunc(Dgm[dgmRow,0],Dgm[dgmRow,1], i,j,delta, epsilon)
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
#	@param featureFunc
#		If we want to use a different feature function than the tent
#		function, it can be set here.  This function needs to accept
#		inputs in the same format as the `tent` function.
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
def build_G(Dgms, 
			d = 10, 
			delta=1, 
			epsilon = 0,
			maxPower = 1,
			featureFunc = tent):

	M = len(Dgms)

	G = np.zeros((M,d**2+d))
	for m, Dgm in enumerate(Dgms):
		G[m,:] = build_Gm(Dgm,d,delta,epsilon,featureFunc = featureFunc)

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




#----------------------------------------------------#
#----------------------------------------------------#
#----------------------------------------------------#
#----------------------------------------------------#
#----------------------------------------------------#

## Main function to run ML with tents on persistence diagrams.
# Takes data frame DgmsDF and specified persistence diagram column labels, 
# computes the G matrix using build_G.
# Does classification using labels from labels_col in the data frame.
# Returns instance of estimator
# @warning this is not finished yet!
#
# 	@param DgmsDF
# 		A pandas data frame containing, at least, a column of 
# 		diagrams and a column of labels
# 	@param labels_col
# 		A string.  The label for the column in DgmsDF containing
# 		the training labels.
# 	@param dgm_col 
# 		The label(s) for the column containing the diagrams given as a string or list of strings.  
# 	@param params
# 		A class of type ParameterBucket
# 		Should store:
# 			- **d**:
# 				An integer, the number of elements for griding up 
# 				the x and y axis of the diagram.  Will result in 
# 				d*(d+1) tent functions 
# 			- **delta**, **epsilon**:
# 				Controls location and width of mesh elements for x and y axis of the 
# 				diagram. 
# 			- **clfClass**:
# 				The class which will be used for classification.  Currently tested
#				using `sklearn.RidgeClassifierCV` and `sklearn.RidgeCV`.
# @return 
# 	The classifier object. Coefficients can be found from clf.coef_
def TentML(DgmsDF,
			labels_col = 'trainingLabel',  
			dgm_col = 'Dgm1',
			params = None
			):
	#Choosing epsilon

	if params == None:
		print('You need to pass in a ParameterBucket. Exiting....')
		return 
	if params.d == None or params.delta == None or params.epsilon == None or params.maxPower == None:
		print('You need to finish filling the parameter bucket. Currently:')
		print('params.d = ', params.d)
		print('params.delta = ', params.delta)
		print('params.epsilon = ', params.epsilon)
		print('params.maxPower = ', params.maxPower)
		print('Exiting....')
		return

	clf = params.clfClass()
	# if params.classificationType == 'Ridge Classifier':
	# 	clf = RidgeClassifierCV() #(store_cv_values=True)
	# elif params.classificationType == 'Ridge Regression':
	# 	clf = RidgeCV()
	# elif params.classificationType == 'Lasso Classifier':
	# 	clf = LassoCV()
	# else:
	# 	print("You didn't specify a valid classification type.")
	# 	print("Currently, you have params.classificationType =", params.classificationType)
	# 	print("Exiting...")
	# 	return


	print('Training estimator.') # ' Type is ' + params.clfClass + '...')
	print('Max power is', params.maxPower,'...')
	# print('The column used for labels is: ' + labels_col + '...')
	startTime = time.time()

	#check to see if only one column label was passed. If so, turn it into a list.
	if type(dgm_col) == str:
		dgm_col = [dgm_col,]


	listOfG = []
	for dgmColLabel in dgm_col:
		G = build_G(DgmsDF[dgmColLabel],params.d,params.delta,params.epsilon,params.maxPower)
		listOfG.append(G)

	G = np.concatenate(listOfG,axis = 1)


	# Remove columns (features) that are entirely zero
	# if params.remove0cols:
	# 	numCols = np.shape(G)[1]
	# 	zeroCols = np.where(G.sum(0) == 0)[0]
	# 	nonzeroCols = list(  set(range(numCols)) - set(list(zeroCols)) )

	# 	G = G[:,nonzeroCols]
	# 	print('Removed features who evaluate entirely to 0...')
	# 	print('\tWent from', numCols, 'down to', np.shape(nonzeroCols)[0], 'features...')
	# else:
	# 	print('\tUsing', np.shape(G)[1], 'features...')

	clf.fit(G,list(DgmsDF[labels_col]))

	endTime = time.time()
	print('Trained estimator. Time taken is ' + printPrettyTime(endTime-startTime) + '.\n')
	# Get score on training set

	print('Checking score on training set...')
	score = clf.score(G,list(DgmsDF[labels_col]))
	print('Score on training set: ' + str(score) + '.\n')

	# clf.delta = params.delta
	# clf.epsilon = params.epsilon
	clf.trainingScore = score
	# clf.d = params.d
	# if params.remove0cols:
	# 	clf.remainingCols = nonzeroCols

	return clf


