## @package teaspoon.ML.tents
# Machine learning featurization method
# 
# If you make use of this code, please cite the following paper:<br/>
# J.A. Perea, E. Munch, and F. Khasawneh.  "Approximating Continuous Functions On Persistence Diagrams." Preprint, 2017.
#
# 
#


from teaspoon.Misc import printPrettyTime
import teaspoon.TDA.Persistence as pP

import time
import numpy as np
import pandas as pd 


from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, RidgeClassifierCV, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from scipy.special import comb
import itertools




## 	A class for storing all of the chosen parameters for a data frame and computation of ML.
#
class ParameterBucket(object):
	def __init__(self, description = '',
					d=10,
					delta = 1,
					epsilon = 0,
					maxPower = 1,
					clfClass = RidgeClassifierCV,
					seed = None,
					test_size = .33):
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
	    @param seed
	    	The seed for the pseudo-random number generator.  Pass None if you don't want it fixed; otherwise, pass an integer.
	    @param test_size
	    	A number in \f$[0,1]\f$.  Gives the percentage of data points to be reserved for the testing set if this is being used for a train/test split experiment.  Otherwise, ignored. 

	    """
		self.description = description
		self.d = d
		self.delta = delta
		self.epsilon = epsilon

		self.maxPower = maxPower
		self.clfClass = clfClass

		self.seed = seed

		self.test_size = test_size

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
		output += 'Variables in parameter bucket\n'
		output += '---\n'
		for key in attrs.keys():
			output += str(key) + ' : '
			output += str(attrs[key])+ '\n'
		output += '---\n'
		return output

	def encloseDgms(self, DgmSeries):
		'''!
		@brief Tests to see if the parameters enclose the persistence diagrams in the DgmSeries

		@returns boolean
		'''

		# Height of parallelogram; equivalently maximum lifetime enclosed
		height = self.d * self.delta + self.epsilon
		width = self.d * self.delta

		minBirth = pP.minBirthSeries(DgmSeries)
		if minBirth <0:
			print("This code assumes positive birth times.")
			return False

		maxBirth = pP.maxBirthSeries(DgmSeries)
		if maxBirth > width:
			print('There are birth times outside the bounding box.')
			return False

		minPers = pP.minPersistenceSeries(DgmSeries)
		if minPers < self.epsilon:
			print('There are points below the epsilon shift.')
			return False

		maxPers = pP.maxPersistenceSeries(DgmSeries)
		if maxPers > height:
			print('There are points above the box.')
			return False

		return True

	def chooseDeltaEpsWithPadding(self, DgmsSeries, pad = 0):
		'''
		
		DgmsSeries is pd.series
		d is number of grid elements in either direction
		pad is the additional padding outside of the points in the diagrams

		Sets the needed delta and epsilon 


		'''

		topPers = pP.maxPersistenceSeries(DgmsSeries)
		bottomPers = pP.minPersistenceSeries(DgmsSeries)
		topBirth = max(DgmsSeries.apply(pP.maxBirth))

		height = max(topPers,topBirth)

		bottomBirth = min(DgmsSeries.apply(pP.minBirth))
		if bottomBirth < 0:
			print('This code assumes that birth time is always positive\nbut you have negative birth times....')
			print('Your minimum birth time was', bottomBirth)

		epsilon = bottomPers/2

		delta = (height + pad - epsilon) / self.d

		self.delta = delta
		self.epsilon = epsilon



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
# 	@param Dgm
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
		print('\tCreating row ' + str(m) + ' out of ' + str(M) + '...', end ='\r' )
		G[m,:] = build_Gm(Dgm,d,delta,epsilon,featureFunc = featureFunc)
	print('')
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


## Vectorized (much faster!) version of the tents function
# @warning Returns features in different order than older versions of the code. The coefficients can be returned into square form using 
# ```row.reshape((d,d+1)).T ```
# @param Dgm
# 	A persistence diagram, given as a $K \times 2$ numpy array
# @param params
# 	An tents.ParameterBucket object.  Really, we need d, delta, epsilon, and maxPower from that.
# @param type
#	This code accepts diagrams either 
#	* in (birth, death) coordinates, in which case `type = 'BirthDeath'`, or 
#	* in (birth, lifetime) = (birth, death-birth) coordinates, in which case `type = 'BirthLifetime'`
# @return \f$\sum_{x,y \in \text{Dgm}}g_{i,j}(x,y)\f$ where
# \f[g_{i,j}(x,y) = 
# \bigg| 1- \max\left\{ \left|\frac{x}{\delta} - i\right|, \left|\frac{y-x}{\delta} - j\right|\right\} \bigg|_+\f]
# where
# \f$| * |_+\f$ is positive part; equivalently, min of \f$*\f$ and 0.
def tentVectorized(Dgm, params, type = 'BirthDeath'):
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

	Irepeated = I.repeat(Dgm.shape[0])
	Jrepeated = J.repeat(Dgm.shape[0])

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

	if params.maxPower >1:
		BigOuts = [out]
		for i in range(params.maxPower):
			BigOuts.append(np.outer(out,BigOuts[-1]).flatten())
		out = np.concatenate(BigOuts)
	return out 

def build_G_Vectorized(DgmSeries, params):
	applyTents = lambda x: tentVectorized(x,params = params)
	G = np.array(list(DgmSeries.apply(applyTents )))
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
		print('You need to finish filling the parameter bucket. ')
		print(params)
		# print('params.d = ', params.d)
		# print('params.delta = ', params.delta)
		# print('params.epsilon = ', params.epsilon)
		# print('params.maxPower = ', params.maxPower)
		print('Exiting....')
		return

	clf = params.clfClass()

	print('Training estimator.') # ' Type is ' + params.clfClass + '...')
	# print('The column used for labels is: ' + labels_col + '...')
	startTime = time.time()

	#check to see if only one column label was passed. If so, turn it into a list.
	if type(dgm_col) == str:
		dgm_col = [dgm_col,]
	
	for col in dgm_col:	
		print('Check that parameters enclose diagrams in', col,'col:', params.encloseDgms(DgmsDF[col])	)

	print('Making G...')

	listOfG = []
	for dgmColLabel in dgm_col:
		# G = build_G(DgmsDF[dgmColLabel],params.d,params.delta,params.epsilon,params.maxPower)
		G = build_G_Vectorized(DgmsDF[dgmColLabel],params)
		listOfG.append(G)

	G = np.concatenate(listOfG,axis = 1)

	numFeatures = np.shape(G)[1]
	print('Number of features used is', numFeatures,'...')

	clf.fit(G,list(DgmsDF[labels_col]))

	endTime = time.time()
	print('Trained estimator. Time taken is ' + printPrettyTime(endTime-startTime) + '.\n')
	# Get score on training set

	print('Checking score on training set...')
	score = clf.score(G,list(DgmsDF[labels_col]))
	print('Score on training set: ' + str(score) + '.\n')

	clf.delta = params.delta
	clf.epsilon = params.epsilon
	clf.trainingScore = score
	clf.d = params.d

	return clf




## Main testing function for classification or regression methods.  
# Does train/test split, creates classifier, and returns score on test.
#
# 	@param DgmsDF
# 		A pandas data frame containing, at least, a column of 
# 		diagrams and a column of labels
# 	@param labels_col
# 		A string.  The label for the column in DgmsDF containing the training labels.
# 	@param dgm_col 
# 		A string or list of strings giving the label for the column containing the diagrams.
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
#			- **seed**:
#				None if we don't want to mess with the seed for the train_test_split function. Else, pass integer.
#			- **test_split**: 
#				The percentage of the data to be reserved for the test part of the train/test split.
#
# 	@return
#		Returned as a dictionary of entries:
# 		- **score**
# 			The percent correct when predicting on the test set.
# 		- **DgmsDF**
# 			The original data frame passed back with a column labeled
# 			'Prediction' added with the predictions gotten for the
# 			test set. Data points in the training set will have an 
# 			entry of NaN
# 		- **clf**
# 			The classifier object.  
#
def getPercentScore(DgmsDF, 
					labels_col = 'trainingLabel',  
					dgm_col = 'Dgm1',
					params = ParameterBucket(),
					):

	print('---')
	print('Beginning experiment.')
	print(params)

	#check to see if only one column label was passed. If so, turn it into a list.
	if type(dgm_col) == str:
		dgm_col = [dgm_col,]

	# Run actual train/test experiment using sklearn
	D_train, D_test, L_train,L_test = train_test_split(DgmsDF,
													DgmsDF[labels_col],
													test_size=params.test_size,
													random_state = params.seed
													)

	#--------Training------------#
	print('Using ' + str(len(L_train)) + '/' + str(len(DgmsDF)) + ' to train...')
	clf = TentML(D_train,
					labels_col = labels_col,
					dgm_col = dgm_col,
					params = params)

	#--------Testing-------------#
	print('Using ' + str(len(L_test)) + '/' + str(len(DgmsDF)) + ' to test...')
	listOfG = []
	for dgmColLabel in dgm_col:
		G = build_G_Vectorized(D_test[dgmColLabel],params)
		listOfG.append(G)

	G = np.concatenate(listOfG,axis = 1)


	# Compute predictions and add to DgmsDF data frame
	L_predict = pd.Series(clf.predict(G),index = L_test.index)
	DgmsDF['Prediction'] = L_predict

	# Compute score
	score = clf.score(G,list(L_test))

	print('Score on testing set: ' + str(score) +"...\n")

	print('Finished with train/test experiment.')

	output = {}
	output['score'] = score
	output['DgmsDF'] = DgmsDF
	output['clf'] = clf

	return output




