"""
This is the main code for running ML code in teaspoon.

"""


# Here, we start with an instance of the `ParameterBucket` class. The intention of this
# object is to keep all determined parameters in one easy to use object. A new
# `ParameterBucket` subclass can be defined to inform any featurization method of interest.
# For instance, a simple example of using tent functions as defined in *Approximating
# Continuous Functions on Persistence Diagrams Using Template Functions* (Perea, Munch,
# Khasawneh 2018) is shown below.
#
# import teaspoon.ML.Base as Base
# import teaspoon.MakeData.PointCloud as gPC
# import teaspoon.ML.feature_functions as fF
# from sklearn.linear_model import RidgeClassifierCV
#
# params = Base.TentParameters(clf_model = RidgeClassifierCV,
# 							feature_function = fF.tent,
# 							test_size = .33,
# 							seed = 48824,
# 							d = 10,
# 							delta = 1,
# 							epsilon = 0
# 							)
#
# DgmsDF = gPC.testSetClassification(N = 20,
# 								  numDgms = 50,
# 								  muRed = (1,3),
# 								  muBlue = (2,5),
# 								  sd = 1,
# 								   seed = 48824
# 								  )
#
# out = Base.getPercentScore(DgmsDF,dgm_col = 'Dgm', labels_col = 'trainingLabel', params = params )


"""
.. module: Base
"""

from teaspoon.Misc import printPrettyTime
from teaspoon.ML import feature_functions as fF
from teaspoon.TDA import Persistence as pP
from teaspoon.TSP.adaptivePart import Partitions

import time
import numpy as np
import pandas as pd


from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, RidgeClassifierCV, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.preprocessing import scale, PolynomialFeatures
from scipy.special import comb
import itertools
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ParameterBucket(object):
	def __init__(self, description = '',
					clf_model = RidgeClassifierCV,
					feature_function = fF.tent,
					test_size = .33,
					seed = None,
					**kwargs):

		'''
		Creates a new ParameterBucket object.

		This object is being used to keep track of all the parameters needed
		for the tents ML featurization.

		Parameters:

			description (str):
				A description, has no effect on code. This can be set on initialization.

			clf_model (function):
				The choice of tool used for classification or regression, passed as
				the function.  This code has been tested using `sklearn` functions `
				RidgeClassiferCV` for classification and `RidgeCV` for regression.

			feature_function (function):
				The function you want to use for featurization.  This should be a
				function that takes as inputs a diagram and a ParameterBucket, and returns
				a vector of features. The default is ML.feature_functions.tents()

			test_size (float):
				A number in :math:`[0,1]`.  Gives the percentage of data points
				to be reserved for the testing set if this is being used for a train/test
				split experiment.  Otherwise, ignored.

			seed (int):
				The seed for the pseudo-random number generator.  Pass None if
				you don't want it fixed; otherwise, pass an integer.

			kwargs: Any leftover inputs are stored as attributes.

		'''

		self.description = description
		self.clf_model = clf_model
		self.seed = seed
		self.test_size = test_size
		self.feature_function = feature_function
		self.__dict__.update(kwargs)


	def __str__(self):
		'''
		Nicely prints all currently set values in the ParameterBucket.
		'''
		attrs = vars(self)
		output = ''
		output += 'Variables in parameter bucket\n'
		output += '---\n'
		for key in attrs.keys():
			output += str(key) + ' : '
			output += str(attrs[key])+ '\n'
		output += '---\n'
		return output


	def makeAdaptivePartition(self, DgmsPD, dgm_type = 'BirthDeath', meshingScheme = 'DV', partitionParams = {}):
		'''
		Combines all persistence diagrams in the series together, then generates an adaptive partition mesh and
		includes it in the parameter bucket as self.partitions.

		Note: If the data is passed in as a pd.Series then self.partitions is a dictionary with one entry with key
		'All' meaning it is the parititions for all persistence diagrams handed in. If the parameter **split** is True
		and the data is passed in as a pd.DataFrame, then it will partition the diagrams in each column separately.
		This is useful if you want to pass a column of 0-dim diagrams and 1-dim diagrams and partition them separately.
		In this case, self.partitions is a dictionary where the keys match the labels of the columns in the pd.DataFrame.

		Parameters:
			DgmsPD (pd.Series or pd.DataFrame):
				Structure of type pd.Series containing persistence diagrams or pd.DataFrame
				containing many columns of persistence diagrams.

			type (str):
				String specifying the type of persistence diagrams given,
				options are 'BirthDeath' or 'BirthLifetime'.

			meshingScheme (str):
				The type of meshing scheme. Options include 'DV' and 'clustering'.
				See Partitions class for explanation of these. If anything else is
				passed for this parameter, it will just use the bounding box of all
				points.

			partitionParams (dict):
				Dictionary with parameters for the partitioning algorithms. If an empty
				dictionary is passed, partitioning algorithms will just use the defaults.

		To Do:
			This can't handle infinite points in the diagram yet

		'''

		if hasattr(self, 'split'):
		    partitionParams['split'] = self.split
		else:
		    partitionParams['split'] = False

		self.partitions = {}

		# if meshingScheme == 'clustering':
		# 	if 'numClusters' in partitionParams:
		# 		if not isinstance(partitionParams['numClusters'], dict):
		# 			numClustersDict = {}
		# 			for col in DgmPD.columns:
		# 				numClustersDict.update({col: partitionParams['numClusters']})
		# 			partitionParams['numClusters'] = numClustersDict

		if self.split and isinstance(DgmsPD, pd.DataFrame):
			for col in DgmsPD.columns:
				AllPoints = np.concatenate(list(DgmsPD[col]))

				# Remove inifinite points
				AllPoints = pP.removeInfiniteClasses(AllPoints)

				x = AllPoints[:,0]
				y = AllPoints[:,1]
				if dgm_type == 'BirthDeath':
					life = y-x
				else:
					life = y
				fullData = np.column_stack((x,life))

				self.partitions.update({col: Partitions(data = fullData, meshingScheme =  meshingScheme, partitionParams = partitionParams)})

				if hasattr(self.partitions[col],'xFloats'):
					# create new attribute to keep the index of the floats for the partition bucket
					self.partitions[col].partitionBucketInd = deepcopy(self.partitions[col].partitionBucket)

					# convert nodes in partitions to floats in partition bucket
					for partition in self.partitions[col].partitionBucket:
						self.partitions[col].convertOrdToFloat(partition)
		else:

			try:
				AllDgms = []
				for label in DgmsPD.columns:
					DgmsSeries = DgmsPD[label]
					AllDgms.extend(list(DgmsSeries))
				AllPoints = np.concatenate(AllDgms)
			except:
				# you had a series to start with
				AllPoints = np.concatenate(list(DgmsPD))

			col = 'All'

			# Remove inifinite points
			AllPoints = pP.removeInfiniteClasses(AllPoints)

			x = AllPoints[:,0]
			y = AllPoints[:,1]
			if dgm_type == 'BirthDeath':
				life = y-x
			else:
				life = y
			fullData = np.column_stack((x,life))

			self.partitions.update({col: Partitions(data = fullData, meshingScheme = meshingScheme, partitionParams = partitionParams)})

			# If we used ordinals to begin with, save the ordinal partitions and
			# convert them back to floats
			if hasattr(self.partitions[col],'xFloats'):
				# create new attribute to keep the index of the floats for the partition bucket
				self.partitions[col].partitionBucketInd = deepcopy(self.partitions[col].partitionBucket)

				# convert nodes in partitions to floats in partition bucket
				for partition in self.partitions[col].partitionBucket:
					self.partitions[col].convertOrdToFloat(partition)


	def check_params(self):
		'''
		Function to check the parameters and make sure they are set up correctly to be able to run the code.

		This doesn't mean the code will run correctly, it just means there's nothing wrong with the parameters.
		'''
		d = self.d

		# If d is a dictionary, check to make sure it contains the proper keys
		# If split is False, it needs the key 'All'
		# If split is True, the keys need to match the entries in dgm_col
		# All other entries in the dictionary will be ignored
		if isinstance(d,dict):
			if self.split:
				dgm_col = self.dgm_col
			else:
				dgm_col = ['All']
			dgm_col.sort()
			d_keys = list(d.keys())
			d_keys.sort()
			if not set(dgm_col).issubset(set(d_keys)):
				print("Error: There's a problem with the dictionary used for parameter d.")
				print("d has keys: ", d_keys)
				print("dgm_col are: ", dgm_col)
				print("These need to match.")
				return

		# If you have specified partition parameters and are using the clustering algorithm
		# and numClusters is dictionary, check to make sure it contains the proper keys
		# If split is False, it needs the key 'All'
		# If split is True, the keys need to match the entries in dgm_col
		# All other entries in the dictionary will be ignored
		if hasattr(self, 'partitionParams'):
			partitionParams = self.partitionParams
			if 'numClusters' in partitionParams.keys():
				if isinstance(partitionParams['numClusters'], dict):
					if self.split:
						dgm_col = self.dgm_col
					else:
						dgm_col = ['All']
					dgm_col.sort()
					numCluster_keys = list(partitionParams['numClusters'].keys())
					numCluster_keys.sort()
					if not set(dgm_col).issubset(set(numCluster_keys)):
						print('Error: The dictionary you used for partition parameter')
						print('numClusters does not match the diagram dimensions used.')
						print('Fix these to match before continuing.')
						return

		print('Parameters checked.')


## A new type of parameter ParameterBucket
#
# This is the type specially built for Interpolating Polynomials
class InterpPolyParameters(ParameterBucket):

	def __init__(self, d = 3,
				useAdaptivePart = False,
				meshingScheme = 'DV',
				jacobi_poly = 'cheb1',
				clf_model = RidgeClassifierCV,
				test_size = .33,
				seed = None,
				maxPower = 1,
				split = False,
				**kwargs):
		'''
		Creates a new subclass of ParameterBucket specifically for the interpolating polynomials and sets all necessary parameters.

		This object is being used to keep track of all the parameters needed for the interpolating polynomial ML featurization.

		Parameters that are included in the ParameterBucket initially:

		Parameters:
			d (int): Number of mesh points in each direction.

			useAdaptivePart (bool):
				Determine whether you want to adaptively partition the
				persistence diagrams. By default it is set to False.

			meshingScheme (str):
				The type of meshing scheme. Options include 'DV' and 'clustering'.
				See Partitions class for explanation of these. If anything else is
				passed for this parameter, it will just use the bounding box of all
				points.

			jacobi_poly (str):
				The type of interpolating polynomial to use. Options are 'cheb1' and 'legendre'.

			clf_model (function):
				The choice of tool used for classification or regression, passed
				as the function.  This code has been tested using `sklearn`
				functions `RidgeClassiferCV` for classification and `RidgeCV` for regression.

			feature_function (function):
				The function you want to use for featurization.  This should be a
				function that takes as inputs a diagram and a ParameterBucket, and
				returns a vector of features. The default is ML.feature_functions.tents()

			test_size (float):
				A number in :math:`[0,1]`.  Gives the percentage of data points to be
				reserved for the testing set if this is being used for a train/test
				split experiment.  Otherwise, ignored.

			seed (int):
				The seed for the pseudo-random number generator.  Pass None if
				you don't want it fixed; otherwise, pass an integer.

			kwargs: Any leftover inputs are stored as attributes.

		'''


		self.feature_function = fF.interp_polynomial
		self.partitions = None
		self.jacobi_poly = jacobi_poly
		self.d = d
		self.useAdaptivePart = useAdaptivePart #This should be boolean
		self.meshingScheme = meshingScheme
		self.split = split
		self.clf_model = clf_model
		self.seed = seed
		self.test_size = test_size
		self.maxPower = maxPower
		self.__dict__.update(kwargs)




	def check(self):
		# Check for all the parameters required for tents function
		# TODO
		print("This hasn't been made yet. Ask me later.")
		pass

	def calcD(self, verbose=False):
		'''
		Sets delta and epsilon for tent function mesh.
		It also assigns d to each partition and adds it to the partition bucket as another dictionary element.
		Currently the only option is to use the same d for each partition.
		You can choose different number of divisions in the mesh for x and y directions.
		This works whether you are using adaptive partitions or not.

		Parameters:

			verbose (bool):
				If true will print additional messages and warnings.
		'''

		# choose delta to be the max of the width or the height of the partition divided by d
		# Note need to iterate over partitionBucket (not just Partitions class) so we can add dictionary elements
		for key in self.partitions:
			for partition in self.partitions[key].partitionBucket:
				# add or subtract padding if needed
				xmin = partition['nodes'][0] #- pad
				xmax = partition['nodes'][1] #+ pad
				ymin = partition['nodes'][2] #- pad
				ymax = partition['nodes'][3] #+ pad

				# d can be an integer meaning use same d in all directions,
				# or a list of d in each direction
				d = self.d
				if isinstance(d, list):
					dx = d[0]
					dy = d[1]
				elif isinstance(d, int):
					dx = d
					dy = d

				xdiff = xmax - xmin
				ydiff = ymax - ymin

				# calculate delta in each direction and choose the max
				deltax = xdiff / dx
				deltay = ydiff / dy
				# delta = max(deltax, deltay)

				if deltax == 0:
					deltax = np.inf
				if deltay == 0:
					deltay == np.inf
				if (deltax == 0) and (deltay == 0):
					print('Uh oh the partition consists of a single point...')
					print('Something is wrong with the paritioning scheme...')
					print('Exiting...')
					return

				delta = min(deltax, deltay)
				if deltax > deltay:
					delta = deltay

					dx = round(xdiff / delta)
				elif deltay > deltax:
					delta = deltax

					dy = round(ydiff / delta)
				else:
					delta = deltax

				# Assign d as an element in the dictionary for each partition
				d = [int(dx),int(dy)]
				partition['d'] = d



## A new type of parameter ParameterBucket
#
# This is the type specially built for tents
class TentParameters(ParameterBucket):

	def __init__(self, d = 10, delta = 1, epsilon = 0,
				useAdaptivePart = False,
				meshingScheme = 'DV',
				clf_model = RidgeClassifierCV,
				test_size = .33,
				seed = None,
				maxPower = 1,
				split = False,
				dgm_col = [],
				**kwargs):


		'''
		Creates a new ParameterBucket object.

		This object is being used to keep track of all the parameters needed
		for the tents ML featurization. Parameters describe initial attributes
		that start in the class, however throughout the use of the parameters,
		additional attributes will be added.

		Parameters:
			d (int):
				Initial starting value of dimensions of mesh for locations of tents.

			delta (float):
				Radius of all tents.

			epsilon (float):
				Shift up from y-axis (the diagonal in birth-death coordinates)
				to ensure tent supports do not touch y axis.

			clf_model (function):
				The choice of tool used for classification or regression, passed
				as the function.  This code has been tested using `sklearn`
				functions `RidgeClassiferCV` for classification and `RidgeCV`
				for regression.

			feature_function (function):
				The function you want to use for featurization.  This should be
				a function that takes as inputs a diagram and a ParameterBucket,
				and returns a vector of features. The default is
				ML.feature_functions.tents()

			test_size (float):
				A number in :math:`[0,1]`.  Gives the percentage of data points
				to be reserved for the testing set if this is being used for a
				train/test split experiment.  Otherwise, ignored.

			seed (int):
				The seed for the pseudo-random number generator.  Pass None if
				you don't want it fixed; otherwise, pass an integer.

			maxPower (int):
				The maximum degree used for the monomial combinations of the
				tent functions.  Testing suggests we usually want this to be 1.
				Increasing causes large increase in number of features.

			split (boolean):
				Boolean to decide if you want to partition different dimensional
				diagrams separately. If True, it partitions separately. If False
				it combines them all and then partitions.

			kwargs:
				Any leftover inputs are stored as attributes.

		'''

		## Old documentation:
		## The bounding box for the persistence diagram in the (birth, lifetime) coordinates is [0,d * delta] x [epsilon, d* delta + epsilon].  In the usual coordinates, this creates a parallelogram.

		# Set all the necessary parameters for tents function
		self.feature_function = fF.tent
		self.useAdaptivePart = useAdaptivePart #This should be boolean
		self.meshingScheme = meshingScheme
		self.split = split
		self.dgm_col = dgm_col

		self.d = d
		self.delta = delta
		self.epsilon = epsilon
		self.clf_model = clf_model
		self.seed = seed
		self.test_size = test_size
		self.maxPower = maxPower
		self.__dict__.update(kwargs)


	def check(self):

		print("This hasn't been made yet. Ask me later.")
		pass



	def chooseDeltaEpsForPartitions(self, pad=0, verbose=False):
		'''
		Sets delta and epsilon for tent function mesh.
		It also assigns d to each partition and adds it to the partition bucket as another dictionary element.
		Currently the only option is to use the same d for each partition.
		You can choose different number of divisions in the mesh for x and y directions.
		This works whether you are using adaptive partitions or not.

		Parameters:
			pad (int): The additional padding outside of the points in the bounding box/partition

			verbose (bool): If true will print additional messages and warnings.

		'''

		epsilon = self.epsilon
		if epsilon != 0:
			epsilon = 0
			self.epsilon = 0
			print("Sorry only option for epsilon is zero right now... This could be updated later...")
		if pad != 0:
			print("Sorry only option for pad is zero right now... This could be updated later...")
			pad = 0

		# choose delta to be the max of the width or the height of the partition divided by d
		# Note need to iterate over partitionBucket (not just Partitions class) so we can add dictionary elements
		for key in self.partitions:
			for partition in self.partitions[key].partitionBucket:
				# add or subtract padding if needed
				xmin = partition['nodes'][0] #- pad
				xmax = partition['nodes'][1] #+ pad
				ymin = partition['nodes'][2] #- pad
				ymax = partition['nodes'][3] #+ pad

				# d can be:
				#  - an integer meaning use same d in all directions,
				#  - a list of d in each direction
				#  - a dictionary with an integer/list for each key
				d = self.d
				if isinstance(d, dict):
					d = d[key]

				if isinstance(d, list):
					dx = d[0]
					dy = d[1]
				elif isinstance(d, int):
					dx = d
					dy = d
				else:
					print("There's a problem with the parameter d...")
					print("Exiting...")
					return


				xdiff = xmax - xmin
				ydiff = ymax - ymin


				# if dx == 0 and dy == 0:
				# 	delta = max(xdiff/2, ydiff/2)
				#
				# 	partition['delta'] = delta
				# 	partition['d'] = [int(dx), int(dy)]
				#
				# 	continue

				# calculate delta in each direction and choose the max
				deltax = xdiff / dx
				deltay = ydiff / dy
				# delta = max(deltax, deltay)

				if deltax == 0:
					deltax = np.inf
				if deltay == 0:
					deltay == np.inf
				if (deltax == 0) and (deltay == 0):
					print('Uh oh the partition consists of a single point...')
					print('Something is wrong with the paritioning scheme...')
					print('Exiting...')
					return

				delta = min(deltax, deltay)
				if deltax > deltay:
					delta = deltay

					dx = round(xdiff / delta)
				elif deltay > deltax:
					delta = deltax

					dy = round(ydiff / delta)
				else:
					delta = deltax

				# check if support will cross the diagonal
				# if it does, crop the bottom of the partition and recalculate d

				if partition['nodes'][2] - delta < 0:
					# partition['nodes'][2] =
					ymin = delta + epsilon

					if ( ( ymin + (dy * delta) ) > ( ymax + (0.5*delta) ) ) and (dy > 1):
						dy = dy-1

					if ( ( xmin + (dx * delta) ) > ( xmax + (0.55*delta) ) ) and (dx > 1):
						dx = dx-1

				# assign this delta to the partition
				partition['delta'] = delta

				# supportNodes contain the nodes of the bounding box for where tent functions are supported
				tempSuppNodes = [xmin - delta, xmin + ( (dx+1) * delta ), ymin - delta, ymin + ( (dy+1) * delta )]

				partition['supportNodes'] = tempSuppNodes

				# Assign d as an element in the dictionary for each partition
				d = [int(dx),int(dy)]
				partition['d'] = d

				# Just assigns epsilon based on what you want it to be
				# TO DO: could implement another method here to calculate it
				partition['epsilon'] = epsilon

		#print('\nParameters d, delta and epsilon have all been assigned to each partition...\n')


	def plotTentSupport(self, keys ,c = []):
		'''
		Plots the bounding box of the support of all the tent functions.

		Parameters:
			keys (list or str): List of the keys for the partitions you want tent support plotted for.

			c (list): List of colors you want to use in order *(optional)*
		'''

		if not c:
			c = ['r','b','g','m','y','k','c','orange','springgreen','darkred']

		cInd = 0
		if isinstance(keys,str):
			keys = [keys]

		for key in keys:
			# plot the partitions
			for binNode in self.partitions[key]:
				suppXmin = binNode['supportNodes'][0]
				suppXmax = binNode['supportNodes'][1]
				suppYmin = binNode['supportNodes'][2]
				suppYmax = binNode['supportNodes'][3]

				# plt.xlim([suppXmin - 1, suppXmax+1])
				# plt.ylim([suppYmin - 1, suppYmax+1])

				plt.hlines([suppYmin, suppYmax], suppXmin, suppXmax, color=c[cInd], linestyles='dashed')
				plt.vlines([suppXmin, suppXmax], suppYmin, suppYmax, color=c[cInd], linestyles='dashed')

				cInd = cInd+1



	def calcTentCenters(self, col):
		'''
		Calculates the points on the mesh where a tent function is centered. Mainly useful for debugging and
		making figures

		Returns:
			A list of :math:`(dx+1)*(dy+1)\\times 2` numpy arrays, one for each partition containing the centers of the mesh where tents can be centered

		'''

		tent_centers = []

		for partition in self.partitions[col]:
			partition_centers = []
			d = partition['d']
			if isinstance(d, int):
				d = list([d,d])
			delta = partition['delta']
			xmin = partition['supportNodes'][0] +delta
			ymin = partition['supportNodes'][2] +delta
			for i in range(d[0]+1):
				for j in range(d[1]+1):
					partition_centers.append( [xmin + (i*delta) , ymin + (j*delta)] )
			tent_centers.append(np.array(partition_centers))

		return tent_centers





#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#--------ML on diagrams using featurization ------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------



def build_G(DgmSeries, params, dgmColLabel):
	'''
	Applies the passed featurization function to all diagrams in the series and outputs the feature matrix

	Parameters:
		DgmSeries (pd.Series): A pd.Series holding the persistence diagrams. Must be in BirthDeath coordinates.

		params (ParameterBucket): A parameter bucket used for calculations.

		dgmColLabel (str): Label of the column of the dataframe (i.e. type of diagram) you are using.

	Returns:
		Feature matrix where each row corresponds to the featurization of a persistence diagram from the series.

	'''

	# if not hasattr(params,'partitions'):
	# 	print('You have to have a partition bucket set in the')
	# 	print('params.  I should probably tell you how to do')
	# 	print('this here.  Exiting...')
	# 	return

	if params.split == False:
		dgmColLabel = 'All'

	applyFeaturization = lambda x: params.feature_function(x,params = params, dgmColLabel = dgmColLabel)

	G = np.array(list(DgmSeries.apply(applyFeaturization )))
	# Include powers if necessary
	try:
		if params.maxPower>1:
			poly = PolynomialFeatures(params.maxPower)
			G = poly.fit_transform(G)
	except:
		pass
	return G


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#--------ML on diagrams using featurization ------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def ML_via_featurization(DgmsDF,
			labels_col = 'trainingLabel',
			dgm_col = 'Dgm1',
			params = TentParameters(),
			normalize = False,
			verbose = True
			):
	'''
	Main function to run ML with featurization on persistence diagrams.
	Takes data frame DgmsDF and specified persistence diagram column labels,
	computes the G matrix using build_G.
	Does classification using labels from labels_col in the data frame.
	Returns trained model.

	Parameters:
		DgmsDF (pd.DataFrame): A pandas data frame containing, at least, a column of diagrams and a column of labels. Diagrams must be in BirthDeath coordinates.

		labels_col (str): The label for the column in DgmsDF containing the training labels.

		dgm_col (str or list): The label(s) for the column containing the diagrams given as a string or list of strings.

		params (TentParameters or InterpPolyParameters):
			A class of type TentParameters (subclasses of class ParameterBucket). Should store:

				- **d**: An integer, the number of elements for griding up the x and y axis of the diagram.  Will result in (d+1)*(d+1) tent functions
				- **delta**, **epsilon**: Controls location and width of mesh elements for x and y axis of the diagram.
				- **clfClass**: The class which will be used for classification.  Currently tested using `sklearn.RidgeClassifierCV` and `sklearn.RidgeCV`.

	Returns:
		The classifier object. Coefficients can be found from clf.coef.

	'''

	clf = params.clf_model()

	if verbose:
		print('Training estimator.')
		startTime = time.time()

	#check to see if only one column label was passed. If so, turn it into a list.
	if type(dgm_col) == str:
		dgm_col = [dgm_col]

	if verbose:
		print('Making G...')

	numFeatures = {}
	nnzFeatures = {}
	listOfG = []
	for dgmColLabel in dgm_col:
		G = build_G(DgmsDF[dgmColLabel],params,dgmColLabel)
		listOfG.append(G)

		numFeatures[dgmColLabel] = np.shape(G)[1]
		nnzFeatures[dgmColLabel] = len(np.where(G.any(axis=0))[0])

	G = np.concatenate(listOfG,axis = 1)

	numFeatures['Total'] = np.shape(G)[1]
	nnzFeatures['Total'] = len(np.where(G.any(axis=0))[0])

	# Normalize G
	if normalize:
		G = scale(G)


	if verbose:
		print('Number of features: ', numFeatures,'...')
		print('Number of nonzero features: ', nnzFeatures )

	params.num_features = numFeatures
	params.nnz_features = nnzFeatures

	clf.fit(G,list(DgmsDF[labels_col]))

	if verbose:
		print('Checking score on training set...')

	score = clf.score(G,list(DgmsDF[labels_col]))

	if verbose:
		print('Score on training set: ' + str(score) + '.\n')

	clf.trainingScore = score


	return clf




#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#---------------Get percent score-----------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def getPercentScore(DgmsDF,
					labels_col = 'trainingLabel',
					dgm_col = 'Dgm1',
					params = TentParameters(),
					normalize = False,
					verbose = True,
					):
	'''
	Main testing function for classification or regression methods.
	Does train/test split, creates classifier, and returns score on test.

	Parameters:
		DgmsDF (pd.DataFrame): A pandas data frame containing, at least, a column of diagrams and a column of labels

		labels_col (str): A string.  The label for the column in DgmsDF containing the training labels.

		dgm_col (str or list): A string or list of strings giving the label for the column containing the diagrams.

		params(Parameter Bucket): A class of type ParameterBucket. Should store at least:

			- **featureFunction**: The function use for featurizing the persistence diagrams. Should take in a diagram and a ParameterBucket and output a vector of real numbers as features.
			- **clfClass**: The model which will be used for classification.  Currently tested using `sklearn.RidgeClassifierCV` and `sklearn.RidgeCV`.
			- **seed**: None if we don't want to mess with the seed for the train_test_split function. Else, pass integer.
			- **test_split**: The percentage of the data to be reserved for the test part of the train/test split.

	Returns (dict): Returned as a dictionary of entries:

		- **score**: The percent correct when predicting on the test set.
		- **DgmsDF**: The original data frame passed back with a column labeled 'Prediction' added with the predictions gotten for the test set. Data points in the training set will have an entry of NaN
		- **clf**: The fitted model
	'''


	if verbose:
		print('---')
		print('Beginning experiment.')
		print(params)


	#check to see if only one column label was passed. If so, turn it into a list.
	if type(dgm_col) == str:
		dgm_col = [dgm_col]

	# time1 = time.time()
	# Run actual train/test experiment using sklearn
	D_train, D_test, L_train,L_test = train_test_split(DgmsDF,
													DgmsDF[labels_col],
													test_size=params.test_size,
													random_state = params.seed
													)

	# time2 = time.time()
	# print('train test split time: ', time2 - time1)

	# Get the portions of the test data frame with diagrams and concatenate into giant series:
	if not params.split:
		allDgms = pd.concat((D_train[label] for label in dgm_col))
	else:
		allDgms = pd.concat((D_train[label] for label in dgm_col), axis =1, sort=False)
	# time3 = time.time()

	if params.useAdaptivePart == True:

		# if you passed parameters for the partitions, pass them through
		# else use the defaults
		if hasattr(params, 'partitionParams'):
			partitionParams = params.partitionParams
		else:
			partitionParams = {}

		params.makeAdaptivePartition(allDgms, meshingScheme = params.meshingScheme, partitionParams=partitionParams)

	# if not using adaptive partitions, just get the bounding box
	else:
		params.makeAdaptivePartition(allDgms, meshingScheme = None)

	# time4 = time.time()
	# print('adaptive partitioning time: ', time4 - time3)


	# If using tent functions, calculate delta parameter
	if params.feature_function.__name__ == 'tent':
		params.chooseDeltaEpsForPartitions(verbose=verbose)
	elif params.feature_function.__name__ == 'interp_polynomial':
	 	params.calcD(verbose=verbose)

	# time5 = time.time()
	# print('choose delta eps time: ', time5-time4)
	#--------Training------------#
	if verbose:
		print('Using ' + str(len(L_train)) + '/' + str(len(DgmsDF)) + ' to train...')
	clf = ML_via_featurization(D_train,
					labels_col = labels_col,
					dgm_col = dgm_col,
					params = params,
					normalize = normalize,
					verbose = verbose)

	# time6 = time.time()
	# print('training time: ', time6 - time5)

	#--------Testing-------------#
	if verbose:
		print('Using ' + str(len(L_test)) + '/' + str(len(DgmsDF)) + ' to test...')
	listOfG = []
	for dgmColLabel in dgm_col:
		G = build_G(D_test[dgmColLabel],params,dgmColLabel)
		listOfG.append(G)

	G = np.concatenate(listOfG,axis = 1)

	# Normalize G
	if normalize:
		G = scale(G)

	# Compute predictions and add to DgmsDF data frame
	L_predict = pd.Series(clf.predict(G),index = L_test.index)
	DgmsDF['Prediction'] = L_predict

	# Compute score
	score = clf.score(G,list(L_test))
	if verbose:
		print('Score on testing set: ' + str(score) +"...\n")

		print('Finished with train/test experiment.')

	# time7 = time.time()
	# print('testing time: ', time7 - time6)



	output = {}
	output['score'] = score
	output['DgmsDF'] = DgmsDF
	output['clf'] = clf

	return output, allDgms, D_test[dgmColLabel]
