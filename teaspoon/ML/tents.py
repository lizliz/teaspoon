## @package teaspoon.ML.tents
# Machine learning featurization method
#
# If you make use of this code, please cite the following paper:<br/>
# J.A. Perea, E. Munch, and F. Khasawneh.  "Approximating Continuous Functions On Persistence Diagrams." Preprint, 2017.
#
# An example workflow to ensure that classification is working:
# \code{.py}
# import teaspoon.MakeData.PointCloud as gPC
# import teaspoon.ML.tents as tents
# df = gPC.testSetClassification()
# tents.getPercentScore(df,dgm_col = 'Dgm')





from teaspoon.Misc import printPrettyTime
#import teaspoon.TDA.Persistence as pP
from teaspoon.TDA import Persistence as pP
#import teaspoon.ML.feature_functions as fF
from teaspoon.ML import feature_functions as fF

import time
import numpy as np
import pandas as pd


from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, RidgeClassifierCV, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.preprocessing import scale, PolynomialFeatures
from scipy.special import comb
import itertools







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



## Applies the tent function to all diagrams in the series and outputs the feature matrix
# \f$G_{a,b}\f$ is the \f$b^{th}\f$ tent function (after flattening the order) evaluated on the \f$a^{th}\f$ diagram in the series.
# @param DgmSeries : pd.Series
#	The structure holding the persistence diagrams.
# @param params : tents.ParameterBucket
# 	A parameter bucket used for calculations.
def build_G(DgmSeries, params):
    applyTents = lambda x: params.feature_function(x,params = params)
    G = np.array(list(DgmSeries.apply(applyTents )))

    # Include powers if necessary
    if params.maxPower>1:
        poly = PolynomialFeatures(params.maxPower)
        G = poly.fit_transform(G)
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
			params = None,
			normalize = False,
			verbose = True
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

    if verbose:
        print('Training estimator.')

#    startTime = time.time()

    #check to see if only one column label was passed. If so, turn it into a list.
    if type(dgm_col) == str:
        dgm_col = [dgm_col]

    if verbose:
        print('Making G...')

    listOfG = []
    for dgmColLabel in dgm_col:
        G = build_G(DgmsDF[dgmColLabel],params)
        listOfG.append(G)

    G = np.concatenate(listOfG,axis = 1)

    numFeatures = np.shape(G)[1]

    # Normalize G
    if normalize:
        G = scale(G)


    if verbose:
        print('Number of features used is', numFeatures,'...')

    clf.fit(G,list(DgmsDF[labels_col]))

    if verbose:
        print('Checking score on training set...')

    score = clf.score(G,list(DgmsDF[labels_col]))

    if verbose:
        print('Score on training set: ' + str(score) + '.\n')

    clf.delta = params.delta
    clf.epsilon = params.epsilon
    clf.trainingScore = score
    clf.d = params.d

    return clf
