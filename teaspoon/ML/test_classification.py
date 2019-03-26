# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:43:11 2018

@author: khasawn3
"""

# this file tests the classification code
# cd to the teaspoon folder, load the package and import the needed modules
import numpy as np
#import teaspoon.MakeData.PointCloud as gpc
from teaspoon.MakeData import PointCloud as gpc
from Base import ParameterBucket, getPercentScore, InterpPolyParameters, TentParameters
import feature_functions as fF
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, RidgeClassifierCV



import pandas as pd
from teaspoon.TDA import Persistence as pP
#from teaspoon.TSP.adaptivePart import Partitions
#import teaspoon.TDA.Draw as Draw

# to reload a module
from imp import reload
reload(fF)
reload(gpc)


#----------------------------------
# define a parameters bucket
#----------------------------------
params = TentParameters()
#params = InterpPolyParameters()
params.useAdaptivePart = True


#----------------------------------
# generate a data frame for testing
#----------------------------------

#--- Test classification
# df = gpc.testSetClassification()
# dgmColLabel = 'Dgm'
# params.clfClass = RidgeClassifierCV



#--- Test regression
# df = gpc.testSetRegressionBall()
# dgmColLabel = 'Dgm'
# params.clfClass = RidgeCV


#--- Test manifold classification
df = gpc.testSetManifolds(numDgms = 100, numPts = 200) # numpTs=200
# dgmColLabel = ['Dgm0', 'Dgm1']
dgmColLabel = ['Dgm1']
params.clfClass = RidgeClassifierCV

# dgmColLabel = ['Dgm1']
# #params.clfClass = RidgeClassifierCV


# # get a diagram
# Dgms = df.iloc[0:5]

# Dgms = Dgms[dgmColLabel]


# try:
# 	AllDgms = []
# 	for label in Dgms.columns:
# 		DgmsSeries = Dgms[label]
# 		AllDgms.extend(list(DgmsSeries))
# 	AllPoints = np.concatenate(AllDgms)
# except:
# 	# you had a series to start with
# 	AllPoints = np.concatenate(list(DgmsPD))

# AllPoints = pP.removeInfiniteClasses(AllPoints)

# type = 'BirthDeath'

# x = AllPoints[:,0]
# y = AllPoints[:,1]
# if type == 'BirthDeath':
# 	life = y-x
# else:
# 	life = y
# fullData = np.column_stack((x,life))

#print('\n', params, '\n')

# get a diagram
#Dgm = df['Dgm'][0]

#----------------------------------
# Find bounding box
#----------------------------------
#params.setBoundingBox(df[dgmColLabel], pad = .05)
#params.jacobi_poly = 'cheb1'  # choose the interpolating polynomial

#----------------------------------
# define the number of base points
#----------------------------------
params.d = 4
#params.chooseDeltaEpsWithPadding()


#----------------------------------
# choose a feature function
#----------------------------------
#params.feature_function = fF.interp_polynomial
#params.feature_function = fF.tent


#----------------------------------
# Run the experiment
#----------------------------------
num_runs = 2
yy = np.zeros((num_runs))
for i in np.arange(num_runs):
	xx, Dgms_train, Dgms_test = getPercentScore(df,
					labels_col = 'trainingLabel',
					dgm_col = dgmColLabel,
					params = params,
					normalize = False,
					verbose = False
					)
	yy[i] = xx['score']

print('\navg success rate = {}\nStdev = {}'.format(np.mean(yy), np.std(yy)))

# kernel = stats.gaussian_kde(yy)
# values = np.linspace(yy.min(), yy.max(), 1000)
# plotting
#plt.plot(values, kernel(values), '.')
#plt.show()

# AllDgms_train = np.row_stack(Dgms_train.values)

# AllDgms_test = np.row_stack(Dgms_test.values)


# params.partitions.plot()
# plt.scatter(AllDgms_train[:,0], AllDgms_train[:,1] - AllDgms_train[:,0])
# plt.scatter(AllDgms_test[:,0], AllDgms_test[:,1] - AllDgms_test[:,0])


# plt.show()