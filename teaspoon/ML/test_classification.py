# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:43:11 2018

@author: khasawn3
"""

# this file tests the classification code
# cd to the teaspoon folder, load the package and import the needed modules
import numpy as np
import teaspoon.MakeData.PointCloud as gpc
from tents import ParameterBucket, getPercentScore
import feature_functions as fF

# to reload a module
from imp import reload
reload(fF)
reload(gpc)

# define a parameters bucket
params = ParameterBucket()

# generate a data frame for testing
df = gpc.testSetClassification()
#df = gpc.testSetRegressionBall()
#df = gpc.testSetManifolds()

# get a diagram
#Dgm = df['Dgm'][0]

# Find bounding box
params.findBoundingBox(df['Dgm'], pad = .05)
params.jacobi_poly = 'cheb1'  # choose the interpolating polynomial

# define the number of base points
params.d = 100
params.feature_function = fF.interp_polynomial
#params.feature_function = fF.tent

num_runs = 100
yy = np.zeros((num_runs))
for i in np.arange(num_runs):
	xx = getPercentScore(df, 
					labels_col = 'trainingLabel',  
					dgm_col = 'Dgm',
					params = params,
					normalize = True, 
					verbose = False
					)
	yy[i] = xx['score']

print('\navg success rate = {}'.format(np.mean(yy)))

