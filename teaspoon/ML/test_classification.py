# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:43:11 2018

@author: khasawn3
"""

# this file tests the classification code
# cd to the teaspoon folder, load the package and import the needed modules
import teaspoon.MakeData.PointCloud as gpc
from tents import ParameterBucket
import feature_functions as fF

# to reload a module
from imp import reload
reload(fF)
reload(gpc)

# define a parameters bucket
params = ParameterBucket()

# generate a data frame for testing
df = gpc.testSetClassification()

# get a diagram
Dgm = df['Dgm'][0]

# define the number of base points
params.d = 5
params.feature_function = fF.interp_polynomial
xx = fF.interp_polynomial(Dgm, params)




