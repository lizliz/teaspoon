## @package teaspoon.ML.Base
# Basic code used for doing ML on persistence diagrams
# by featurization.  Includes TODO
#


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
