Packages
========


.. toctree::
   :caption: Packages:

.. autosummary::
   teaspoon.MakeData.PointCloud
   teaspoon.ML.feature_functions
   teaspoon.ML.Base
   teaspoon.TDA.Distance
   teaspoon.TDA.Draw
   teaspoon.TDA.Persistence

teaspoon.MakeData.PointCloud
____________________________
.. automodule:: teaspoon.MakeData.PointCloud
   :members: Circle, Sphere, Annulus, Torus, Cube, Clusters, normalDiagram, testSetClassification, testSetRegressionLine, testSetRegressionBall, testSetManifolds

teaspoon.ML.feature_functions
_____________________________
.. automodule:: teaspoon.ML.feature_functions
   :members: tent, sub2ind, ind2sub, quad_cheb1, quad_legendre, bary_weights, bary_diff_matrix, interp_polynomial

teaspoon.ML.Base
________________
.. automodule:: teaspoon.ML.Base
   :members: build_G, ML_via_featurization, getPercentScore

teaspoon.TDA.Distance
_____________________
.. automodule:: teaspoon.TDA.Distance
   :members: dgmDist_Hera

teaspoon.TDA.Draw
_________________
.. automodule:: teaspoon.TDA.Draw
   :members: drawDgm, drawPtCloud

teaspoon.TDA.Persistence
________________________
.. automodule:: teaspoon.TDA.Persistence
   :members: prepareFolders, readPerseusOutput, readRipserString, readRipserOutput, VR_Ripser, writePointCloudFileForPerseus, VR_Perseus, distMat_Ripser, distMat_Perseus, writeMatrixFileForPerseus, Cubical_Perseus, minPers, maxPers, maxBirth, minBirth, minPersistenceSeries, maxPersistenceSeries, minBirthSeries, maxBirthSeries, removeInfiniteClasses
