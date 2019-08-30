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

MakeData.PointCloud
________________________________
.. automodule:: teaspoon.MakeData.PointCloud
   :members: Circle, Sphere, Annulus, Torus, Cube, Clusters, normalDiagram, testSetClassification, testSetRegressionLine, testSetRegressionBall, testSetManifolds

ML.feature_functions
________________________________
.. automodule:: teaspoon.ML.feature_functions
   :members: tent, interp_polynomial

ML.Base
__________
.. automodule:: teaspoon.ML.Base
   :members: build_G, ML_via_featurization, getPercentScore

TDA.Distance
______________
.. automodule:: teaspoon.TDA.Distance
   :members: dgmDist_Hera

TDA.Draw
___________
.. automodule:: teaspoon.TDA.Draw
   :members: drawDgm, drawPtCloud

TDA.Persistence
________________
.. automodule:: teaspoon.TDA.Persistence
   :members: prepareFolders, readPerseusOutput, readRipserString, readRipserOutput, VR_Ripser, writePointCloudFileForPerseus, VR_Perseus, distMat_Ripser, distMat_Perseus, writeMatrixFileForPerseus, Cubical_Perseus, minPers, maxPers, maxBirth, minBirth, minPersistenceSeries, maxPersistenceSeries, minBirthSeries, maxBirthSeries, removeInfiniteClasses
