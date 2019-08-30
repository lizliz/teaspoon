Classes
=======

.. toctree::
   :caption: Classes:

ParameterBucket
________________________________
.. autoclass:: teaspoon.ML.Base.ParameterBucket
   :members: __init__, __str__, makeAdaptivePartition

InterpPolyParameters
______________________________________
.. autoclass:: teaspoon.ML.Base.InterpPolyParameters
   :members: __init__, calcD

TentParameters
________________________________
.. autoclass:: teaspoon.ML.Base.TentParameters
   :members: __init__, check_params, chooseDeltaEpsForPartitions, plotTentSupport, calcTentCenters


Partitions
____________________________________
.. autoclass:: teaspoon.TSP.adaptivePart.Partitions
  :members: __init__, setParameters, convertOrdToFloat, __len__, __getitem__, getOrdinal, __iter__, iterOrdinal, __str__, plot, isOrdinal, return_partition_DV, return_partition_clustering
