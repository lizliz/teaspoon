Classes
=======

.. toctree::
   :caption: Classes:

teaspoon.ML.Base.ParameterBucket
________________________________
.. autoclass:: teaspoon.ML.Base.ParameterBucket
   :members: __init__, __str__, makeAdaptivePartition

teaspoon.ML.Base.InterpPolyParameters
______________________________________
.. autoclass:: teaspoon.ML.Base.InterpPolyParameters
   :members: __init__, calcD

teaspoon.ML.Base.TentParameters
________________________________
.. autoclass:: teaspoon.ML.Base.TentParameters
   :members: __init__, check_params, chooseDeltaEpsForPartitions, plotTentSupport, calcTentCenters


teaspoon.TSP.adaptivePart.Partitions
____________________________________
.. autoclass:: teaspoon.TSP.adaptivePart.Partitions
  :members: __init__, setParameters, convertOrdToFloat, getOrdinal, __iter__, iterOrdinal, __str__, plot, isOrdinal, return_partition_DV, return_partition_clustering
