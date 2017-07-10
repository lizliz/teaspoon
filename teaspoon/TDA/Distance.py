"""
Created on Jun 9, 2016

Has code for computing pairwise distance for diagrams.

@author: liz
"""

import numpy as np
import os
import subprocess
# import TSAwithTDA.pyPerseus as pyPerseus
# import TSAwithTDA.SlidingWindows as SW
from .Persistence import  prepareFolders

def dgmDist_Hera(D1,D2, wassDeg = 'Bottleneck', relError = None, internal_p = None):
    '''
    TODO: This documentation needs cleaning and updating.

    Computes the distance using the geom_bottleneck and geom_matching code
    from hera, found at https://bitbucket.org/grey_narn/hera

    Need to build the C++ code, then add bottleneck_dist folder and wasserstein_dist to path
    On Liz's machine, this involves adding the following lines to the ~/.bashrc
        export PATH="/home/liz/Dropbox/Math/Code/hera/geom_matching/wasserstein/build:$PATH"
        export PATH="/home/liz/Dropbox/Math/Code/hera/geom_bottleneck/build/example:$PATH"


    Default values for wasserstein_dist from Hera: 
    wasserstein_degree  = 1.0, 
    relative_error = 0.01, 
    internal_p = infinity.

    Valid values: 
    wasserstein_degree must be in $[1.0, \infinity)$, 
    relative_error must be positive,
    internal_p must be in $[1.0, \infinity]$ (to explicitly set internal_p to $\infinity$, supply inf).By default wasserstein degree is 1.0, relative error is 0.01, internal norm is l_infinity.

    hera input format:
    wasserstein_dist file1 file2  [wasserstein degree] [relative error] [internal norm] 


    Parameters
    ----------
    D1, D2
        Persistence diagrams given as Lx2 numpy arrays.
    wassDeg
        Options are:
        - 'Bottleneck'
        - np.inf (also returns bottleneck distance)
        - an integer p for computing the p-th Wasserstein distance

    Returns
    -------
    distance
        - a float for the distance between the diagrams

    '''

    prepareFolders()

    inputFile1 = '.teaspoonData/input/Dgm1.txt'
    inputFile2 = '.teaspoonData/input/Dgm2.txt'

    np.savetxt(inputFile1, D1, fmt = '%1.5f')
    np.savetxt(inputFile2, D2, fmt = '%1.5f')

    if wassDeg.lower() == 'bottleneck' or wassDeg == np.inf:
        cmd = 'bottleneck_dist ' + inputFile1 + ' ' + inputFile2 
    elif type(wassDeg) == str:
        print('Options for wassDeg are:\n-"bottleneck"\n-np.inf\n-A number in [1,\infty).')
        print('You passed something funky.  Exiting...')
        return
    elif wassDeg >= 1:
        cmd = 'wasserstein_dist ' + inputFile1 + ' ' + inputFile2
        cmd += ' ' + str(wassDeg)
    else:
        print('For computing the Wasserstein distance, wassDeg must be in [1,\infty)...')
        print('Exiting...')
        return

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell = True)
    # proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,shell = True)
    (out,err) = proc.communicate()
    out = out.decode().split('\n')
    # print(out)
    distance = out[-2]
    distance = float(distance)


    return distance