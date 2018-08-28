# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 09:35:30 2018

@author: khasawn3
"""

import numpy as np
from scipy.stats import binned_statistic_2d, chisquare, rankdata
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# data: is a manyx2 numpy array that contains all the original data
# borders: a dictionary that contains 'nodes' with a numpyu array of Xmin, Xmax, Ymin, Ymax,
# and 'npts' which contains the number of points in the bin
# r: is the number of partitions
# alpha: the significance level to test for independence
def adaptive_partition_DV(data, borders, r=2, alpha=0.05):
    # extract the bin boundaries
    Xmin = borders['nodes'][0]
    Xmax = borders['nodes'][1]
    Ymin = borders['nodes'][2]
    Ymax = borders['nodes'][3]
    
    # find the number of bins
#    numBins = r ** 2  
    idx = np.where((data[:, 0] >= Xmin) 
                   & (data[:, 0] <= Xmax )
                   & (data[:, 1] >= Ymin)
                   & (data[:, 1] <= Ymax))
    
    # extract the points in the bin
    Xsub = data[idx, 0]
    Ysub = data[idx, 1]
    
#    print(Xsub.shape, '\t', Ysub.shape)
    
    # find the indices of the points in the x- and y-patches
    idx_x = np.where((data[:, 0] >= Xmin) & (data[:, 0] <= Xmax))
    idx_y = np.where((data[:, 1] >= Ymin) & (data[:, 1] <= Ymax))
    
    # get the subpartitions
    ai = np.floor(np.percentile(data[idx_x, 0], 1/r * np.arange(1, r) * 100))
    bj = np.floor(np.percentile(data[idx_y, 1], 1/r * np.arange(1, r) * 100))
    
    # get the bin edges
    edges1 = np.concatenate(([Xmin], ai, [Xmax]))
    edges2 = np.concatenate(([Ymin], bj, [Ymax]))
    
    # first exit criteria: we cannot split inot unique boundaries any more
    # preallocate the partition list
    partitions = []
    if (len(np.unique(edges1, return_counts=True)[1]) < r + 1 or
         len(np.unique(edges2, return_counts=True)[1])< r + 1):
        
        # reject futher partitions, and return original bin
        partitions.insert(0, {'nodes': np.array([Xmin, Xmax, Ymin, Ymax]), 
                  'npts': len(idx[0])})
        return partitions
    
    # figure out the shift in the edges so that boundaries do not overlap
    xShift = np.zeros( (2 * r, 2 * r))    
    yShift = xShift
    xShift[:, 1:-1] = np.tile(np.array([[-1, 0]]), (2 * r, r - 1))
    yShift = xShift.T
    
    # find the boundaries for each bin
    # duplicate inner nodes for x mesh
    dupMidNodesX = np.append(np.insert(np.repeat((edges1[1:-1]), 2, axis=0), 
                                      0, edges1[0]), edges1[-1])
    
    # duplicate inner nodes for y mesh
    dupMidNodesY = np.append(np.insert(np.repeat((edges2[1:-1]), 2, axis=0), 
                                      0, edges2[0]), edges2[-1])
    # reshape
    dupMidNodesY = np.reshape(dupMidNodesY, (-1, 1))
    
    # now find the nodes for each bin
    xBinBound = dupMidNodesX + xShift
    yBinBound = dupMidNodesY + yShift
    
    # find the number of points in each bin, and put this info into array
    binned_data = binned_statistic_2d(Xsub.flatten(), Ysub.flatten(), None, 'count', 
                                      bins=[edges1, edges2])
    # get the counts. Flatten columnwise to match the bin definition in the
    # loop that creates the dictionaries below
    binCounts = binned_data.statistic.flatten('F')
    
    # define an empty list to hold the dictionaries of the fresh partitions
    bins = []
    # create dictionaries for each bin
    # start with the loop over y
    # note how the loop counts were obtained above to match the convention
    # here
    for yInd in np.arange(r):
        # this is the loop over x
        for xInd in np.arange(r):
            # get the bin number
            binNo = yInd * r  + xInd
            xLow, xHigh = xBinBound[yInd, 2*xInd + np.arange(2)]
            yLow, yHigh = yBinBound[2*yInd + np.arange(2), xInd]
            bins.append({'nodes': np.array([xLow, xHigh, yLow, yHigh]), 
                'npts': binCounts[binNo] })
    
    # calculate the chi square statistic
    chi2 = chisquare(binCounts)    
  
    # check for independence and start recursion
    # if the chi2 test fails, do further partitioning:
    if (chi2.pvalue < alpha and Xmax!=Xmin and Ymax!=Ymin).all(): 
        for binInfo in bins:
            if binInfo['npts'] !=0:  # if the bin is not empty:
                # append entries to the tuple
                partitions.extend(adaptive_partition_DV(data=data, 
                                                        borders=binInfo,
                                                        r=r, alpha=alpha))
                              
    # Second exit criteria: 
    # if the partitions are independent, reject further partitioning and 
    # save the orignal, unpartitioned bin
    elif len(idx[0]) !=0:        
        partitions.insert(0, {'nodes': np.array([Xmin, Xmax, Ymin, Ymax]), 
                  'npts': len(idx[0])})
    return partitions
        
#--------------------------------------------------

# this part tests the adaptive meshing algorithm
if __name__ == "__main__":
    # generate a bivariate Gaussian
    
    # fix the seed For reproducibility
    np.random.seed(48824)
    
    # create a bivariate Gaussian
    mu = np.array([0, 0])  # the means
    cov = 0.7 # covariance
    sigma = np.array([[1, cov], [cov, 1]])  # covariance matrix
    
    # create the multivariate random variable
    nsamples = 2000  # number of random samples
    x, y = np.random.multivariate_normal(mu, sigma, nsamples).T

    
    # perform ordinal sampling (ranking) transformation
    xRanked = rankdata(x, method='ordinal')
    yRanked = rankdata(y, method='ordinal')
    
    # obtain the adaptive mesh
    numParts = 3
    
    # define bin0, which is the whole initial data set
    bin0 = {'nodes': np.array([1, xRanked.size, 1, yRanked.size]), 
            'npts': xRanked.size}
            
    # get the adaptive partition of the data
    partitions = adaptive_partition_DV(np.column_stack((xRanked, yRanked)),
                                       borders = bin0, 
                                       r=numParts)
    
    # plot the partitions
    fig1, ax1 = plt.subplots()
    for binNode in partitions:
        # get the bottom left corner
        corner = (binNode['nodes'][0], binNode['nodes'][2])
        
        # get the width and height
        width = binNode['nodes'][1] - binNode['nodes'][0]
        height = binNode['nodes'][3] - binNode['nodes'][2]
        
        # add the corresponding rectangle
        ax1.add_patch(patches.Rectangle(corner, width, height, fill=False)) 
        
    # overlay the data
    plt.plot(xRanked, yRanked, 'r*')
    
    # add formatting
    plt.axis('tight')    
    # show the figure
    plt.show()
    