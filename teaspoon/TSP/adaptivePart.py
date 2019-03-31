# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 09:35:30 2018

@author: khasawn3
"""

import numpy as np
from scipy.stats import binned_statistic_2d, chisquare, rankdata
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy

class Partitions:
    def __init__(self, data = None,
                 meshingScheme = None,
                 numParts=3,
                 alpha=0.05):
        '''
        A data structure for storing a partition coming from an adapative meshing scheme.
        
        :Parameter data:
            A numpy array of type many by 2
        
        :Parameter meshingScheme:
            The type of meshing scheme. Only option currently is 'DV', a method based on this paper (add paper). Any other input here will only use the bounding box of all points in the Dgms in the training set.     
        
        :Parameter numParts:
            Number of partitions in each direction

        :Parameter alpha:
            The significance level to test for independence

        TODO: Finish documentation
        '''

        if data is not None:
            # check that the data is in ordinal coordinates
            # data converted to ordinal and stored locally if not already
            if not self.isOrdinal(data):
                print("Converting the data to ordinal...")
                # perform ordinal sampling (ranking) transformation
                xRanked = rankdata(data[:,0], method='ordinal')
                yRanked = rankdata(data[:,1], method='ordinal')


                xFloats = np.copy(data[:,0])
                xFloats.sort()
                yFloats = np.copy(data[:,1])
                yFloats.sort()

                self.xFloats = xFloats
                self.yFloats = yFloats


                data = np.column_stack((xRanked,yRanked))

                # and return an empty partition bucket

            # If there is data, set the bounding box to be the max and min in the data


            xmin = data[:,0].min()
            xmax = data[:,0].max()
            ymin = data[:,1].min()
            ymax = data[:,1].max()

            # self.borders stores x and y min and max of overall bounding box in 'nodes' and the number of points in the bounding box in 'npts'
            self.borders = {}
            self.borders['nodes'] = np.array([xmin, xmax, ymin, ymax])
            self.borders['npts'] = data.shape[0]
            self.numParts = numParts
            self.alpha = alpha


            # If there is data, use the chosen meshing scheme to build the partitions.
            if meshingScheme == 'DV' and self.isOrdinal(data):
                # Figure out
                self.partitionBucket = self.return_partition_DV(data = data,
                                        borders = self.borders,
                                        r = self.numParts,
                                        alpha = self.alpha)
            else: # meshingScheme == None
            # Note that right now, this will just do the dumb thing for every other input
                self.partitionBucket = [self.borders]
                #  set the partitions to just be the bounding box


        else:
            self.partitionBucket = []


    def convertOrdToFloat(self,partitionEntry):
        '''
        Converts to nodes of a partition entry from ordinal back to floats. 

        :Parameter partitionEntry:
            The partition that you want to convert.

        :returns:
            Partition entry with converted nodes. Also sets dictionary element to the converted version.
        '''
        bdyList = partitionEntry['nodes'].copy()
        # Need to subtract one to deal with counting from
        # 0 vs counting from 1 problems
        xLow = int(bdyList[0])-1
        xHigh = int(bdyList[1])-1
        yLow = int(bdyList[2])-1
        yHigh = int(bdyList[3])-1


        if hasattr(self, 'xFloats'):
            xLowFloat = self.xFloats[xLow]
            xHighFloat= self.xFloats[xHigh]
            yLowFloat = self.yFloats[yLow]
            yHighFloat = self.yFloats[yHigh]
            convertedBdyList = [xLowFloat, xHighFloat, yLowFloat,yHighFloat]
            partitionEntry['nodes'] = convertedBdyList
            return partitionEntry
        else:
            print("You're trying to convert your ordinal data")
            print("back to floats, but you must have had ordinal")
            print("to begin with so I can't.  Exiting...")

    # magic method to get number of partitions
    def __len__(self):
        return len(self.partitionBucket)

    # magic method ot extract partitions
    def __getitem__(self,key):
        if hasattr(self,'xFloats'): #if the data wasn't ordinal
            entry = self.partitionBucket[key].copy()
            entry = self.convertOrdToFloat(entry)
            return entry
        else:
            return self.partitionBucket[key]


    def getOrdinal(self,key):
        '''
        Overrides the builtin magic method in the case where you had non-ordinal data but still want the ordinal stuff back.
        If the data wasn't ordinal, this has the exact same effect as self[key].
        '''

        return self.partitionBucket[key]

    def __iter__(self):
        '''
        Iterates over the converted entries in the parameter bucket
        '''

        # if hasattr(self,'xFloats'):
        #     return map( self.convertOrdToFloat, deepcopy(self.partitionBucket)  )
        # else:
        return iter(self.partitionBucket)

    def iterOrdinal(self):
        '''
        Functions just like iter magic method without converting each entry back to its float
        '''
        return iter(self.partitionBucketInd)

    def __str__(self):
        '''
        Nicely prints all currently set values in the bucket.
        '''
        attrs = vars(self)
        output = ''
        output += 'Variables in partition bucket\n'
        output += '---\n'
        for key in attrs.keys():
            output += str(key) + ' : '
            output += str(attrs[key])+ '\n'
            output += '---\n'
        return output


    def plot(self):
        '''
        Plot the partitions. Can plot in ordinal or float, whichever is in the partition bucket when it's called.
        '''

        fig1, ax1 = plt.subplots()
        for binNode in self:
            # print(binNode)
            # get the bottom left corner
            corner = (binNode['nodes'][0], binNode['nodes'][2])

            # get the width and height
            width = binNode['nodes'][1] - binNode['nodes'][0]
            height = binNode['nodes'][3] - binNode['nodes'][2]

            # add the corresponding rectangle
            ax1.add_patch(patches.Rectangle(corner, width, height, fill=False))

        # Doesn't show unless we do this
        plt.axis('tight')


    def isOrdinal(self, dd):
        '''
        Helper function for error checking. Used to make sure input is in ordinal coordinates. 
        It checks that when the two data columns are sorted they are each equal to an ordered vector with the same number of rows.
        '''
        return np.all(np.equal(np.sort(dd, axis=0),
                        np.reshape(np.repeat(np.arange(start=1,stop=dd.shape[0]+1),
                                             2), (dd.shape[0], 2))))





    def return_partition_DV(self, data, borders, r=2, alpha=0.05):
        '''
        Recursive method that partitions the data based on the DV method. 

        :Parameter data:
            A manyx2 numpy array that contains all the original data

        :Parameter borders:
            A dictionary that contains 'nodes' with a numpy array of Xmin, Xmax, Ymin, Ymax,
        
        :Parameter r:
            The number of partitions

        :Parameter alpha:
            The significance level to test for independence

        :returns:
            List of dictionaries. Each dictionary corresponds to a partition and contains 'nodes', a numpy array of Xmin, Xmax, Ymin, Ymax of the partition, and 'npts', the number of points in the partition.
        
        '''
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

        # first exit criteria: we cannot split into unique boundaries any more
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
                    partitions.extend(self.return_partition_DV(data=data,
                                                            borders=binInfo,
                                                            r=r, alpha=alpha))

        # Second exit criteria:
        # if the partitions are independent, reject further partitioning and
        # save the orignal, unpartitioned bin
        elif len(idx[0]) !=0:
            partitions.insert(0, {'nodes': np.array([Xmin, Xmax, Ymin, Ymax]),
                      'npts': len(idx[0])})

        return partitions

# #--------------------------------------------------

# # this part tests the adaptive meshing algorithm
# if __name__ == "__main__":
#     # generate a bivariate Gaussian

#     # fix the seed For reproducibility
#     np.random.seed(48824)

#     # create a bivariate Gaussian
#     mu = np.array([0, 0])  # the means
#     cov = 0.7 # covariance
#     sigma = np.array([[1, cov], [cov, 1]])  # covariance matrix

#     # create the multivariate random variable
#     nsamples = 2000  # number of random samples
#     x, y = np.random.multivariate_normal(mu, sigma, nsamples).T


#     # perform ordinal sampling (ranking) transformation
#     xRanked = rankdata(x, method='ordinal')
#     yRanked = rankdata(y, method='ordinal')

#     # obtain the adaptive mesh
#     numParts = 4

#     # get the adaptive partition of the data
#     partitionList = Partitions(np.column_stack((xRanked, yRanked)),
#                                        meshingScheme = "DV", numParts=numParts)

#     # plot the partitions
#     partitionList.plot()

#     # overlay the data
#     plt.plot(xRanked, yRanked, 'r*')

#     # add formatting
#     plt.axis('tight')
#     # show the figure
#     plt.show()
