# @package teaspoon.TDA.TopologicalConsistency
"""
Topological Consistency via Kernel Estimation
==============================================

This module includes code for running the ideas from

`Omer Bobrowski, Sayan Mukherjee, and Jonathan E. Taylor, Topological consistency via kernel estimation. Bernoulli, Vol. 23, No. 1. Bernoulli Society for Mathematical Statistics and Probability. p. 288-328, 2017. <https://projecteuclid.org/journals/bernoulli/volume-23/issue-1/Topological-consistency-via-kernel-estimation/10.3150/15-BEJ744.full>`_

"""


import numpy as np
from ripser import ripser
from scipy.spatial import distance_matrix


def OmerApprox(func, L, eps, r, domain=[-2, 2, -2, 2], N=300):
    """
    This function takes an :math:`\mathbb{R}` function defined on the provided domain :math:`\subseteq \mathbb{R}^2`, and a level :math:`L` of interest. TODO FINISH DOCUMENTING

    Args:
       todo

    Returns:
       todo
    """

    # Randomly sample $N$ points from the domain, defined as the box [a1,a2] \times [b1,b2]
    a1, a2, b1, b2 = domain
    points = np.random.random((N, 2))

    points[:, 0] = points[:, 0]*(a2-a1) - (a2-a1)/2 + np.average((a1, a2))
    points[:, 1] = points[:, 1]*(b2-b1) - (b2-b1)/2 + np.average((b1, b2))

    # Make an array storing the value of the function applied to each random point
    vals = np.array([func(points[i, 0], points[i, 1]) for i in range(N)])

    # Helper code that extracts subsets of the point cloud

    def extractPoints(points, vals, valueLow, valueHigh=np.inf):
        # only keep the points in the (many x 2) list with associated vals entry, v, satisfying
        #        valueLow <= v <valueHigh
        pointsKept = points[np.where(
            (vals >= valueLow) & (vals < valueHigh))[0], :]
        return pointsKept

    # Extract two point clouds: One for points with func >= L+ eps,
    # the other with values between L-eps and L+eps.
    # The combination of these two point clouds is what is called $\chi_n^{L-\e}$ in the paper.
    pointsPlus = extractPoints(points, vals, L+eps)
    pointsMinus = extractPoints(points, vals, L-eps, L+eps)

    # Make a point cloud with the pointsPlus points first, then the pointsMinus
    numPointsInPlus = pointsPlus.shape[0]
    P = np.concatenate([pointsPlus, pointsMinus])

    # Compute the distance matrix for this collection of point clouds
    D = distance_matrix(P, P)

    # We will build a new matrix given by
    #           |  0 if i = j, so diagonal entries
    #  D[i,j] = |  1 if ||p_i - p_j|| <= r and p_i, p_j both in pointsPlus
    #           |  2 if ||p_i - p_j|| <= r and one of p_i or  p_j in pointsMinus
    #           |  4 if ||p_i - p_j|| > r
    D = D <= r
    D = D*2
    D[:numPointsInPlus, :numPointsInPlus] = D[:numPointsInPlus, :numPointsInPlus]/2
    D[np.where(D == 0)] = 4
    np.fill_diagonal(Dnew, 0)

    # Compute persistence on this matrix

    dgms = ripser(Dnew, distance_matrix=True)

    # We are interested in the 1-dimensional classes that are born at 1 and die entering 4
    # (Really, we just need them to live past 2)
    important = dgms['dgms'][1]
    important = important[np.where(important[:, 0] == 1)]
    important = important[np.where(important[:, 1] == 4)]

    return important.shape[0]
