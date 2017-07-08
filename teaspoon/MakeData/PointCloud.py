'''
Generates point clouds of different types.  

Currently goals are:
1) Circle
2) Annulus
3) Ellipse (Buggy)
4) Torus

All point clouds should be returned as a numpy array, 
'''

import random
import os
import numpy as np




#-------------Circles and Annuli---------------------------------------#
def Circle(N = 100, r=1, seed = None):
    """
    Generate N points in R^2 from the circle centered
    at the origin with radius r.

    Parameters
    ----------
    N -
        Number of points to generate
    r -
        Radius of the circle
    seed -
        Fixes the seed.  Good if we want to replicate results.


    Returns
    -------
    P -  
        A Nx2 numpy array with the points drawn as the rows.

    """
    P = []
    np.random.seed(seed)
    theta = np.random.rand(N,1)
    theta = theta.reshape((N,))
    P = np.zeros([N,2])
    P[:,0] = r*np.cos(2*np.pi*theta)
    P[:,1] = r*np.sin(2*np.pi*theta)

    return P    


def Sphere(N = 100, r = 1, noise = 0, seed = None):
    """
    Generate N points in R^3 from the sphere centered
    at the origin with radius r.
    If noise is set to a positive number, the points 
    can be at distance r +/- noise from the origin.

    Parameters
    ----------
    N -
        Number of points to generate
    r -
        Radius of the sphere
    seed -
        Fixes the seed.  Good if we want to replicate results.


    Returns
    -------
    P -  
        A Nx3 numpy array with the points drawn as the rows.

    """
    np.random.seed(seed)

    Rvect = 2*noise*np.random.random(N) + r
    thetaVect =   np.pi * np.random.random(N)
    phiVect = 2 * np.pi * np.random.random(N)

    P = np.zeros((N,3))
    P[:,0] = Rvect * np.sin(thetaVect) * np.cos(phiVect)
    P[:,1] = Rvect * np.sin(thetaVect) * np.sin(phiVect)
    P[:,2] = Rvect * np.cos(thetaVect)

    return P


def Annulus(N=200,r=1,R=2, seed = None):
    '''
    Returns point cloud sampled from uniform distribution on  
    annulus in R^2 of inner radius r and outer radius R

    Parameters
    ----------
    N -
        Number of points to generate
    r -
        Inner radius of the annulus
    R -
        Outer radius of the annulus
    seed -
        Fixes the seed.  Good if we want to replicate results.


    Returns
    -------
    P -  
        A Nx2 numpy array with the points drawn as the rows.

    '''
    np.random.seed(seed)
    P = np.random.uniform(-R,R,[2*N,2])
    S = P[:,0]**2 + P[:,1]**2
    P = P[np.logical_and(S>= r**2, S<= R**2)]
    #print np.shape(P)

    while P.shape[0]<N:
        Q = np.random.uniform(-R,R,[2*N,2])
        S = Q[:,0]**2 + Q[:,1]**2
        Q = Q[np.logical_and(S>= r**2, S<= R**2)]
        P = np.append(P,Q,0)
        #print np.shape(P)
    
    return P[:N,:]


#-------------Torus a la Diaconis paper--------------------------------#

def Torus(N = 100, r = 1,R = 2,  seed = None):
    '''
    Sampling method taken from Sampling from a Manifold by Diaconis, 
    Holmes and Shahshahani, arXiv:1206.6913

    Generates torus with points
    x = ( R + r*cos(theta) ) * cos(psi),  
    y = ( R + r*cos(theta) ) * sin(psi),
    z = r * sin(theta)

    Need to draw theta with distribution

    g(theta) = (1+ r*cos(theta)/R ) / (2pi) on 0 <= theta < 2pi

    and psi with uniform density on [0,2pi).

    For theta, draw theta uniformly from [0,2pi) and 
    eta from [1-r/R,1+r/R].  If eta< 1 + (r/R) cos(theta), return theta.
    
    Parameters
    ----------
    N -
        Number of points to generate
    r -
        Inner radius of the torus
    R -
        Outer radius of the torus
    seed -
        Fixes the seed.  Good if we want to replicate results.


    Returns
    -------
    P -  
        A Nx3 numpy array with the points drawn as the rows.

    ''' 

    np.random.seed(seed)
    psi = np.random.rand(N,1)
    psi = 2*np.pi*psi
    
    outputTheta = []
    while len(outputTheta)<N:
        theta = np.random.rand(2*N,1)
        theta = 2*np.pi*theta

        eta = np.random.rand(2*N,1)
        eta = eta / np.pi

        fx = (1+ r/float(R)*np.cos(theta)) / (2*np.pi)

        outputTheta = theta[eta<fx]


    theta = outputTheta[:N]
    theta = theta.reshape(N,1)


    x = ( R + r*np.cos(theta) ) * np.cos(psi)  
    y = ( R + r*np.cos(theta) ) * np.sin(psi)
    z = r * np.sin(theta)
    x = x.reshape((N,))
    y = y.reshape((N,))
    z = z.reshape((N,))

    P = np.zeros([N,3])
    P[:,0] = x
    P[:,1] = y
    P[:,2] = z

    return P




#----------------------------------------------------------------------#

def Cube(N = 100, diam = 1, dim = 2, seed = None):
    """
    Generate N points in R^dim from the box
    [0,diam]x[0,diam]x...x[0,diam]

    Parameters
    ----------
    N -
        Number of points to generate
    diam -
        Points are pulled from the box 
        [0,diam]x[0,diam]x...x[0,diam]
    dim -
        Points are embedded in R^dim

    Returns
    -------
    P -  
        A Nxdim numpy array with the points drawn as the rows.

    """
    np.random.seed(seed)

    P = diam*np.random.random((N,dim))

    return P.T



#----------------------------------------------------------------------#

def Clusters(N = 100, centers = np.array(((0,0),(3,3))),sd = 1, seed = None):
    """
    Generate k clusters of points, N points in total (evenly divided?)
    centers is a k x d numpy array, where centers[i,:] is the center of 
    the ith cluster in R^d.
    Points are drawn from a normal distribution with std dev = sd

    Parameters
    ----------
    N -
        Number of points to be generated
    centers -
        k x d numpy array, where centers[i,:] is the center of 
        the ith cluster in R^d.

    sd - 
        standard deviation of clusters.
        TODO: Make this enterable as a vector so each cluster can have
        a different sd?
    seed -
        Fixes the seed.  Good if we want to replicate results.

    Returns
    -------
    P -  
        A Nxd numpy array with the points drawn as the rows.

    """

    np.random.seed(seed)


    # Dimension for embedding
    d = np.shape(centers)[1]

    # Identity matrix for covariance
    I = np.eye(d)

    # Number of clusters
    k = np.shape(centers)[0]

    ptsPerCluster = N//k
    ptsForLastCluster = N//k + N%k

    for i in range(k):
        if i == k-1:
            newPts = np.random.multivariate_normal(centers[i,:], I, ptsForLastCluster)
        else:
            newPts = np.random.multivariate_normal(centers[i,:], I, ptsPerCluster)

        if i == 0:
            P = newPts[:]
        else:
            P = np.concatenate([P,newPts])

    return P



#----------------------------------------------------------------------#


