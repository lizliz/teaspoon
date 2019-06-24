
'''
Machine learning featurization method

If you make use of this code, please cite the following paper:

J.A. Perea, E. Munch, and F. Khasawneh.  "Approximating Continuous Functions On Persistence Diagrams." Preprint, 2017.
'''


import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------- #
# -------------------------------------------- #
# ----------Featurization--------------------- #
# -------------------------------------------- #
# -------------------------------------------- #

"""
.. module::feature_functions
"""

def tent(Dgm, params, dgmColLabel, dgm_type='BirthDeath'):
    '''
    Applies the tent function to a diagram.

    Parameters:
        Dgm (np.array):	A persistence diagram, given as a :math:`K \times 2` numpy array

        params (ParameterBucket):
            A Base.ParameterBucket object. Really, we need d, delta, epsilon, and the partitions from that.

        type (str): This code accepts diagrams either

            1. in (birth, death) coordinates, in which case `type = 'BirthDeath'`, or
            2. in (birth, lifetime) = (birth, death-birth) coordinates, in which case `type = 'BirthLifetime'`

    Returns:

        :math:`\sum_{x,y \in \text{Dgm}}g_{i,j}(x,y)` where

        .. math::

            g_{i,j}(x,y) = \\bigg| 1- \\max\\left\\{ \\left|\\frac{x}{\\delta} - i\\right|, \\left|\\frac{y-x}{\\delta} - j\\right|\\right\\} \\bigg|_+]

        where :math:`| * |_+` is positive part; equivalently, min of * and 0.

    .. note:: This code does not take care of the maxPower polynomial stuff.  The build_G() function does it after all the rows have been calculated.

    '''


    # delta = params.delta
    #epsilon = params.epsilon
    # print(Dgm[:3])
    # Move to birth,lifetime plane
    if dgm_type == 'BirthDeath':
        T = np.array(((1, -1), (0, 1)))
        A = np.dot(Dgm, T)

    elif dgm_type == 'BirthLifetime':

        A = Dgm
    else:
        print('Your choices for type are "BirthDeath" or "BirthLifetime".')
        print('Exiting...')
        return

    all_out = []

    # first, get the entries in Dgm that are within each partition
    for partition in params.partitions[dgmColLabel]:

        if isinstance(partition['d'], list):
            dx = partition['d'][0]
            dy = partition['d'][1]
        elif isinstance(partition['d'],int):
            dx = partition['d']
            dy = partition['d']
        else:
            print('Parameter d needs to be an int or a list of 2 ints (one for x direction and one for y direction.) ')
            print('Exiting...')
            return

        # Get delta and epsilon from partition bucket
        delta = partition['delta']
        epsilon = partition['epsilon']

        # get subset of points in the diagram that are in the
        # support of all tents in the partition
        Supp_xmin, Supp_xmax, Supp_ymin, Supp_ymax = partition['supportNodes']
        Asub = A[(A[:, 0] >= Supp_xmin) &
                (A[:, 0] <= Supp_xmax) &
                (A[:, 1] >= Supp_ymin) &
                (A[:, 1] <= Supp_ymax), :]

        # if there are no dgm points in the partition just add a vector of zeros and move on to the next partition
        if len(Asub) == 0:
            all_out = np.concatenate((all_out, np.zeros((dx + 1) * (dy + 1))), axis=0)
            continue

        # get bounding box of the tent tent centers
        # ie. the mesh of tent centers stays within this range, including on the boundary
        # ie. tent centers are at (xmin, ymin), (xmin + delta, ymin), ... , (xmin + dx*delta, ymin)=(xmax,ymin)
        #                         (xmin, ymin + delta), ... , (xmin + dx*delta, ymin + delta)
        #                         ...
        #                         (xmin, ymin + dy*delta)=(xmin,ymax), ... , (xmin + dx*delta, ymin+dy*delta)=(xmax,ymax)
        # BUT support extends outside that by delta on all sides (unless ymin is zero)
        xmin = Supp_xmin + delta #+ epsilon
        xmax = Supp_xmax - delta
        ymin = Supp_ymin + delta
        ymax = Supp_ymax - delta

        I,J = np.meshgrid(range(dx + 1), range(dy + 1))

        # Iflat = delta * I.reshape(np.prod(I.shape))
        # Jflat = delta * J.reshape(np.prod(I.shape)) + epsilon

        Iflat = xmin + (delta * I.reshape(np.prod(I.shape)))
        Jflat = ymin + (delta * J.reshape(np.prod(J.shape))) + epsilon

        Irepeated = Iflat.repeat(Asub.shape[0])
        Jrepeated = Jflat.repeat(Asub.shape[0])

        DgmRepeated = np.tile(Asub, (len(Iflat), 1))

        BigIJ = np.array((Irepeated, Jrepeated)).T

        B = DgmRepeated - BigIJ
        B = np.abs(B)
        B = np.max(B, axis=1)
        B = delta - B
        B = np.where(B >= 0, B, 0)
        B = B.reshape((Iflat.shape[0], Asub.shape[0]))
        out = np.sum(B, axis=1)

        out = out.reshape((dx + 1, dy + 1)).T.flatten()
        out = out / delta

        if np.count_nonzero(out) == 0 and np.count_nonzero(Asub[:,1]) != 0 and ( not any(np.array(partition['supportNodes']) == Asub) ):
            print('All zero but it shouldnt be...')
            print('Dgm: ', Asub)
            print('Nodes: ', partition['nodes'])
            print('Support Nodes: ', partition['supportNodes'])
            print('Delta: ', delta)
            print(' ')

        all_out = np.concatenate((all_out, out), axis=0)


    return all_out

def sub2ind(array_shape, rows, cols):
    '''
    Converts subscripts to indices
    '''
    ind = rows * array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    return ind


# convert indices to subscripts
def ind2sub(array_shape, ind):
    '''
    Converts indices to subscripts
    '''
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return rows, cols


# find the quadrature/interpolation weights for common orthognal functions
# define the function blocks
# Chebyshev-Gauss points of the first kind
def quad_cheb1(npts=10):
    '''
    Find the quadrature/interpolation weights for common orthognal functions

    Define the function blocks

    Chebyshev-Gauss points of the first kind
    '''
    # get the Chebyshev nodes of the first kind
    x = np.cos(np.pi * np.arange(0, npts + 1) / npts)

    # get the corresponding quadrature weights
    w = np.pi / (1 + np.repeat(npts, npts + 1))

    return x, w


# Computes the Legendre-Gauss-Lobatto nodes, and their quadrate weights.
# The LGL nodes are the zeros of (1-x**2)*P'_N(x), wher P_N is the nth Legendre
# polynomial
def quad_legendre(npts=10):
    '''
    Computes the Legendre-Gauss-Lobatto nodes, and their quadrate weights.

    The LGL nodes are the zeros of (1-x**2)*P'_N(x), wher P_N is the nth Legendre polynomial
    '''
    # Truncation + 1
    nptsPlusOne = npts + 1
    eps = np.spacing(1)  # find epsilon, the machine precision

    # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
    x = np.cos(np.pi * np.arange(0, npts + 1) / npts)

    # Get the Legendre Vandermonde Matrix
    P = np.zeros((nptsPlusOne, nptsPlusOne))

    # Compute P_(N) using the recursion relation
    # Compute its first and second derivatives and
    # update x using the Newton-Raphson method.

    xold = 2

    while np.max(np.abs(x - xold)) > eps:
        xold = x

        P[:, 0] = 1
        P[:, 1] = x

        for k in range(1, npts):
            P[:, k + 1] = ((2 * k + 1) * x * P[:, k] - k * P[:, k - 1]) / (k + 1)

        x = xold - (x * P[:, npts] - P[:, npts - 1]) \
            / (nptsPlusOne * P[:, npts])

    # get the corresponding quadrature weights
    w = 2 / (npts * nptsPlusOne * P[:, npts] ** 2)

    return x, w

'''
Map the inputs to the function block.
You can invoke the desired function block using:
quad_pts_and_weights['name_of_function'](npts)
'''
quad_pts_and_weights = {'cheb1': quad_cheb1,'legendre': quad_legendre}

# map the inputs to the function blocks
# you can invoke the desired function block using:
# quad_pts_and_weights['name_of_function'](npts)
quad_pts_and_weights = {'cheb1': quad_cheb1,
                        'legendre': quad_legendre
                        }


# find the barycentric interplation weights
def bary_weights(x):
    '''
    Find the barycentric interplation weights

    Parameters:
        x: Basepoints for Barycentric weights

    .. note:: this algorithm may be numerically unstable for high degree

    Polynomials (e.g. 15+). If using linear or Chebyshev spacing of base
    points then explicit formulae should then be used. See Berrut and
    Trefethen, SIAM Review, 2004.
    '''

    # find the length of x
    n = x.size

    # Rescale for numerical stability
    eps = np.spacing(1)  # find epsilon, the machine precision
    x = x * (1 / np.prod(x[x > -1 + eps] + 1)) ** (1 / n)

    # find the weights
    w = np.zeros((1, n))
    for i in np.arange(n):
        w[0, i] = np.prod(x[i] - np.delete(x, i))

    return 1 / w


def bary_diff_matrix(xnew, xbase, w=None):
    '''
    Calculate both the derivative and plain Lagrange interpolation matrix
    using Barycentric formulae from Berrut and Trefethen, SIAM Review, 2004.

    Parameters:
        xnew: Interpolation points
        xbase: Base points for interpolation
        w: Weights calculated for base points (optional)

    '''

    # if w is not set, set it

    if w is None:
        w = bary_weights(xbase)

    # get the length of the base points
    n = xbase.size

    # get the length of the requested points
    nn = xnew.size

    # replicate the weights vector into a matrix
    wex = np.tile(w, (nn, 1))

    # Barycentric Lagrange interpolation (from Berrut and Trefethen, SIAM Review, 2004)
    xdiff = np.tile(xnew[np.newaxis].T, (1, n)) - np.tile(xbase, (nn, 1))

    M = wex / xdiff

    divisor = np.tile(np.sum(M, axis=1)[np.newaxis].T, (1, n))
    divisor[np.isinf(divisor)] = np.inf

    M[np.isinf(M)] = np.inf

    M = M / divisor




    #	M[np.isnan(M)] = 0

    M[xdiff == 0] = 1

    #	# Construct the derivative (Section 9.3 of Berrut and Trefethen)
    #	xdiff2 = xdiff ** 2
    #
    #	frac1 = wex / xdiff
    #	frac1[np.isinf(frac1)] = float("inf")
    #
    #	frac2 = wex / xdiff2
    #
    #	DM = (M * np.tile(np.sum(frac2, axis=1)[np.newaxis].T, (1, n)) - frac2) / np.tile(np.sum(frac1, axis=1)[np.newaxis].T, (1, n))
    #	row, col = np.where(xdiff == 0)
    #
    #
    #	if np.all(row == 0):  # or, row.size == 0:
    #		DM[row, ] = (wex[row, ] / np.tile(w[col].T, (1, n))) / xdiff[row, ]
    #		idx = sub2ind(DM.shape, row, col)
    #		DM[idx] = 0
    #		DM[idx] = -np.sum(DM[row, ], axis=1)

    return M


def interp_polynomial(Dgm, params, dgmColLabel, dgm_type='BirthDeath'):
    '''
    Extracts the weights on the interpolation mesh using barycentric Lagrange interpolation.

    Parameters:
        Dgm (np.array):
            A persistence diagram, given as a :math:`K \times 2` numpy array
        params (ParameterBucket):
            A Base.ParameterBucket object.  Really, we need d and the partitions from that.
        type (str): This code accepts diagrams either

        	1. in (birth, death) coordinates, in which case `type = 'BirthDeath'`, or
        	2. in (birth, lifetime) = (birth, death-birth) coordinates, in which case `dgm_type = 'BirthLifetime'`

    Returns:
        **interp_weight** *(np.array)* - a matrix with each entry representiting the weight of an interpolation
        function on the base mesh. This matrix assumes that on a 2D mesh the functions
        are ordered row-wise.

    '''
    #	jacobi_func = params.jacobi_func

    # # check if we asked for a square mesh or not
    # if isinstance(params.d, int):
    #     nx = params.d
    #     ny = params.d
    # else:
    #     nx, ny = params.d
    # check if the Dgm is empty. If it is, pass back zeros
    # if Dgm.size == 0:
    #     return  np.zeros((nx + 1) * (ny + 1))

    # Move to birth,lifetime plane
    if dgm_type == 'BirthDeath':
        T = np.array(((1, -1), (0, 1)))
        A = np.dot(Dgm, T)
    elif dgm_type == 'BirthLifetime':
        A = Dgm
    else:
        print('Your choices for type are "BirthDeath" or "BirthLifetime".')
        print('Exiting...')

    all_weights = []
    # first, get the entries in Dgm that are within each partition
    for partition in params.partitions[dgmColLabel]:

        ##### Might want this back at some point...
        ##### Use the same d for every partition
        # # check if we asked for a square mesh or not
        # if isinstance(params.d, int):
        #     nx = params.d
        #     ny = params.d
        # else:
        #     nx, ny = params.d

        # use different d depending on the partititon
        if isinstance(partition['d'], list):
            nx = partition['d'][0]
            ny = partition['d'][1]
        elif isinstance(partition['d'],int):
            nx = partition['d']
            ny = partition['d']
        else:
            print('Parameter d needs to be an int or a list of 2 ints (one for x direction and one for y direction.) ')
            print('Exiting...')
            return

        query_Dgm_pts = getSubset(A, partition)

        # get the number of query points
        num_query_pts = len(query_Dgm_pts)

        # check if the intersection of the Dgm and the partition is empty.
        # If it is, pass back zeros
        if num_query_pts == 0:
            all_weights = np.concatenate((all_weights, np.zeros((nx + 1) * (ny +1))), axis=0)
            continue
            #return np.zeros((nx + 1) * (ny + 1))

        # get the query points. xq are the brith times, yq are the death times.
        xq, yq = query_Dgm_pts[:, 0], query_Dgm_pts[:, 1]
        # xq, yq = np.sort(query_Dgm_pts[:, 0]), np.sort(query_Dgm_pts[:, 1])

        # 1) Get the base nodes:
        # get the 1D base nodes in x and y

        xmesh, w = quad_pts_and_weights[params.jacobi_poly](nx)
        ymesh, w = quad_pts_and_weights[params.jacobi_poly](ny)
        xmesh = np.sort(xmesh)
        ymesh = np.sort(ymesh)

        # shift the base mesh points to the interval of interpolation [ax, bx], and
        # [ay, by]
        ax, bx, ay, by = partition['nodes']

        xmesh = (bx - ax) / 2 * xmesh + (bx + ax) / 2

        ymesh = (by - ay) / 2 * ymesh + (by + ay) / 2

        # define a mesh on the base points
        x_base, y_base = np.meshgrid(xmesh, ymesh, sparse=False, indexing='ij')

        # get the x and y interpolation matrices
        # get the 1D interpolation matrix for x
        x_interp_mat = bary_diff_matrix(xnew=xq, xbase=xmesh)
        x_interp_mat = x_interp_mat.T  # transpose the x-interplation matrix

        # get the 1D interpolation matrix for y
        y_interp_mat = bary_diff_matrix(xnew=yq, xbase=ymesh)

        # replicate each column in the x-interpolation matrix n times
        Gamma = np.repeat(x_interp_mat, ny + 1, axis=1)

        # unravel, then replicate each row in the y-interpolation matrix m times
        y_interp_mat.shape = (1, y_interp_mat.size)
        Phi = np.repeat(y_interp_mat, nx + 1, axis=0)

        # element-wise multiply Gamma and Phi
        Psi = Gamma * Phi

        # split column-wise, then concatenate row-wise
        # if Psi.size > 0:  # check that Psi is not empty
        Psi = np.concatenate(np.split(Psi, num_query_pts, axis=1), axis=0)

        # now reshape Psi so that each row corresponds to the weights of one query pt
        Psi = np.reshape(Psi, (num_query_pts, -1))

        # get the weights for each interpolation function/base-point
        interp_weights = np.sum(np.abs(Psi), axis=0)

        # plt.figure(10)
        # plt.plot(np.abs(all_weights),'x')
        # plt.show()

        all_weights = np.concatenate((all_weights, np.abs(interp_weights)), axis=0)
        #return np.abs(interp_weights)

    return all_weights


def getSubset(querySet, baseRectangle):
    '''
    Helper function that gets the subset of points that are contained within a certain region.

    :Parameter querySet:
        Set of all points

    :Parameter baseRectange:
        Dictionary containing key 'nodes' which is a list of [xmin, xmax, ymin, ymax].

    :returns:
        Many by 2 numpy array of the query points within the rectangle.
    '''

    # get the rectangle corners
    xmin = baseRectangle['nodes'][0];
    xmax = baseRectangle['nodes'][1];
    ymin = baseRectangle['nodes'][2];
    ymax = baseRectangle['nodes'][3];

    # subset
    return querySet[(querySet[:, 0] >= xmin) &
                    (querySet[:, 0] <= xmax) &
                    (querySet[:, 1] >= ymin) &
                    (querySet[:, 1] <= ymax), :]
