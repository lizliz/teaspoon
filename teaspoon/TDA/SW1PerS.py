import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
import numpy as np
from ripser import ripser
from scipy.interpolate import UnivariateSpline


def N_sw1pers(signal, cutoff=0.1, max_dim=10, plotting=False):
    """This function estimates the correct dimensions for a sliding window reconstruction using truncated fourier transform reconstructions as 
    suggested in "Sliding Windows and Persistence: An Application of Topological Methods to Signal Analysis."

    Args:
        signal (1-D array): one dimensional signal to be used fort the sliding window reconstruction.

    Other Parameters:
        cutoff (Optional[float]): cutoff used to determine accepted percent similar between reconstructed truncated fourier transform and signal. Default is cutoff = 0.1.
        max_dim (Optional[int]): maximum dimension N considered in reconstruction. Default is 10.
        plotting (Optional[boolean]): optional True/False for plotting stages in reconstructing the signal with i-terms from truncated fourier transform.

    Returns:
        [int]: N number of fourier terms to accurately reconstruct signal.
    """

    # this function uses a fourier reconstruction to determine number of terms to get a low error.

    y = np.array(signal)  # set signal as numpy array y
    error = cutoff+1  # initially set error to be greater than cutoff

    # mean center and normalize signal
    y = y-np.mean(y)
    y = y/max(y)
    # adds small amount of noise for filtering signal.
    y = y + np.random.normal(0, 0.001, len(y))

    # length of signal
    n = len(y)

    fhat = np.fft.fft(y, n)  # computes the fft
    psd = fhat * np.conj(fhat)/n  # compute power spectral density
    i_peaks, _ = find_peaks(psd)  # get peaks in psd
    # remove repeated peaks and sort them
    peak_vals = np.flip(np.sort(psd[i_peaks]))[::2]

    i = 0
    while error > cutoff:  # while the error between reconstructed and original signal is too large
        i += 1

        # Filter out noise
        # set noise filter by truncating FFT to each peak
        threshold = peak_vals[i]*1.01
        psd_idxs = psd > threshold  # array of 0 and 1 #apply filter
        fhat_filtered = psd_idxs * fhat  # used to retrieve the signal

        y_filtered = np.fft.ifft(fhat_filtered)  # inverse fourier transform

        error = sum(abs((y-np.real(y_filtered)))) / \
            sum(abs(y))  # find error between two

        if plotting == True:
            plt.plot(y_filtered, label=str(i) +
                     '-term recon., Err. = '+str(round(100*error, 2))+'%')

        # store number of terms needed for error less than cutoff (10%) as N
        N = len(peak_vals[peak_vals > threshold])

        if N > max_dim:  # no more than max dimension
            error = 0

    if plotting == True:
        print('k/n: ', N/n)
        plt.plot(y, 'k--', label='Actual Time Series (normalized)')
        plt.legend(loc='upper right', bbox_to_anchor=(1.65, 1.0))
        plt.show()

    return N


def sw1pers(signal, w, M, nT=300, point_wise_normalize=True, plotting=False):
    """This function calculates the persistence diagram of the original version of SW1PerS defined in "Sliding Windows and Persistence: An Application of Topological Methods to Signal Analysis."

    Args:
        signal (1-D array): one dimensional signal to be used fort the sliding window reconstruction.
        w (float): window size for sliding window.
        M (int): sliding window reconstruction dimension.

    Other Parameters:
        nT (Optional[int]): number of points in sliding window embedding point cloud.
        point_wise_normalize (Optional[boolean]): option in default SW1PerS for when the signals RMS amplitude is approximately not constant (e.g. a damped free response signal).
        plotting (Optional[boolean]): optional True/False for plotting resulting signal, embedding, and persistence diagram.

    Returns:
        [list]: dgms (persistence diagrams -- 0 and 1 dimensional)

    """

    M = M-1
    # w = 2*np.pi*M/(L*(M+1)) #The theory behind SW1PerS implies that a good window size (w) ~ 2piM/(L*(M+1))
    tau = w/M  # time delay ~ w/M from theory behind SW1PerS
    nS = len(signal)  # number of data points in signal
    t = np.linspace(0, 2*np.pi, nS)  # maps the time into [0,2pi]
    T = (2*np.pi - M*tau)*np.linspace(0, 1, nT)
    # window values of time
    Y = (np.ones(M+1)*T.reshape(len(T), 1))
    Z = (tau*(np.arange(0, M+1))*np.ones(nT).reshape(len(np.ones(nT)), 1))
    tt = (Y + Z).T  # array of time series for each window
    # curve fits data from signal to cubic spline over desired length in time
    interpolation = UnivariateSpline(t, signal, s=0)
    # uses curved fit response to find data points (interpolated) for tt
    X = np.apply_along_axis(interpolation, 1, tt)

    X = X.T
    X_mean = np.sum(X, axis=1, keepdims=True)/M
    if point_wise_normalize == True:
        X_bar = (X - X_mean) / \
            np.sqrt(np.sum((X - X_mean)**2, axis=1, keepdims=True))
    else:
        X_bar = (X - X_mean)/np.max(np.sqrt(np.sum((X - X_mean)**2, axis=1)))

    x_cloud, y_cloud = X_bar.T[0], X_bar.T[1]
    # transpose and apply square form in preparation to give cloud to Ripser

    X_bar = scipy.spatial.distance.pdist(X_bar)
    dist_matrix = scipy.spatial.distance.squareform(X_bar)
    # calls for ripser to calculate persistence homology
    dgms = ripser(dist_matrix, maxdim=1, coeff=11,
                  distance_matrix=True, metric='euclidean')['dgms']

    if plotting == True:
        TextSize = 13
        gs = gridspec.GridSpec(2, 2)
        plt.figure(1)
        MS = 10
        plt.figure(figsize=(7, 7))

        ax = plt.subplot(gs[0, 0:2])
        ax.plot(t, signal, 'k', label='Time Series $x(t)$')
        ax.plot([0, w], [0, 0], 'r',
                label=r'Window Size $w = \frac{2\pi M}{L(M+1)}$')
        ax.plot(t, interpolation(t), label='1-D spline $g(t)$')
        plt.xlabel(r'$t$', size=TextSize)
        plt.ylabel(r'$x(t)$', size=TextSize)
        plt.xticks(size=TextSize)
        plt.yticks(size=TextSize)
        plt.legend(loc='lower right')

        ax = plt.subplot(gs[1, 0])
        ax.plot(x_cloud, y_cloud, 'ko', markersize=2)
        plt.xlabel(r'$g(t)$', size=TextSize)
        plt.ylabel(r'$g(t+\tau)$', size=TextSize)
        plt.xticks(size=TextSize)
        plt.yticks(size=TextSize)

        ax = plt.subplot(gs[1, 1])
        ax.plot(dgms[0][:-1].T[0], dgms[0][:-1].T[1],
                'rX', markersize=MS, label='$H_0$')
        ax.plot([0, 2], [0, 2], 'k--')
        ax.plot(dgms[1].T[0], dgms[1].T[1], 'g^', markersize=MS, label='$H_1$')
        plt.xlabel('Birth', size=TextSize)
        plt.ylabel('Death', size=TextSize)
        plt.xticks(size=TextSize)
        plt.yticks(size=TextSize)
        plt.legend(loc='lower right')

        plt.subplots_adjust(wspace=0.4)
        plt.show()

    return dgms


def periodicity_score(dgms):
    """This function calculates the periodicity score from SW1PerS.

    Args:
        dgms (list of 2-D arrays): persistence diagram ouputs from ripser for both 0 and 1 dimensional persistence.

    Returns:
        [float]: score, the periodicity score from the SW1PerS theory. This value is between 0 and 1 with a value near 0 signifying a periodic signal.
    """

    if len(dgms[1]) == 0:
        score = np.nan
    else:
        pd = dgms[1].T
        lifetimes = pd[1]-pd[0]
        max_lifetime = max(lifetimes)
        score = 1-max_lifetime/(3**0.5)
    return score


# In[ ]:
if __name__ == '__main__':

    #import packages
    import numpy as np
    from teaspoon.TDA.SW1PerS import sw1pers, N_sw1pers

    t = np.linspace(0, 20, 1000)
    ts = np.sin(2*np.pi*t)

    # _Set parameters for SW1PerS__
    nT = 300  # number of points in point cloud
    N = N_sw1pers(ts, cutoff=0.25, plotting=True)
    M = 2*N  # embedding dimension
    L = 20  # number of periods in signal
    w = 2*np.pi*M/(L*(M+1))  # optimal window size from sw1pers theory

    # ______________Plotting____________________
    dgms = sw1pers(ts, w=w, M=M, nT=nT, plotting=True)

    # _______________Printing Results________________________________
    print('Number of significant frequencies (N): ', N)
    print('Window Size:                           ', w)
    print('embedding dimension (M = 2N):          ', M)
    print('Periodicity Score:                     ', periodicity_score(dgms))
