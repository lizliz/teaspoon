## @package teaspoon.MakeData.TimeSeries.stdSignals
# Generates standard signals with some easy to access noise insertion.
#


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy
import scipy.signal
import scipy.io
import pandas as pd


def uni(length, halfHeight = 1,seed = None):
    np.random.seed(seed)
    noise = np.random.rand(length)*2*halfHeight-halfHeight
    return noise

def uniPos(length, Height = 1,seed = None):
    np.random.seed(seed)
    noise = np.random.rand(length)*Height
    return noise

def norm(length, sd, seed = None):
    np.random.seed(seed)
    noise = np.random.normal(size = length, scale = sd)
    return noise


def pulse(t,T=1,duty = .1, max_pulse_height = 1):
    '''
    Creates a pulse wave with period T.

    Function values are:
    | max_pulse_height    if t mod T < T*duty
    |     0               else
    '''
    y = max_pulse_height * .5 *(scipy.signal.square(2*np.pi/T*t,duty = duty)+1)

    return y


def pulse_with_noise(t,T, duty,
                    alpha_percent = 0,
                    beta_percent =0,
                    epsilon_percent=0,
                    seed = None,
                    max_pulse_height = 1,
                    beta_noise_function = uni,):
    '''
    Creates a pulse with alpha, beta, and epsilon noise present.  See paper for details.
    Note that this assumes these values are given as a percentage of other parameters.
    Reasonable options for each:
      - alpha_percent is a percentage of tau, and gives the digital noise inputs.
        This can be [0,.5].
      - beta_percent is the y-noise given as a percentage of the amplitude
    '''
    np.random.seed(seed)

    # find the width of the spike
    tau = duty * T

    # noise to add in the x-direction (causes digital noise)
    alpha = tau * alpha_percent

    # noise to add in the y-direction
    beta = max_pulse_height * beta_percent

    # accordian noise
    epsilon = epsilon_percent * T

    if epsilon_percent>0:
        # initialize Q
        Q = []
        while sum(Q) < t[-1]:
            # get the list of random period lengths drawn from a uniform distribution
            # on [T-epsilon, T+epsilon]
            Q.extend(T + uni(10, epsilon))

        # Find the locations of the beginning of each period
        C = [sum(Q[:i]) for i in range(len(Q)+1)]
        C = np.array(C)

        def getPeriodIndex(x):
            if x == 0:
                return 0
            else:
                return np.where(C<x)[0][-1]

        # Tag each entry in the time vector t by which period it belongs to
        index = [getPeriodIndex(x) for x in list(t)]

        # phi respaces the real line to match the period
        #phi = [T*i+(T/Q[i])*(t[j] - C[i]) for j,i in enumerate(index)]
        phi = [(T/Q[i])*(t[j] - C[i]) for j,i in enumerate(index)]
        phi = np.array(phi)

    else:
        phi = t

    # Add digital noise
    if alpha_percent > 0:
        phi = phi+uni(len(phi), halfHeight=alpha, seed=seed)

    #
    Y = pulse(phi, T=T, duty = duty, max_pulse_height = max_pulse_height)

    if beta_percent > 0:
        Y += beta_noise_function(len(Y),beta,seed = seed)
        # Y += uni(len(Y), halfHeight=beta, seed=seed)

    return Y
