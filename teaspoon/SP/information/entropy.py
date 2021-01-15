def MsPE(ts, n = 5, delay_end = 200, plotting = False, normalize = False):
    """This function takes a time series and calculates Multi-scale Permutation Entropy (MsPE) 
        over multiple time scales
    
    Args:
       ts (array):  Time series (1d).
       n (int): Permutation dimension. Default is 5.
       delay_end (int): maximum delay in search. default is 200.
       
    Kwargs:
       plotting (bool): Plotting for user interpretation. defaut is False.
       normalize (bool): Normalizes the permutation entropy on scale from 0 to 1. defaut is False.

    Returns:
       (array): MsPE, the permutation entropy over multiple time scales.

    """
    from pyentrp import entropy as ent
    import math
    import numpy as np
    
    MSPE = []
    delays = []
    m = n
    for delay in np.arange(1,delay_end):
        PE = ent.permutation_entropy(ts, m, delay)
        ME = 1
        if normalize == True: ME = np.log2(math.factorial(n))
        PE = PE/ME
        MSPE = np.append(MSPE, PE)
        delays.append(delay)
    
    if plotting == True:
        import matplotlib.pyplot as plt
        plt.figure(2) 
        TextSize = 17
        plt.figure(figsize=(8,3))
        plt.plot(delays, MSPE, marker = '.')
        plt.xticks(size = TextSize)
        plt.yticks(size = TextSize)
        plt.ylabel(r'$h(3)$', size = TextSize)
        plt.xlabel(r'$\tau$', size = TextSize)
        plt.show()
    return delays, MSPE


def PE(ts, n = 5, tau = 10, normalize = False):
    """This function takes a time series and calculates Permutation Entropy (PE).
    
    Args:
       ts (array):  Time series (1d).
       n (int): Permutation dimension. Default is 5.
       tau (int): Permutation delay. Default is 10.
       
    Kwargs:
       normalize (bool): Normalizes the permutation entropy on scale from 0 to 1. defaut is False.

    Returns:
       (float): PE, the permutation entropy.

    """
    from pyentrp import entropy as ent
    import math
    import numpy as np
    
    PE = ent.permutation_entropy(ts, n, tau)
    ME = 1
    if normalize == True: ME = np.log2(math.factorial(n))
    PE = PE/ME

    return PE


def WPE(ts, n = 5, tau = 10, normalize = False):
    """This function takes a time series and calculates Weighted Permutation Entropy (WPE) from "Weighted-permutation entropy: A complexity measure for time series incorporating amplitude information" by Bilal Fadlallah, Badong Chen, Andreas Keil, and José Príncipe.
    
    Args:
       ts (array):  Time series (1d).
       n (int): Permutation dimension. Default is 5.
       tau (int): Permutation delay. Default is 10.
       
    Kwargs:
       normalize (bool): Normalizes the permutation entropy on scale from 0 to 1. defaut is False.

    Returns:
       (float): h, the weighted permutation entropy.

    """
    
    time_series = ts
    
    if n == None:
        from teaaspoon.parameter_selection import MsPE
        tau = int(MsPE.MsPE_tau(ts)) 
        n = MsPE.MsPE_n(ts, tau)
        
    if tau == None:
        from teaaspoon.parameter_selection import MsPE
        tau = int(MsPE.MsPE_tau(ts))
    
    m, delay = n, tau
    
    import itertools
    import numpy as np
    def util_hash_term(perm): #finds permutation type
        deg = len(perm)
        return sum([perm[k]*deg**k for k in range(deg)])
    L = len(time_series) #total length of time series
    perm_order = [] #prepares permutation sequence array
    weights = []
    permutations = np.array(list(itertools.permutations(range(m)))) #prepares all possible permutations for comparison
    hashlist = [util_hash_term(perm) for perm in permutations] #prepares hashlist
    for i in range(L - delay * (m - 1)): 
    #For all possible permutations in time series
        v = time_series[i:i + delay * m:delay]
        sorted_index_array = np.array(np.argsort(v, kind='quicksort'))
        X_bar = np.mean(v)
        weight = (1/n)*np.sum((v-X_bar)**2)
        weights.append(weight)
        #sort array for catagorization
        hashvalue = util_hash_term(sorted_index_array);
        #permutation type
        perm_order = np.append(perm_order, np.argwhere(hashlist == hashvalue)[0][0])
        #appends new permutation to end of array
    perm_seq = perm_order.astype(int)+1 #sets permutation type as integer where $p_i \in \mathbb{z}_{>0}$
    perms, counts = np.unique(perm_seq, return_counts = True)
    weighted_probs = []
    perm_seq, weights = np.array(perm_seq), np.array(weights)
    d = np.sum(weights)
    weighted_probs = [np.sum(weights[perm_seq == p] )/d for p in perms]
    probs = np.array(weighted_probs)
    H = -np.sum(probs*np.log2(probs))
    WPE = H
    if normalize == True:
        WPE = H/np.log2(np.math.factorial(n))
        
    return WPE #returns sequence of permutations



def PersistentEntropy(lifetimes, normalize = False):
    """This function takes a time series and calculates Permutation Entropy (PE).
    
    Args:
       lifetimes (array):  Lifetimes from persistence diagram (1d).
       
    Kwargs:
       normalize (bool): Normalizes the entropy on scale from 0 to 1. defaut is False.

    Returns:
       (float): PerEn, the persistence diagram entropy.

    """
    import numpy as np
    lt = lifetimes
    if len(lt) > 1:
        L = sum(lt)
        p = lt/L
        E = sum(-p*np.log2(p))
    if len(lt) == 1:
        E = 0
    if len(lt) == 0:
        E = np.nan
    Emax = 1
    if normalize == True: Emax = np.log2(sum(lt))
    PerEn = E/Emax

    return PerEn



# In[ ]:
    
if __name__ == "__main__": #Only runs if running from this file (This will show basic example)

    #-----------------------------------PE-----------------------------------------
    
    import numpy as np
    t = np.linspace(0,100,2000)
    ts = np.sin(t)  #generate a simple time series
    
    from teaspoon.SP.information.entropy import PE
    h = PE(ts, n = 6, tau = 15, normalize = True)
    print('Permutation entropy: ', h)
    
    
    #-----------------------------------MsPE-----------------------------------------
    
    import numpy as np
    t = np.linspace(0,100,2000)
    ts = np.sin(t)  #generate a simple time series
    
    from teaspoon.SP.information.entropy import MsPE
    delays,H = MsPE(ts, n = 6, delay_end = 40, normalize = True)
    
    import matplotlib.pyplot as plt
    plt.figure(2) 
    TextSize = 17
    plt.figure(figsize=(8,3))
    plt.plot(delays, H, marker = '.')
    plt.xticks(size = TextSize)
    plt.yticks(size = TextSize)
    plt.ylabel(r'$h(3)$', size = TextSize)
    plt.xlabel(r'$\tau$', size = TextSize)
    plt.show()
    
    
    
    #-----------------------------------persistent entropy---------------------------
    
    import numpy as np
    #generate a simple time series with noise
    t = np.linspace(0,20,200)
    ts = np.sin(t) +np.random.normal(0,0.1,len(t)) 
    
    from teaspoon.SP.tsa_tools import takens
    #embed the time series into 2 dimension space using takens embedding
    embedded_ts = takens(ts, n = 2, tau = 15)
    
    from ripser import ripser
    #calculate the rips filtration persistent homology
    result = ripser(embedded_ts, maxdim=1)
    diagram = result['dgms']
    
    
    
    #--------------------Plot embedding and persistence diagram---------------
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1, 2) 
    plt.figure(figsize = (12,5)) 
    TextSize = 17
    MS = 4
    
    ax = plt.subplot(gs[0, 0]) 
    plt.yticks( size = TextSize)
    plt.xticks(size = TextSize)
    plt.xlabel(r'$x(t)$', size = TextSize)
    plt.ylabel(r'$x(t+\tau)$', size = TextSize)
    plt.plot(embedded_ts.T[0], embedded_ts.T[1], 'k.')
    
    ax = plt.subplot(gs[0, 1]) 
    top = max(diagram[1].T[1])
    plt.plot([0,top*1.25],[0,top*1.25],'k--')
    plt.yticks( size = TextSize)
    plt.xticks(size = TextSize)
    plt.xlabel('Birth', size = TextSize)
    plt.ylabel('Death', size = TextSize)
    plt.plot(diagram[1].T[0],diagram[1].T[1] ,'go', markersize = MS+2)
    
    plt.show()
    #-------------------------------------------------------------------------
    
    #get lifetimes (L) as difference between birth (B) and death (D) times
    B, D = diagram[1].T[0], diagram[1].T[1]
    L = D - B
    
    from teaspoon.SP.information.entropy import PersistentEntropy
    h = PersistentEntropy(lifetimes = L)
    
    print('Persistent entropy: ', h)
    
    
    
    
    
    
    
    
    
    