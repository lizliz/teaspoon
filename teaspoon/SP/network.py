
def Adjaceny_OP(perm_seq, n, delay = 1): #Gets Adjacency Matrix (weighted and direction) using permutation sequence
    import numpy as np
    N = np.math.factorial(n) #number of possible nodes
    A = np.zeros((N,N)) #prepares A
    for i in range(len(perm_seq)-delay): #go through all permutation transitions (This could be faster wiuthout for loop)
        A[perm_seq[i]-delay][perm_seq[i+delay]-1] += 1 #for each transition between permutations increment A_ij
    return A #this A is directional and weighted


def Adjacency_KNN(indices):
    import numpy as np
    A = np.zeros((len(indices.T[0]),len(indices.T[0])))
    for h in range(len(indices.T[0])):
        KNN_i = indices[h][indices[h]!=h] #indices of k nearest neighbors
        A[h][KNN_i]+=1 #increment A_ij for kNN indices
        A.T[h][KNN_i]+=1
    A[A>0] = 1
    return A


def Adjacency_CGSS(state_seq, N, delay = 1): #Coarse Grained State Space adjacency matrix
    import numpy as np
    N = (np.max(state_seq)+1)
    A = np.zeros((N,N)) #prepares A
    for i in range(len(state_seq)-delay): #go through all permutation transitions (This could be faster wiuthout for loop)
        A[state_seq[i]][state_seq[i+delay]] += 1 #for each transition between permutations increment A_ij
    return A #this A is directional and weighted





def knn_graph(ts, n = None, tau = None, k = 4):
    
    """This function creates an k-NN network represented as an adjacency matrix A using a 1-D time series
    
    Args:
        ts (1-D array): 1-D time series signal
    
    Other Parameters:
        n (Optional[int]): embedding dimension for state space reconstruction. Default is uses FNN algorithm from parameter_selection module.
        tau (Optional[int]): embedding delay fro state space reconstruction. Default uses MI algorithm from parameter_selection module.
        k (Optional[int]): number of nearest neighbors for graph formation.
        
    Returns:
        [2-D square array]: A (2-D weighted and directed square adjacency matrix)
    """
    
    #import sub modules
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
    from teaspoon.SP import tsa_tools
    
    if tau == None:
        from parameter_selection import MI_delay
        tau = MI_delay.MI_for_delay(ts, method = 'basic', h_method = 'sturge', k = 2, ranking = True)
    if n == None:
        from parameter_selection import FNN_n
        perc_FNN, n = FNN_n.FNN_n(ts, tau)
    
    ETS = tsa_tools.takens(ts, n, tau) #get embedded time series
    
    distances, indices = tsa_tools.k_NN(ETS, k= k) 
    #gets distances between embedded vectors and the indices of the nearest neighbors for every vector
        
    A = Adjacency_KNN(indices) #get adjacency matrix (weighted, directional)
    
    return A


def ordinal_partition_graph(ts, n = None, tau = None):
    
    """This function creates an ordinal partition network represented as an adjacency matrix A using a 1-D time series
    
    Args:
        ts (1-D array): 1-D time series signal
    
    Other Parameters:
        n (Optional[int]): embedding dimension for state space reconstruction. Default is uses false nearest negihbor algorithm from parameter_selection module.
        tau (Optional[int]): embedding delay from state space reconstruction. Default uses mutual information algorithm from parameter_selection module.
        k (Optional[int]): number of nearest neighbors for graph formation.
        
    Returns:
        [2-D square array]: A (2-D weighted and directed square adjacency matrix)
    """
    
    #import sub modules
    import os
    import sys
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','..'))
    from teaspoon.SP.tsa_tools import permutation_sequence
    
    if n == None:
        from parameter_selection import MsPE
        tau = int(MsPE.MsPE_tau(ts)) 
        n = MsPE.MsPE_n(ts, tau)
        
    if tau == None:
        from parameter_selection import MsPE
        tau = int(MsPE.MsPE_tau(ts)) 
    
    PS = permutation_sequence(ts, n, tau)
        
    A = Adjaceny_OP(PS, n) #gets adjacency matrix from permutation sequence transtitions
    
    return A
    

def cgss_graph(ts, n = None, tau = None, B = 10, embedding_method = 'standard', binning_method = 'equal_frequency'):
    """This function creates a coarse grained state space (cgss) network represented as an adjacency matrix A using a 1-D time series
    
    Args:
        ts (1-D array): 1-D time series signal
    
    Other Parameters:
        n (Optional[int]): embedding dimension for state space reconstruction. Default is uses false nearest negihbor algorithm from parameter_selection module.
        tau (Optional[int]): embedding delay from state space reconstruction. Default uses mutual information algorithm from parameter_selection module.
        B (Optional[int]): number of bins per dimension for graph formation. Default is 10.
        
    Returns:
        [2-D square array]: A (2-D weighted and directed square adjacency matrix)
    """
    #import sub modules
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
    from teaspoon.SP import tsa_tools
    import numpy as np
    
    def equalObs(x, B): #define function to calculate equal-frequency bins based on interpolation
        return np.interp(np.linspace(0, len(x), B), np.arange(len(x)), np.sort(x))
    
    #----------------Get embedding parameters if not defined----------------
    if tau == None:
        from parameter_selection import MI_delay
        tau = MI_delay.MI_for_delay(ts, method = 'basic', h_method = 'sturge', k = 2, ranking = True)
    if n == None:
        from parameter_selection import FNN_n
        perc_FNN, n = FNN_n.FNN_n(ts, tau)
    
    #get state space reconstruction from signal (SSR)
    SSR = tsa_tools.takens(ts, n, tau) 
    
    
    #----------------Define how to use the embedding----------------
    if embedding_method == 'difference': #uses standard state space reconstruction
        delta = np.diff(SSR, axis = 1) #get differences in SSR vectors along coordinate axis
        delta = delta.T #transpose delta array to put in columns
        embedding = delta
        N = B**n
        basis = B**(np.arange(n-1)) #basis for assigning symbolic value
    if embedding_method == 'standard': #uses differences along axis of state space reconstruction
        embedding  = np.array(SSR).T
        N = B**(n-1)
        basis = B**(np.arange(n)) #basis for assigning symbolic value
        
        
    #----------------Define how to partition the embedding----------------
    if binning_method == 'equal_frequency':
        #define bins with equal-frequency or probability (approximately) 
        B_array = equalObs(embedding.flatten(), B+1)
        B_array[-1] = B_array[-1]*(1 + 10**-10)
    if binning_method == 'equal_size': #define bins based on equal spacing
        B_array = np.linspace(np.amin(embedding), np.amax(embedding)*(1+10**-10), B+1) 
        
        
    #----------------digitize the embedding to a sequence----------------
    digitized_embedding = [] #prime the digitized version of deltas
    for e_i in embedding: #nloop through n-1 delta positions
        digitzed_vector = np.digitize(e_i, bins = B_array) # digitalize column delta_i
        digitized_embedding.append(digitzed_vector) #append to digitalized deltas data structure
    digitized_embedding = np.array(digitized_embedding).T - 1 #digitalize and stacked delta vectors
    symbol_seq = np.sum(np.array(basis)*digitized_embedding, axis = 1) # symbolic sequence from basis and D
    
    
    #get adjacency matrix from sequence
    A = Adjacency_CGSS(symbol_seq, N)
   
    return A
    
# In[ ]:
    
if __name__ == "__main__": #Only runs if running from this file (This will show basic example)

    #import needed packages
    import numpy as np
    t = np.linspace(0,30,200) #define time array
    ts = np.sin(t) + np.sin(2*t) #generate a simple time series
    
    A_knn = knn_graph(ts) #ordinal partition network from time series
    
    A_op = ordinal_partition_graph(ts) #knn network from time series
    
    A_cgss = cgss_graph(ts) #coarse grained state space netwrok from time series
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    