import numpy as np
from itm.distributions import empirical_dist
from itm.distributions import marginal
from itm.distributions import conditional

# Liam Paninski. 2003. Estimation of entropy and mutual information.
# Neural Comput. 15, 6 (June 2003), 1191–1253.
# DOI:https://doi.org/10.1162/089976603321780272
#
# Use entropy_miller or entropy_jackknified for continuous data
# the data must be binned first before calling the functions.
#
def entropy(prob_x):
    '''Shannon Entropy

       MLE estimator, also called the “plug-in”'''
    probabilites = np.fromiter((v for _,v in prob_x.items()), dtype=np.float128 )
    non_zeros_probabilites = probabilites[np.nonzero(probabilites)]
    return -np.sum(non_zeros_probabilites*np.log2(non_zeros_probabilites))

def entropy_miller(data_x):
    '''Shannon Entropy for binned continuous data.

    Miller-Madow bias correction :
    H(p_N) + (m-1)/N, m is the number of non-zero bins
    '''
    prob_x = empirical_dist(data_x)
    N = len(data_x)
    m = len(prob_x.keys())
    return entropy(prob_x) + (m-1)/(2*N)

def entropy_jackknified(data_x):
    '''Shannon Entropy for binned continuous data.

    H(p_N) - (N-1)/N H(p_{N-1})
    '''
    mle_entropy = entropy(empirical_dist(data_x))
    N = len(data_x)
    data_x_1 = (data_x[:i] + data_x[i+1:] for i in range(N))
    bias = sum([ entropy(empirical_dist(data))
                 for data in data_x_1
              ])
    return N*mle_entropy - (N-1)*bias/N

def conditional_entropy(prob_xy, *cols):
    '''Conditional Shannon Entrop
       Conditioned on the given columns of distribution'''
    conditional_prob = conditional(prob_xy, *cols)
    marginal_prob = marginal(prob_xy, *cols)
    return sum([ marginal_prob[k]*entropy(dist) for k, dist in conditional_prob.items()])



def kullback_leibler(dist_p, dist_q):
    '''Kullback–Leibler distance'''
    probabilites = np.array([
        (p1, dist_q[k]) for k,p1 in dist_p.items() if p1 !=0 and dist_q[k] != 0])
    return np.sum(probabilites[:,0]*np.log2(probabilites[:, 0]/probabilites[:, 1]))
