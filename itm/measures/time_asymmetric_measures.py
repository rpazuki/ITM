from itm.utils import sliding_window
from itm.distributions import empirical_dist
from itm.measures import mutual_information3
from itm.measures import conditional_mutual_information

def active_information_storage(x, k):
    '''Active Information Storage
    I(x_{n-k+1}, ..., x_{n}; x_{n+1})'''
    # create a sliding window, size k
    x_window_k = sliding_window(x[:-1], k)
    # find the create the tuple
    # of (x_{n+1}; x_{n}, ..., x_{n-k+1})
    # from the main time series and the
    # moving window and create its probability
    # distribution
    prob_x_k_x = empirical_dist(x_window_k, x[k:])
    # find the mutual information
    return mutual_information3(prob_x_k_x)

def transfer_entropy(x, y, k, i):
    ''' Transfer Entropy: x = source, y = target
        I(x_{n-k+1}, ..., x_{n}; y_{n+1} | y_{n-i+1}, ..., y_{n})'''

    # create a sliding window, size k from source
    x_k = sliding_window(x[:-1], k)
    # create a sliding window, size i from target
    y_i = sliding_window(y[:-1], i)
    # mix both signals and the y_{n+1}
    # as ((y_{n-i+1}, ..., y_{n}), y_{n+1}, x_{n-k+1}, ..., x_{n})
    y_i_1_x_k = map(lambda xy: (xy[0], xy[1]) + tuple(xy[2]), zip(y_i, y[i:], x_k))
    # create the joint distributions
    prob_y_i_1_x_k = empirical_dist(y_i_1_x_k)
    # find the conditional mutual entropy
    return conditional_mutual_information(prob_y_i_1_x_k, 0)

def conditional_transfer_entropy(x, y, z, k, i, j):
    ''' Conditional Transfer Entropy: X = source, Y = target
        I(x_{n-k+1},...,x_{n};y_{n+1}|y_{n-i+1},...,y_{n},z_{n-j+1},...,z_{n})'''
    # create a sliding window, size k from source
    x_k = sliding_window(x[:-1], k)
    # create a sliding window, size i from target
    y_i = sliding_window(y[:-1], i)
    # create a sliding window, size j from target
    z_j = sliding_window(z[:-1], j)
    # mix three signals and the y_{n+1}
    # as ((y_{n-i+1},...,y_{n}),(z_{n-j+1},...,z_{n}), y_{n+1},x_{n-k+1},...,x_{n})
    y_i_1_z_j_x_k = map(
        lambda xyz: (xyz[0], xyz[1], xyz[2]) + tuple(xyz[3]),
        zip(y_i, z_j, y[i:], x_k))
    # create the joint distributions
    prob_y_i_1_z_j_x_k = empirical_dist(y_i_1_z_j_x_k)
    # find the conditional mutual entropy
    return conditional_mutual_information(prob_y_i_1_z_j_x_k, 0, 1)
