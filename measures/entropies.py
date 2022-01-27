import numpy as np
from utils import sliding_window
from distributions import empirical_dist
from distributions import marginal
from distributions import conditional

def H(prob_x):
    '''Shannon Entropy'''
    probabilites = np.fromiter((v for _,v in prob_x.items()), dtype=np.float128 )
    non_zeros_probabilites = probabilites[np.nonzero(probabilites)]
    return -np.sum(non_zeros_probabilites*np.log2(non_zeros_probabilites))

def H_conditional(prob_xy, *cols):
    '''Conditional Shannon Entrop
       Conditioned on the given columns of distribution'''
    conditional_prob = conditional(prob_xy, *cols)
    marginal_prob = marginal(prob_xy, *cols)
    return sum([ marginal_prob[k]*H(dist) for k, dist in conditional_prob.items()])

def I(prob_xy):
    ''' Mutual Information
        I(X,Y) = H(X) + H(Y) - H(X,Y)
        Assumes there are two columns in prob_xy'''
    prob_x = marginal(prob_xy, 0)
    prob_y = marginal(prob_xy, 1)
    return H(prob_x) + H(prob_y) - H(prob_xy)

def I2(prob_xy):
    ''' Mutual Information
        I(X,Y) = H(X, Y) - H(X|Y) - H(Y|X)
        Assumes there are two columns in prob_xy'''
    return  H(prob_xy)- H_conditional(prob_xy, 0)- H_conditional(prob_xy, 1)

def I3(prob_xy):
    ''' Mutual Information
        I(X,Y) = H(X) - H(X|Y)
        Assumes there are two columns in prob_xy'''
    prob_x = marginal(prob_xy, 0)
    return H(prob_x) - H_conditional(prob_xy, 0)

def I_conditional(prob_xyz, *cols):
    ''' Conditional Mutual Information
        I(X,Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
        Assumes, after conditioning on cols, there are two columns in prob_XYZ'''
    xy_cols = [c for c in range(len(cols)+2) if c not in cols]
    x_col = xy_cols[0]
    y_col = xy_cols[1]
    cols_and_x = [c for c in cols if c < x_col] + [x_col] + [c for c in cols if c > x_col]
    cols_and_y = [c for c in cols if c < y_col] + [y_col] + [c for c in cols if c > y_col]

    prob_x_z = marginal(prob_xyz, *cols_and_x)
    prob_y_z = marginal(prob_xyz, *cols_and_y)
    prob_z = marginal(prob_xyz, *cols)

    return (
        H(prob_x_z) + H(prob_y_z) - H(prob_xyz) - H(prob_z)
    )

def I2_conditional(prob_xyz, *cols):
    ''' Conditional Mutual Information
        I(X,Y|Z) = H(X |Z) - H(X|Y, Z)
        Assumes, after conditioning on cols, there are two columns in prob_xyz'''
    xy_cols = [c for c in range(len(cols)+2) if c not in cols]
    x_col = xy_cols[0]
    y_col = xy_cols[1]
    cols_and_x = [c for c in cols if c < x_col] + [x_col] + [c for c in cols if c > x_col]
    cols_and_y = [c for c in cols if c < y_col] + [y_col] + [c for c in cols if c > y_col]
    cols_shifted = [ c for c in cols if c < y_col] + [c-1 for c in cols if c > y_col]
    prob_xz = marginal(prob_xyz, *cols_and_x)
    return (
        H_conditional(prob_xz, *cols_shifted)
        - H_conditional(prob_xyz, *cols_and_y)
    )

def I3_conditional(prob_xyz, *cols):
    ''' Conditional Mutual Information
        I(X,Y|Z) = H(X |Z) + H(Y |Z) - H(X, Y|Z)
        Assumes, after conditioning on cols, there are two columns in prob_xyz'''
    xy_cols = [c for c in range(len(cols)+2) if c not in cols]
    x_col = xy_cols[0]
    y_col = xy_cols[1]
    cols_and_x = [c for c in cols if c < x_col] + [x_col] + [c for c in cols if c > x_col]
    cols_and_y = [c for c in cols if c < y_col] + [y_col] + [c for c in cols if c > y_col]
    cols_shifted_by_x = [
        c for c in cols if c < x_col] + [c-1 for c in cols if c > x_col]
    cols_shifted_by_y = [
        c for c in cols if c < y_col] + [c-1 for c in cols if c > y_col]
    prob_yz = marginal(prob_xyz, *cols_and_y)
    prob_xz = marginal(prob_xyz, *cols_and_x)
    return (
          H_conditional(prob_yz, *cols_shifted_by_x)
        + H_conditional(prob_xz, *cols_shifted_by_y)
        - H_conditional(prob_xyz, *cols)
    )

def AIS(x, k):
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
    return I3(prob_x_k_x)

def TE(x, y, k, i):
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
    return I_conditional(prob_y_i_1_x_k, 0)

def TE_Condition(x, y, z, k, i, j):
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
    return I_conditional(prob_y_i_1_z_j_x_k, 0, 1)
