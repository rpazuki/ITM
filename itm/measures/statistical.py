import numpy as np
from itm.utils import sliding_window
from itm.distributions import empirical_dist


def correlation(x, y):
    '''Correlation between X and Y'''
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)
    prob_xy = empirical_dist(x, y)
    mean_xy = np.sum(  x*y*p for (x,y), p in prob_xy.items())
    return (mean_xy - mean_x*mean_y)/(std_x*std_y + 1e-32)

def autocorrelation(x, max_lag=1):
    '''Autoforelation in x'''
    # the row of matrix are the sliding_windows element
    x_columns = np.array([row for row in sliding_window(x, max_lag+1)])
    return np.array([
        correlation(x_columns[:, 0], x_columns[:, index])
        for index in range(1, max_lag+1)
        ])
