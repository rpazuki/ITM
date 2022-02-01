from collections import Counter
from itertools import  zip_longest, groupby
from operator import itemgetter
import numpy as np

def empirical_dist(*data, fillvalue='-'):
    '''Empirical distribution'''
    if len(data) == 1:
        hist = Counter([row[0] for row in zip_longest(*data, fillvalue=fillvalue)])
    else:
        hist = Counter([row for row in zip_longest(*data, fillvalue=fillvalue)])
    values_sum = sum(hist.values())
    return Counter({ k:v/values_sum for k,v in hist.items()})

def moments(dist, degree):
    '''Find the moment degree of the dist

       it assumes the columns of the dist are
       numbers'''
    return sum([ (np.prod(k)**degree)*p for k,p in dist.items() ])

# Important note:
#
# The marginal of X is not equal to the direct prob of X
# since the moving average construct a shorter version than
# the original time series.
# If it is important to create a joint dist that has the same marginal
# We can assumea periodic boundary condition, by attaching a segment from the start
# of the time series with length equal to the lag-1 to get an equal marginal from
# the joint probability dist
# e.g. if
# dist = find_dist(sliding_window(signal+signal[:2],3))
# then
# marginal(dist,0) ==  find_dist(signal)
# marginal(dist,1) ==  find_dist(signal)
# marginal(dist,2) ==  find_dist(signal)

def marginal(prob_x, *cols):
    '''Marginal distribution'''
    operation = itemgetter(*cols)
    selected_cols = sorted([(operation(k),v) for k,v in prob_x.items()],
                           key=itemgetter(0))
    return { k:sum(map(lambda row: row[1] , group))
        for k, group in groupby(selected_cols, lambda item: item[0])
    }

def conditional(prob_x, *cols):
    '''Conditional Distribution'''
    # Take the first key
    key_exam = next(iter(prob_x.keys()), None)
    # getter for the conditioned columns
    operation = itemgetter(*cols)
    # getter of the negate of the conditioned columns
    inv_op = itemgetter(* [i for i in range(len(key_exam)) if i not in cols]  )
    # map the  rows by disecting the columns
    selected_cols = sorted([ (operation(k),inv_op(k),v)
                            for k,v in prob_x.items()], key=itemgetter(0))
    # First, group by the conditional columns
    # next , intenally group by on similar values tofind the summs
    cond = {
        k: { k2:sum(map(lambda row:row[2], group2))
           for k2, group2 in
            groupby(sorted(group, key=itemgetter(1)), lambda item2: item2[1])
        }
        for k, group in groupby(selected_cols, lambda item: item[0])
    }
    # Normalised internall dists before return
    totals = {
        k: sum(dist.values())
        for k, dist in cond.items()
    }
    return {
        k: {k2:v/totals[k]
            for k2,v in dist.items()
        }
        for k, dist in cond.items()
    }

def marginal_conditional(prob_x_z, *cols):
    ''' prob_X_Z is conditioned on cols'''
    return { k:marginal(dist, *cols)
          for k, dist in  prob_x_z.items()
    }
