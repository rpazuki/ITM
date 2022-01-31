import numpy as np
from itm.measures.entropies import kullback_leibler

def test_kullback_leibler():
    # zero distance
    dist_1 = {0: .25, 1:.25, 2:.5}
    dist_2 = {0: .25, 1:.25, 2:.5}
    assert kullback_leibler(dist_1, dist_2) == 0.0
    # distribution with zero probability
    dist_1 = {0: .25, 1:.25, 2:.5, 3:0}
    dist_2 = {0: .25, 1:.25, 2:.5, 3:0}
    assert kullback_leibler(dist_1, dist_2) == 0.0
    #
    dist_1 = {0: (9/25), 1:(12/25), 2:(4/25)}
    dist_2 = {0: (1/3),  1:(1/3),   2:(1/3)}

    distance_1 = ((9/25)*np.log2((9/25)/(1/3))
        +(12/25)*np.log2((12/25)/(1/3))
        +(4/25)*np.log2((4/25)/(1/3)))
    assert kullback_leibler(dist_1, dist_2) == distance_1

    distance_2 = ((1/3)*np.log2((1/3)/(9/25))
        +(1/3)*np.log2((1/3)/(12/25))
        +(1/3)*np.log2((1/3)/(4/25)))
    assert kullback_leibler(dist_2, dist_1) == distance_2
