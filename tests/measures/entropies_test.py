import numpy as np
from measures.entropies import H
from measures.entropies import H_conditional
from measures.entropies import I
from measures.entropies import I2
from measures.entropies import I3
from measures.entropies import I_conditional
from measures.entropies import I2_conditional
from measures.entropies import I3_conditional

def test_shannon_entropy():
    '''Test Entropy'''
    dist1 = {0:.5, 1:.5}
    assert H(dist1) == 1.0

    dist1 = {0:1, 1:0}
    assert H(dist1) == 0.0

def test_conditional_shannon_entropy():
    '''Test Conditional Entropy'''
    dist1={(0, 0):.5, (1, 1):.5}
    assert H_conditional(dist1, 0) == 0.0
    assert H_conditional(dist1, 1) == 0.0

    dist1={(0, 0):.25, (0, 1):.25, (1, 0):.25, (1, 1):.25}
    assert H_conditional(dist1, 0) == 1.0
    assert H_conditional(dist1, 1) == 1.0

    dist1={(0, 0):1/8, (0, 1):1/8, (1, 0):1/8, (1, 1):5/8}
    h_0 = H_conditional(dist1, 0)
    assert np.isclose(h_0, (1/4)*(.5*np.log2(2) + .5*np.log2(2))
                        +(3/4)*((1/6)*np.log2(6) + (5/6)*np.log2(6/5))
                    )
    h_1 = H_conditional(dist1, 1)
    assert np.isclose(h_1, (1/4)*(.5*np.log2(2) + .5*np.log2(2))
                        +(3/4)*((1/6)*np.log2(6) + (5/6)*np.log2(6/5))
                    )

def test_mutual_information():
    '''Test Mutual Information'''
    dist1={(0, 0):.5, (1, 1):.5}
    assert I(dist1) == 1.0
    assert I2(dist1) == 1.0
    assert I3(dist1) == 1.0

    dist1={(0, 0):.25, (0, 1):.25, (1, 0):.25, (1, 1):.25}
    assert I(dist1) == 0.0
    assert I2(dist1) == 0.0
    assert I3(dist1) == 0.0

    dist1={(0, 0):1/8, (0, 1):1/8, (1, 0):1/8, (1, 1):5/8}
    entropy_1 = (1/4)*np.log2(4) + (3/4)*np.log2(4/3)
    entropy_2 = 3*(1/8)*np.log2(8) + (5/8)*np.log2(8/5)

    assert np.isclose(I(dist1), 2*entropy_1 - entropy_2)
    assert np.isclose(I2(dist1), 2*entropy_1 - entropy_2)
    assert np.isclose(I3(dist1), 2*entropy_1 - entropy_2)

def test_conditional_mutual_information():
    '''Test Conditional Mutual Information'''
    dist1={(0, 0, 0):.5, (1, 1, 1):.5}
    assert I_conditional(dist1, 2) == 0.0
    assert I2_conditional(dist1, 2) == 0.0
    assert I3_conditional(dist1, 2) == 0.0

    dist1={(0, 0, 0):.5, (1, 1, 0):.5}
    assert I_conditional(dist1, 2) == 1.0
    assert I2_conditional(dist1, 2) == 1.0
    assert I3_conditional(dist1, 2) == 1.0

    dist1={(0, 0, 0):.25, (0, 1, 0):.25, (1, 0, 0):.25, (1, 1, 0):.25}
    assert I_conditional(dist1, 2) == 0.0
    assert I2_conditional(dist1, 2) == 0.0
    assert I3_conditional(dist1, 2) == 0.0
