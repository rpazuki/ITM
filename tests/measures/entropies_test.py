import numpy as np
from itm.measures import entropy
from itm.measures import entropy_miller
from itm.measures import entropy_jackknified
from itm.measures import conditional_entropy
from itm.measures import mutual_information
from itm.measures import mutual_information2
from itm.measures import mutual_information3
from itm.measures import conditional_mutual_information
from itm.measures import conditional_mutual_information2
from itm.measures import conditional_mutual_information3

def test_shannon_entropy():
    '''Test Entropy'''
    dist1 = {0:.5, 1:.5}
    assert entropy(dist1) == 1.0

    dist1 = {0:1, 1:0}
    assert entropy(dist1) == 0.0

def test_entropy_millery():
    '''Test Entropy'''
    data = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert np.isclose(entropy_miller(data), (1.0 + 1/20))

    data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert np.isclose(entropy_miller(data), 0.0)

def test_entropy_jackknified():
    '''Test Entropy'''
    entropy_1 = entropy({0:2/5, 1:3/5})
    data = [0, 0, 0, 1, 1, 1]
    assert np.isclose(entropy_jackknified(data), (6*1.0 - 5*entropy_1))

    data = [1, 1, 1, 1, 1, 1]
    assert entropy_jackknified(data) == 0.0

def test_conditional_shannon_entropy():
    '''Test Conditional Entropy'''
    dist1={(0, 0):.5, (1, 1):.5}
    assert conditional_entropy(dist1, 0) == 0.0
    assert conditional_entropy(dist1, 1) == 0.0

    dist1={(0, 0):.25, (0, 1):.25, (1, 0):.25, (1, 1):.25}
    assert conditional_entropy(dist1, 0) == 1.0
    assert conditional_entropy(dist1, 1) == 1.0

    dist1={(0, 0):1/8, (0, 1):1/8, (1, 0):1/8, (1, 1):5/8}
    h_0 = conditional_entropy(dist1, 0)
    assert np.isclose(h_0, (1/4)*(.5*np.log2(2) + .5*np.log2(2))
                        +(3/4)*((1/6)*np.log2(6) + (5/6)*np.log2(6/5))
                    )
    h_1 = conditional_entropy(dist1, 1)
    assert np.isclose(h_1, (1/4)*(.5*np.log2(2) + .5*np.log2(2))
                        +(3/4)*((1/6)*np.log2(6) + (5/6)*np.log2(6/5))
                    )

def test_mutual_information():
    '''Test Mutual Information'''
    dist1={(0, 0):.5, (1, 1):.5}
    assert mutual_information(dist1) == 1.0
    assert mutual_information2(dist1) == 1.0
    assert mutual_information3(dist1) == 1.0

    dist1={(0, 0):.25, (0, 1):.25, (1, 0):.25, (1, 1):.25}
    assert mutual_information(dist1) == 0.0
    assert mutual_information2(dist1) == 0.0
    assert mutual_information3(dist1) == 0.0

    dist1={(0, 0):1/8, (0, 1):1/8, (1, 0):1/8, (1, 1):5/8}
    entropy_1 = (1/4)*np.log2(4) + (3/4)*np.log2(4/3)
    entropy_2 = 3*(1/8)*np.log2(8) + (5/8)*np.log2(8/5)

    assert np.isclose(mutual_information(dist1), 2*entropy_1 - entropy_2)
    assert np.isclose(mutual_information2(dist1), 2*entropy_1 - entropy_2)
    assert np.isclose(mutual_information3(dist1), 2*entropy_1 - entropy_2)

def test_conditional_mutual_information():
    '''Test Conditional Mutual Information'''
    dist1={(0, 0, 0):.5, (1, 1, 1):.5}
    assert conditional_mutual_information(dist1, 2) == 0.0
    assert conditional_mutual_information2(dist1, 2) == 0.0
    assert conditional_mutual_information3(dist1, 2) == 0.0

    dist1={(0, 0, 0):.5, (1, 1, 0):.5}
    assert conditional_mutual_information(dist1, 2) == 1.0
    assert conditional_mutual_information2(dist1, 2) == 1.0
    assert conditional_mutual_information3(dist1, 2) == 1.0

    dist1={(0, 0, 0):.25, (0, 1, 0):.25, (1, 0, 0):.25, (1, 1, 0):.25}
    assert conditional_mutual_information(dist1, 2) == 0.0
    assert conditional_mutual_information2(dist1, 2) == 0.0
    assert conditional_mutual_information3(dist1, 2) == 0.0
