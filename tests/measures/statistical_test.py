import numpy as np
from numpy.random import default_rng
from itm.measures.statistical import correlation
from itm.measures.statistical import autocorrelation

def test_correlation():
    '''Test correlation function'''
    rng = default_rng(43)
    x = rng.standard_normal(100)
    y = x
    assert np.isclose(correlation(x,y), 1.0)
    y = -x
    assert np.isclose(correlation(x,y), -1.0)
    x = rng.standard_normal(1000)
    y = rng.standard_normal(1000)
    assert correlation(x,y) < 0.001

def test_autocorrelation():
    '''Test autocorrelation function'''
    rng = default_rng(43)
    x = rng.standard_normal(10000)
    autocorrelation_1 = autocorrelation(x, 1)
    assert autocorrelation_1[0] < 0.01
    autocorrelation_4 = autocorrelation(x, 4)
    assert autocorrelation_4[0] < 0.01
    assert autocorrelation_4[1] < 0.01
    assert autocorrelation_4[2] < 0.01
    assert autocorrelation_4[3] < 0.01
