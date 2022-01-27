import numpy as np
from measures.entropies import TE
from measures.entropies import TE_Condition


def test_transfer_entropy():
    '''Test Transfer Entropy'''
    np.random.seed(43)
    x = np.random.randint(0,2, 1000000)
    y = np.r_[[0], x[:-1]]

    # For a copied r.v., we expect TE~1
    # X -> Y
    assert 1 - TE(x, y, 1, 1) < 0.001
    # Y -> X
    assert TE(y, x, 1, 1) < 0.001

    x = np.random.randint(0,2, 1000000)
    y = np.random.randint(0,2, 1000000)
    # For two r.v., we expect TE~0
    # X -> Y
    assert TE(x, y, 1, 1) < 0.001
    # Y -> X
    assert TE(y, x, 1, 1) < 0.001

def test_conditional_transfer_entropy():
    '''Test Conditional Transfer Entropy'''
    np.random.seed(43)
    x = np.random.randint(0,2, 1000000)
    y = np.r_[[0], x[:-1]]

    # For a copied r.v. conditioned on the source, we expect TE~0
    # X -> Y | X
    assert TE_Condition(x, y, x, 1, 1, 1) < 0.001
    # Y -> X | Y
    assert TE_Condition(y, x, y, 1, 1, 1) < 0.001

    x = np.random.randint(0,2, 1000000)
    y = np.r_[[0], x[:-1]]
    z = np.random.randint(0,2, 1000000)

    # For a copied r.v. conditioned on a random r.v., we expect TE~1
    # X -> Y | Z
    assert 1 - TE_Condition(x, y, z, 1, 1, 1) < 0.001
    # Y -> X | Z
    assert TE_Condition(y, x, z, 1, 1, 1) < 0.001
