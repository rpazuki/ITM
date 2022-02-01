import numpy as np
from itm.measures import transfer_entropy
from itm.measures import conditional_transfer_entropy


def test_transfer_entropy():
    '''Test Transfer Entropy'''
    np.random.seed(43)
    x = np.random.randint(0,2, 1000000)
    y = np.r_[[0], x[:-1]]

    # For a copied r.v., we expect TE~1
    # X -> Y
    assert 1 - transfer_entropy(x, y, 1, 1) < 0.001
    # Y -> X
    assert transfer_entropy(y, x, 1, 1) < 0.001

    x = np.random.randint(0,2, 1000000)
    y = np.random.randint(0,2, 1000000)
    # For two r.v., we expect TE~0
    # X -> Y
    assert transfer_entropy(x, y, 1, 1) < 0.001
    # Y -> X
    assert transfer_entropy(y, x, 1, 1) < 0.001

def test_conditional_transfer_entropy():
    '''Test Conditional Transfer Entropy'''
    np.random.seed(43)
    x = np.random.randint(0,2, 1000000)
    y = np.r_[[0], x[:-1]]

    # For a copied r.v. conditioned on the source, we expect TE~0
    # X -> Y | X
    assert conditional_transfer_entropy(x, y, x, 1, 1, 1) < 0.001
    # Y -> X | Y
    assert conditional_transfer_entropy(y, x, y, 1, 1, 1) < 0.001

    x = np.random.randint(0,2, 1000000)
    y = np.r_[[0], x[:-1]]
    z = np.random.randint(0,2, 1000000)

    # For a copied r.v. conditioned on a random r.v., we expect TE~1
    # X -> Y | Z
    assert 1 - conditional_transfer_entropy(x, y, z, 1, 1, 1) < 0.001
    # Y -> X | Z
    assert conditional_transfer_entropy(y, x, z, 1, 1, 1) < 0.001
