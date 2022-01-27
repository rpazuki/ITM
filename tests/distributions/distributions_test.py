from distributions import empirical_dist
from distributions import marginal
from distributions import conditional
from distributions import marginal_conditional

def test_dist():
    dist1 = empirical_dist([0,1])
    assert dist1[0] == 0.5
    assert dist1[1] == 0.5
    dist1 = empirical_dist([0,1,1])
    assert dist1[0] == 1/3
    assert dist1[1] == 2/3
    dist1 = empirical_dist([0,1,0,1], [0,0,1,1])
    assert dist1[(0,0)] == .25
    assert dist1[(0,1)] == .25
    assert dist1[(1,0)] == .25
    assert dist1[(1,1)] == .25

def test_marginal():
    dist0 = {(0, 0):.25, (0, 1):.25, (1, 0):.25, (1, 1):.25}
    dist1 = marginal(dist0, 0)
    assert dist1[0] == 0.5
    assert dist1[1] == 0.5
    dist1 = marginal(dist0, 1)
    assert dist1[0] == 0.5
    assert dist1[1] == 0.5

    dist0 = empirical_dist([0,1,0,1,0,1,0,1], [0,0,1,1,0,0,1,1],[0,0,0,0,1,1,1,1])
    dist1 = marginal(dist0, 0)
    assert dist1[0] == 0.5
    assert dist1[1] == 0.5
    dist1 = marginal(dist0, 1)
    assert dist1[0] == 0.5
    assert dist1[1] == 0.5
    dist1 = marginal(dist0, 2)
    assert dist1[0] == 0.5
    assert dist1[1] == 0.5
    dist1 = marginal(dist0, 0, 1)
    assert dist1[(0,0)] == .25
    assert dist1[(0,1)] == .25
    assert dist1[(1,0)] == .25
    assert dist1[(1,1)] == .25
    dist1 = marginal(dist0, 0, 2)
    assert dist1[(0,0)] == .25
    assert dist1[(0,1)] == .25
    assert dist1[(1,0)] == .25
    assert dist1[(1,1)] == .25
    dist1 = marginal(dist0, 1, 2)
    assert dist1[(0,0)] == .25
    assert dist1[(0,1)] == .25
    assert dist1[(1,0)] == .25
    assert dist1[(1,1)] == .25
    dist1 = marginal(dist0, 2, 1)
    assert dist1[(0,0)] == .25
    assert dist1[(0,1)] == .25
    assert dist1[(1,0)] == .25
    assert dist1[(1,1)] == .25

def test_conditional():
    dist0 = {(0, 0):.25, (0, 1):.25, (1, 0):.25, (1, 1):.25}
    dist1 = conditional(dist0, 0)
    assert dist1[0][0] == 0.5
    assert dist1[0][1] == 0.5
    assert dist1[1][0] == 0.5
    assert dist1[1][1] == 0.5
    dist1 = conditional(dist0, 1)
    assert dist1[0][0] == 0.5
    assert dist1[0][1] == 0.5
    assert dist1[1][0] == 0.5
    assert dist1[1][1] == 0.5

    dist0 = empirical_dist([0,1,0,1,0,1,0,1], [0,0,1,1,0,0,1,1],[0,0,0,0,1,1,1,1])
    dist1 = conditional(dist0, 0)
    dist_given_0 = dist1[0]
    assert dist_given_0[(0,0)] == .25
    assert dist_given_0[(0,1)] == .25
    assert dist_given_0[(1,0)] == .25
    assert dist_given_0[(1,1)] == .25
    dist_given_1 = dist1[1]
    assert dist_given_1[(0,0)] == .25
    assert dist_given_1[(0,1)] == .25
    assert dist_given_1[(1,0)] == .25
    assert dist_given_1[(1,1)] == .25
    dist1 = conditional(dist0, 1)
    dist_given_0 = dist1[0]
    assert dist_given_0[(0,0)] == .25
    assert dist_given_0[(0,1)] == .25
    assert dist_given_0[(1,0)] == .25
    assert dist_given_0[(1,1)] == .25
    dist_given_1 = dist1[1]
    assert dist_given_1[(0,0)] == .25
    assert dist_given_1[(0,1)] == .25
    assert dist_given_1[(1,0)] == .25
    assert dist_given_1[(1,1)] == .25
    dist1 = conditional(dist0, 2)
    dist_given_0 = dist1[0]
    assert dist_given_0[(0,0)] == .25
    assert dist_given_0[(0,1)] == .25
    assert dist_given_0[(1,0)] == .25
    assert dist_given_0[(1,1)] == .25
    dist_given_1 = dist1[1]
    assert dist_given_1[(0,0)] == .25
    assert dist_given_1[(0,1)] == .25
    assert dist_given_1[(1,0)] == .25
    assert dist_given_1[(1,1)] == .25

    dist1 = conditional(dist0, 0, 1)
    dist_given_0_0 = dist1[(0,0)]
    assert dist_given_0_0[0] == .5
    assert dist_given_0_0[1] == .5
    dist_given_0_1 = dist1[(0,1)]
    assert dist_given_0_1[0] == .5
    assert dist_given_0_1[1] == .5
    dist_given_1_0 = dist1[(1,0)]
    assert dist_given_1_0[0] == .5
    assert dist_given_1_0[1] == .5
    dist_given_1_1 = dist1[(1,1)]
    assert dist_given_1_1[0] == .5
    assert dist_given_1_1[1] == .5

    dist1 = conditional(dist0, 0, 2)
    dist_given_0_0 = dist1[(0,0)]
    assert dist_given_0_0[0] == .5
    assert dist_given_0_0[1] == .5
    dist_given_0_1 = dist1[(0,1)]
    assert dist_given_0_1[0] == .5
    assert dist_given_0_1[1] == .5
    dist_given_1_0 = dist1[(1,0)]
    assert dist_given_1_0[0] == .5
    assert dist_given_1_0[1] == .5
    dist_given_1_1 = dist1[(1,1)]
    assert dist_given_1_1[0] == .5
    assert dist_given_1_1[1] == .5

    dist1 = conditional(dist0, 1, 2)
    dist_given_0_0 = dist1[(0,0)]
    assert dist_given_0_0[0] == .5
    assert dist_given_0_0[1] == .5
    dist_given_0_1 = dist1[(0,1)]
    assert dist_given_0_1[0] == .5
    assert dist_given_0_1[1] == .5
    dist_given_1_0 = dist1[(1,0)]
    assert dist_given_1_0[0] == .5
    assert dist_given_1_0[1] == .5
    dist_given_1_1 = dist1[(1,1)]
    assert dist_given_1_1[0] == .5
    assert dist_given_1_1[1] == .5

def test_asymetric_conditional():
    dist0 = empirical_dist([0,1,0,1,0,1,0,1], [0,0,1,1,0,0,1,1],[1,0,0,0,1,1,1,1])
    dist1 = conditional(dist0, 0)
    dist_given_0 = dist1[0]
    assert dist_given_0[(0,1)] == .5
    assert dist_given_0[(1,0)] == .25
    assert dist_given_0[(1,1)] == .25
    dist_given_1 = dist1[1]
    assert dist_given_1[(0,0)] == .25
    assert dist_given_1[(0,1)] == .25
    assert dist_given_1[(1,0)] == .25
    assert dist_given_1[(1,1)] == .25

    dist1 = conditional(dist0, 1)
    dist_given_0 = dist1[0]
    assert dist_given_0[(0,1)] == .5
    assert dist_given_0[(1,0)] == .25
    assert dist_given_0[(1,1)] == .25
    dist_given_1 = dist1[1]
    assert dist_given_1[(0,0)] == .25
    assert dist_given_1[(0,1)] == .25
    assert dist_given_1[(1,0)] == .25
    assert dist_given_1[(1,1)] == .25

def test_marginal_conditional():
    dist_f = empirical_dist([0,1,0,1,0,1,0,1], [0,0,1,1,0,0,1,1],[0,0,0,0,1,1,1,1])
    dist0 = conditional(dist_f, 0)
    # dist0 is conditioned on the first column, so the conditional
    # is a joint dist with two r.v. and we can marginalise on each
    dist1 = marginal_conditional(dist0, 0)
    dist_given_0 = dist1[0]
    assert dist_given_0[0] == .5
    assert dist_given_0[1] == .5
    dist_given_1 = dist1[1]
    assert dist_given_1[0] == .5
    assert dist_given_1[1] == .5
    dist1 = marginal_conditional(dist0, 1)
    dist_given_0 = dist1[0]
    assert dist_given_0[0] == .5
    assert dist_given_0[1] == .5
    dist_given_1 = dist1[1]
    assert dist_given_1[0] == .5
    assert dist_given_1[1] == .5

    dist_f = empirical_dist([0,1,0,1,0,1,0,1], [0,0,1,1,0,0,1,1],[0,0,0,0,1,1,1,1])
    dist0 = conditional(dist_f, 1)
    # dist0 is conditioned on the first column, so the conditional
    # is a joint dist with two r.v. and we can marginalise on each
    dist1 = marginal_conditional(dist0, 0)
    dist_given_0 = dist1[0]
    assert dist_given_0[0] == .5
    assert dist_given_0[1] == .5
    dist_given_1 = dist1[1]
    assert dist_given_1[0] == .5
    assert dist_given_1[1] == .5
    dist1 = marginal_conditional(dist0, 1)
    dist_given_0 = dist1[0]
    assert dist_given_0[0] == .5
    assert dist_given_0[1] == .5
    dist_given_1 = dist1[1]
    assert dist_given_1[0] == .5
    assert dist_given_1[1] == .5

    dist_f = empirical_dist([0,1,0,1,0,1,0,1], [0,0,1,1,0,0,1,1],[0,0,0,0,1,1,1,1])
    dist0 = conditional(dist_f, 2)
    # dist0 is conditioned on the first column, so the conditional
    # is a joint dist with two r.v. and we can marginalise on each
    dist1 = marginal_conditional(dist0, 0)
    dist_given_0 = dist1[0]
    assert dist_given_0[0] == .5
    assert dist_given_0[1] == .5
    dist_given_1 = dist1[1]
    assert dist_given_1[0] == .5
    assert dist_given_1[1] == .5
    dist1 = marginal_conditional(dist0, 1)
    dist_given_0 = dist1[0]
    assert dist_given_0[0] == .5
    assert dist_given_0[1] == .5
    dist_given_1 = dist1[1]
    assert dist_given_1[0] == .5
    assert dist_given_1[1] == .5

    dist_f = empirical_dist([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1], 
                     [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                     [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                     [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
    dist0 = conditional(dist_f, 0, 1)
    # dist0 is conditioned on the first and second columns, so the conditional
    # is a joint dist with two r.v. and we can marginalise on each
    dist1 = marginal_conditional(dist0, 0)
    dist_given_0_0 = dist1[(0,0)]
    assert dist_given_0_0[0] == .5
    assert dist_given_0_0[1] == .5
    dist_given_0_1 = dist1[(0,1)]
    assert dist_given_0_1[0] == .5
    assert dist_given_0_1[1] == .5
    dist_given_1_0 = dist1[(1,0)]
    assert dist_given_1_0[0] == .5
    assert dist_given_1_0[1] == .5
    dist_given_1_1 = dist1[(1,1)]
    assert dist_given_1_1[0] == .5
    assert dist_given_1_1[1] == .5
    dist1 = marginal_conditional(dist0, 1)
    dist_given_0_0 = dist1[(0,0)]
    assert dist_given_0_0[0] == .5
    assert dist_given_0_0[1] == .5
    dist_given_0_1 = dist1[(0,1)]
    assert dist_given_0_1[0] == .5
    assert dist_given_0_1[1] == .5
    dist_given_1_0 = dist1[(1,0)]
    assert dist_given_1_0[0] == .5
    assert dist_given_1_0[1] == .5
    dist_given_1_1 = dist1[(1,1)]
    assert dist_given_1_1[0] == .5
    assert dist_given_1_1[1] == .5


    dist_f = empirical_dist([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1], 
                     [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                     [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                     [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
    dist0 = conditional(dist_f, 0, 2)
    # dist0 is conditioned on the first and second columns, so the conditional
    # is a joint dist with two r.v. and we can marginalise on each
    dist1 = marginal_conditional(dist0, 0)
    dist_given_0_0 = dist1[(0,0)]
    assert dist_given_0_0[0] == .5
    assert dist_given_0_0[1] == .5
    dist_given_0_1 = dist1[(0,1)]
    assert dist_given_0_1[0] == .5
    assert dist_given_0_1[1] == .5
    dist_given_1_0 = dist1[(1,0)]
    assert dist_given_1_0[0] == .5
    assert dist_given_1_0[1] == .5
    dist_given_1_1 = dist1[(1,1)]
    assert dist_given_1_1[0] == .5
    assert dist_given_1_1[1] == .5
    dist1 = marginal_conditional(dist0, 1)
    dist_given_0_0 = dist1[(0,0)]
    assert dist_given_0_0[0] == .5
    assert dist_given_0_0[1] == .5
    dist_given_0_1 = dist1[(0,1)]
    assert dist_given_0_1[0] == .5
    assert dist_given_0_1[1] == .5
    dist_given_1_0 = dist1[(1,0)]
    assert dist_given_1_0[0] == .5
    assert dist_given_1_0[1] == .5
    dist_given_1_1 = dist1[(1,1)]
    assert dist_given_1_1[0] == .5
    assert dist_given_1_1[1] == .5

    dist_f = empirical_dist([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1], 
                     [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                     [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                     [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
    dist0 = conditional(dist_f, 1, 2)
    # dist0 is conditioned on the first and second columns, so the conditional
    # is a joint dist with two r.v. and we can marginalise on each
    dist1 = marginal_conditional(dist0, 0)
    dist_given_0_0 = dist1[(0,0)]
    assert dist_given_0_0[0] == .5
    assert dist_given_0_0[1] == .5
    dist_given_0_1 = dist1[(0,1)]
    assert dist_given_0_1[0] == .5
    assert dist_given_0_1[1] == .5
    dist_given_1_0 = dist1[(1,0)]
    assert dist_given_1_0[0] == .5
    assert dist_given_1_0[1] == .5
    dist_given_1_1 = dist1[(1,1)]
    assert dist_given_1_1[0] == .5
    assert dist_given_1_1[1] == .5
    dist1 = marginal_conditional(dist0, 1)
    dist_given_0_0 = dist1[(0,0)]
    assert dist_given_0_0[0] == .5
    assert dist_given_0_0[1] == .5
    dist_given_0_1 = dist1[(0,1)]
    assert dist_given_0_1[0] == .5
    assert dist_given_0_1[1] == .5
    dist_given_1_0 = dist1[(1,0)]
    assert dist_given_1_0[0] == .5
    assert dist_given_1_0[1] == .5
    dist_given_1_1 = dist1[(1,1)]
    assert dist_given_1_1[0] == .5
    assert dist_given_1_1[1] == .5


    dist_f = empirical_dist([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1], 
                     [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                     [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                     [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
    dist0 = conditional(dist_f, 0)
    dist1 = marginal_conditional(dist0, 0, 1)
    dist_given_0 = dist1[0]
    assert dist_given_0[(0,0)] == .25
    assert dist_given_0[(0,1)] == .25
    assert dist_given_0[(1,0)] == .25
    assert dist_given_0[(1,1)] == .25
    dist_given_1 = dist1[1]
    assert dist_given_1[(0,0)] == .25
    assert dist_given_1[(0,1)] == .25
    assert dist_given_1[(1,0)] == .25
    assert dist_given_1[(1,1)] == .25
    dist1 = marginal_conditional(dist0, 0, 2)
    dist_given_0 = dist1[0]
    assert dist_given_0[(0,0)] == .25
    assert dist_given_0[(0,1)] == .25
    assert dist_given_0[(1,0)] == .25
    assert dist_given_0[(1,1)] == .25
    dist_given_1 = dist1[1]
    assert dist_given_1[(0,0)] == .25
    assert dist_given_1[(0,1)] == .25
    assert dist_given_1[(1,0)] == .25
    assert dist_given_1[(1,1)] == .25
    dist1 = marginal_conditional(dist0, 2, 1)
    dist_given_0 = dist1[0]
    assert dist_given_0[(0,0)] == .25
    assert dist_given_0[(0,1)] == .25
    assert dist_given_0[(1,0)] == .25
    assert dist_given_0[(1,1)] == .25
    dist_given_1 = dist1[1]
    assert dist_given_1[(0,0)] == .25
    assert dist_given_1[(0,1)] == .25
    assert dist_given_1[(1,0)] == .25
    assert dist_given_1[(1,1)] == .25


    dist0 = conditional(dist_f, 1)
    dist1 = marginal_conditional(dist0, 0, 1)
    dist_given_0 = dist1[0]
    assert dist_given_0[(0,0)] == .25
    assert dist_given_0[(0,1)] == .25
    assert dist_given_0[(1,0)] == .25
    assert dist_given_0[(1,1)] == .25
    dist_given_1 = dist1[1]
    assert dist_given_1[(0,0)] == .25
    assert dist_given_1[(0,1)] == .25
    assert dist_given_1[(1,0)] == .25
    assert dist_given_1[(1,1)] == .25
    dist1 = marginal_conditional(dist0, 0, 2)
    dist_given_0 = dist1[0]
    assert dist_given_0[(0,0)] == .25
    assert dist_given_0[(0,1)] == .25
    assert dist_given_0[(1,0)] == .25
    assert dist_given_0[(1,1)] == .25
    dist_given_1 = dist1[1]
    assert dist_given_1[(0,0)] == .25
    assert dist_given_1[(0,1)] == .25
    assert dist_given_1[(1,0)] == .25
    assert dist_given_1[(1,1)] == .25
    dist1 = marginal_conditional(dist0, 2, 1)
    dist_given_0 = dist1[0]
    assert dist_given_0[(0,0)] == .25
    assert dist_given_0[(0,1)] == .25
    assert dist_given_0[(1,0)] == .25
    assert dist_given_0[(1,1)] == .25
    dist_given_1 = dist1[1]
    assert dist_given_1[(0,0)] == .25
    assert dist_given_1[(0,1)] == .25
    assert dist_given_1[(1,0)] == .25
    assert dist_given_1[(1,1)] == .25

    dist0 = conditional(dist_f, 2)
    dist1 = marginal_conditional(dist0, 0, 1)
    dist_given_0 = dist1[0]
    assert dist_given_0[(0,0)] == .25
    assert dist_given_0[(0,1)] == .25
    assert dist_given_0[(1,0)] == .25
    assert dist_given_0[(1,1)] == .25
    dist_given_1 = dist1[1]
    assert dist_given_1[(0,0)] == .25
    assert dist_given_1[(0,1)] == .25
    assert dist_given_1[(1,0)] == .25
    assert dist_given_1[(1,1)] == .25
    dist1 = marginal_conditional(dist0, 0, 2)
    dist_given_0 = dist1[0]
    assert dist_given_0[(0,0)] == .25
    assert dist_given_0[(0,1)] == .25
    assert dist_given_0[(1,0)] == .25
    assert dist_given_0[(1,1)] == .25
    dist_given_1 = dist1[1]
    assert dist_given_1[(0,0)] == .25
    assert dist_given_1[(0,1)] == .25
    assert dist_given_1[(1,0)] == .25
    assert dist_given_1[(1,1)] == .25
    dist1 = marginal_conditional(dist0, 2, 1)
    dist_given_0 = dist1[0]
    assert dist_given_0[(0,0)] == .25
    assert dist_given_0[(0,1)] == .25
    assert dist_given_0[(1,0)] == .25
    assert dist_given_0[(1,1)] == .25
    dist_given_1 = dist1[1]
    assert dist_given_1[(0,0)] == .25
    assert dist_given_1[(0,1)] == .25
    assert dist_given_1[(1,0)] == .25
    assert dist_given_1[(1,1)] == .25

    dist1 = marginal_conditional(dist0, 0, 1, 2)
    dist_given_0 = dist1[0]
    assert dist_given_0[(0,0,0)] == .125
    assert dist_given_0[(0,0,1)] == .125
    assert dist_given_0[(0,1,0)] == .125
    assert dist_given_0[(0,1,1)] == .125
    assert dist_given_0[(1,0,0)] == .125
    assert dist_given_0[(1,0,1)] == .125
    assert dist_given_0[(1,1,0)] == .125
    assert dist_given_0[(1,1,1)] == .125
    dist_given_1 = dist1[1]
    assert dist_given_1[(0,0,0)] == .125
    assert dist_given_1[(0,0,1)] == .125
    assert dist_given_1[(0,1,0)] == .125
    assert dist_given_1[(0,1,1)] == .125
    assert dist_given_1[(1,0,0)] == .125
    assert dist_given_1[(1,0,1)] == .125
    assert dist_given_1[(1,1,0)] == .125
    assert dist_given_1[(1,1,1)] == .125