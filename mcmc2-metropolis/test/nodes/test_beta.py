import numpy
import pytest
import random

from nodes import (Fixed, Beta, Normal)

def test_make_beta_node():
    node = Beta('test node', alpha=3, beta=3)

    assert node.name == 'test node (Beta)'
    assert node.alpha.val == 3
    assert node.beta.val == 3

def test_beta_likelihood_of_neg_alpha():
    numpy.random.seed(0)

    alpha = Fixed('alpha', val=-1)
    beta = Fixed('beta', val=0.25)
    node = Beta('test node', alpha=alpha, beta=beta, val=.25)

    assert node.likelihood() == 0

def test_beta_likelihood_of_neg_beta():
    numpy.random.seed(0)

    alpha = Fixed('alpha', val=1)
    beta = Fixed('beta', val=-1)
    node = Beta('test node', alpha=alpha, beta=beta, val=.25)

    assert node.likelihood() == 0

def test_beta_likelihood():
    alpha = Fixed('alpha', val=11)
    beta = Fixed('beta', val=0.25)
    node = Beta('test node', alpha=alpha, beta=beta, val=.25)

    assert node.likelihood() == pytest.approx(-14.34, .01)

def test_beta_sample():
    numpy.random.seed(20)
    random.seed(1)

    alpha = Fixed('alpha', val=11)
    beta = Fixed('beta', val=11)
    node = Beta(
            'test node',
            alpha=alpha, beta=beta,
            val=.25,
            candidate_standard_deviation=.2
        )

    node.sample()

    assert node.val == pytest.approx(.42, .1)

#def test_beta_sample_with_children():
#    numpy.random.seed(40)
#    random.seed(1)
#
#    alpha = Fixed('alpha', val=11)
#    beta = Fixed('beta', val=0.25)
#    parent = Beta(
#            'test node',
#            alpha=alpha,
#            beta=beta,
#            val=.01,
#            candidate_standard_deviation=3,
#            )
#
#    child = Normal(
#            'test node child',
#            mean=parent,
#            var=1,
#            val=-1,
#            )
#
#    parent.sample()
#
#    assert parent.val == pytest.approx(.01, .001)
