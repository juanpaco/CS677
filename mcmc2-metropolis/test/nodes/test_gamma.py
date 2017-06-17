import numpy
import pytest
import random

from nodes import (Fixed, Gamma, Normal)

def test_make_gamma_node():
    node = Gamma('test node', alpha=3, beta=3)

    assert node.name == 'test node (Gamma)'
    assert node.alpha.val == 3
    assert node.beta.val == 3

def test_gamma_likelihood():
    numpy.random.seed(0)

    alpha = Fixed('alpha', val=11)
    beta = Fixed('beta', val=0.25)
    node = Gamma('test node', alpha=alpha, beta=beta, val=.25)

    assert node.likelihood() == pytest.approx(-44.27, .01)

def test_gamma_sample():
    numpy.random.seed(20)
    random.seed(1)

    alpha = Fixed('alpha', val=11)
    beta = Fixed('beta', val=0.25)
    node = Gamma('test node', alpha=alpha, beta=beta, val=2)

    node.sample()

    assert node.val == pytest.approx(2.88, .01)

def test_gamma_sample_with_children():
    numpy.random.seed(40)
    random.seed(1)

    alpha = Fixed('alpha', val=11)
    beta = Fixed('beta', val=0.25)
    parent = Gamma(
            'test node',
            alpha=alpha,
            beta=beta,
            val=.01,
            candidate_standard_deviation=3,
            )

    child = Normal(
            'test node child',
            mean=parent,
            var=1,
            val=-1,
            )

    parent.sample()

    assert parent.val == pytest.approx(.01, .001)
