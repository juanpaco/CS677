import numpy
import pytest
import random

from nodes import (Fixed, Poisson, Normal)

def test_make_poisson_node():
    node = Poisson('test node', rate=3)

    assert node.name == 'test node (Poisson)'
    assert node.rate.value() == 3

def test_poisson_likelihood_of_neg():
    numpy.random.seed(0)

    rate = Fixed('rate', val=3)
    node = Poisson('test node', rate=rate, val=-2)

    assert node.likelihood() == 0

def test_poisson_likelihood():
    numpy.random.seed(0)

    rate = Fixed('rate', val=3)
    node = Poisson('test node', rate=rate, val=2)

    assert node.likelihood() == pytest.approx(-1.49, .01)

def test_poisson_sample():
    numpy.random.seed(10)
    random.seed(1)

    rate = Fixed('rate', val=3)
    node = Poisson('test node', rate=rate, val=4)

    node.sample()

    assert node.value() == pytest.approx(5, .01)

#def test_poisson_sample_with_children():
#    numpy.random.seed(40)
#    random.seed(1)
#
#    alpha = Fixed('alpha', val=11)
#    beta = Fixed('beta', val=0.25)
#    parent = Poisson(
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
