import numpy
import pytest

from nodes import (Bernoulli)

def test_likelihood_bernoulli_alone():
    node = Bernoulli(name='not js', ps=0.3)

    assert node.likelihood(target=1) == .3
    assert node.likelihood(target=0) == .7

def test_sample_bernoulli_alone():
    numpy.random.seed(0)

    node = Bernoulli(name='not js', ps=0.1)
    node.sample()

    assert node.val == 0

def test_likelihood_bernoulli_with_parent():
    parent = Bernoulli(name='parent', ps=0.3, val=1, observed=True)
    child = Bernoulli(name='child', ps=[ 0.2, 0.7 ])
    child.add_parent(parent)

    assert child.likelihood(target=1) == .7

# A   B
#  \ /
#   v
#   C
def test_likelihood_bernoulli_with_parent_2():
    A = Bernoulli(name="A", ps=.3, val=1)
    B = Bernoulli(name="B", ps=.4, val=0)
    C = Bernoulli(name="C", ps=[ .3, .9, .2, .7 ], val=1)

    A.add_child(C)
    B.add_child(C)
    C.add_parent(A)
    C.add_parent(B)

    # .7 * .3
    assert A.complete_conditional(target=0) == pytest.approx(.21, abs=.01)
    # .3 * .2
    assert A.complete_conditional(target=1) == pytest.approx(.06, abs=.01)
    # .6 * .2
    assert B.complete_conditional(target=0) == pytest.approx(.12, abs=.01)
    # .4 * .7
    assert B.complete_conditional(target=1) == pytest.approx(.28, abs=.01)
    assert C.complete_conditional(target=0) == .8
    assert C.complete_conditional(target=1) == .2

# A   B
#  \ /
#   v
#   C
def test_sample_bernoulli_with_parent():
    numpy.random.seed(5)

    A = Bernoulli(name="A", ps=.7, val=1)
    B = Bernoulli(name="B", ps=.4, val=0)
    C = Bernoulli(name="C", ps=[ .3, .9, .2, .7 ], val=0)

    A.add_child(C)
    B.add_child(C)
    C.add_parent(A)
    C.add_parent(B)

    assert A.sample() == 1
    assert B.sample() == 1
    assert C.sample() == 1

def test_sample_observed_node():
    numpy.random.seed(0)

    A = Bernoulli(name="A", ps=.99, val=0, observed=True)

    assert A.sample() == 0
