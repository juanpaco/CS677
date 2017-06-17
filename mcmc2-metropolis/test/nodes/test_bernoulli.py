import numpy
import pytest

from nodes import (Bernoulli, Fixed)

def test_bernoulli_likelihood_alone():
    node = Bernoulli(name='not js', ps=Fixed('ps', val=0.3))

    # log of .3
    assert node.likelihood(target=1) == pytest.approx(-1.20, abs=.01)
    # log of .7
    assert node.likelihood(target=0) == pytest.approx(-0.35, abs=.01)

def test_bernoulli_p_index():
    parent_1 = Fixed('parent 1', val=1)
    parent_2 = Fixed('parent 2', val=0)

    ps = { (1,1): 0.1, (1,0): 0.2, (0,1): 0.3, (0,0): 0.4 }

    node = Bernoulli('not js', ps=ps, parents=[ parent_1, parent_2 ])

    assert node.p_index() == (1,0)

def test_bernoulli_p_index_one_parent():
    parent_1 = Fixed('parent 1', val=1)

    ps = { (1,): 0.1, (0,): 0.2 }

    node = Bernoulli('not js', ps=ps, parents=[ parent_1 ])

    assert node.p_index() == (1,)

def test_bernoulli_p_value():
    parent_1 = Fixed('parent 1', val=1)
    parent_2 = Fixed('parent 2', val=0)

    tt = Fixed('tt', val=0.1)
    tf = Fixed('tf', val=0.2)
    ft = Fixed('ft', val=0.3)
    ff = Fixed('ff', val=0.4)

    ps = { (1,1): tt, (1,0): tf, (0,1): ft, (0,0): ff }

    node = Bernoulli('not js', ps=ps, parents= [ parent_1, parent_2 ])

    assert node.p() == ps[(1,0)].value()

def test_bernoulli_p_value_no_parents():
    node = Bernoulli('not js', ps=Fixed('ps', 0.2))

    assert node.p() == 0.2

def test_bernoulli_sample_alone():
    numpy.random.seed(0)

    node = Bernoulli(name='not js', val=0, ps=0.1)
    node.sample()

    assert node.value() == 0

def test_bernoulli_likelihood_with_parent():
    parent = Bernoulli(
            name='parent',
            ps=Fixed('parent p', val=0.3),
            val=1,
            observed=True,
        )

    ps = { (1,): Fixed('t', val=0.7), (0,): Fixed('f', val=0.3) }
    child = Bernoulli(name='child', ps=ps, parents=[ parent ])

    assert child.non_log_likelihood(target=1) == .7
    assert child.likelihood(target=1) == pytest.approx(-0.35, abs=.01)

# A   B
#  \ /
#   v
#   C
def test_bernoulli_likelihood_with_parent_2():
    A = Bernoulli(name="A", ps=Fixed('A-ps', val=.3), val=1)
    B = Bernoulli(name="B", ps=Fixed('B-ps', val=.4), val=0)

    tt = Fixed('tt', val=0.7)
    tf = Fixed('tf', val=0.2)
    ft = Fixed('ft', val=0.9)
    ff = Fixed('ff', val=0.3)

    ps = { (1,1): tt, (1,0): tf, (0,1): ft, (0,0): ff }
    C = Bernoulli(name="C", ps=ps, val=1, parents=[ A, B ])

    # log(.7) + log(.3)
    assert A.complete_conditional(target=0) == pytest.approx(-1.56, abs=.01)
    # log(.3) + log(.2)
    assert A.complete_conditional(target=1) == pytest.approx(-2.81, abs=.01)
    # log(.6) + log(.2)
    assert B.complete_conditional(target=0) == pytest.approx(-2.12, abs=.01)
    # log(.4) + log(.7)
    assert B.complete_conditional(target=1) == pytest.approx(-1.27, abs=.01)
    assert C.complete_conditional(target=0) == pytest.approx(-0.22, abs=.01)
    assert C.complete_conditional(target=1) == pytest.approx(-1.60, abs=.01)

# A   B
#  \ /
#   v
#   C
def test_bernoulli_sample_with_parent():
    numpy.random.seed(5)

    A = Bernoulli(name="A", ps=Fixed('A-ps', val=.3), val=1)
    B = Bernoulli(name="B", ps=Fixed('B-ps', val=.4), val=0)

    tt = Fixed('tt', val=0.7)
    tf = Fixed('tf', val=0.2)
    ft = Fixed('ft', val=0.9)
    ff = Fixed('ff', val=0.3)

    ps = { (1,1): tt, (1,0): tf, (0,1): ft, (0,0): ff }
    C = Bernoulli(name="C", ps=ps, val=1, parents=[ A, B ])

    assert A.sample() == 1
    assert B.sample() == 0
    assert C.sample() == 0

def test_bernoulli_sample_observed_node():
    numpy.random.seed(0)

    A = Bernoulli(name="A", ps=.99, val=0, observed=True)

    assert A.sample() == 0
