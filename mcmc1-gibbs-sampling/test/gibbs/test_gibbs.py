import numpy
import pytest

from nodes import (Bernoulli)
from gibbs import tick

def test_tick():
    numpy.random.seed(0)

    A = Bernoulli(name="A", ps=.7, val=1)
    B = Bernoulli(name="B", ps=.4, val=0)
    C = Bernoulli(name="C", ps=[ .3, .9, .2, .7 ], val=0)

    A.add_child(C)
    B.add_child(C)
    C.add_parent(A)
    C.add_parent(B)

    nodes = [ A, B, C ]

    tick(nodes)

    assert A.val == 1
    assert B.val == 1
    assert C.val == 1
#def test_tick_that_should_have_a_false():
#    numpy.random.seed(0)
#
#    network = [
#            create_clamped(0),
#            create_bernoulli(ps=[ .1, .75 ], parents=[ 0 ]),
#        ]
#
#    next_vals = tick(network)
#
#    assert next_vals[0] == 0
#    assert next_vals[1] == 0
