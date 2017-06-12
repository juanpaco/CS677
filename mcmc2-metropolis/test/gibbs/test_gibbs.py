import numpy
import pytest

from nodes import (Bernoulli)
from gibbs import tick

def test_tick():
    numpy.random.seed(4)

    A = Bernoulli(name="A", ps=.7, val=1)
    B = Bernoulli(name="B", ps=.4, val=0)
    C = Bernoulli(name="C", ps=[ .3, .9, .2, .7 ], val=0)

    A.add_child(C)
    B.add_child(C)
    C.add_parent(A)
    C.add_parent(B)

    nodes = [ A, B, C ]

    tick(nodes)

    assert A.val == 0
    assert B.val == 0
    assert C.val == 1
