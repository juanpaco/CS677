import numpy
import pytest
import random

from nodes import (Fixed, Normal)

def test_make_normal_node():
    node = Normal('test node', mean = 3, var = 3)

    assert node.name == 'test node (Normal)'
    assert node.mean.val == 3
    assert node.var.val == 3

def test_normal_likelihood():
    numpy.random.seed(0)

    mean = Fixed('mean', val=0)
    var = Fixed('var', val=1)
    node = Normal('test node', mean=mean, var=var, val=0)

    # ln(1/sqrt(2*pi))
    assert node.likelihood(node.val) == pytest.approx(-0.91, .01)

def test_normal_sample():
    numpy.random.seed(0)
    random.seed(0)

    mean = Fixed('mean', val=1)
    var = Fixed('var', val=1)
    node = Normal(
            'test node',
            mean=mean,
            var=var,
            val=0
            )

    node.sample()

    assert node.val == pytest.approx(1.76, .01)

def test_normal_sample_with_children():
    numpy.random.seed(0)
    random.seed(0)

    mean = Fixed('mean', val=1)
    var = Fixed('var', val=1)
    parent = Normal(
            'test node parent',
            mean=mean,
            var=var,
            val=0
            )

    child = Normal(
            'test node child',
            mean=parent,
            var=var,
            val=-1,
            )

    parent.sample()

    assert parent.val == 0

