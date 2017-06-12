import numpy
import pytest
import random

from nodes import (Fixed, InverseGamma, Normal)

def test_make_inverse_gamma_node():
    node = InverseGamma('test node', alpha=3, beta=3)

    assert node.name == 'test node (Inverse Gamma)'
    assert node.alpha.val == 3
    assert node.beta.val == 3

def test_inverse_gamma_likelihood():
    numpy.random.seed(0)

    alpha = Fixed('alpha', val=11)
    beta = Fixed('beta', val=0.25)
    node = InverseGamma('test node', alpha=alpha, beta=beta, val=.25)

    assert node.likelihood() == pytest.approx(0.78, .01)

def test_inverse_gamma_sample():
    numpy.random.seed(2)
    random.seed(1)

    alpha = Fixed('alpha', val=11)
    beta = Fixed('beta', val=0.25)
    node = InverseGamma('test node', alpha=alpha, beta=beta, val=2)

    node.sample()

    assert node.val == pytest.approx(1.58, .01)

def test_inverse_gamma_sample_with_children():
    numpy.random.seed(2)
    random.seed(0)

    alpha = Fixed('alpha', val=11)
    beta = Fixed('beta', val=0.25)
    parent = InverseGamma(
            'test node',
            alpha=alpha,
            beta=beta,
            val=2,
            candidate_standard_deviation=3,
            )

    child = Normal(
            'test node child',
            mean=parent,
            var=1,
            val=-1,
            )

    parent.sample()

    assert parent.val == pytest.approx(.749, .001)
