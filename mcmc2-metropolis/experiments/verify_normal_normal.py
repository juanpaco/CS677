from nodes import (Normal)
from sample import sample

A = Normal('A', mean=0, var=1, val=0, candidate_standard_deviation=2)
B = Normal(
        'second',
        val=1.2,
        mean=A+2,
        var=1,
        observed=True,
    )

nodes = [ A, B ]

sample(nodes, num_samples=10000)

A.mixplot()
A.plot_posterior()

