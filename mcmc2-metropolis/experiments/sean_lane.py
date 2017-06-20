from nodes import (Beta, Fixed, Gamma, Normal)
from sample import (sample)

A = Gamma('A', alpha=2, beta=2, val=0.5, candidate_standard_deviation=1)
B = Beta('B', alpha=3, beta=1, val=1, candidate_standard_deviation=0.25)
C = Normal('C', mean=A, var=B, val=A.value(), candidate_standard_deviation=1.25)

nodes = [ A, B, C ]

sample(nodes, burn=1000, num_samples=10000)

A.mixplot()
A.plot_posterior()
B.mixplot()
B.plot_posterior()
C.mixplot()
C.plot_posterior()
