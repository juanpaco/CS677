from nodes import (Fixed, Beta, Normal, Poisson)
from sample import sample

A = Poisson('A', rate=4, val=5, candidate_standard_deviation=1)
B = Beta('B', alpha=(A ** 2), beta=1, val=.5, candidate_standard_deviation=.25)
C = Normal('C', mean=4, var=B, val=5, candidate_standard_deviation=1)

nodes = [ A, B, C ]

sample(nodes, num_samples=10000)

print('A reject', A.rejected)
A.mixplot(True)
A.plot_posterior(True)
B.mixplot(True)
B.plot_posterior(True)
C.mixplot(True)
C.plot_posterior(True)

