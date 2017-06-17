from nodes import (Bernoulli, Beta, Fixed)
from sample import sample

B = Beta('B', alpha=2, beta=3, val=0.3, candidate_standard_deviation=.1)

X = Bernoulli('X', ps=B, val=0, observed=True)

nodes = [ B, X ]

sample(nodes, num_samples=10000)

print('rejected:', B.rejected)

B.mixplot()
B.plot_posterior()
