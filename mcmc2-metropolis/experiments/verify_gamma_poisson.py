from nodes import (Fixed, Gamma, Poisson)
from sample import sample

lambduh = Gamma('lambduh', alpha=2, beta=3, val=0.4, candidate_standard_deviation=1)

X = Poisson('X', rate=lambduh, val=3, observed=False)

nodes = [ lambduh, X ]

sample(nodes, num_samples=10000)

#lambduh.mixplot()
#lambduh.plot_posterior()

print('x rejected', X.rejected)
X.mixplot()


