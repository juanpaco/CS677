from nodes import (Fixed, Beta, Normal, Poisson)
from sample import sample

A = Poisson('A', rate=4, val=5, candidate_standard_deviation=.5)
B = Beta('B', alpha=(A ** 2), beta=1, val=.5, candidate_standard_deviation=.5)
C = Normal('C', mean=A, var=B, val=5, candidate_standard_deviation=8)

nodes = [ A, B, C ]

sample(nodes, num_samples=10000)

print('A reject', A.rejected)
A.mixplot()
#B.mixplot()
#C.mixplot()

