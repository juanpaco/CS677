from nodes import (Beta, Fixed, InverseGamma, Normal)
from sample import (sample)

A = Normal('A', mean=0.3, var=0.2, val=0.3, candidate_standard_deviation=0.7)
B = InverseGamma('B', alpha=11, beta=0.5, val=0.04, candidate_standard_deviation=.15)
C = Normal('C', mean=A, var=B, val=10, candidate_standard_deviation=0.05)
D = Beta('D', alpha=10, beta=1, val=1, candidate_standard_deviation=0.4)
E = Normal('E', mean=D, var=C, val=1, candidate_standard_deviation=20)

nodes = [ A, B, C, D, E ]

sample(nodes, burn=1000, num_samples=10000)

#A.mixplot()
#A.plot_posterior()
#B.mixplot()
#B.plot_posterior()
C.mixplot()
#C.plot_posterior()
#D.mixplot()
#D.plot_posterior()
#E.mixplot()
#E.plot_posterior()
