import math
import numpy

from nodes import (
        Bernoulli,
        Beta,
        Fixed,
        Gamma,
        InverseGamma,
        Normal,
        Poisson,
        )
from sample import (sample)

#A = NormalNode(mean=20, var=1)
#E = BetaNode(alpha=1, beta=1)
#B = GammaNode(shape=A**pi, invscale=7)
#D = BetaNode(alpha=A, beta=E)
#C = BernoulliNode(D: True, (1-D): False)
#F = PoissonNode(rate=D)
#G = NormalNode(mean=E, var=F)

# Cannot tune A for the life of me.
A = Normal('A', mean=20, var=1, val=20, candidate_standard_deviation=0.05)
E = Beta('E', alpha=1, beta=1, val=.5, candidate_standard_deviation=0.1)
B = Gamma(
        'B',
        alpha=A**math.pi,
        beta=7,
        val=21000,
        candidate_standard_deviation=40,
    )
#D = Beta('D', alpha=A, beta=E, val=0.9, candidate_standard_deviation=.025)
#C = Bernoulli('C', ps=D, val=1)
#F = Poisson('F', rate=D, val=2, candidate_standard_deviation=.5)
#G = Normal('G', mean=E, var=F, val=E.value(), candidate_standard_deviation=5)
#G = Normal('G', mean=E, var=F, val=5, candidate_standard_deviation=1, observed=True)

nodes = [ A, E, B]#, D, C, F, G ]

sample(nodes, burn=10000, num_samples=10000)

for node in nodes:
    print(node.name, 'rejected', node.rejected)
    print(node.name, 'accepted', node.accepted)
    print(node.name, 'stayed', node.stayed)
    node.mixplot()
    #node.plot_posterior()
