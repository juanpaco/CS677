from nodes import (Fixed, Gamma, Normal, Poisson)
from sample import (sample)

c_samples = [
        3.5063,
        11.7901,
        6.3254,
        5.6002,
        12.8518,
        7.9177,
        14.6427,
        5.1759,
        9.4447,
        7.5178,
        7.9308,
        10.3868,
        6.5013,
        7.5152,
        9.1474,
        9.1673,
        12.0740,
        7.2019,
        15.8837,
        13.8034,
    ]

a_hyper = Gamma(
        'a_hyper',
        alpha=Fixed('a_hyper_alpha', val=2),
        beta=Fixed('a_hyper_beta', val=3),
        val=0.4,
        candidate_standard_deviation=1,
    )

b_hyper = Gamma(
        'b_hyper',
        alpha=Fixed('b_hyper_alpha', val=2),
        beta=Fixed('b_hyper_beta', val=3),
        val=0.4,
        candidate_standard_deviation=1,
    )

A = Poisson('A', rate=a_hyper, val=8, candidate_standard_deviation=.1)
B = Poisson('B', rate=b_hyper, val=2, candidate_standard_deviation=.1)

nodes = [ a_hyper, b_hyper, A, B ]

for s in c_samples:
    new_node = Normal(
            'C',
            mean=A,
            var=B,
            val=s,
            candidate_standard_deviation=3,
            observed=True,
        )

    nodes.append(new_node)

sample(nodes, burn=1000, num_samples=10000)

a_hyper.mixplot()
a_hyper.plot_posterior()
b_hyper.mixplot()
b_hyper.plot_posterior()
A.mixplot()
A.plot_posterior()
B.mixplot()
B.plot_posterior()
