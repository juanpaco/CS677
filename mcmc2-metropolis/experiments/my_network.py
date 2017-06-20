from nodes import (Fixed, Gamma, InverseGamma, Normal, Poisson)
from sample import sample

medical_costs = [
        3.43258523,
        7.40850029,
        9.26409888,
        2.18740696,
        5.17707892,
        12.35317243,
        7.24685147,
        8.3705836,
        6.29274266,
        -0.71217019,
    ]

SoldiersKicked = Poisson('SoldiersKicked', rate=5, val=5, candidate_standard_deviation=10)
Var = Gamma('Var', alpha=11, beta=1/2, val=2.5, candidate_standard_deviation=10)

nodes = [ SoldiersKicked, Var ]

for i, val in enumerate(medical_costs):
    MedicalCosts = Normal(
            'MedicalCosts-{}'.format(i),
            mean=SoldiersKicked,
            var=Var,
            val=val,
            candidate_standard_deviation=1,
            observed=True
        )

    nodes.append(MedicalCosts)

sample(nodes, num_samples=10000)

write = True

SoldiersKicked.mixplot(write=write)
SoldiersKicked.plot_posterior(write=write)
Var.mixplot(write=write)
Var.plot_posterior(write=write)
