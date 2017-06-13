import numpy

from nodes import (Fixed, InverseGamma, Normal)
from sample import (sample)

data = [float(line) for line in open('experiments/fac-data.txt')]
estimated_mean = numpy.mean(data)
estimated_var = numpy.var(data)

nodes = []

mean = Normal(
        'mean',
        mean=5,
        var=1/9,
        val=estimated_mean,
        candidate_standard_deviation=1/3,
    )
nodes.append(mean)

variance = InverseGamma(
        'variance',
        alpha=11,
        beta=.25,
        val=estimated_var,
        candidate_standard_deviation=.387,
    )
nodes.append(variance)

for datum in data:
    node = Normal(
            'data - ' + str(datum),
            mean=mean,
            var=variance,
            val=datum,
            observed=True
        )
    nodes.append(node)

sample(nodes, num_samples=10000)

#mean.mixplot()
mean.plot_posterior()
