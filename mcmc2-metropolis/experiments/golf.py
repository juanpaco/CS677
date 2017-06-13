import numpy

from nodes import (Fixed, InverseGamma, Normal)
from sample import (sample)

def load_line(line):
    elements = line.rstrip().split()

    return (elements[0], float(elements[1]), elements[2])

def load_data():
  return [ load_line(line) for line in open('experiments/golfdataR.dat') ]

data = load_data()
golfers = set([ datum[0] for datum in data])
tournaments = set([ datum[2] for datum in data])

nodes = []

hyper_tourn_mean = Normal(
    'Tournament hyper mean',
    mean=72,
    var=2,
    val=0,
    candidate_standard_deviation=2,
)
nodes.append(hyper_tourn_mean)

hyper_tourn_var = InverseGamma(
    'Tournament hyper variance',
    alpha=18,
    beta=.015,
    val=3,
    candidate_standard_deviation=2,
)
nodes.append(hyper_tourn_var)

tournament_means = {}
for tournament in tournaments:
    tournament_means[tournament] = Normal(
            tournament + ' mean',
            mean=hyper_tourn_mean,
            var=hyper_tourn_var,
            val=0,
            candidate_standard_deviation=1.414,
        )

    nodes.append(tournament_means[tournament])

hyper_golfer_var = InverseGamma(
    'Golfer hyper variance',
    alpha=18,
    beta=.015,
    val=3,
    candidate_standard_deviation=2,
)
nodes.append(hyper_golfer_var)

golfer_means = {}
for golfer in golfers:
    golfer_means[golfer] = Normal(
            golfer + ' mean',
            mean=0,
            var=hyper_golfer_var,
            val=0,
            candidate_standard_deviation=1.414,
        )

    nodes.append(golfer_means[golfer])

obsvar = InverseGamma(
    'obsvar',
    alpha=72,
    beta=.0014,
    val=3,
    candidate_standard_deviation=2,
)
nodes.append(obsvar)

for (name, score, tourn) in data:
    node = Normal(
            'Result ' + name + ' ' + tourn,
            mean=tournament_means[tourn] + golfer_means[name],
            var=obsvar,
            observed=True,
            val=score,
        )

    nodes.append(node)

print(obsvar.children[-1].name)



#estimated_mean = numpy.mean(data)
#estimated_var = numpy.var(data)
#
#nodes = []
#
#mean = Normal(
#        'mean',
#        mean=5,
#        var=1/9,
#        val=estimated_mean,
#        candidate_standard_deviation=1/3,
#    )
#nodes.append(mean)
#
#variance = InverseGamma(
#        'variance',
#        alpha=11,
#        beta=.25,
#        val=estimated_var,
#        candidate_standard_deviation=.387,
#    )
#nodes.append(variance)
#
#for datum in data:
#    node = Normal(
#            'data - ' + str(datum),
#            mean=mean,
#            var=variance,
#            val=datum,
#            observed=True
#        )
#    nodes.append(node)
#
#sample(nodes, num_samples=10000)
#
##mean.mixplot()
#mean.plot_posterior()
