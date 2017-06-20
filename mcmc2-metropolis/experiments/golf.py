import numpy

from nodes import (Fixed, InverseGamma, Normal)
from sample import (sample)

nsamples = 10000

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
    val=72,
    candidate_standard_deviation=.20,
)
nodes.append(hyper_tourn_mean)

hyper_tourn_var = InverseGamma(
    'Tournament hyper variance',
    alpha=18,
    beta=.015,
    val=.00075,
    candidate_standard_deviation=.8,
)
nodes.append(hyper_tourn_var)

tournament_means = {}
for tournament in tournaments:
    tournament_means[tournament] = Normal(
            tournament + ' mean',
            mean=hyper_tourn_mean,
            var=hyper_tourn_var,
            val=72,
            candidate_standard_deviation=1.414,
        )

    nodes.append(tournament_means[tournament])

hyper_golfer_var = InverseGamma(
    'Golfer hyper variance',
    alpha=18,
    beta=.015,
    val=.00075,
    candidate_standard_deviation=.5,
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
    alpha=83,
    beta=.0014,
    val=.000015,
    candidate_standard_deviation=.04,
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

sample(nodes, burn=1000, num_samples=nsamples)

ability = []
for name, mean in golfer_means.items():
    mean.mixplot()
    mean.plot_posterior()
    samples = mean.posteriors[:]
    samples.sort()
    median = samples[nsamples//2]
    low = samples[int(.05 * nsamples)]
    high = samples[int(.95 * nsamples)]
    ability.append( (name, low, median, high) )
    
ability.sort(key=lambda a: a[2])
i = 1
for golfer, low, median, high in ability:
    print('{}: {} {}; 90%% interval: ({}, {})'.format(i, golfer, median, low, high))
    i += 1

hyper_tourn_mean.mixplot()
hyper_tourn_mean.plot_posterior()
hyper_tourn_var.mixplot()
hyper_tourn_var.plot_posterior()
hyper_golfer_var.mixplot()
hyper_golfer_var.plot_posterior()
obsvar.mixplot()
obsvar.plot_posterior()

