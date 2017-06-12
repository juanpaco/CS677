from functools import reduce

from nodes import Bernoulli
from gibbs import sample

RushHour = Bernoulli(name='Rush Hour', ps=.25)
Rain = Bernoulli(name='Rain', ps=.1)
Traffic = Bernoulli(name='Traffic', ps=[ .05, .3, .8, .95 ])
Accident = Bernoulli(name='Accident', ps=[ .05, .15, .15, .3 ], val=1, observed=True)

RushHour.add_child(Traffic)
Traffic.add_parent(RushHour)
Rain.add_child(Traffic)
Traffic.add_parent(Rain)
Rain.add_child(Accident)
Accident.add_parent(Traffic)
Accident.add_parent(Rain)

nodes = [ RushHour, Rain, Traffic, Accident ]

samples = sample(nodes)

# Check the Rain count
count = reduce(lambda count, s: count + s[1], samples, 0)

print(count)

# P(Rush hour = true) = 0.25
# P(Rain = true) = 0.1
# P(Traffic = true | Rush hour = true & Rain = true) = 0.95
# P(Traffic = true | Rush hour = true & Rain = false) = 0.8
# P(Traffic = true | Rush hour = false & Rain = true) = 0.3
# P(Traffic = true | Rush hour = false & Rain = false) = 0.05
# P(Accident = true | Traffic = true & Rain = true) = 0.3
# P(Accident = true | Traffic = true & Rain = false) = 0.15
# P(Accident = true | Traffic = false & Rain = true) = 0.15
# P(Accident = true | Traffic = false & Rain = false) = 0.05
# 
# Find P(Rain | Accident = true)
