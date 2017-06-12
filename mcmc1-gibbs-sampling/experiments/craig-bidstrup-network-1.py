from functools import reduce

from nodes import Bernoulli
from gibbs import sample

TreeFalls = Bernoulli(name='Tree Falls', ps=.02)
InTheWoods = Bernoulli(name='InTheWoods', ps=.4)
HearsTree = Bernoulli(name='HearsTree', ps=[ .003, .7, .2, .8 ], val=0, observed=True)

TreeFalls.add_child(HearsTree)
HearsTree.add_parent(TreeFalls)
InTheWoods.add_child(HearsTree)
HearsTree.add_parent(InTheWoods)

nodes = [ TreeFalls, InTheWoods, HearsTree ]

samples = sample(nodes)

#P(A|C=false)
# Check the TreeFalls
count = reduce(lambda count, s: count + s[0], samples, 0)

print(count)
