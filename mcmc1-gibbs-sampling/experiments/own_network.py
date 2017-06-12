from functools import reduce

from nodes import Bernoulli
from gibbs import sample

BigShotPresent = Bernoulli(name='BigShotPresent', ps=.05)
DemoFailed = Bernoulli(name='DemoFailed', ps=[ .4, .9 ])
DeletedProductionDb = Bernoulli(name='DeletedProductionDb', ps=.2)
Fired = Bernoulli(name='Fired', ps=[ .001, .9, .2, .999 ], val=1, observed=True)
#Fired = Bernoulli(name='Fired', ps=[ .001, .9, .2, .999 ])

BigShotPresent.add_child(DemoFailed)
DemoFailed.add_parent(BigShotPresent)

DeletedProductionDb.add_child(Fired)
Fired.add_child(DeletedProductionDb)

DemoFailed.add_child(Fired)
Fired.add_parent(DemoFailed)
DeletedProductionDb.add_child(Fired)
Fired.add_parent(DeletedProductionDb)

nodes = [ BigShotPresent, DemoFailed, DeletedProductionDb, Fired ]

samples = sample(nodes)

#p(DemoFailed = true | Fired = true)
count = reduce(lambda count, s: count + s[1], samples, 0)

print(count)

# p(BigShotPresent = true) = .05
#p(DemoFailed = true | BigShotPresent = true) = .9
#p(DemoFailed = true | BigShotPresent = false) = .4
#p(DeletedProductionDatabase = true) = .2
#p(Fired | DemoFailed = false, DeletedProductionDatabase = false) = .001
#p(Fired | DemoFailed = false, DeletedProductionDatabase = true) = .9
#p(Fired | DemoFailed = true, DeletedProductionDatabase = false) = .2
#p(Fired | DemoFailed = true, DeletedProductionDatabase = true) = .999
