from functools import reduce

from nodes import Bernoulli
from gibbs import sample

Burglary = Bernoulli(name='Burglary', ps=.001)
Earthquake = Bernoulli(name='Earthquake', ps=.002)
Alarm = Bernoulli(name='Alarm', ps=[ .001, .29, .94, .95 ])
JohnCalls = Bernoulli(name='JohnCalls', ps=[ .05, .90 ], observed=False)
MaryCalls = Bernoulli(name='MaryCalls', ps=[ .01, .70 ], val=1, observed=True)

Burglary.add_child(Alarm)
Alarm.add_parent(Burglary)
Earthquake.add_child(Alarm)
Alarm.add_parent(Earthquake)

Alarm.add_child(JohnCalls)
JohnCalls.add_parent(Alarm)
Alarm.add_child(MaryCalls)
MaryCalls.add_parent(Alarm)

nodes = [ Burglary, Earthquake, Alarm, JohnCalls, MaryCalls ]

samples = sample(nodes)

# Check the Burglary count
count = reduce(lambda count, s: count + s[0], samples, 0)

print(count)

#P(Burglary=true) = 0.001
#P(Eathquake=true) = 0.002
#P(Alarm=true | Burglary=true, Earthquake=true) = 0.95
#P(Alarm=true | Burglary=true, Earthquake=false) = 0.94
#P(Alarm=true | Burglary=false, Earthquake=true) = 0.29
#P(Alarm=true | Burglary=false, Earthquake=false) = 0.001
#P(JohnCalls=true | Alarm=true) = 0.90
#P(JohnCalls=true | Alarm=false) = 0.05
#P(MaryCalls=true | Alarm=true) = 0.70
#P(MaryCalls=true | Alarm=false) = 0.01

