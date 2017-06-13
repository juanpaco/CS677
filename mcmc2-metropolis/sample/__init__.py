from functools import reduce

def collect_samples(nodes):
    return [ node.val for node in nodes]

def tick(nodes, isBurn = False):
    for node in nodes:
        node.sample(isBurn)

    return collect_samples(nodes)

def sample(nodes, burn=1000, num_samples=1000):
    for i in range(burn):
        tick(nodes, isBurn = True)

    return [ tick(nodes) for i in range(num_samples) ]


