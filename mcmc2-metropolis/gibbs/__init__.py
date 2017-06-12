from functools import reduce

def collect_samples(nodes):
    return [ node.val for node in nodes]

def tick(nodes):
    for node in nodes:
        node.sample()

    return collect_samples(nodes)

def sample(nodes, burn=1000, num_samples=1000):
    for i in range(burn):
        tick(nodes)

    return [ tick(nodes) for i in range(num_samples) ]


