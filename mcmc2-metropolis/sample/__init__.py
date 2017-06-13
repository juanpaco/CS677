from functools import reduce

def collect_samples(nodes):
    return [ node.value() for node in nodes]

def tick(nodes, isBurn = False):
    for node in nodes:
        node.sample(isBurn)

    return collect_samples(nodes)

def sample(nodes, burn=1000, num_samples=1000):
    for i in range(burn):
        if i % 10 == 0:
            print('burn', i)
        tick(nodes, isBurn = True)

    samples = []

    for i in range(num_samples):
        samples.append(tick(nodes))

    return samples

