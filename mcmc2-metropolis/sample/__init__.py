from functools import reduce
from tqdm import tqdm

def collect_samples(nodes):
    return [ node.value() for node in nodes]

def tick(nodes, isBurn = False):
    for node in nodes:
        node.sample(isBurn)

    return collect_samples(nodes)

def sample(nodes, burn=1000, num_samples=1000):
    for i in tqdm(range(burn), desc='(BURN) '):
        tick(nodes, isBurn = True)

    samples = []

    for i in tqdm(range(num_samples)):
        samples.append(tick(nodes))

    return samples

