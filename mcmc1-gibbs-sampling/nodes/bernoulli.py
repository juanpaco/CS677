import numpy
from functools import reduce

class Bernoulli:
    def __init__(self, name, ps=0.5, val=None, observed=False):
        self.name = name + "(Bernoulli)"
        self.ps = ps
        self.val = numpy.random.binomial(1, 0.5) if val is None else val
        self.observed = observed
        self.parents = []
        self.children = []

    def likelihood(self, target, depth=0):
        likelihood = None

        if len(self.parents) > 0:
            index_key = reduce(
                    lambda key, p: key + str(p.val),
                    self.parents,
                    '',
                )
            index = int(index_key, 2)

            #print('index_key', index_key)
            #print('index', index)

            likelihood = self.ps[index]
        else:
            likelihood = self.ps

        # If we're asking about False, then it's 1 - p.
        if target == 0:
            likelihood = 1 - likelihood

        if depth > 0 and len(self.children) > 0:
            likelihood = reduce(
                    lambda l, child: l * child.likelihood(child.val),
                    self.children,
                    likelihood,
                )

        return likelihood

    def sample(self):
        if self.observed:
            return self.val

        # TODO: I don't think this is the right P for these resamples
        self.val = numpy.random.binomial(1, self.likelihood(1, depth=1))

        return self.val

    def add_parent(self, node):
        self.parents.append(node)

    def add_child(self, node):
        self.children.append(node)
