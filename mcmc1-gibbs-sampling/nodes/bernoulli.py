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

        # This hack may require some repentance, but it gets the job done.
        #   When doing the likelihood for the children, they'll need to use the
        #   hypothetical value of this node, and not the actualy. So, we'll set
        #   the value of this node to the hypothetical, get the child
        #   likelihood, and then set it back.
        #
        # I can do this and sleep at night, because the value gets changed back
        #   within this function, and the operation I use it for is read only.
        old_val = self.val
        self.val = target

        if depth > 0 and len(self.children) > 0:
            likelihood = reduce(
                    lambda l, child: l * child.likelihood(child.val),
                    self.children,
                    likelihood,
                )

        self.val = old_val

        return likelihood

    def sample(self):
        if self.observed:
            return self.val

        # TODO: I don't think this is the right P for these resamples
        t_likelihood = self.likelihood(1, depth=1)
        f_likelihood = self.likelihood(0, depth=1)
        normalized_likelihood = t_likelihood / (t_likelihood + f_likelihood)
        # I don't think the normalization works properly.  It's using the actual
        # value of the parent and not the proposed one.

        #print('t like', t_likelihood)
        #print('f like', f_likelihood)
        #print('norm like', normalized_likelihood)

        self.val = numpy.random.binomial(1, normalized_likelihood)

        return self.val

    def add_parent(self, node):
        self.parents.append(node)

    def add_child(self, node):
        self.children.append(node)
