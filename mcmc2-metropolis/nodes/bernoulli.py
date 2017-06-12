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

    def likelihood(self, target=None):
        likelihood = None
        target_val = self.val if target is None else target

        #print(self.name, 'target', target_val)
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

        #print(self.name, 'like before', likelihood)
        # If we're asking about False, then it's 1 - p.
        if target_val == 0:
            likelihood = 1 - likelihood

        #print(self.name, target, likelihood)

        return likelihood

    def complete_conditional(self, target):
        #print(self.name, 'complete', target)
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

        likelihood = self.likelihood(target=target)

        prob = reduce(
                lambda p, child: p * child.likelihood(),
                self.children,
                likelihood,
            )

        self.val = old_val

        return prob

    def sample(self):
        if self.observed:
            return self.val

        t_prob = self.complete_conditional(target=1)
        #print(self.name, 't_prob', t_prob)
        f_prob = self.complete_conditional(target=0)
        #print(self.name, 'f_prob', f_prob)
        normalized_prob = t_prob / (t_prob + f_prob)

        #print(self.name, 'sample', t_prob, f_prob, normalized_prob)

        self.val = numpy.random.binomial(1, normalized_prob)

        #print('****', self.name, 'got sample', self.val)

        return self.val

    def add_parent(self, node):
        self.parents.append(node)

    def add_child(self, node):
        self.children.append(node)
