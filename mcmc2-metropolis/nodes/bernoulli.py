from functools import reduce
import math
import numbers
import numpy
from scipy.misc import logsumexp

from .node import (Fixed, Node)

class Bernoulli(Node):
    def __init__(self, name, val=None, ps=0.5, parents=None, observed=False):
        Node.__init__(
                self,
                '{} (Bernoulli)'.format(name),
                val=val,
                observed=observed,
            )

        if isinstance(ps, numbers.Number):
            self.ps = Fixed('p for {}'.format(self.name), val=ps)
        else:
            self.ps = ps

        self.val = numpy.random.binomial(1, 0.5) if val is None else val
        self.observed = observed
        self.parents = [] if parents is None else parents

        for parent in self.parents:
            parent.add_child(self)

        if isinstance(self.ps, dict):
            for val in self.ps.values():
                val.add_child(self)
        else:
            self.ps.add_child(self)

    def p_index(self):
        return tuple([ p.value() for p in self.parents ])

    def p(self):
        if len(self.parents) == 0:
            return self.ps.value()
        else:
            return self.ps[self.p_index()].value()

    def likelihood(self, target=None):
        return math.log(self.non_log_likelihood(target))

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

        #print(self.name, 'children:', self.children)

        prob = reduce(
                lambda p, child: p + child.likelihood(),
                self.children,
                likelihood,
            )

        self.val = old_val

        return prob

    def non_log_likelihood(self, target=None):
        target_val = self.val if target is None else target

        p = self.p()

        #print(self.name, 'like before', p)
        # If we're asking about False, then it's 1 - p.
        if target_val == 0:
            p = 1 - p 

        print(self.name, target, p)

        return p

    def sample(self, isBurn=False):
        if self.observed:
            return self.val

        t_like = self.complete_conditional(target=1)
        #print(self.name, 't_like', t_like)
        f_like = self.complete_conditional(target=0)
        #print(self.name, 'f_like', f_like)
        denominator = logsumexp([ t_like, f_like ])

        t_prob = t_like - denominator

        log_rand = math.log(numpy.random.random())

        #print(self.name, 'sample: t_prob', t_prob, 'log_rand', log_rand)

        if log_rand < t_prob:
            self.val = 1
        else:
            self.val = 0

        if isBurn is False:
            self.posteriors.append(self.val)

        #print('****', self.name, 'got sample', self.val)

        return self.val
