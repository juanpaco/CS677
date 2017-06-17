from functools import reduce
from math import (log, sqrt)
import matplotlib.pyplot as plt
import matplotlib.pylab as mlab
import numpy
import random
import scipy.stats

class Node:
    def __init__(
            self,
            name,
            val=None,
            observed=False,
            candidate_standard_deviation=1
        ):
        self.name = name
        self.val = val
        self.observed = observed
        self.candidate_standard_deviation = candidate_standard_deviation
        self.children = []
        self.posteriors = []
        self.rejected = 0

    def likelihood(self):
        raise NotImplementedError

    def complete_conditional(self, target):
        return reduce(
                lambda l, child: l * child.likelihood(),
                self.children,
                self.likelihood()
            )

    def sample(self, isBurn=False):
        if self.observed:
            return self.val

        # get a candidate value
        cand = numpy.random.normal(self.val, self.candidate_standard_deviation)
        cand = self.cleanse_val(cand)

        #print(self.name, 'cand', cand)

        if not self.in_support(cand):
            self.rejected = self.rejected + 1
            return self.val

        old_val = self.val

        reject_likelihood = self.likelihood(old_val)
        accept_likelihood = self.likelihood(cand)

        # factor in the children with the curernt value
        for child in self.children:
            reject_likelihood += child.likelihood()

        # get the likelihood of the candidate value
        self.val = cand

        for child in self.children:
            accept_likelihood += child.likelihood()

        u = log(random.random())

        #print(self.name, 'r', reject_likelihood)
        #print(self.name, 'a', accept_likelihood)
        #print(self.name, 'u', u)

        # set it back if staying is more likely
        if u >= accept_likelihood - reject_likelihood:
            #print(self.name, 'set it back')
            self.val = old_val

        if not isBurn:
          self.posteriors.append(self.val)
        
        return self.val

    def cleanse_val(self, val):
        return val

    # Need a function to handle the Add node's value retrieval
    def value(self):
        return self.val

    def add_child(self, child):
        self.children.append(child)

    def mixplot(self):
        xs, ys = zip(*enumerate(self.posteriors))

        plt.plot(xs, ys)
        plt.title('{} mixing'.format(self.name))
        plt.show()
        #plt.savefig(self.name + '-mixplot.png')
        #plt.close()

    def plot_posterior(self):
        sample_min = min(self.posteriors)
        sample_max = max(self.posteriors)

        xs = mlab.frange(sample_min, sample_max, (sample_max - sample_min) / 100)
        ys = [self.pdf(x) for x in xs]
        plt.plot(xs, ys, label='Priot Dist ' + self.name)

        plt.title('Prior Dist {}'.format(self.name))
        plt.hist(self.posteriors, bins=30, normed=True, label="Posterior Dist " + self.name)
        plt.show()
        #plt.savefig(self.name + '-posterior.png')
        #plt.close()

    def __add__(self, other):
        return Add(self, other)

    def __pow__(self, other):
        return Power(self, other)

class Add(Node):
    def __init__(self, *args):
        def map_args(n):
            if isinstance(n, Node):
                return n
            else:
                return Fixed('Fixed ({})'.format(n), val=n)

        self.parents = [ map_args(n) for n in list(args)]

        Node.__init__(
                self,
                ':'.join([ p.name for p in self.parents ]) + ' (Add)',
            )

    def add_child(self, child):
        for p in self.parents:
            p.add_child(child)

    def value(self):
        return reduce(lambda total, p: total + p.value(), self.parents, 0)

# The purpose of this node is to just have something that gives a fixed value
#  With a probability of 1.  This is useful for priors.
class Fixed(Node):
    def __init__(self, name, val=None):
        Node.__init__(
                self,
                name + ' (Fixed)',
                val=val
            )

    def likelihood(self):
        # It's in log space, remember
        return 0

class Power(Node):
    def __init__(self, base, exponent):
        if isinstance(base, Node):
            self.base = base
        else:
            self.base = Fixed('base {}'.format(base), val=base)

        if isinstance(exponent, Node):
            self.exponent = exponent 
        else:
            self.exponent = Fixed('exponent {}'.format(exponent), val=exponent)

        name = '{}:{} (Pow)'.format(self.base.name, self.exponent.name)

        Node.__init__(
                self,
                name,
            )

        self.parents = [ self.base, self.exponent ]

    def add_child(self, child):
        for p in self.parents:
            p.add_child(child)

    def value(self):
        return self.base.value() ** self.exponent.value()
