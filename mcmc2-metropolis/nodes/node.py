from functools import reduce
from math import (log, sqrt)
import matplotlib.pyplot as plt
import matplotlib.pylab as mlab
import numpy
import random

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

        #print(self.name, 'cand', cand)

        if not self.in_support(cand):
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

    # Need a function to handle the Add node's value retrieval
    def value(self):
        return self.val

    def add_child(self, child):
        self.children.append(child)

    def mixplot(self):
        xs, ys = zip(*enumerate(self.posteriors))

        plt.plot(xs, ys)
        plt.title('{} mixing'.format(self.name))
        plt.savefig(self.name + '-mixplot.png')
        #plt.show()
        plt.close()

    def plot_posterior(self):
        xs = mlab.frange(5, 6.5, (6.5-5) / 100)
        ys = [self.pdf(x) for x in xs]
        plt.plot(xs, ys, label='Priot Dist ' + self.name)

        plt.hist(self.posteriors, bins=30, normed=True, label="Posterior Dist " + self.name)
        plt.ylim(ymin=0)
        plt.xlim(5,6.5)
        plt.savefig(self.name + '-posterior.png')
        #plt.show()
        plt.close()

    def __add__(self, other):
        return Add(self, other)

class Add(Node):
    def __init__(self, *args):
        Node.__init__(
                self,
                ':'.join([ p.name for p in list(args) ]) + ' (Add)',
            )

        self.parents = list(args)

    def add_child(self, child):
        for p in self.parents:
            p.add_child(child)

    def value(self):
        return reduce(lambda total, p: total + p.value(), self.parents, 0)

