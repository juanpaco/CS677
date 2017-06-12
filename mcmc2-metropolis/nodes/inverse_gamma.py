from math import (log, sqrt)
import numpy
import random
import scipy.stats
import scipy.misc

from .node import (Node)
from .fixed import (Fixed)

class InverseGamma(Node):
    def __init__(
            self,
            name,
            alpha=0,
            beta=1,
            val=None,
            observed=False,
            candidate_standard_deviation=1,
            ):
        Node.__init__(
                self,
                name + ' (Inverse Gamma)',
                val=val,
                observed=observed,
                candidate_standard_deviation=candidate_standard_deviation,
            )

        if isinstance(alpha, Node):
            self.alpha = alpha
        else:
            self.alpha = Fixed('Alpha of ' + name, val=alpha)

        if isinstance(beta, Node):
            self.beta = beta
        else:
            self.beta = Fixed('Beta of ' + name, val=beta)

        if val is None:
            self.val = 3

        self.alpha.add_child(self)
        self.beta.add_child(self)

    def likelihood(self, value=None):
        target = self.val if value is None else value

        theta = 1 / self.beta.val

        #print(self.name, 'scale', theta)
        #print(self.name, 'val', target)
        #print(self.name, 'alpha', self.alpha.val)
        #print(self.name, 'not log', scipy.stats.invgamma.pdf(target, self.alpha.val, scale=theta))

        return scipy.stats.invgamma.logpdf(target, self.alpha.val, scale=theta)

    def sample(self):
        # get a candidate value
        cand = numpy.random.normal(self.val, self.candidate_standard_deviation)

        #print(self.name, 'cand', cand)

        if cand <= 0:
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
        
        return self.val


