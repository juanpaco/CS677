from math import (log, sqrt)
import numpy
import random
import scipy.stats
import scipy.misc

from .node import (Node)
from .fixed import (Fixed)

class Normal(Node):
    def __init__(
            self,
            name,
            mean=0,
            var=1,
            val=None,
            observed=False,
            candidate_standard_deviation=1,
            ):
        Node.__init__(
                self,
                name + ' (Normal)',
                val=val,
                observed=observed,
                candidate_standard_deviation=candidate_standard_deviation,
            )

        if isinstance(mean, Node):
            self.mean = mean
        else:
            self.mean = Fixed('Mean of ' + name, val=mean)

        if isinstance(var, Node):
            self.var = var 
        else:
            self.var = Fixed('Variance of ' + name, val=var)

        if val is None:
            self.val = numpy.random.normal(
                    self.mean.val,
                    sqrt(self.var.val),
                    1
                )

        self.mean.add_child(self)
        self.var.add_child(self)

    def likelihood(self, value=None):
        target = self.val if value is None else value

        return scipy.stats.norm.logpdf(
                target,
                self.mean.val,
                sqrt(self.var.val)
            )

    def sample(self):
        # get a candidate value
        cand = numpy.random.normal(self.val, self.candidate_standard_deviation)

        #print('cand', cand)

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

        #print('r', reject_likelihood)
        #print('a', accept_likelihood)
        #print('u', u)

        # set it back if staying is more likely
        if u >= accept_likelihood - reject_likelihood:
            #print('set it back')
            self.val = old_val
        
        return self.val


