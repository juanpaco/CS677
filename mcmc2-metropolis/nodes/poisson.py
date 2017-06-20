import numpy
import scipy.stats

from .node import (Fixed, Node)

class Poisson(Node):
    def __init__(
            self,
            name,
            rate=0,
            val=None,
            observed=False,
            candidate_standard_deviation=1,
            ):
        Node.__init__(
                self,
                name + '-(Poisson)',
                val=val,
                observed=observed,
                candidate_standard_deviation=candidate_standard_deviation,
            )

        if isinstance(rate, Node):
            self.rate = rate 
        else:
            self.rate = Fixed('Rate of ' + name, val=rate)

        if val is None:
            self.val = 1

        self.rate.add_child(self)

    def likelihood(self, value=None):
        target = self.value() if value is None else value

        if target <= 0:
            return numpy.NINF

        #print(self.name, 'targetf', target)

        #print(self.name, 'rate', self.rate.value())
        #print(self.name, 'target', target)

        return scipy.stats.poisson.logpmf(
                target,
                self.rate.value(),
            )

    def pdf(self, val):
        return scipy.stats.poisson.pmf(
                val,
                self.rate.value(),
            )

    def in_support(self, val):
        return val >= 0

    def cleanse_val(self, val):
        #print(self.name, 'cleanse', val)
        return round(val)
