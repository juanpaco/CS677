from math import (sqrt)
import numpy
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
        target = self.value() if value is None else value

        return scipy.stats.norm.logpdf(
                target,
                self.mean.value(),
                sqrt(self.var.value())
            )

    def pdf(self, val):
        return scipy.stats.norm.pdf(
                val,
                self.mean.value(),
                sqrt(self.var.value())
            )

    def in_support(self, val):
        return True
