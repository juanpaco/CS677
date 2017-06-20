from math import (log, pi, sqrt)
import numpy
import scipy.stats
import scipy.misc

from .node import (Fixed, Node)

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
                name + '-(Normal)',
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
                    self.mean.value(),
                    sqrt(self.var.value()),
                    1,
                )[0]

        self.mean.add_child(self)
        self.var.add_child(self)

        self.original_mean = self.mean.value()
        self.original_var = self.var.value()

    def likelihood(self, value=None):
        #print(self.name, self.mean.value(), self.var.value(), self.value())
        if (self.var.value() <= 0):
            return -100

        target = self.value() if value is None else value

        #print(self.name, 'target', target, 'mean', self.mean.value(), 'var', self.var.value())
        pi_term = log(2 * pi * self.var.value())
        mean_term = (target - self.mean.value()) ** 2

        return -0.5 * (pi_term + (mean_term / self.var.value()))

    def pdf(self, val):
        return scipy.stats.norm.pdf(
                val,
                self.original_mean,
                sqrt(self.original_var),
            )

    def in_support(self, val):
        return True
