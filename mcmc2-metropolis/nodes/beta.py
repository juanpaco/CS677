import numpy
import scipy.stats
import scipy.misc

from .node import (Fixed, Node)

class Beta(Node):
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
                name + '-(Beta)',
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
        target = self.value() if value is None else value

        if not self.in_support(target):
            return -100

        #if self.alpha.value() <= 0:
        #    return -100

        #if self.beta.value() <= 0:
        #    return -100

        #print(self.name, 'scale', theta)
        #print(self.name, 'val', target)
        #print(self.name, 'alpha', self.alpha.val)
        #print(self.name, 'not log', scipy.stats.invbeta.pdf(target, self.alpha.val, scale=theta))

        return scipy.stats.beta.logpdf(
                target,
                self.alpha.value(),
                self.beta.value(),
            )

    def pdf(self, val):
        return scipy.stats.beta.pdf(
                val,
                self.alpha.value(),
                self.beta.value(),
            )

    def in_support(self, val):
        return val > 0 and val < 1
