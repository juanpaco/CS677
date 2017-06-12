# The purpose of this node is to just have something that gives a fixed value
#  With a probability of 1.  This is useful for priors.
from .node import (Node)

class Fixed(Node):
    def __init__(self, name, val=None):
        Node.__init__(
                self,
                name + ' (Fixed)',
                val=val
            )

    def likelihood(self):
        return 0

