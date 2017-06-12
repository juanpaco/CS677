from functools import reduce

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

    def likelihood(self):
        raise NotImplementedError

    def complete_conditional(self, target):
        return reduce(
                lambda l, child: l * child.likelihood(),
                self.children,
                self.likelihood()
            )

    def sample():
        raise NotImplementedError

    def add_child(self, child):
        self.children.append(child)

