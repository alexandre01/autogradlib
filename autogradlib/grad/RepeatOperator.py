from .GradOperator import GradOperator
from autogradlib import Variable as lib


class RepeatOperator(GradOperator):
    """
    Repeat a tensor along a given axis `dim` (a new dimension is created), `repetitions` times
    """
    def __init__(self, child, dim, repetitions):
        super().__init__()

        self.dim = dim
        self.repetitions = repetitions

        dimensions = [1] * (len(child.size()) + 1)
        dimensions[self.dim] = self.repetitions

        self.children = [child.grad_op]
        self.variable = lib.Variable(child.tensor.unsqueeze(self.dim).repeat(*dimensions), grad_op=self, is_leaf=False)

    def pass_grad(self, gradwrtoutput):
        super().pass_grad(gradwrtoutput)

        self.children[0].pass_grad(gradwrtoutput.sum(dim=self.dim))

    def __repr__(self):
        return "RepeatOperator ({}x, dim={})".format(self.repetitions, self.dim)
