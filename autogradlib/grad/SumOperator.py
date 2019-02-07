from torch import FloatTensor
from .GradOperator import GradOperator
from autogradlib import Variable as lib


class SumOperator(GradOperator):
    def __init__(self, child, dim=None):
        super().__init__()

        self.children = [child.grad_op]
        self.dim = dim  # Dimension (Integer) along which to perform the summation

        if dim is None or len(child.size()) == 1:
            # The output is a scalar, we need to wrap it into a Tensor
            self.variable = lib.Variable(FloatTensor([child.tensor.sum()]), grad_op=self, is_leaf=False)
        else:
            self.variable = lib.Variable(child.tensor.sum(dim=dim), grad_op=self, is_leaf=False)

    def pass_grad(self, gradwrtoutput):
        super().pass_grad(gradwrtoutput)

        if self.dim is None or len(self.children[0].variable.size()) == 1:
            # The output is a scalar, we need to wrap it into a Tensor
            ones = FloatTensor(self.children[0].variable.size()).fill_(1)
            self.children[0].pass_grad(ones * gradwrtoutput)
        else:
            dimensions = [1] * len(self.children[0].variable.size())
            dimensions[self.dim] = self.children[0].variable.size(self.dim)

            self.children[0].pass_grad(gradwrtoutput.unsqueeze(self.dim).repeat(*dimensions))


    def __repr__(self):
        return "SumOperator" + (" (dim={})".format(self.dim) if self.dim is not None else "")
