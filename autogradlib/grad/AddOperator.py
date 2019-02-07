from .GradOperator import GradOperator
from autogradlib import Variable as lib


class AddOperator(GradOperator):
    def __init__(self, a, b):
        super().__init__()

        self.children = [a.grad_op, b.grad_op]
        self.variable = lib.Variable(a.tensor + b.tensor, grad_op=self, is_leaf=False)

    def pass_grad(self, gradwrtoutput):
        super().pass_grad(gradwrtoutput)

        self.children[0].pass_grad(gradwrtoutput)
        self.children[1].pass_grad(gradwrtoutput)

    def __repr__(self):
        return "AddOperator"
