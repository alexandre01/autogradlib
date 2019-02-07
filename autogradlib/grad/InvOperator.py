from .GradOperator import GradOperator
from autogradlib import Variable as lib


class InvOperator(GradOperator):
    def __init__(self, child):
        super().__init__()

        self.children = [child.grad_op]
        self.variable = lib.Variable(1 / child.tensor, grad_op=self, is_leaf=False)

    def pass_grad(self, gradwrtoutput):
        super().pass_grad(gradwrtoutput)

        self.children[0].pass_grad(-1 / self.children[0].variable.tensor.pow(2) * gradwrtoutput)

    def __repr__(self):
        return "InvOperator"
