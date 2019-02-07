from .GradOperator import GradOperator
from autogradlib import Variable as lib


class TransposeOperator(GradOperator):
    def __init__(self, child):
        super().__init__()

        self.children = [child.grad_op]
        self.variable = lib.Variable(child.tensor.t(), grad_op=self, is_leaf=False)

    def pass_grad(self, gradwrtoutput):
        super().pass_grad(gradwrtoutput)

        self.children[0].pass_grad(gradwrtoutput.t())

    def __repr__(self):
        return "TransposeOperator"
