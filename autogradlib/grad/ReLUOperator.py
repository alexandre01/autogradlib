from .GradOperator import GradOperator
from autogradlib import Variable as lib


class ReLUOperator(GradOperator):
    def __init__(self, child):
        super().__init__()

        self.children = [child.grad_op]
        self.variable = lib.Variable(child.tensor.clamp(min=0), grad_op=self, is_leaf=False)

    def pass_grad(self, gradwrtoutput):
        super().pass_grad(gradwrtoutput)

        gradwrtoutput[self.children[0].variable.tensor < 0] = 0
        self.children[0].pass_grad(gradwrtoutput)

    def __repr__(self):
        return "ReLUOperator"
