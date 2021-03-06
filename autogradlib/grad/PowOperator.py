from .GradOperator import GradOperator
from autogradlib import Variable as lib


class PowOperator(GradOperator):
    def __init__(self, child, power):
        super().__init__()

        self.children = [child.grad_op]
        self.power = power
        self.variable = lib.Variable(child.tensor.pow(power), grad_op=self, is_leaf=False)

    def pass_grad(self, gradwrtoutput):
        super().pass_grad(gradwrtoutput)

        self.children[0].pass_grad(self.children[0].variable.tensor * self.power * gradwrtoutput)

    def __repr__(self):
        return "PowOperator^{}".format(self.power)
