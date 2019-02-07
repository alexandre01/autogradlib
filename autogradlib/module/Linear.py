from .Module import Module
from autogradlib import Variable
from torch import Tensor
import math


class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.w = Variable(Tensor(out_dim, in_dim), name="w")
        self.b = Variable(Tensor(out_dim), name="b")
        self.__initParams()

    def __initParams(self):
        """
        Initialize weights using Xavier initialization
        """
        std = math.sqrt(2.0 / (self.w.tensor.size(0) + self.w.tensor.size(1)))
        self.w.tensor.normal_(0, std)
        self.b.tensor.normal_(0, std)

    def forward(self, input):
        return input.mm(self.w.t()) + self.b.repeat(0, input.size(dim=0))

    def __repr__(self):
        return "Linear(in_dim={}, out_dim={})".format(self.w.tensor.size(1), self.w.tensor.size(0))
