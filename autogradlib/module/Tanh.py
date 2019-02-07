from .Module import Module


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (-(-x * 2).exp() + 1) / ((-x * 2).exp() + 1)

    def __repr__(self):
        return "Tanh"