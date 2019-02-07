from .Module import Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ((x * (-1)).exp() + 1).inv()

    def __repr__(self):
        return "Sigmoid"
