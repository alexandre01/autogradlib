from .Module import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.relu()

    def __repr__(self):
        return "ReLU"