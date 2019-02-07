from .Module import Module


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

    def __repr__(self):
        str = "Sequential(\n"
        for i, module in enumerate(self.modules):
            str += "  {}: {}\n".format(i, module.__repr__())
        str += ")"
        return str