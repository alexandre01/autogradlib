from .Module import Module


class Softmax(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = x.exp()

        # We must repeat the y.sum vector `n` times along the given dimension
        # in order to multiply similar-sized tensors (where n = y.size(dim=self.dim))
        z = y / (y.sum(dim=self.dim).repeat(self.dim, y.size(dim=self.dim)))
        return z

    def __repr__(self):
        return "Softmax(dim={})".format(self.dim)
