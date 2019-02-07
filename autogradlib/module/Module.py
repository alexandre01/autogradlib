from autogradlib import Variable


class Module(object):
    def __init__(self):
        pass

    def forward(self, input):
        """
        Virtual method that should be overridden by all subclasses
        """
        raise NotImplementedError

    def __call__(self, input):
        """
        Making modules callable
        """
        return self.forward(input)

    def zero_grad(self):
        for param in self.params():
            param.zero_grad()

    @staticmethod
    def __computeParams(v):
        if isinstance(v, Variable):
            return [v]

        if isinstance(v, Module):
            return v.params()

        return []

    def params(self):
        """
        Automatically compute the list of parameters of the module by checking its attributes
        Supports Variable, Module and list of Variable or Module
        """
        params = []

        for v in vars(self).values():
            params.extend(self.__computeParams(v))

            if isinstance(v, list):
                for p in v:
                    params.extend(self.__computeParams(p))

        return params

    def __repr__(self):
        return "Module"
