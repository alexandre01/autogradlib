from graphviz import Digraph
from autogradlib.grad import *
from torch import FloatTensor


class Variable:
    """
    Variable is a Tensor wrapper that is used to perform automatic gradient computation
    
    Attributes:
        - tensor: the embedded tensor (Tensor object)
        - grad: stores the accumulated gradient (Tensor object) of the loss w.r.t. the tensor
        - grad_op: the associated GradOperator object that actually performs the backpropagation of gradients
        - name: node name used for drawing the operations graph
    
    """
    def __init__(self, x, grad_op=None, name=None, is_leaf=True):

        if not isinstance(x, FloatTensor):
            raise Exception("A non Tensor object was provided to Variable")

        self.tensor = x
        self.grad = FloatTensor([0])

        if grad_op is not None:
            self.grad_op = grad_op
        else:
            self.grad_op = AccumulateOperator(self)

        if is_leaf and name is None:
            self.name = "Input"
        else:
            self.name = name

    def backward(self):
        if not self.__is_scalar():
            raise Exception("The backward method is being called on a non-scalar Variable")

        # The gradient of the loss w.r.t. itself is 1
        return self.grad_op.pass_grad(FloatTensor([1]))

    def __is_scalar(self):
        """
        Is the associated tensor a scalar?
        """
        return list(self.tensor.size()) == [1]

    def size(self, dim=None):
        if dim is None:
            return self.tensor.size()

        return self.tensor.size(dim)

    def zero_grad(self):
        self.grad.zero_()

    def draw_graph(self):
        if self.name is None:
            self.name = "Output"

        dot = Digraph()
        x = self.draw_node(dot)
        y = self.grad_op.draw_node(dot, [x])
        dot.edge(x, y)

        return dot

    def draw_node(self, dot):
        node_id = str(self.__hash__())
        dot.node(node_id, self.name, shape='diamond', style='filled', color='lightblue2')
        return node_id

    def __add__(self, other):
        if not isinstance(other, Variable):
            other = Variable(FloatTensor([other]), name=other.__repr__())

        return AddOperator(self, other).variable

    def __mul__(self, other):
        if not isinstance(other, Variable):
            other = Variable(FloatTensor([other]), name=other.__repr__())

        return MulOperator(self, other).variable

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + other.__neg__()

    def __truediv__(self, other):
        if not isinstance(other, Variable):
            return self * (1 / other)

        return self * other.inv()

    def exp(self):
        return ExpOperator(self).variable

    def sum(self, dim=None):
        return SumOperator(self, dim).variable

    def mm(self, other):
        return MatrixMulOperator(self, other).variable

    def pow(self, power):
        return PowOperator(self, power).variable

    def inv(self):
        return InvOperator(self).variable

    def t(self):
        return TransposeOperator(self).variable

    def relu(self):
        return ReLUOperator(self).variable

    def repeat(self, dim, repetitions):
        return RepeatOperator(self, dim, repetitions).variable

    def __repr__(self):
        return "Variable containing: " + self.tensor.__repr__()
