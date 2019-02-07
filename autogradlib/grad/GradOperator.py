class GradOperator:
    """
    Wrapper of Variable objects that builds the gradient operations DAG on the fly
    
    Attributes:
        - variable: the wrapped output variable (Variable object)
        - children: the children nodes (GradOperator objects) in the gradient operations DAG
    """
    def __init__(self, variable=None, children=[]):
        self.variable = variable
        self.children = children

    def pass_grad(self, gradwrtoutput):
        """
        Performs the backpropagation algorithm:
        - Receives as argument the gradient of the loss w.r.t. the output (Tensor object)
        - Accumulates the Variable's grad
        - Recursively calls pass_grad on its children
        """

        # Accumulate the Variable's grad
        self.variable.grad = self.variable.grad + gradwrtoutput

    def draw_node(self, dot, node_ids):
        """
        Draw the GradientOperator node on the given graph and return its id
        
        Parameters:
            - dot: the graphviz graph
            - node_ids: the ids of all the already drawn nodes, to avoid drawing twice a same Variable or GradOperator
        """

        node_id = str(self.__hash__())
        dot.node(node_id, self.__repr__())

        for child in self.children:

            child_id = str(child.__hash__())

            if child_id not in node_ids:
                child.draw_node(dot, node_ids)
                node_ids.append(child_id)

            dot.edge(node_id, child_id)

        return node_id

    def __repr__(self):
        return "GradOperator"
