from .GradOperator import GradOperator


class AccumulateOperator(GradOperator):
    """
    A dummy GradOperator that does nothing else than accumulating the gradients
    Used for leaf nodes in the gradient operations DAG
    """
    def __init__(self, variable):
        super().__init__()
        self.variable = variable

    def pass_grad(self, gradwrtoutput):
        super().pass_grad(gradwrtoutput)

    def draw_node(self, dot, node_ids):
        x = super().draw_node(dot, node_ids)

        node_id = str(self.variable.__hash__())

        if self.variable.name:
            if node_id not in node_ids:
                node_ids.append(node_id)

            self.variable.draw_node(dot)
            dot.edge(x, node_id)

        return x

    def __repr__(self):
        return "AccumulateOperator"
