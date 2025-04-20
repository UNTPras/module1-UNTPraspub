from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    # Convert vals to a mutable list
    vals = list(vals)

    # Perturb the value at position `arg` positively
    vals[arg] += epsilon
    f_plus = f(*vals)

    # Perturb negatively
    vals[arg] -= 2 * epsilon
    f_minus = f(*vals)

    # Central difference formula
    return (f_plus - f_minus) / (2 * epsilon)



variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    visited = set()
    order: List[Variable] = []

    def visit(v: Variable):
        if v.unique_id in visited or v.is_constant():
            return
        visited.add(v.unique_id)
        for parent in v.parents:
            visit(parent)
        order.append(v)

    visit(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    derivatives = {variable.unique_id: deriv}

    for var in topological_sort(variable):
        d_output = derivatives.get(var.unique_id, 0.0)
        if var.is_leaf():
            var.accumulate_derivative(d_output)
        else:
            for parent, d_parent in var.chain_rule(d_output):
                derivatives[parent.unique_id] = (
                    derivatives.get(parent.unique_id, 0.0) + d_parent
                )


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
