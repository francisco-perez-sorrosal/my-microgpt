"""Autograd engine for my microgpt: scalar-valued automatic differentiation. This is what Pytorch provides for free under the hood."""

import math


class Value:
    """A scalar value that tracks its computation graph for automatic differentiation."""

    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data: float, children: tuple["Value", ...] = (), local_grads: tuple[float, ...] = ()):
        self.data: float = data  # forward pass: scalar value calculated for this node
        self.grad: float = 0.0  # backward pass: gradient (derivative) of the loss with respect to this node's output (0.0 by default)
        self._children: tuple["Value", ...] = children  # list of child nodes that contribute to this node's output
        self._local_grads: tuple[float, ...] = local_grads  # local gradients (derivatives) of this node's output with respect to each child's input

    def __add__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other: float) -> "Value":
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self) -> "Value":
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self) -> "Value":
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self) -> "Value":
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self) -> "Value":
        return self * -1

    def __radd__(self, other: "Value | float") -> "Value":
        return self + other

    def __sub__(self, other: "Value | float") -> "Value":
        return self + (-other)

    def __rsub__(self, other: "Value | float") -> "Value":
        return other + (-self)

    def __rmul__(self, other: "Value | float") -> "Value":
        return self * other

    def __truediv__(self, other: "Value | float") -> "Value":
        return self * other**-1

    def __rtruediv__(self, other: "Value | float") -> "Value":
        return other * self**-1

    def backward(self) -> None:
        """Backward pass: Compute gradients via reverse-mode autodiff."""
        topo: list[Value] = []
        visited: set[int] = set()

        def build_topo(v: Value) -> None:
            """Build a topological order of the computation graph."""
            # If we haven't visited this node yet, add it to the visited set and recursively add its children to the topological order.
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed (topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad}), children={self._children}, local_grads={self._local_grads}"


def main() -> None:
    # Example: L = a*b + a, with a=2.0, b=3.0
    a = Value(2.0)
    b = Value(3.0)
    c = a * b       # c = 6.0
    big_l = c + a   # L = 8.0
    big_l.backward()
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"L = a*b + a = {big_l.data}")
    print(f"dL/da = {a.grad} (expected 4.0: b + 1 = 3 + 1, via both paths)")
    print(f"dL/db = {b.grad} (expected 2.0: a = 2)")


if __name__ == "__main__":
    main()
