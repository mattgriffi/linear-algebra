"""
This class defines a Euclidean vector. Vectors are immutable.
"""


import math


ERROR = "Cannot {} vectors that are not of the same dimension."


class DimensionError(ValueError):
    pass


class Vector:

    def __init__(self, *args):
        # If args is 1 argument, then assume that argument is an iterable
        # containing elements
        if len(args) == 1:
            self.elements = tuple(args[0])
        # Otherwise, assume args is an iterable of elements
        else:
            self.elements = tuple(args)

        self.non_zero = any(self.elements)

    def dimension(self):
        return len(self)

    def dot(self, other):
        self._check_length(other, "dot")
        if not self or not other:  # <u, O> = <O, u> = 0
            return 0
        return sum(i * j for i, j in zip(self, other))

    def norm(self):
        if not self:  # |O| = 0
            return 0
        return math.sqrt(self.norm2())

    def norm2(self):
        if not self:  # |O|^2 = 0
            return 0
        return sum(i * i for i in self)

    def normalize(self):
        if not self:
            return 0
        return self / self.norm()

    def project_onto(self, other):
        self._check_length(other, "project")
        if not other:
            raise ValueError("Cannot project onto a zero vector.")
        return other.dot(self) / other.norm2() * other

    def _check_length(self, other, message):
        if len(other) != len(self):
            raise DimensionError(ERROR.format(message))

    def __len__(self):
        return len(self.elements)

    def __eq__(self, other):
        return self.elements == other.elements

    def __add__(self, other):
        self._check_length(other, "add")
        return Vector(i + j for i, j in zip(self, other))

    def __sub__(self, other):
        self._check_length(other, "subtract")
        return Vector(i - j for i, j in zip(self, other))

    def __mul__(self, k):
        return Vector(k * i for i in self)

    def __rmul__(self, k):
        return self * k

    def __truediv__(self, k):
        return Vector(i / k for i in self)

    def __floordiv__(self, k):
        return Vector(i // k for i in self)

    def __neg__(self):
        return -1 * self

    def __bool__(self):
        """The zero vector is False, any other vector is True."""
        return self.non_zero

    def __getitem__(self, index):
        return self.elements[index]

    def __contains__(self, value):
        return value in self.elements

    def __iter__(self):
        return iter(self.elements)

    def __next__(self):
        return next(self.elements)

    def __str__(self):
        return str(self.elements)
