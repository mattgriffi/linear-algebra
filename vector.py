"""This class defines a Euclidean vector. Vectors are immutable.
"""


import math
import numbers


ERROR = "Cannot {} vectors that are not of the same dimension."


class DimensionError(ValueError):
    """Raised by functions that require Vectors of the same dimension.
    """
    pass


def check_dimensions(u, v, reason):
    """Raises a DimensionError with the given reason if u and v are not
    the same dimension.
    """
    if len(u) != len(v):
        raise DimensionError(ERROR.format(reason))


class Vector:

    def __init__(self, *args, zero=None):
        """An immutable Euclidean vector.

        Parameters
        ----------
        *args
            If a single argument is given, that argument should be an iterable
            containing the elements of the Vector. If multiple arguments are
            given, those arguments will be the elements of the Vector.
        zero: int, optional
            If provided, a zero Vector of the given dimension will be constructed,
            and all other parameters will be ignored.
        """
        # Initialize to a zero vector if "zero" is given
        if zero is not None:
            self.elements = (0.0,) * zero
        # Initialize self.elements from *args
        elif len(args) == 1:
            # If the input is a vector, just copy its elements
            if isinstance(args[0], Vector):
                self.elements = args[0].elements
            else:
                self.elements = tuple(map(float, args[0]))
        else:
            self.elements = tuple(map(float, args))

        # Initialize self.non_zero
        self.non_zero = any(not math.isclose(e, 0, abs_tol=1e-15) for e in self.elements)

    def dimension(self):
        """Returns the dimension of the Vector.
        """
        return len(self)

    def __len__(self):
        return len(self.elements)

    def __eq__(self, other):
        return isinstance(other, Vector) and self.elements == other.elements

    def __add__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        check_dimensions(self, other, "add")
        return Vector(i + j for i, j in zip(self, other))

    def __sub__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        check_dimensions(self, other, "subtract")
        return Vector(i - j for i, j in zip(self, other))

    def __mul__(self, k):
        if not issubclass(k.__class__, numbers.Real):
            return NotImplemented
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
        """The zero vector is False, any other vector is True.
        """
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
        # Create a string with each of the elements, rounded to four decimal
        # places, left justified with up to 10 spaces of padding, with a blank
        # before nonnegative numbers.
        return "".join(
            (("" if e < 0 else " ") + "{}").format(round(e, 4)).ljust(10) for e in self
            )
