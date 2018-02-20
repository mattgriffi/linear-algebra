"""This class defines a matrix consisting of Vectors.
Matrices are immutable.
"""


import itertools
import math


from vector import Vector, check_dimensions
import vmath


ERROR = "Cannot {} matrices that are not of the same dimension."


class DimensionError(ValueError):
    """Raised by functions that require Matrices of the same dimension.
    """
    pass


class Matrix:
    def __init__(self, *args, **kwargs):
        """A Matrix consisting of Vectors. Matrices are immutable.

        Parameters
        ----------
        *args

        **kwargs

        """
        # Initialize to a zero matrix if "zero" is given
        if "zero" in kwargs:
            # TODO implements zero matrix initialization
            self.cs = (0,) * kwargs["zero"]  # (0,) makes a tuple
        # Initialize self.cs from *args
        elif len(args) == 1:
            self.cs = tuple(args[0])
        else:
            self.cs = args

        self.rs = tuple(Vector(x) for x in zip(self.cs))

        self.non_zero = any(bool(col) for col in self.cs)

    def dimension(self):
        """Returns the dimension of the Matrix as a tuple (rows, columns).
        """
        return len(self)

    def __len__(self):
        """Returns the dimension of the Matrix as a tuple (rows, columns).
        """
        pass

    def __eq__(self, other):
        pass

    def __add__(self, other):
        # TODO be sure to check dimensions
        pass

    def __sub__(self, other):
        # TODO be sure to check dimensions
        pass

    def __mul__(self, k):
        pass

    def __rmul__(self, k):
        pass

    def __truediv__(self, k):
        pass

    def __floordiv__(self, k):
        pass

    def __neg__(self):
        pass

    def __bool__(self):
        """The zero Matrix is False, any other Matrix is True.
        """
        return self.non_zero

    def __getitem__(self, index):
        pass

    def __contains__(self, value):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def __str__(self):
        # TODO make it look pretty
        pass
