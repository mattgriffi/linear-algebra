"""This class defines a Euclidean vector. Vectors are immutable.
"""


import math


ERROR = "Cannot {} vectors that are not of the same dimension."


class DimensionError(ValueError):
    """Raised by functions that require Vectors of the same dimension.
    """
    pass


class Vector:

    def __init__(self, *args, **kwargs):
        """An immutable Euclidean vector.

        Parameters
        ----------
        *args
            If a single argument is given, that argument should be an iterable
            containing the elements of the Vector. If multiple arguments are
            given, those arguments will be the elements of the Vector.
        **kwargs
        zero: int
            If provided, a zero Vector of the given dimension will be
            constructed. *args will be ignored.

        """
        # Initialize to a zero vector if "zero" is given
        if "zero" in kwargs:
            self.elements = (0,) * kwargs["zero"]  # (0,) makes a tuple
        # Initialize self.elements from *args
        elif len(args) == 1:
            self.elements = tuple(args[0])
        else:
            self.elements = args

        # Initialize self.non_zero
        for e in self.elements:
            if not math.isclose(e, 0, abs_tol=1e-15):
                self.non_zero = True
                break
        else:
            self.non_zero = False

    def dimension(self):
        """Returns the dimension of the Vector.
        """
        return len(self)

    def dot(self, other):
        """Performs the standard dot product with the given Vector.

        Parameters
        ----------
        other : Vector
            The Vector to dot this one with. Both Vectors' elements must define
            addition and multiplication.

        Returns
        -------
            The result of taking the dot product of the given Vector with this
            one. This will be of whatever type is given by the multiplication
            and subsequent addition of the Vectors' elements. Rounds to 15
            decimal places.

        Raises
        ------
        DimensionError
            If the Vectors' dimensions differ.
        """
        self._check_length(other, "dot")
        if not self or not other:  # <u, O> = <O, u> = 0
            return 0
        # Round the result to 15 decimal places, otherwise u.dot(v) == 0 can
        # be False for some orthogonal vectors
        return round(sum(i * j for i, j in zip(self, other)), 15)

    def is_normal(self):
        """Determines whether this Vector is normal.

        Returns
        -------
        bool
            True if Vector is normal, otherwise False.

        Notes
        -----
        This compares the norm of the vector to 1 with a tolerance of 15
        decimal places.
        """
        return math.isclose(self.norm(), 1, abs_tol=1e-15)

    def norm(self):
        """Returns the Euclidean norm.
        """
        if not self:  # |O| = 0
            return 0
        return math.sqrt(self.norm2())

    def norm2(self):
        """Returns the square of the Euclidean norm.
        """
        if not self:  # |O|^2 = 0
            return 0
        return sum(i * i for i in self)

    def normalize(self):
        """Returns the normalization of the Vector.
        """
        if not self:
            return self
        return self / self.norm()

    def project_onto(self, other):
        """Projects this Vector onto the given one.

        Parameters
        ----------
        other : Vector
            The Vector to project this one onto.

        Returns
        -------
        Vector
            The result of projecting this Vector onto other.

        Raises
        ------
        DimensionError
            If the Vectors' dimensions differ.
        ValueError
            If other is the zero vector of the appropriate dimension.
        """
        self._check_length(other, "project")
        if not other:
            raise ValueError("Cannot project onto a zero vector.")
        return other.dot(self) / other.norm2() * other

    def _check_length(self, other, message):
        """Raises DimensionError if self and other differ in length.
        """
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
        return str(self.elements)
