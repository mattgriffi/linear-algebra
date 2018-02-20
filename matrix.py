"""This class defines a matrix consisting of Vectors.
Matrices are immutable.
"""


from vector import Vector
import vmath


ERROR = "Cannot {} matrices that are not of the appropriate dimension."


class DimensionError(ValueError):
    """Raised by functions that require Matrices of the same dimension.
    """
    pass


def check_dimensions(A, B, reason):
    """Raises a DimensionError with the given reason if u and v are not
    the same dimension.
    """
    if A.dimension() != B.dimension():
        raise DimensionError(ERROR.format(reason))


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
            # TODO implement zero matrix initialization
            self.columns = (0,) * kwargs["zero"]  # (0,) makes a tuple
        # Initialize columns vectors from *args
        elif len(args) == 1:
            self.columns = tuple(args[0])
        else:
            self.columns = args

        # Initialize row vectors from the column vectors
        self.rows = tuple(Vector(x) for x in zip(*self.columns))

        if "transpose" in kwargs:
            if kwargs["transpose"]:
                self.rows, self.columns = self.columns, self.rows

        # It is a zero matrix if all of the column vectors are zero
        self.non_zero = any(bool(col) for col in self.columns)

    def dimension(self):
        """Returns the dimension of the Matrix as a tuple (rows, columns).
        """
        return len(self.rows), len(self.columns)

    def __eq__(self, other):
        if not isinstance(other, Matrix) or self.dimension() != other.dimension():
            return False
        return all(u == v for u, v in zip(self.columns, other.columns))

    def __add__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        check_dimensions(self, other, "add")
        return Matrix(u + v for u, v in zip(self.columns, other.columns))

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        check_dimensions(self, other, "subtract")
        return Matrix(u - v for u, v in zip(self.columns, other.columns))

    def __mul__(self, other):
        if isinstance(other, Vector):
            # TODO implement matrix-vector multiplication
            return NotImplemented
        elif isinstance(other, Matrix):
            # Self must have same number of columns as other has rows
            if self.dimension()[1] != other.dimension()[0]:
                raise DimensionError(ERROR.format("multiply"))
            return Matrix(Vector(vmath.dot(r, c) for r in self.rows) for c in other.columns)
        else:
            # Scalar multiplication
            return Matrix(other * v for v in self.columns)

    def __rmul__(self, other):
        # TODO implement for matrix-vector multiplication which is not commutative
        return self * other

    def __truediv__(self, k):
        return (1 / k) * self

    def __floordiv__(self, k):
        return (1 // k) * self

    def __neg__(self):
        return -1 * self

    def __bool__(self):
        """The zero Matrix is False, any other Matrix is True.
        """
        return self.non_zero

    def __getitem__(self, index):
        return self.rows[index]

    def __contains__(self, value):
        if isinstance(value, Vector):
            return value in self.rows or value in self.columns
        else:
            return any(value in row for row in self.rows)

    def __str__(self):
        # Get the str of each row with a new line after every except the last
        return "".join(str(r) + ("\n" if i < len(self.rows) - 1 else "")
                       for i, r in enumerate(self.rows))
