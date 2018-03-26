"""This class defines a matrix consisting of Vectors.
Matrices are immutable.
"""


from collections import namedtuple

from vector import Vector
import vmath


ERROR = "Cannot {} matrices that are not of the appropriate dimension."

Dimension = namedtuple("Dimension", ("rows", "columns"))


class DimensionError(ValueError):
    """Raised by functions that require Matrices of the same dimension.
    """
    pass


def check_dimensions(A, B, reason):
    """Raises a DimensionError with the given reason if u and v are not
    the same dimension.
    """
    if A.dim != B.dim:
        raise DimensionError(ERROR.format(reason))


class Matrix:
    def __init__(self, *args, columns=False, zero=None, identity=None):
        """A Matrix consisting of Vectors. Matrices are immutable.

        Parameters
        ----------
        *args
            If a single argument is given, that argument should be an iterable
            of iterables, with the internal iterables containing the elements
            of the rows of the Matrix.
            If multiple arguments are given, those arguments should be iterables
            containing the elements of the rows of the Matrix.
        columns : bool, optional
            If True, *args will be used to construct the columns of the Matrix
            instead of the rows. This is equivalent to setting columns to False,
            then taking the transpose of the resulting Matrix. Default False.
        zero : 2-tuple, optional
            If given, should be a tuple of ints (row, columns) indicating the
            dimensions of the zero Matrix to construct. If zero is specified,
            all other parameters are ignored.
        identity : int, optional
            If given, an identity Matrix of the given dimension will be constructed.
            All other parameters will be ignored.
        """
        if zero is not None and identity is not None:
            raise ValueError("Matrix cannot be both zero and identity.")
        # Initialize a zero matrix if zero is given
        if zero is not None:
            m, n = zero
            self.rows = (Vector(zero=n),) * m
        # Initialize an identity matrix if identity is given
        elif identity is not None:
            n = identity
            self.rows = tuple(vmath.get_standard_unit_vectors(n))
        # Initialize row vectors from *args
        elif len(args) == 1:
            self.rows = tuple(Vector(x) for x in args[0])
        else:
            self.rows = tuple(Vector(x) for x in args)

        # Initialize column vectors from the row vectors
        self.columns = tuple(Vector(x) for x in zip(*self.rows))

        # Swap rows and columns if we were given column vectors
        if columns and zero is None:
            self.rows, self.columns = self.columns, self.rows

        # It is a zero matrix if all of the row vectors are zero
        if zero is not None:
            self.non_zero = False
        elif identity is not None:
            self.non_zero = True
        else:
            self.non_zero = any(bool(r) for r in self.rows)

        # Initialize dim to (rows, columns)
        self.dim = Dimension(len(self.rows), len(self.columns))

    def __eq__(self, other):
        if not isinstance(other, Matrix) or self.dim != other.dim:
            return False
        return all(u == v for u, v in zip(self.rows, other.rows))

    def __add__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        check_dimensions(self, other, "add")
        return Matrix(u + v for u, v in zip(self.rows, other.rows))

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        check_dimensions(self, other, "subtract")
        return Matrix(u - v for u, v in zip(self.rows, other.rows))

    def __mul__(self, other):
        if isinstance(other, Vector):
            # TODO implement the case of a column matrix times a row vector
            if self.dim.columns != other.dim:
                raise DimensionError(
                    "Cannot apply Matrix to Vector that is not of appropriate dimension.")
            return Vector(vmath.dot(row, other) for row in self.rows)
        elif isinstance(other, Matrix):
            # Self must have same number of columns as other has rows
            if self.dim.columns != other.dim.rows:
                raise DimensionError(ERROR.format("multiply"))
            return Matrix((vmath.dot(r, c) for c in other.columns) for r in self.rows)
        else:
            # Scalar multiplication
            return Matrix(other * r for r in self.rows)

    def __rmul__(self, other):
        if isinstance(other, Vector):
            # TODO implement the case of a column vector times a row matrix
            if self.dim.columns != other.dim:
                raise DimensionError(
                    "Cannot apply Matrix to Vector that is not of appropriate dimension.")
                return Vector(vmath.dot(col, other) for col in self.columns)
        # Assume scalar multiplication
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
            return any(value in r for r in self.rows)

    def __str__(self):
        return "".join(str(r) + '\n' for r in self.rows)[:-1]
    
    def str_fractions(self):
        return "".join(r.str_fractions() + '\n' for r in self.rows)[:-1]
