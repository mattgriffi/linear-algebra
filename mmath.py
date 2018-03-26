"""This file defines some useful functions for performing calculations
with Matrices.
"""

import functools
import math
import operator

import vmath
from matrix import Matrix, DimensionError
from vector import Vector


def transpose(A):
    """Calculates the transpose of Matrix A.
    """
    return Matrix(A.columns)


def trace(A):
    """Calculates the sum of the leading diagonal of Matrix A.
    """
    return sum(A[i][i] for i in range(min(A.dim)))


def rank(A, is_rref=False):
    """Returns the rank of Matrix A.
    """
    return len(row_space(A, is_rref))


def nullity(A, is_rref=False):
    """Returns the nullity of Matrix A.
    """
    return A.dim.columns - rank(A, is_rref)


def invert(A):
    """Returns the inverse of Matrix A. Raises Dimension error if A
    is not square.
    """
    if A.dim.rows != A.dim.columns:
        raise DimensionError("Cannot invert non-square Matrix.")
    
    A_augment_I = augment(A, Matrix(identity=A.dim.columns))
    _, A_inverse = deaugment(rref(A_augment_I), A.dim.columns)
    return A_inverse


def power(A, n):
    """Returns square Matrix A raised to the n'th power.
    """
    if A.dim.rows != A.dim.columns:
        raise DimensionError("Cannot exponentiate non-square Matrix.")

    def r(B, n):
        # Use exponentiation by squaring method
        if n <= 1:
            return B
        elif n % 2:  # Odd exponent
            return B * r(B * B, (n - 1) / 2)
        else:  # Even exponent
            return r(B * B, n / 2)

    return r(A, n)


def augment(A, B):
    """Augments Matrix A with Matrix or Vector B.

    Parameters
    ----------
    A : Matrix
        The Matrix to augment with B.
    B : Matrix or Vector
        The Matrix or Vector to attach to A.

    Returns
    -------
    Matrix
        The result of augmenting A with B.

    Raises
    ------
    DimensionError
        If A and B do not have the same number of rows (Matrix), or if
        B does not have as many elements as A has rows (Vector).
    """
    if isinstance(B, Vector):
        if A.dim.rows != B.dim:
            raise DimensionError("Cannot augment Matrix with Vector of innapropriate dimension.")
        return Matrix(*A.columns, B, columns=True)
    else:
        if A.dim.rows != B.dim.rows:
            raise DimensionError("Cannot augment Matrix with Matrix of innapropriate dimension.")
        return Matrix(*A.columns, *B.columns, columns=True)


def deaugment(A, n):
    """Splits the last n columns off of A. Returns the resulting Matrices or
    Matrix and Vector if n is 1.
    """
    if n == 1:
        return Matrix(A.columns[:-1], columns=True), A.columns[-1]
    return Matrix(A.columns[:-n], columns=True), Matrix(A.columns[-n:], columns=True)


def factorize(A):
    """Performs QR factorization on invertible Matrix A. Returns (Q, R).
    """
    if A.dim.rows != A.dim.columns:
        raise DimensionError("Matrix A must be square.")

    # Use Gram-Schidt to calculate orthogonal Matrix Q from columns of A
    Q = Matrix(vmath.gs(A.columns), columns=True)

    # If A is not invertible, vmath.gs will not return a square matrix
    if A.dim != Q.dim:
        raise DimensionError("Matrix A must be invertible.")

    # Upper triangular Matrix R = (Q^T)A
    R = transpose(Q) * A

    return Q, R


def rref(A):
    """Performs Gauss-Jordan Elimination, returns the reduced row
    echelon form of A.
    """
    # This will be very inefficient with immutable matrices and vectors
    m, n = A.dim
    r, c = 0, 0
    not_zero = lambda x: not math.isclose(x, 0, abs_tol=1e-15)

    while r < m and c < n:
        try:
            # Get the first unprocessed row index of a nonzero element in column c
            nz = next(i for i, x in enumerate(A.columns[c][r:]) if not_zero(x)) + r
            # Swap that row with row r
            A = row_swap(A, r, nz)
        except StopIteration:
            # If there are no nonzero elements left, skip column
            c += 1
            continue

        # Divide row r by its leading element so it becomes 1
        assert not_zero(A[r][c])
        A = row_multiply(A, r, 1 / A[r][c])

        # Eliminate nonzero elements of all other rows in column c
        for i in (x for x in range(m) if x != r):
            if not_zero(A[i][c]):
                A = row_add_mul(A, r, i, -1 * A[i][c])

        # Go to next row and column
        r += 1
        c += 1

    return A


def get_transformation_matrix(T, B, C):
    """Returns the transformation Matrix for T with respect to domain
    basis B and codomain basis C.
    """
    W = Matrix(C, columns=True)
    result_columns = (deaugment(rref(augment(W, v)), 1)[1] for v in transform_all(T, B))
    return Matrix(result_columns, columns=True)


def change_basis(v, B):
    """Returns Vector v in terms of basis Vector set B.
    """
    A = Matrix(*B, v, columns=True)
    _, vb = deaugment(rref(A), 1)
    return vb


def transform_all(A, V):
    """Returns an iterator that applies Matrix A to each Vector in V.
    """
    for v in V:
        yield A * v


def compose(M):
    """Returns the resulting Matrix from multiplying the Matrices in
    iterable M together. The result is M[0] * M[1] * ... * M[len(M) - 1]
    """
    return functools.reduce(operator.mul, M)


def row_space(A, is_rref=False):
    """Returns an iterator of Vectors forming a basis for the row space of Matrix A.
    """
    R = A if is_rref else rref(A)
    for row in R:
        if row:
            yield row


def column_space(A):
    """Returns an iterator of Vectors forming a basis for the column space of Matrix A.
    """
    return row_space(transpose(A))


def row_swap(A, row1, row2):
    """Swaps row1 with row2 and returns a new Matrix.
    """
    if row1 == row2:
        return A
    rows = list(A.rows)
    rows[row1], rows[row2] = rows[row2], rows[row1]
    return Matrix(rows)


def row_add(A, row1, row2):
    """Adds row1 to row2 and returns a new Matrix.
    """
    rows = list(A.rows)
    rows[row2] = rows[row1] + rows[row2]
    return Matrix(rows)


def row_multiply(A, row, k):
    """Multiplies row by scalar k and returns a new Matrix.
    """
    if k == 1:
        return A
    rows = list(A.rows)
    rows[row] = k * rows[row]
    return Matrix(rows)


def row_add_mul(A, row1, row2, k):
    """Adds a multiple of row1 to row2 and returns a new Matrix.
    """
    if k == 0:
        return A
    rows = list(A.rows)
    rows[row2] = k * rows[row1] + rows[row2]
    return Matrix(rows)
