"""This file defines some useful functions for performing calculations
with Matrices.
"""

import math

from contextlib import suppress

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
    """Performs Gauss-Jordan Elimination return the reduced row
    echelon form of A.
    """
    # This will be very inefficient with immutable matrices and vectors
    m, n = A.dim
    r, c = 0, 0
    not_zero = lambda x: not math.isclose(x, 0, abs_tol=1e-15)

    while r < m and c < n:
        # TODO make this less bad

        # If column c is all 0, skip it
        if not A.columns[c]:
            c += 1
            continue

        with suppress(StopIteration):
            # Get the row index of the first non-zero element of column c
            # below row r
            nz = next(i for i, x in enumerate(A.columns[c][r + 1:]) if not_zero(x))
            # Swap that row with row r
            A = row_swap(A, r, nz)

        # Divide row r by its leading element so it becomes 1
        A = row_multiply(A, r, 1 / A[r][c])

        # Eliminate nonzero elements of all other rows in column c
        for i in (x for x in range(m) if x != r):
            if not_zero(A[i][c]):
                val_to_eliminate = A[i][c]
                A = row_multiply(A, r, -1 * val_to_eliminate)
                A = row_add(A, r, i)
                A = row_multiply(A, r, -1 / val_to_eliminate)

        # Go to next row and column
        r += 1
        c += 1

    return A


def row_swap(A, row1, row2):
    """Swaps row1 with row2 and returns a new Matrix.
    """
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
    rows = list(A.rows)
    rows[row] = k * rows[row]
    return Matrix(rows)
