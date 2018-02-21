"""This file defines some useful functions for performing calculations
with Matrices.
"""

import vmath
from matrix import Matrix, DimensionError


def transpose(A):
    """Calculates the transpose of Matrix A.
    """
    return Matrix(A.columns)


def trace(A):
    """Calculates the sum of the leading diagonal of Matrix A.
    """
    return sum(A[i][i] for i in range(min(A.dim)))


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
