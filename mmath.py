"""This file defines some useful functions for performing calculations
with Matrices.
"""

import itertools
import math

import vmath
from matrix import Matrix, check_dimensions
from vector import Vector


def transpose(A):
    """Calculates the transpose of Matrix A.
    """
    return Matrix(A.columns)


def trace(A):
    """Calculates the sum of the leading diagonal of Matrix A.
    """
    return sum(A[i][i] for i in range(min(A.dimension())))


def factorize(A):
    """Performs QR factorization on invertible Matrix A. Returns (Q, R).
    """
    # Use Gram-Schidt to calculate orthogonal Matrix Q from columns of A
    Q = Matrix(vmath.gs(A.columns), columns=True)
    # Upper triangular Matrix R = (Q^T)A
    R = transpose(Q) * A
    return Q, R
