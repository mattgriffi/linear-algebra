"""This file defines some useful functions for performing calculations
with Matrices.
"""

import itertools
import math

from matrix import Matrix, check_dimensions
from vector import Vector


def transpose(A):
    """Calculates the transpose of Matrix A.
    """
    return Matrix(A.rows)


def trace(A):
    """Calculates the sum of the leading diagonal of Matrix A.
    """
    return sum(A[i][i] for i in range(min(A.dimension())))
