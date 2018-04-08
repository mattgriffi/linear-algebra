"""
This program allows for various linear algebra calculations.
"""


import cProfile
import math
import pstats

import mmath
import vmath
from matrix import Matrix
from vector import Vector


def main():
    A = Matrix(
        (1, 0, 4),
        (0, 4, 0),
        (3, 5, -3)
    )

    eigenvalues = mmath.eig(A, precision=0)
    printm(eigenvalues)
    printm(mmath.poly(eigenvalues))


def printm(A):
    print(A)
    print()


def printf(A):
    print(A.str_fractions())
    print()


if __name__ == "__main__":
    main()
