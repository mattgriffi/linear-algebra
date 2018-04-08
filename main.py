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
        (1, 2, 1),
        (2, 1, 1),
        (1, 1, 2)
    )

    printf(mmath.eig(A, n=5))


def printm(A):
    print(A)
    print()


def printf(A):
    print(A.str_fractions())
    print()


if __name__ == "__main__":
    main()
