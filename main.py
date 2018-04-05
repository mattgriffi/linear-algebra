"""
This program allows for various linear algebra calculations.
"""


import math

import mmath
import vmath
from matrix import Matrix
from vector import Vector


def main():
    A = Matrix(
        (1, 4, 5, 3),
        (5, 22, 27, 11),
        (6, 19, 27, 31),
        (5, 28, 35, -8)
    )
    L, U = mmath.factor_LU(A)
    printm(L)
    printm(U)
    printm(L * U)


def printm(A):
    print(A)
    print()


def printf(A):
    print(A.str_fractions())
    print()


if __name__ == "__main__":
    main()
