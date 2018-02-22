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
        (1, 0, 4, 5, 6, -12),
        (0, 0, 6, 3, 2, 0),
        (-3, 0, -10, -2, 2, -3),
        (1, 0, 1, 0, 0, 3),
        (0, 0, 1, 0, 0, 3),
        (1, 0, 1, -3, 0, 3),
        (0, 0, 0, 0, 10, 0),
        (3, 0, 1, -7, 0, 3),
    )

    printm(mmath.rref(A))


def printm(A):
    print(A)
    print()


if __name__ == "__main__":
    main()
